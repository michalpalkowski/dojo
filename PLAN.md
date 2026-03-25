# Sound Realm Sharding — Full Integration Plan

## Overview

Implement a complete, sound, gas-efficient sharding pipeline across four repositories: **dojo core** (slot ownership verification + ShardSettled event), **sharding operator** (per-slot metadata in settlement calls), **Eternum game contracts** (generic realm scope declaration + entity enumeration), and **Torii fork** (ShardSettled event processor). The system guarantees that no settled slot can escape entity lock scope, Torii sees all settled changes, and the game can generically declare what belongs to a shard.

## Goals

- **Soundness**: On-chain verification that every settled slot belongs to a locked entity (hash proof + lock check)
- **Torii visibility**: All settled values visible to clients via new `ShardSettled` event + custom Torii processor
- **Gas efficiency**: Total settlement < 300K steps (target: ~70K for 13 changed slots)
- **Generic realm scope**: Game declares scope at runtime — no hardcoded models in dojo core
- **Pre-allocation**: Building positions pre-locked deterministically so new entities created on shard are covered
- **Global state safety**: Shared models (Market, Liquidity) use Add CRDT without locking — concurrent shard access safe

## Non-Goals

- Changing Torii upstream (we use our fork at ~/Repos/torii)
- Reverting to baseline — we build on current `feature/dynamic-structures` branch
- Hardcoding Eternum-specific logic in dojo core — game declares scope generically
- Full production TEE flow changes — slot verification is orthogonal to SP1/attestation

## Assumptions and Constraints

- **Gas budget**: < 300K steps per settlement invoke on default Katana
- **Building grid**: ~30-50 positions per realm (hex grid, max_level ~4-5, center at 10,10)
- **Changed slots**: Typically ~10-20 per settlement (out of ~600+ committed)
- **Entity lock covers all models**: Locking entity_id X blocks writes to ALL models for entity X
- **Operator is trusted but verified**: Operator provides metadata, contract verifies cryptographically
- **Torii fork**: ~/Repos/torii — we control it fully

## Requirements

### Functional

- F1: Settlement verifies each changed slot belongs to a locked entity via `compute_dojo_field_slot` / `compute_dojo_packed_slot` hash recomputation
- F2: Settlement emits `ShardSettled` event with per-slot (model_selector, entity_id, member_selector, value) data
- F3: Torii fork processes `ShardSettled` event and updates its index identically to `StoreUpdateMember`
- F4: Operator computes and passes per-slot metadata (model, entity, member) alongside settlement data
- F5: Game declares shard scope generically: list of entity_ids (including pre-allocated positions)
- F6: Global/shared models use Add CRDT, are NOT locked, deltas merged at settlement
- F7: `settle_dev` (dev mode) also performs slot ownership verification + emits ShardSettled

### Non-Functional

- NF1: Settlement gas < 300K steps (currently ~50K, adding ~20K for verification + events = ~70K)
- NF2: No changes to existing Torii event processing (StoreSetRecord etc. untouched)
- NF3: Backward compatible — existing tests continue to pass
- NF4: Slot ownership verification is not bypassable (no dev-mode skip)

## Technical Design

### Data Model

#### New/Modified On-Chain Storage (dojo core)

No new storage needed. Slot ownership is verified by recomputing hashes — stateless check.

#### New Event (dojo core)

```cairo
#[derive(Drop, starknet::Event)]
pub struct ShardSettled {
    #[key]
    pub shard_id: felt252,
    pub model_selectors: Span<felt252>,    // per changed slot
    pub entity_ids: Span<felt252>,          // per changed slot
    pub member_selectors: Span<felt252>,    // per changed slot (0 = packed)
    pub values: Span<felt252>,              // per changed slot (final value after CRDT merge)
}
```

#### Operator Data Flow Addition

```
SlotPlanEntry {
    slot_key: Felt,
    model_selector: Felt,
    entity_id: Felt,
    member_selector: Felt,  // 0 for packed
}
```

`slot_planner.rs` already computes (model, entity, field) per slot. We surface this mapping to the settlement pipeline.

### API Design

#### Modified: `IShardingSettlement.settle()` and `IShardingSettlementDev.settle_dev()`

```cairo
fn settle(
    ref self: T,
    shard_id: felt252,
    all_committed_keys: Span<felt252>,
    changed_indices: Span<u32>,
    changed_values: Span<felt252>,
    state_diff_hash: felt252,
    global_state_root: felt252,
    end_block_number: u64,
    // NEW: per-slot ownership metadata
    slot_model_selectors: Span<felt252>,
    slot_entity_ids: Span<felt252>,
    slot_member_selectors: Span<felt252>,
);

fn settle_dev(
    ref self: T,
    shard_id: felt252,
    all_committed_keys: Span<felt252>,
    changed_indices: Span<u32>,
    changed_values: Span<felt252>,
    end_block_number: u64,
    // NEW: per-slot ownership metadata
    slot_model_selectors: Span<felt252>,
    slot_entity_ids: Span<felt252>,
    slot_member_selectors: Span<felt252>,
);
```

#### New: Torii EventProcessor

```rust
pub struct ShardSettledProcessor;
// Implements EventProcessor<P> for StarkNet provider P
// Registered in initialize_event_processors() for ContractType::WORLD
```

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        GAME (Eternum)                            │
│  1. register_shard_policy(model, crdt, overrides)  [deploy]      │
│  2. request_sharding(all_realm_entity_ids)         [runtime]     │
│     └── includes pre-allocated building positions                │
└───────────────────────────┬──────────────────────────────────────┘
                            │ ShardingRequested event
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                      OPERATOR (Rust)                              │
│  3. Parse event → entity_ids, fork block                         │
│  4. Resolve model layouts (IModel.definition)                    │
│  5. Plan slots: entity_id × model × field → SlotPlanEntry[]      │
│  6. Read CRDT policies → partition Add/Set slots                 │
│  7. register_commitment(shard_id, H(sorted_slots), ...)          │
│  8. Fork Katana, game plays                                      │
│  9. State diff → changed keys                                    │
│ 10. Build settlement: intersect committed ∩ changed              │
│ 11. Per changed slot: lookup SlotPlanEntry → metadata             │
│ 12. settle(keys, indices, values, metadata)                      │
└───────────────────────────┬──────────────────────────────────────┘
                            │ On-chain transaction
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                     DOJO CORE (Cairo)                             │
│ 13. verify_commitment_and_diff (existing)                        │
│ 14. NEW: per-slot verify:                                        │
│     a. compute_dojo_field_slot(model, entity, member) == key     │
│     b. entity_lock.read(entity) == shard_id                      │
│ 15. Write value (CRDT merge: Set or Add delta)                   │
│ 16. NEW: emit ShardSettled(shard_id, models[], entities[],       │
│          members[], final_values[])                               │
│ 17. Unlock entities, clear shard state                           │
└───────────────────────────┬──────────────────────────────────────┘
                            │ ShardSettled event
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                      TORII (fork)                                │
│ 18. ShardSettledProcessor receives event                         │
│ 19. Per slot: lookup model from cache                            │
│ 20. Deserialize value, update entity in storage                  │
│ 21. Clients see settled data                                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Serial Dependencies (Must Complete First)

#### Phase 0: Dojo Core — Slot Ownership Verification + ShardSettled Event
**Prerequisite for:** All subsequent phases (operator, Eternum, Torii depend on new settle() signature)

| Task | Description | Output |
|------|-------------|--------|
| 0.1 | **Modify `apply_settle` in `component.cairo`** to accept per-slot metadata (`slot_model_selectors`, `slot_entity_ids`, `slot_member_selectors`) and verify ownership. For each changed slot: (a) recompute expected key via `compute_dojo_field_slot(model, entity, member)` or `compute_dojo_packed_slot(model, entity)` based on `member_selector != 0`, (b) assert `expected_key == actual_key`, (c) assert `entity_lock.read(entity_id) == shard_id`. | Modified `apply_settle` with ownership verification |
| 0.2 | **Add `ShardSettled` event** to `sharding_component::Event` enum. Emit after all writes complete, with final values (post-CRDT-merge). Collect final values in a separate array during the write loop. | `ShardSettled` event struct + emission in `apply_settle` |
| 0.3 | **Update `IContractComponent` trait** in `component.cairo` — add metadata params to `settle()`, `settle_dev()`, `end_shard()` (add `ShardSettled` to event enum). | Updated trait signatures |
| 0.4 | **Update `IShardingSettlement` and `IShardingSettlementDev` traits** in `world_sharding.cairo` — add 3 new Span params to `settle()` and `settle_dev()`. | Updated interface file |
| 0.5 | **Update `world_contract.cairo`** — wire new params through `ShardingSettlementImpl` and `ShardingSettlementDevImpl` to the component. | Updated world contract |
| 0.6 | **Update ALL existing tests** in `component.cairo`, `settlement_events.cairo`, `request.cairo` — add empty metadata spans `[].span()` to existing settle/settle_dev calls so they compile. Then add new tests for ownership verification. | All tests pass with new signature |
| 0.7 | **Add new tests for slot ownership verification**: (a) test_settle_rejects_wrong_model_selector — provide wrong model_selector, expect 'slot ownership mismatch', (b) test_settle_rejects_unlocked_entity — provide entity_id not in shard scope, expect 'entity not in shard', (c) test_settle_with_correct_metadata_succeeds — happy path with full metadata. | 3+ new tests passing |
| 0.8 | **Run `scarb test`** in `crates/dojo/core-tests` — all tests green. | CI-clean test results |

**Key implementation detail for 0.1** — The `apply_settle` write loop becomes:

```cairo
let mut final_values: Array<felt252> = ArrayTrait::new();
let mut i: u32 = 0;
while i < changes_len {
    let idx = *changed_indices[i];
    let key = *all_committed_keys[idx];
    let value = *changed_values[i];
    let model_sel = *slot_model_selectors[i];
    let entity_id = *slot_entity_ids[i];
    let member_sel = *slot_member_selectors[i];

    // ── SLOT OWNERSHIP VERIFICATION ──
    let expected_key = if member_sel != 0 {
        compute_dojo_field_slot(model_sel, entity_id, member_sel)
    } else {
        compute_dojo_packed_slot(model_sel, entity_id)
    };
    assert(expected_key == key, 'slot ownership mismatch');
    assert(self.entity_lock.read(entity_id) == shard_id, 'entity not in shard');

    // ── CRDT WRITE (existing logic) ──
    let storage_address: StorageAddress = key.try_into().unwrap();
    let final_value = if self.add_slot_flag.read((shard_id, key)) {
        // Add CRDT delta merge
        let current = storage_read_syscall(0, storage_address).unwrap_syscall();
        let initial = self.add_initial_values.read((shard_id, key));
        let delta = value.into() - initial.into();  // u256 math
        let sum = current.into() + delta;
        sum.try_into().expect('overflow')
    } else {
        value
    };
    storage_write_syscall(0, storage_address, final_value).unwrap_syscall();
    final_values.append(final_value);
    i += 1;
};

// ── EMIT SHARD SETTLED ──
self.emit(ShardSettled {
    shard_id,
    model_selectors: slot_model_selectors,
    entity_ids: slot_entity_ids,
    member_selectors: slot_member_selectors,
    values: final_values.span(),
});

self.unlock_entities(shard_id);
self.clear_shard_state(shard_id);
```

**Note on Add CRDT slots and entity lock bypass:** Global/shared models (Market, Liquidity) use Add CRDT. Their entities are NOT locked by any shard. The ownership check `entity_lock.read(entity_id) == shard_id` would FAIL for them. We need a bypass:

```cairo
// For Add CRDT slots: entity may not be locked (global state).
// Soundness comes from delta merge, not exclusive lock.
if self.add_slot_flag.read((shard_id, key)) {
    // Add CRDT: skip entity lock check — delta merge is safe without lock
} else {
    // Set CRDT: entity MUST be locked by this shard
    assert(self.entity_lock.read(entity_id) == shard_id, 'entity not in shard');
}
```

This is sound because Add CRDT computes `current + (shard_value - initial_value)`. Even if mainchain modified the value concurrently, the delta is applied correctly. No data loss.

---

### Parallel Workstreams

These workstreams can be executed independently after Phase 0.

#### Workstream A: Operator — Settlement Pipeline Update
**Dependencies:** Phase 0 (new settle signature)
**Can parallelize with:** Workstreams B, C

| Task | Description | Output |
|------|-------------|--------|
| A.1 | **Extend `SlotPlanEntry` in `slot_planner.rs`** — currently `plan_entity_slots` returns `Vec<Felt>` (slot keys). Change to return `Vec<SlotPlanEntry>` where `SlotPlanEntry { key: Felt, model_selector: Felt, entity_id: Felt, member_selector: Felt }`. The data is already computed during traversal — just capture it. | `SlotPlanEntry` struct, updated `plan_entity_slots` |
| A.2 | **Build slot→metadata lookup map** in settlement pipeline — after planning all entity slots, build `HashMap<Felt, SlotPlanEntry>` keyed by slot_key. This allows O(1) lookup when mapping state diff changed keys to their metadata. | `build_slot_metadata_map()` function |
| A.3 | **Update `build_settle_call` in `calls.rs`** — add 3 new `&[Felt]` params: `slot_model_selectors`, `slot_entity_ids`, `slot_member_selectors`. Encode as `[len, elements...]` in calldata after existing params. | Updated call builder + calldata encoding |
| A.4 | **Update `build_settle_dev_call` in `calls.rs`** — same 3 new params. | Updated dev call builder |
| A.5 | **Update `build_commitment_settlement_calls` in `settlement.rs`** — after computing `changed_values`, lookup each changed key in the metadata map to get `(model_selector, entity_id, member_selector)`. Pass these to the call builders. | Metadata wired through settlement pipeline |
| A.6 | **Update `CommitmentSettlementConfig`** — add `slot_metadata: HashMap<Felt, SlotPlanEntry>` field. Populate from slot planning step. | Updated config struct |
| A.7 | **Update DryRunSettler** in `production.rs` — pass metadata through `settle_dev` call path. | Dev mode works with ownership verification |
| A.8 | **Update tests in `calls.rs`** — add metadata to test calldata assertions. | Tests pass |
| A.9 | **Run `cargo test`** in sharding_operator — all tests pass. | CI-clean |

**Key detail for A.1** — Currently `plan_entity_slots` traverses the Layout tree recursively, computing field slots. At each leaf:

```rust
// CURRENT: just collect keys
entity_slots.push(compute_dojo_field_slot(model_selector, entity_id, field_selector));

// NEW: collect full metadata
entries.push(SlotPlanEntry {
    key: compute_dojo_field_slot(model_selector, entity_id, field_selector),
    model_selector,
    entity_id,
    member_selector: field_selector,  // 0 for packed
});
```

#### Workstream B: Torii Fork — ShardSettled Processor
**Dependencies:** Phase 0 (ShardSettled event definition)
**Can parallelize with:** Workstreams A, C

| Task | Description | Output |
|------|-------------|--------|
| B.1 | **Add `selector!("ShardSettled")` to `DOJO_RELATED_EVENTS`** in `~/Repos/torii/crates/indexer/engine/src/engine.rs` line 34-42. | Engine recognizes ShardSettled events |
| B.2 | **Create `shard_settled.rs`** processor in `~/Repos/torii/crates/processors/src/processors/`. Implement `EventProcessor` trait: parse event data (model_selectors[], entity_ids[], member_selectors[], values[]), iterate per-slot, look up model from cache, deserialize value, call `storage.set_entity()`. | New processor file |
| B.3 | **Register processor** in `~/Repos/torii/crates/processors/src/processors/mod.rs` `initialize_event_processors()` — add to WORLD contract type processors list. | Processor registered |
| B.4 | **Handle member vs packed updates** — if `member_selector != 0`, extract the specific member from model schema (like `StoreUpdateMember` does at lines 105-118 of `store_update_member.rs`). If `member_selector == 0`, treat as full packed update (like `StoreUpdateRecord`). | Correct member-level indexing |
| B.5 | **Add task dependencies** — `ShardSettled` depends on model registration (like `StoreUpdateRecord` does). Use `task_dependencies()` to return the model registration task ID. | Correct task ordering in parallel processing |
| B.6 | **Test with existing Torii test infrastructure** — write unit test for event parsing and processor logic. | Tests pass |

**Key detail for B.2** — The processor must handle the Span<felt252> encoding:

```rust
// Event data layout (Cairo Span encoding):
// [model_selectors_len, model_selectors...,
//  entity_ids_len, entity_ids...,
//  member_selectors_len, member_selectors...,
//  values_len, values...]

fn process(&self, ctx: &EventProcessorContext) -> Result<()> {
    let data = &ctx.event.data;
    let mut offset = 0;

    let models = parse_felt_span(data, &mut offset)?;
    let entities = parse_felt_span(data, &mut offset)?;
    let members = parse_felt_span(data, &mut offset)?;
    let values = parse_felt_span(data, &mut offset)?;

    for i in 0..models.len() {
        let model = ctx.cache.model(&ctx.contract_address, models[i])?;

        if members[i] != Felt::ZERO {
            // Member update — find member in schema, deserialize single value
            let member_schema = find_member_by_selector(&model.schema, members[i])?;
            let deserialized = member_schema.deserialize(&[values[i]])?;
            // Wrap in struct for storage.set_entity compatibility
            let wrapped = wrap_member_in_struct(&model.schema, members[i], deserialized);
            ctx.storage.set_entity(&model, entities[i], &wrapped).await?;
        } else {
            // Packed update — deserialize full entity from single packed felt
            let deserialized = model.schema.deserialize(&[values[i]])?;
            ctx.storage.set_entity(&model, entities[i], &deserialized).await?;
        }
    }
    Ok(())
}
```

#### Workstream C: Eternum Game — Realm Scope Declaration
**Dependencies:** Phase 0 (entity lock + verification guarantees scope correctness)
**Can parallelize with:** Workstreams A, B

| Task | Description | Output |
|------|-------------|--------|
| C.1 | **Create `realm_scope.cairo`** in Eternum sharding system — a helper that enumerates all entity_ids for a realm. Uses the game's knowledge of its own model structure. Takes `realm_entity_id` and returns `Array<felt252>` with: (a) realm entity_id itself, (b) all building positions on hex grid (pre-allocated), (c) StructureBuildings entity_id, (d) resource entity_ids, (e) any other realm-scoped entities. | `get_realm_entity_ids(world, realm_entity_id) -> Array<felt252>` |
| C.2 | **Implement hex grid pre-allocation** — given realm position `(outer_col, outer_row)` and max_level from config, enumerate all valid `(inner_col, inner_row)` positions on hex grid. Compute `entity_id_from_keys((outer_col, outer_row, inner_col, inner_row))` for each. These entity_ids are locked even if no building exists there yet. | `enumerate_building_positions(outer_col, outer_row, max_level) -> Array<felt252>` |
| C.3 | **Update `request_shard` in Eternum sharding contract** — call `get_realm_entity_ids()` internally, then `world.request_sharding(entities)`. The game caller just passes `realm_entity_id`. | `request_shard_realm(realm_entity_id)` |
| C.4 | **Register CRDT policies at deploy** — ensure `register_shard_policy` is called for ALL models that might be modified on a shard. At minimum: Structure(Set), Building(Set), StructureBuildings(Set), Resource(Add for amounts). Global models: Market(Add), Liquidity(Add). | Policy registration in Eternum deploy/init |
| C.5 | **Test realm scope enumeration** — verify the hex grid enumeration produces correct entity_ids, test that all e2e building positions are covered. | Cairo unit tests |
| C.6 | **Update e2e bot** in operator `src/e2e_bot/eternum/calls.rs` — change `request_shard_call` to use the new `request_shard_realm(realm_entity_id)` signature instead of passing raw entity_ids. | Updated e2e bot |

**Key detail for C.2** — Hex grid enumeration using BFS from center:

```cairo
fn enumerate_building_positions(
    outer_col: u32,
    outer_row: u32,
    max_rings: u32,
) -> Array<felt252> {
    let mut positions: Array<felt252> = ArrayTrait::new();
    let center = BuildingImpl::center(); // Coord { x: 10, y: 10 }

    // BFS or iterative ring enumeration
    // Ring 0 = center (excluded — buildings can't be at center)
    // Ring 1..=max_rings = all hex neighbors at each distance
    // For each position, compute entity_id
    let mut visited: Felt252Dict<bool> = Default::default();
    let mut queue: Array<Coord> = ArrayTrait::new();
    queue.append(center);
    visited.insert(coord_hash(center), true);

    // BFS outward, tracking ring depth
    // Each ring: take all current positions, compute their 6 neighbors
    // Stop after max_rings rings
    // For each non-center position: append entity_id
    // ...

    // For each valid position:
    let entity_id = entity_id_from_serialized_keys(
        [outer_col.into(), outer_row.into(), pos.x.into(), pos.y.into()].span()
    );
    positions.append(entity_id);

    positions
}
```

**Gas analysis for C.2**: For max_level=5 (6 rings), ~90 hex positions. Each `entity_id_from_serialized_keys` = 1 Poseidon hash. ~90 hashes ≈ ~45K steps. Plus ~90 entity locks in `request_sharding` ≈ ~90K steps. Total request_sharding ≈ ~135K steps. Within 300K budget.

---

### Merge Phase

After parallel workstreams complete, these tasks integrate the work.

#### Phase N: End-to-End Integration
**Dependencies:** Workstreams A, B, C

| Task | Description | Output |
|------|-------------|--------|
| N.1 | **Integration test: dojo core + operator** — write a test that: (a) deploys world + Foo model, (b) requests sharding with entities, (c) registers commitment with metadata, (d) settles with correct metadata → values written + ShardSettled emitted. Verify event data matches expected. | Dojo-level integration test |
| N.2 | **Integration test: operator builds correct calldata** — unit test that mocks the full pipeline: plan slots → compute metadata map → build settle call → verify calldata encoding matches what Cairo expects. | Operator calldata test |
| N.3 | **E2E test: full pipeline** — run the e2e bot with updated Eternum: (a) deploy Eternum with policies, (b) request_shard_realm(realm_a), (c) create buildings on shard, (d) finish shard, (e) settle with metadata, (f) verify via Torii that buildings appear on mainchain. | E2E test passing |
| N.4 | **Torii integration test** — verify that Torii processes ShardSettled events: (a) index a world with sharding, (b) settle a shard, (c) query Torii SQL for settled entities, (d) verify values match. | Torii integration test |
| N.5 | **Gas measurement** — run `scarb test` with gas reporting on settlement tests. Verify total < 300K steps for realistic workloads (13 changed slots with metadata). | Gas report |
| N.6 | **Update PLAN.md** — mark completed, document any deviations. | Updated plan |

---

## Testing and Validation

### Unit Tests (per workstream)

**Dojo Core (Phase 0):**
- `test_settle_rejects_wrong_model_selector` — wrong metadata → panic
- `test_settle_rejects_unlocked_entity` — entity not in shard → panic
- `test_settle_with_correct_metadata_succeeds` — happy path
- `test_settle_add_crdt_skips_lock_check` — Add CRDT slot with unlocked entity works
- `test_settle_emits_shard_settled_event` — verify event data
- All existing tests updated with empty metadata spans

**Operator (Workstream A):**
- `test_slot_plan_entry_captures_metadata` — SlotPlanEntry has correct fields
- `test_build_settle_call_encodes_metadata` — calldata format correct
- `test_metadata_map_lookup` — changed key → correct SlotPlanEntry

**Torii (Workstream B):**
- `test_parse_shard_settled_event` — event data parsing
- `test_processor_updates_entity` — storage.set_entity called correctly

**Eternum (Workstream C):**
- `test_enumerate_building_positions` — correct hex grid enumeration
- `test_realm_scope_includes_all_entities` — no entities missing
- `test_request_shard_realm_locks_all` — all entity_ids locked

### End-to-End Test

Full pipeline: Eternum deploy → register policies → request_shard_realm → play on shard (create building) → finish_shard → settle with metadata → Torii shows building on mainchain.

## Rollout and Migration

### Rollout Order
1. Dojo core changes (Phase 0) — must be first, defines new ABI
2. Operator changes (Workstream A) — adapts to new ABI
3. Torii fork changes (Workstream B) — independent, deploy when ready
4. Eternum changes (Workstream C) — uses new request_shard_realm

### Migration
- **Breaking change**: `settle()` and `settle_dev()` signatures change (3 new params)
- All callers must be updated simultaneously (operator + tests)
- No data migration needed (stateless verification)

### Rollback Plan
- Baseline commit 523e7b7 in sharding_operator works
- Git revert to current `feature/dynamic-structures` state if needed
- Dual-mode strategy available (commitment_mode flag) per PLAN.md history

## Verification Checklist

- [x] `cd ~/Repos/sharding_operator/externals/dojo/crates/dojo/core-tests && scarb test` — 450 passed, 0 failed
- [x] `cd ~/Repos/sharding_operator && cargo test --lib` — 330 passed, 0 failed
- [x] `cd ~/Repos/torii && cargo check -p torii-processors` — compiles clean
- [x] `cd ~/Repos/sharding_operator/externals/eternum/contracts/game && scarb check` — compiles clean
- [x] Ownership verification: 3 new tests (wrong model selector, unlocked entity, metadata length) — all pass
- [x] Add CRDT: global model slot settled without entity lock — test passes
- [x] Torii fork built: `~/Repos/torii/target/release/torii` (commit 942bcb46)
- [x] Torii fork pushed: `https://github.com/michalpalkowski/torii.git` branch `feature/sharding`
- [ ] Gas measurement: settlement with 13 changed slots + metadata < 300K steps (needs runtime measurement)
- [ ] E2E: building created on shard visible in Torii after settlement (needs full deployment test)

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Calldata size for metadata (3×13 = 39 extra felts) approaches tx limits | Low | Med | 39 felts ≈ 1.2KB. Starknet tx limit is 128KB. No issue. |
| Hex grid enumeration misses edge positions | Med | High | Exhaustive test: create buildings at all positions, verify entity_ids in scope |
| Torii ShardSettled parsing breaks on multi-felt members | Med | Med | Start with single-felt members (felt252 fields). Extend to multi-felt if needed. |
| Gas estimate wrong for large realms (100+ entities) | Low | Med | Pre-allocation is one-time. Measure actual gas in test. Fallback: smaller scope. |
| CRDT bypass for Add slots could be exploited | Low | High | Add slots are registered at commit time (controlled by operator, verified on-chain). Operator is owner-only. |
| Poseidon hash mismatch between Rust and Cairo | Low | Critical | Already verified in previous session. Existing tests cover this. |
| Building positions change between Eternum versions | Med | Med | Pre-allocation is game-specific (Eternum contract). Version pinned in externals/. |

## Open Questions

- [ ] **Multi-felt members**: Some model fields span multiple felt252 values (u256, structs). Current plan assumes 1 value per slot. Need to verify that all shardable fields are single-felt, or extend ShardSettled to support multi-felt member values.
- [ ] **Building grid exact size**: Need to verify max_level config value in Eternum test deployment. This determines pre-allocation count and gas.
- [ ] **Resource model key structure**: Need to verify Resource model keys to include in realm scope enumeration. Is it `(entity_id, resource_type)` or just `(entity_id)`?
- [ ] **Event size limits**: ShardSettled event carries 4 Span<felt252>. For 13 changed slots = ~52 felts in event data. Verify Starknet event size limits.
- [ ] **Torii member deserialization**: Does `storage.set_entity()` with a partial struct (single member) work correctly for all model types? Verify with Torii's existing `StoreUpdateMember` behavior.

## Decision Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Slot ownership via hash recomputation | Stateless, no extra storage, cryptographically sound. Operator can't forge — hash verification on-chain. | (a) Store slot→entity mapping on-chain — too expensive. (b) Trust operator without verification — not sound. |
| Add CRDT slots bypass entity lock check | Delta merge is mathematically safe without exclusive access. Locking globals would block concurrent shards. | (a) Lock global entities — kills concurrent shard access. (b) Exclude globals from settlement — game can't modify Market on shard. |
| ShardSettled as single event (not per-slot) | One event is cheaper than 13 StoreUpdateMember events. Custom Torii processor handles batched data. | (a) Emit StoreUpdateMember per slot — works without Torii changes but more gas + pollutes existing event stream. (b) Emit ShardSettled per entity — grouping complexity for unclear benefit. |
| Pre-allocate building positions via hex BFS | Deterministic, covers new entities created on shard. ~90 entity locks ≈ ~90K steps — within budget. | (a) Dynamic lock extension — extra on-chain call, complexity. (b) Skip pre-allocation — new buildings on shard can't be verified. |
| Generic realm scope (game declares, not dojo core) | Dojo core should be game-agnostic. Game knows its own structure. | (a) Dojo core knows about "realms" — coupling. (b) Operator discovers scope — fragile, depends on Torii queries. |
| Custom Torii fork processor (not upstream events) | Cleaner architecture, one batched event vs N individual events. We control the fork. | (a) Emit standard StoreUpdateMember — no Torii changes but more gas and semantic confusion (settlement looks like normal writes). |
