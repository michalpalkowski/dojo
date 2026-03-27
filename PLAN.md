# CRDT Variant Semantic Refactor — Implementation Plan

**Date:** 2026-03-27
**Branch:** `feature/sharding`
**ADR:** ADR-011 (knowladge-base/systems/sharding/decisions.md)

## Overview

Restructure the four CRDT variants (Set, Add, SetLock, Lock) to properly separate **concurrent** vs **exclusive** locking semantics. Currently, all variants use exclusive `entity_lock`, which makes Add CRDT degenerate to Set (no concurrent writes possible → delta merge is pointless). After this change, Set (LWW) and Add become concurrent (shared lock, no entity_lock), while SetLock and Lock remain exclusive.

## Goals

- **Concurrent Set (LWW)**: Multiple shards can overwrite the same field without entity_lock. Last settle wins.
- **Concurrent Add**: Multiple shards can delta-merge the same field. Mainnet writes allowed during shard — deltas combine correctly.
- **SetLock = current Set behavior**: Exclusive entity_lock, overwrite. Most common variant. No behavioral change, just renamed semantic.
- **Lock = freeze**: Exclusive entity_lock, but reject any settlement slots for that entity. Guarantees immutability.
- **Safety enforced at settle time**: Shared entities can only submit Set/Add CRDT slots (never SetLock/Lock).

## Non-Goals

- Changing SP1 proof structure (initial_value proving is a separate issue tracked as security gap)
- Changing Torii event handling (ShardSettled processor is unaffected)
- Backward compatibility with old encoding (this is a breaking change to CRDT semantics)

## Assumptions and Constraints

- **Gas budget**: Shared lock operations (reference counting) must fit within Katana invoke limits
- **Encoding change**: Cairo enum variant order stays the same (Set=0, Add=1, Lock=2, SetLock=3) but the `encode_crdt` values may change (currently 1=Set, 2=Add, 3=Lock, 4=SetLock)
- **request_shard API change**: Adds `shared_entities` parameter — breaking change for all callers
- **Entity classification is caller responsibility**: The game contract decides which entities are shared vs exclusive based on its CRDT policy knowledge

## Spread Assessment

### Cairo (dojo externals) — 5 files

| File | Changes | Impact |
|------|---------|--------|
| `crates/dojo/core/src/sharding/request.cairo` | CRDVariant enum: no structural change, semantic change only | Low |
| `crates/dojo/core/src/sharding/component.cairo` | **HEAVY**: New storage (shared lock maps), `request_shard` signature + logic, `verify_slot_ownership` dual-mode, `write_slot_with_crdt` Lock rejection, `unlock_entities` shared cleanup, `clear_shard_state` shared cleanup | High |
| `crates/dojo/core/src/world/world_contract.cairo` | `is_entity_locked` behavior unchanged (shared entities don't set entity_lock, so no code change needed) | None |
| `crates/dojo/core-tests/src/tests/sharding/component.cairo` | Update all `request_shard` calls (add empty shared_entities), add new tests for concurrent locking | Medium |
| `examples/spawn-and-move/src/sharding_systems.cairo` | Update `request_shard` calls, update `register_policies` to use SetLock where currently Set, add concurrent access tests | Medium |

### Rust (sharding operator) — 8+ files

| File | Changes | Impact |
|------|---------|--------|
| `src/models/crdts.rs` | CRDT enum: semantics change but structure stays. `is_add()` stays. Add `is_concurrent()` method (true for Add and Set) | Low |
| `src/models/storage_slots_file.rs` | CrdtTypeName enum: no change needed (names stay, semantics change) | None |
| `src/operator/service/policy_reader.rs` | Constants stay same. Add `partition_concurrent_slots()` to separate shared vs exclusive entities. `read_add_initial_values_lazy` also needs to read initial values for concurrent Set slots? No — Set doesn't need initial_value. Only Add does. So no change to initial value reading. | Medium |
| `src/shard/settlement.rs` | SlotEntry construction: no change (initial_value logic already correct for Add vs non-Add). But need to validate Lock slots aren't submitted. | Low |
| `src/shard/calls.rs` | `build_settle_call`: no change to SlotEntry serialization. But `request_shard` call builder needs `shared_entities` param. | Medium |
| `src/e2e_bot/eternum/calls.rs` | `request_shard_call` and `request_shard_realm_call`: add `shared_entities` param | Low |
| `src/client/request_sharding/mod.rs` | `run_request_sharding`: add shared_entities to calldata | Low |
| `src/operator/service/shard_task/lifecycle/gameplay.rs` | `enable_fork_mode`: no change (only cares about active shards) | None |

### Knowledge Base — 3 files (DONE)

| File | Status |
|------|--------|
| `systems/sharding/crdts-and-storage.md` | Updated — new CRDT variant descriptions with concurrent/exclusive semantics |
| `systems/sharding/decisions.md` | Updated — ADR-011 added |
| `systems/sharding/glossary.md` | Updated — entity lock (exclusive) and entity lock (shared) entries |

### Game Contracts (Eternum/spawn-and-move) — 2+ files

| File | Changes | Impact |
|------|---------|--------|
| `examples/spawn-and-move/src/sharding_systems.cairo` | `register_policies`: Set→SetLock for entity-locked models, keep Set for concurrent models. `request_shard`: split entities into exclusive/shared lists. | Medium |
| Eternum sharding systems (external) | Same pattern: SetLock for realm entities, Set/Add for global counters | Medium |

---

## Technical Design

### New Storage (component.cairo)

```cairo
#[storage]
pub struct Storage {
    // ... existing fields ...

    /// Shared lock: entity_id → count of active shards with concurrent access.
    entity_shared_count: Map<felt252, u32>,
    /// Shared shard membership: (entity_id, shard_id) → true if shard has shared access.
    entity_shared_shard: Map<(felt252, felt252), bool>,
    /// Shared entity tracking for unlock: (shard_id, index) → entity_id.
    shard_shared_entities: Map<(felt252, u32), felt252>,
    shard_shared_entity_count: Map<felt252, u32>,
}
```

### Modified `request_shard` Signature

```cairo
fn request_shard(
    ref self: ComponentState<TContractState>,
    entities: Span<felt252>,           // exclusive (SetLock/Lock) — entity_lock
    shared_entities: Span<felt252>,    // concurrent (Set/Add) — shared ref-count
    entity_keys_flat: Span<felt252>,
) -> felt252;
```

### Locking Logic

```
exclusive_entities:
  assert(entity_lock[id] == 0, ENTITY_ALREADY_SHARDED)
  assert(entity_shared_count[id] == 0, 'entity has shared locks')
  entity_lock[id] = shard_id

shared_entities:
  assert(entity_lock[id] == 0, ENTITY_ALREADY_SHARDED)  // no exclusive lock
  entity_shared_count[id] += 1
  entity_shared_shard[(id, shard_id)] = true
```

### Modified `verify_slot_ownership`

```
exclusive_lock = entity_lock[entity_id]
if exclusive_lock == shard_id:
    return OK  // exclusive — any CRDT allowed

if entity_shared_shard[(entity_id, shard_id)]:
    effective_crdt = lookup_crdt(model_selector, member_selector)
    assert(effective_crdt == Set or effective_crdt == Add, 'shared needs concurrent CRDT')
    return OK

panic('entity not in shard')
```

### Modified `write_slot_with_crdt`

```cairo
if is_lock:
    panic('Shard: Lock entity cannot be settled')  // NEW
elif is_add:
    // delta merge (unchanged)
else:
    // overwrite (Set and SetLock — unchanged)
```

### `is_entity_locked` — NO CHANGE

```cairo
fn is_entity_locked(self, entity_id) -> bool {
    self.entity_lock.read(entity_id) != 0
    // shared entities don't set entity_lock → returns false → mainnet writes allowed
}
```

---

## Implementation Plan

### Serial Dependencies (Must Complete First)

#### Phase 0: Dojo Core — Shared Lock Storage + request_shard
**Prerequisite for:** All subsequent phases

| Task | Description | Output |
|------|-------------|--------|
| 0.1 | Add shared lock storage fields to `component.cairo` Storage struct | 4 new Map fields |
| 0.2 | Modify `request_shard` signature: add `shared_entities: Span<felt252>` parameter | Updated trait + impl |
| 0.3 | Implement dual locking in `request_shard`: exclusive for `entities`, ref-counted for `shared_entities`. Mutual exclusion: exclusive blocks shared, shared blocks exclusive. | Locking logic |
| 0.4 | Update `IContractComponent` trait in `component.cairo` | Updated interface |
| 0.5 | Update ALL existing `request_shard` call sites to add empty `[].span()` for shared_entities | Tests compile |
| 0.6 | Run `scarb test` — all existing tests pass with new signature | Green tests |

#### Phase 1: Dojo Core — Settlement Changes
**Prerequisite for:** Operator changes

| Task | Description | Output |
|------|-------------|--------|
| 1.1 | Modify `verify_slot_ownership` to accept both exclusive and shared modes. For shared: check `entity_shared_shard`, enforce Set/Add CRDT only. | Dual verification |
| 1.2 | Modify `write_slot_with_crdt` to reject Lock CRDT slots (panic) | Lock freeze enforcement |
| 1.3 | Modify `unlock_entities` to also unlock shared entities (decrement count, clear membership) | Shared unlock |
| 1.4 | Modify `clear_shard_state` to also clear shared entity tracking | Shared cleanup |
| 1.5 | Add helper: `resolve_effective_crdt(model_selector, member_selector) -> CRDVariant` for use in `verify_slot_ownership` | CRDT resolution helper |
| 1.6 | Run `scarb test` | Green tests |

---

### Parallel Workstreams

#### Workstream A: Dojo Tests + Examples
**Dependencies:** Phase 0, Phase 1
**Can parallelize with:** Workstream B

| Task | Description | Output |
|------|-------------|--------|
| A.1 | Update `sharding_systems.cairo` `register_policies`: change `CRDVariant::Set` to `CRDVariant::SetLock` for entity-locked models (Moves, Position, MockToken). Keep `CRDVariant::Add` overrides as-is. | Updated policies |
| A.2 | Add test: `test_concurrent_set_allows_multiple_shards` — two shards lock same entity as shared, both succeed | New test |
| A.3 | Add test: `test_concurrent_add_allows_multiple_shards` — two shards delta-merge same counter | New test |
| A.4 | Add test: `test_shared_entity_rejects_setlock_slot` — shared entity tries to settle SetLock slot, panics | New test |
| A.5 | Add test: `test_lock_entity_rejects_settlement` — Lock entity tries to settle, panics | New test |
| A.6 | Add test: `test_exclusive_blocks_shared` — exclusive-locked entity can't be shared-locked by another shard | New test |
| A.7 | Add test: `test_shared_blocks_exclusive` — shared-locked entity can't be exclusive-locked | New test |
| A.8 | Update `test_add_entity_still_exclusively_locked` — rename to `test_mixed_entity_still_exclusively_locked`, update comment explaining WHY (mixed Set+Add fields require exclusive) | Updated test |
| A.9 | Add test: `test_mainnet_writes_allowed_for_shared_entities` — write to shared entity on mainnet doesn't panic | New test |
| A.10 | Run `scarb test` in core-tests and examples | All green |

#### Workstream B: Operator Rust Changes
**Dependencies:** Phase 0 (new request_shard signature)
**Can parallelize with:** Workstream A

| Task | Description | Output |
|------|-------------|--------|
| B.1 | Add `is_concurrent()` method to `CRDT` enum in `crdts.rs`: returns true for `Add` and `Set` | New method |
| B.2 | Add `partition_concurrent_entities()` to `policy_reader.rs`: given entity_ids and their model policies, split into exclusive (any SetLock/Lock field) and shared (all Set/Add fields) | New function |
| B.3 | Update `request_shard_call` in `e2e_bot/eternum/calls.rs`: add `shared_entities` param (empty for now — Eternum uses SetLock for everything) | Updated call builder |
| B.4 | Update `run_request_sharding` in `client/request_sharding/mod.rs`: add shared_entities to calldata | Updated client |
| B.5 | Update encoding constants comment in `policy_reader.rs` to document new semantics | Doc update |
| B.6 | Run `cargo test --lib` | All green |

---

### Merge Phase

#### Phase N: Integration
**Dependencies:** Workstreams A, B

| Task | Description | Output |
|------|-------------|--------|
| N.1 | E2E integration test: deploy world with concurrent Add model, two shards lock same entity as shared, both settle with delta merge, final value = initial + delta_A + delta_B | E2E test |
| N.2 | Verify gas: shared lock operations (ref-count increment/decrement) within Katana limits | Gas report |
| N.3 | Update knowledge base if any design changed during implementation | Docs current |

---

## Testing and Validation

### New Tests (Workstream A)

1. `test_concurrent_set_allows_multiple_shards` — happy path concurrent Set
2. `test_concurrent_add_allows_multiple_shards` — happy path concurrent Add with delta merge
3. `test_shared_entity_rejects_setlock_slot` — safety: shared entity + SetLock → panic
4. `test_lock_entity_rejects_settlement` — safety: Lock entity + any slot → panic
5. `test_exclusive_blocks_shared` — mutual exclusion: exclusive then shared → panic
6. `test_shared_blocks_exclusive` — mutual exclusion: shared then exclusive → panic
7. `test_mainnet_writes_allowed_for_shared_entities` — is_entity_locked returns false for shared

### Existing Tests (Updated)

All existing `request_shard` calls get `[].span()` as shared_entities. All existing `register_shard_policy` calls with `CRDVariant::Set` for entity-locked models change to `CRDVariant::SetLock`.

## Verification Checklist

- [ ] `cd crates/dojo/core-tests && scarb test` — all pass
- [ ] `cd examples/spawn-and-move && scarb test` — all pass
- [ ] `cd ~/Repos/sharding_operator && cargo test --lib` — all pass
- [ ] Two shards can lock same entity with Add CRDT and both settle successfully
- [ ] Lock entity rejects any settlement slot
- [ ] Shared entity rejects SetLock/Lock settlement slot
- [ ] Mainnet writes work for shared-locked entities
- [ ] Gas: request_shard with 10 shared entities < 100K steps

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `initial_value` not proven for concurrent Add — operator can inflate delta | Known | High | Documented security gap. Acceptable for counters. Phase 2: prove initial_value in SP1. |
| Set LWW causes data loss (last settle wins) | By design | Med | Developer must consciously choose Set over SetLock. Document clearly. |
| Shared lock storage cost (4 new maps) | Low | Low | Maps are sparse, only written for entities that actually use concurrent access |
| Breaking change to request_shard API | Certain | Med | All callers updated in same PR. Compile-time enforcement (Cairo won't compile with old signature). |
| Game contracts forget to use SetLock (use Set by accident) | Med | High | Default CRDT for `decode_crdt(0)` should be SetLock (not Set). Forces explicit opt-in for concurrent. |

## Open Questions

- [ ] Should `decode_crdt(0)` default to `SetLock` (safe) instead of `Set` (concurrent)? Defaulting to concurrent could be dangerous if someone forgets to register a policy.
- [ ] Should `request_shard` validate entity classification against registered policies? (Gas cost vs safety trade-off — currently caller-trusted, validated at settle time.)
- [ ] How to handle entities that appear in BOTH exclusive and shared model policies? (e.g., entity has Position=SetLock and Score=Add.) Currently must be exclusive (SetLock wins).

## Decision Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Caller classifies entities (not contract) | Gas: contract would need to iterate all registered models per entity. Caller (game contract) already knows its model structure. Safety enforced at settle time. | (a) Contract auto-classifies — expensive, needs model→entity mapping. (b) Single list with per-entity mode — more complex API. |
| Shared lock uses reference counting | Simple, O(1) per lock/unlock. No need to enumerate all shards for an entity. | (a) Bitmap of shard_ids — limited to 252 concurrent shards. (b) Linked list — gas-expensive traversal. |
| Lock rejects settlement slots (not just blocks writes) | True freeze guarantee. If Lock only blocked writes, the shard could still change the value on its Katana fork and try to settle. | (a) Lock = exclusive + Set (current behavior) — no freeze guarantee. (b) Lock check in request_shard instead of settle — doesn't prevent shard writes on fork. |
| Mutual exclusion between exclusive and shared | Prevents incoherent state: if entity has exclusive lock, a shared lock would bypass the entity_lock protection. If entity has shared locks, an exclusive lock would break shared shards' assumption that mainnet writes are allowed. | (a) Allow mixed — complex, potential races. (b) Per-model locking — too granular, gas expensive. |
