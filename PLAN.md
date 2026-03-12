# Atomic Sharding Coverage and Lease Atomicity Implementation Plan

## Overview
This plan delivers full sharding soundness in two required stages. Stage A removes silent partial coverage in Dojo planning (especially for dynamic layouts) and makes model coverage explicit and fail-fast. Stage B introduces lease-level atomic settlement/cancel across Dojo, sharding proxy, and operator so lock release is all-or-nothing at logical group level, not arbitrary slot subsets.

## Goals
- Eliminate silent partial sharding of logical model state.
- Guarantee deterministic lock coverage for all selected members, including dynamic members.
- Guarantee atomic unlock semantics for cancel/settle at logical lease/group level.
- Keep ecosystem compatibility with staged rollout for Dojo, `sharding_operator`, and Eternum.

## Non-Goals
- Dynamic `Add` semantics.
- Implicit migration with no compatibility layer.
- Supporting arbitrary unsafe partial field subsets as default behavior.

## Assumptions and Constraints
- Current operator and proxy are slot-centric (`ShardingRequested.storage_slots`, `cancel_shard(...slots)`).
- Dojo world already supports dynamic member locks and member writes, but operator proxy path does not carry member writes end-to-end yet.
- Existing Eternum helpers currently rely on auto translation and include models with dynamic members.
- Backward compatibility is required during migration, but final target is strict, self-describing, sound APIs.

## Requirements

### Functional
- `request_sharding` must fail when requested coverage is partial and not explicitly allowed by policy.
- Dynamic members must be represented as lockable units (member lock slots) with explicit CRDT restrictions.
- Settlement/cancel must validate lease completeness before unlocking.
- No API may allow unlocking only an arbitrary subset of slots for a lease in final mode.
- Operator must persist and use lease metadata for retries/recovery/cancel.
- Eternum sharding helpers must declare intent explicitly for dynamic fields (lock or exclude by design).

### Non-Functional
- Deterministic planning output for identical `(model, entity, fields, policy)`.
- Stable, explicit error reasons for all fail-fast rejections.
- Bounded calldata/event growth with chunking and limits.
- Zero silent downgrade paths in strict mode.

## Technical Design

### Data Model
- Keep dynamic lock slot primitive:
  - `compute_dynamic_member_lock_slot(model_selector, entity_id, member_selector)`.
- Introduce lease/group metadata (Stage B):
  - `lease_id`, `slot_count`, `slot_hash`, `model_selector`, `entity_id`, `active`.
  - `slot -> lease_id` mapping for integrity checks.
- Persist lease metadata in operator service state for crash-safe recovery.

### API Design
- Stage A (Dojo):
  - Keep `request_sharding(proxy, models)` ABI.
  - Make default builders strict/sound (no silent dynamic skip).
  - Add explicit legacy/partial builder only if absolutely needed, with self-describing naming.
- Stage B (Cross-repo):
  - Add lease-aware proxy entrypoints for cancel/settle.
  - Keep temporary compatibility path for slot-list calls during migration.
  - Extend events with lease metadata (or equivalent verifiable mapping) for operator decoding.

### Architecture
- Stage A:
  - Dojo planner computes complete requested coverage and rejects unsupported implicit gaps.
  - World write guards remain slot/member-lock based, now aligned with strict request semantics.
- Stage B:
  - Proxy/operator treat each request as leases with completeness proofs.
  - Cancel/settle finalize leases atomically, then unlock all covered slots.

## Ecosystem Impact Matrix

| Component | Stage A Impact | Stage B Impact |
|----------|----------------|----------------|
| Dojo `core` | High: planner/request policy behavior changes; tests and helpers update | High: lease metadata, new checks in cancel/settle |
| `sharding_operator` Cairo proxy (`contracts/src/sharding.cairo`) | Low-Med: mostly compatible if event shape unchanged | High: new lease-aware event/call semantics |
| `sharding_operator` Rust service | Low-Med: decode/validation updates possible | High: state model + call builders + recovery logic |
| Eternum sharding helpers/tests | High: replace implicit/legacy builders, explicit dynamic handling | Med: consume lease-aware event/cancel flow indirectly |

Concrete high-risk models in Eternum with dynamic branches under current flow:
- `StructureVillageSlots.directions_left: Span<Direction>`
- `ResourceArrival.slot_1..slot_48: Span<(u8, u128)>`
- `Structure.troop_explorers: Span<ID>` (currently partially selected manually)

---

## Implementation Plan

### Serial Dependencies (Must Complete First)

#### Phase 0: Protocol and Migration Freeze
**Prerequisite for:** All subsequent phases

| Task | Description | Output |
|------|-------------|--------|
| 0.1 | Freeze strict coverage policy semantics (`default strict`, explicit partial mode decision). | Approved policy contract |
| 0.2 | Freeze lease identity and integrity format (`lease_id`, `slot_hash`, `slot_count`). | Cross-repo protocol spec |
| 0.3 | Freeze compatibility window and cutoff for legacy slot-list cancel/settle. | Migration timeline and flags |

---

### Parallel Workstreams

#### Workstream A: Dojo Stage A Coverage Hardening
**Dependencies:** Phase 0  
**Can parallelize with:** Workstreams B, C

| Task | Description | Output |
|------|-------------|--------|
| A.1 | Remove/retire silent dynamic skip as default behavior in request builders. | Strict default builder behavior |
| A.2 | Enforce full coverage planning for selected members with explicit dynamic lock handling. | Planner/request validation updates |
| A.3 | Add explicit naming for any legacy partial mode (if retained). | Self-describing API surface |
| A.4 | Expand tests for strict rejection and dynamic lock requirements. | Green `core-tests` for coverage matrix |

#### Workstream B: Eternum Adaptation for Stage A
**Dependencies:** Phase 0  
**Can parallelize with:** Workstreams A, C

| Task | Description | Output |
|------|-------------|--------|
| B.1 | Replace legacy helper method usage with new self-describing Dojo APIs. | Updated `systems/sharding/contracts.cairo` |
| B.2 | Decide per-model strategy for dynamic fields (`lock`, `exclude explicitly`, or split model). | Model-by-model sharding policy table |
| B.3 | Add tests for `request_shard_all` to verify no silent dynamic gaps remain. | Regression tests for high-risk models |

#### Workstream C: Dojo Internal Refactor for Readability/Modularity
**Dependencies:** Phase 0  
**Can parallelize with:** Workstreams A, B

| Task | Description | Output |
|------|-------------|--------|
| C.1 | Keep `world_contract` orchestration thin; move planning/validation utilities to sharding modules. | Smaller world entrypoints |
| C.2 | Consolidate settlement/cancel helpers into coherent internal units. | Clear separation of concerns |
| C.3 | Keep ABI stable in Stage A while improving maintainability. | Refactored internals, unchanged public flow |

#### Workstream D: Stage B Lease Protocol in Proxy (Cairo)
**Dependencies:** Phase 0  
**Can parallelize with:** Workstreams E, F

| Task | Description | Output |
|------|-------------|--------|
| D.1 | Add lease metadata storage and validation in sharding proxy contract. | Lease-aware proxy state |
| D.2 | Add lease-aware cancel/settle entrypoints with completeness checks. | Atomic lease finalization path |
| D.3 | Emit lease metadata in events for operator reconstruction. | Updated event schema |

#### Workstream E: Stage B Operator Runtime
**Dependencies:** Phase 0  
**Can parallelize with:** Workstreams D, F

| Task | Description | Output |
|------|-------------|--------|
| E.1 | Update event decoder/state to persist leases (not only flat slots). | Lease-aware `ServiceState` |
| E.2 | Update cancel and settlement call builders to lease API. | New call construction path |
| E.3 | Update retry/recovery flows to remain atomic per lease. | Recovery invariants preserved |

#### Workstream F: Cross-Repo Integration and Compatibility
**Dependencies:** Phase 0  
**Can parallelize with:** Workstreams D, E

| Task | Description | Output |
|------|-------------|--------|
| F.1 | Provide dual-mode compatibility (`legacy slots` + `lease mode`) during rollout. | Feature-flagged compatibility layer |
| F.2 | Add cross-repo golden tests for event decoding and calldata encoding. | Stable interoperability vectors |
| F.3 | Cut over to lease mode and deprecate legacy slot-only cancel path. | Final strict protocol mode |

---

### Merge Phase

#### Phase N: End-to-End Cutover
**Dependencies:** Workstreams A, B, C, D, E, F

| Task | Description | Output |
|------|-------------|--------|
| N.1 | Run full e2e on Dojo + proxy + operator + Eternum with strict mode enabled. | End-to-end green pipeline |
| N.2 | Remove legacy default behavior and keep only explicit APIs. | Sound-by-default public API |
| N.3 | Publish migration notes and lock protocol version. | Release-ready docs and versioning |

---

## Testing and Validation

- Dojo core tests:
  - `scarb test --manifest-path crates/dojo/core-tests/Scarb.toml sharding`
- Proxy Cairo tests:
  - `scarb test --manifest-path /home/michal/Repos/sharding_operator/contracts/Scarb.toml`
- Operator Rust tests:
  - `cargo test --manifest-path /home/michal/Repos/sharding_operator/Cargo.toml operator::service::event_decoder`
  - `cargo test --manifest-path /home/michal/Repos/sharding_operator/Cargo.toml operator::service::shard_task`
- Eternum sharding tests:
  - `scarb test --manifest-path /home/michal/Repos/sharding_operator/externals/eternum/contracts/game/Scarb.toml sharding`
- End-to-end:
  - operator lifecycle with request -> gameplay -> settle -> cancel retry path in lease mode.

## Rollout and Migration

- Stage A rollout:
  - Introduce strict planning under feature flag.
  - Update Eternum helpers first, then enable strict-by-default.
- Stage B rollout:
  - Deploy proxy with dual-mode support.
  - Upgrade operator to decode and use lease metadata.
  - Enable lease mode for canary games.
  - Remove legacy slot-only cancel path after canary stabilization.
- Rollback:
  - Keep legacy slot path available during dual-mode window.
  - Disable strict mode flag in case of blocker.

## Verification Checklist

- [x] Stage A strict tests pass in Dojo core.
- [ ] Eternum `request_shard_all` scenarios pass without implicit dynamic skips.
- [ ] Proxy emits event payload that operator decodes in both compatibility modes.
- [ ] Operator recovery and retry preserve lease atomicity.
- [ ] Cancel never leaves partially unlocked lease in strict mode.

## Current Status (Dojo)

- [x] A.1 Remove silent dynamic skip as default behavior.
- [x] A.2 Enforce explicit coverage policy in planner/request path.
- [x] A.3 Keep self-describing request APIs (`shard`, `shard_deterministic`, `shard_dynamic`) without versioned suffixes.
- [x] A.4 Add strict rejection tests and exclusive group atomicity tests in `core-tests`.
- [x] C.1/C.2 Refactor world orchestration by centralizing exclusive-group coverage checks in sharding component.
- [ ] D/E/F Cross-repo lease protocol rollout (`sharding_operator` Cairo proxy + Rust operator + Eternum integration).

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing game flows that relied on silent dynamic skip | High | High | Stage A feature flag + explicit helper migration in Eternum |
| Event schema mismatch between proxy and operator | Med | High | Golden vectors + dual decoder support |
| Lease metadata drift causing false rejections | Med | High | Canonical hash/count derivation + cross-language tests |
| Gas growth from richer metadata and checks | Med | Med | Chunking limits + bounded calldata policy |
| Incomplete migration leaving mixed semantics | Med | High | Time-boxed dual-mode and explicit cutoff |

## Open Questions

- [ ] Should explicit partial mode exist long-term, or be removed after migration?
- [ ] Lease granularity: per `(model, entity)` or per request bundle?
- [ ] Should Stage B include dynamic member-write payload transport in proxy/operator immediately, or in follow-up?

## Decision Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Two-stage rollout (coverage hardening first, lease atomicity second) | Reduces migration risk and isolates concerns | Big-bang protocol rewrite |
| Strict-by-default request semantics | Prevents silent unsound behavior | Keep permissive default with warnings |
| Lease-level atomic finalization | Eliminates partial unlock class of failures | Slot-list cancel with best-effort checks |
