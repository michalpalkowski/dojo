# Dynamic Layout Sharding (Array/ByteArray) Implementation Plan

## Overview
This plan introduces protocol-safe sharding support for dynamic model members (`Layout::Array`, `Layout::ByteArray`) without requiring manual field selection by game developers. The design keeps deterministic slot sharding unchanged and adds a dedicated dynamic settlement path with explicit locking, validation, and fail-fast behavior. The solution is protocol-first: no hidden fallbacks, clear CRDT constraints, and backward-compatible rollout.

## Goals
- Enable automatic sharding for dynamic members in struct models (no manual selector plumbing by game teams).
- Preserve current deterministic sharding behavior and performance.
- Keep settlement safe under concurrent L1 writes (explicit lock semantics for dynamic members).
- Keep Torii event emission correct (`StoreSetRecord`/`StoreUpdateMember`) after dynamic settlement.

## Non-Goals
- Dynamic `Add` CRDT in v1 (delta semantics for variable-length payloads are out of scope).
- Arbitrary deep partial-diff merge for dynamic payloads in v1.
- Replacing existing deterministic settlement path (`update_shard_state`) in v1.
- Silent downgrade when dynamic support is unavailable.

## Assumptions and Constraints
- `world_contract` remains source of truth for model layout validation and event shaping.
- Existing deterministic flow (`request_sharding` + `update_shard_state`) must stay backward compatible.
- Proxy/operator contract can be upgraded to support a v2 settlement payload for dynamic members.
- Dynamic member writes are serialized in canonical Dojo layout format and size-limited.
- Design principles required by project: protocol-first, fail-fast, minimal fallback, trait/generic APIs.

## Requirements

### Functional
- Translator must be able to select dynamic members automatically (policy-controlled), not only deterministic members.
- `request_sharding` must register both:
  - deterministic concrete storage slots (existing behavior),
  - dynamic member lock entries (new behavior).
- Settlement must support combined payload:
  - deterministic slot changes,
  - dynamic member updates (serialized member values).
- Dynamic member settlement must:
  - verify caller is registered proxy,
  - verify member was requested/locked,
  - apply value via canonical layout writer,
  - unlock member lock atomically.
- `cancel_shard_state` must unlock dynamic lock entries as well.
- World write guards must reject normal writes to active dynamic members when lock is exclusive.

### Non-Functional
- Deterministic and dynamic paths must be idempotent and deterministic for the same input.
- Fail-fast errors must be explicit and stable.
- New path must be observable (events + test assertions), with no silent no-op for malformed payload.
- Gas/size controls must bound dynamic settlement payload size.

## Technical Design

### Data Model
- Add dynamic lock key primitive:
  - `compute_dynamic_member_lock_slot(model_selector, entity_id, member_selector) -> felt252`
- Reuse existing component slot maps by storing dynamic lock entries as pseudo-slots (no separate lock table in v1).
- Add v2 settlement payload structs:
  - `DynamicMemberHeader { model_selector, entity_id, member_selector, values_offset, values_len }`
  - `dynamic_values: Span<felt252>` (flat payload buffer)
- Add size limits constants for dynamic payload and per-member payload.

### API Design
- Keep existing APIs unchanged:
  - `request_sharding(proxy, models)`
  - `update_shard_state(storage_changes)`
- Add new world proxy ABI entrypoint:
  - `update_shard_state_v2(storage_changes, dynamic_headers, dynamic_values)`
- Keep `cancel_shard_state(slots)` as shared cancel path (dynamic pseudo-slots included).
- Extend trait-level sharding builders with explicit dynamic-capable methods (or policy flag), e.g.:
  - `shard_with_dynamic(...)`, `shard_add_with_dynamic(...)` (with CRDT restrictions enforced).

### Architecture
```mermaid
graph TD
    A[Game Contract] -->|request_sharding(models)| B[World]
    B -->|validate layout + ACL| B
    B -->|deterministic slots + dynamic lock pseudo-slots| C[Sharding Component]
    C -->|initialize_sharding| D[Proxy/Operator]

    D -->|update_shard_state(slot_changes)| B
    D -->|update_shard_state_v2(slot_changes + dynamic payload)| B

    B -->|deterministic merge| C
    B -->|dynamic member apply via layout writer| B
    B -->|StoreSetRecord / StoreUpdateMember| E[Torii]
```

### UX Flow (if applicable)
- No UI changes required in Dojo core.
- Developer experience change:
  - dynamic-capable auto sharding method available,
  - clear runtime errors for unsupported CRDT/layout combinations.

---

## Implementation Plan

### Serial Dependencies (Must Complete First)

These tasks create foundations that other work depends on. Complete in order.

#### Phase 0: Protocol Freeze and Safety Contract
**Prerequisite for:** All subsequent phases

| Task | Description | Output |
|------|-------------|--------|
| 0.1 | Write ADR/spec for dynamic sharding semantics (lock key derivation, payload format, CRDT support matrix). | Approved protocol note in `PLAN.md` + implementation constants list |
| 0.2 | Define fail-fast error matrix for request/settlement/cancel (invalid selector, unsupported CRDT, payload length mismatch, unauthorized caller, stale lock). | Canonical error cases and test names |
| 0.3 | Define v2 settlement commitment hash shape with proxy team (slot hash + dynamic hash composition). | Cross-repo contract spec (Dojo + operator) |

---

### Parallel Workstreams

These workstreams can be executed independently after Phase 0.

#### Workstream A: Translator and Key Primitives
**Dependencies:** Phase 0
**Can parallelize with:** Workstreams B, C

| Task | Description | Output |
|------|-------------|--------|
| A.1 | Add dynamic lock key helper in sharding slot/key utils. | `compute_dynamic_member_lock_slot(...)` + unit tests |
| A.2 | Extend `IntoShardModel` policies/methods to include dynamic members automatically (policy-driven). | Updated `request.cairo` API |
| A.3 | Enforce CRDT compatibility for dynamic members (v1: `Set`/`SetLock` only). | Fail-fast validation in translator + request path |

#### Workstream B: World Contract Dynamic Request + Guards
**Dependencies:** Phase 0
**Can parallelize with:** Workstreams A, C

| Task | Description | Output |
|------|-------------|--------|
| B.1 | Update `request_sharding` to register dynamic pseudo-lock slots when member layout is dynamic. | Dynamic branch in `world_contract.cairo` request path |
| B.2 | Extend write guards (`assert_model_write_unlocked`, `assert_member_write_unlocked`) to check dynamic lock pseudo-slots. | Lock-safe gameplay writes for dynamic members |
| B.3 | Keep deterministic path unchanged and dedup-safe with mixed deterministic+dynamic fields. | Regression-safe combined request behavior |

#### Workstream C: Settlement V2 and Component Integration
**Dependencies:** Phase 0
**Can parallelize with:** Workstreams A, B

| Task | Description | Output |
|------|-------------|--------|
| C.1 | Add `update_shard_state_v2` world proxy ABI with structured dynamic payload parsing. | New ABI method + serde structs |
| C.2 | Reuse component auth/lock lifecycle for dynamic pseudo-slots; unlock after successful dynamic apply. | Atomic dynamic settlement and unlock behavior |
| C.3 | Emit correct Torii events for dynamic member updates using canonical `read_model_member` after write. | Event parity with deterministic flow |
| C.4 | Ensure cancel path unlocks dynamic pseudo-slots and cleans metadata/entity keys consistently. | Unified cancel semantics |

#### Workstream D: Test Matrix and Compatibility
**Dependencies:** Phase 0
**Can parallelize with:** Workstreams A, B, C

| Task | Description | Output |
|------|-------------|--------|
| D.1 | Add core-tests for dynamic request acceptance/rejection and CRDT compatibility matrix. | New tests in `core-tests/src/tests/sharding/request.cairo` |
| D.2 | Add settlement v2 tests for Array and ByteArray members (set, cancel, partial update failures). | New tests in `settlement_events.cairo`/`component.cairo` |
| D.3 | Add mixed deterministic+dynamic tests (same model, same entity) and duplicate/stale lock failure tests. | Concurrency and lifecycle coverage |
| D.4 | Add backward-compat tests ensuring existing `update_shard_state` behavior unchanged. | No-regression confidence |

---

### Merge Phase

After parallel workstreams complete, these tasks integrate the work.

#### Phase N: Integration
**Dependencies:** Workstreams A, B, C, D

| Task | Description | Output |
|------|-------------|--------|
| N.1 | End-to-end integration with upgraded proxy commitment verification for v2 payload. | Passing integration suite across Dojo + operator repos |
| N.2 | Documentation update for game developers (new auto dynamic behavior + CRDT limits + examples). | Updated docs/changelog snippets |
| N.3 | Rollout checklist and rollback toggles validated in staging. | Release-ready runbook |

---

## Testing and Validation

- Unit tests:
  - dynamic lock key derivation stability,
  - translator policy behavior for dynamic members,
  - payload parser boundary checks.
- Contract tests:
  - `request_sharding` mixed deterministic/dynamic,
  - `update_shard_state_v2` success/failure matrix,
  - cancel/unlock lifecycle for dynamic locks,
  - event correctness after dynamic settlement.
- Regression tests:
  - existing deterministic sharding suite must remain green.

## Rollout and Migration

- Stage 1: Deploy Dojo with both settlement endpoints (`update_shard_state` + `update_shard_state_v2`) behind operator-side feature flag.
- Stage 2: Upgrade proxy/operator to generate/verify v2 commitment and call v2 endpoint when dynamic members are present.
- Stage 3: Enable dynamic auto-sharding for selected games (canary), monitor settlement failures and gas.
- Stage 4: Broaden rollout; keep deterministic path as rollback fallback.
- Rollback plan:
  - disable dynamic flag in proxy,
  - route settlements through legacy deterministic path only,
  - reject dynamic fields at request time until issue resolved.

## Verification Checklist

- `cd /home/michal/Repos/sharding_operator/externals/dojo/crates/dojo/core-tests && scarb test sharding`
- `cd /home/michal/Repos/sharding_operator/externals/dojo/crates/dojo/core-tests && scarb test test_request_sharding_auto_skips_dynamic_fields`
- `cd /home/michal/Repos/sharding_operator/externals/dojo/crates/dojo/core-tests && scarb test test_request_sharding_strict_rejects_mixed_dynamic`
- (After implementation) run new v2 dynamic settlement tests for Array/ByteArray and mixed models.
- (Cross-repo) run operator/proxy settlement tests with v2 commitment verification.

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Protocol mismatch between Dojo and proxy on v2 payload hashing | Med | High | Freeze spec in Phase 0 and implement golden-vector tests in both repos |
| Dynamic payload gas/size blowups | High | High | Hard size caps, fail-fast validation, staged rollout |
| Incorrect lock semantics allowing L1/shard write races | Med | High | Pseudo-lock key model + explicit guard tests for model/member writes |
| Event emission drift for dynamic members | Med | Med | Canonical post-write reads + strict event assertions in tests |
| Backward-compat regression in deterministic path | Low | High | Keep old endpoint unchanged + no-regression suite in every PR |

## Open Questions

- [ ] Should `shard()` default include dynamic members immediately, or only new `*_with_dynamic()` methods first?
- [ ] Do we allow `Lock` CRDT for dynamic members in v1, or only `Set`/`SetLock`?
- [ ] What exact maximum payload size (per member / per settlement tx) is acceptable for production gas budget?
- [ ] Should dynamic settlement support chunking in v1, or fail when payload exceeds cap?

## Decision Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Use dedicated v2 settlement endpoint for dynamic payloads | Keeps legacy deterministic ABI stable and explicit | Overloading existing `update_shard_state` with ambiguous encoding |
| Use pseudo-slot dynamic lock keys | Reuses existing component lock/auth lifecycle with minimal new state surface | Separate dedicated dynamic lock storage tables |
| Dynamic v1 supports only full-value replacement semantics | Simpler, auditable, predictable correctness | Partial diff merge and dynamic Add semantics |
| Keep deterministic flow untouched | Minimizes regression risk and enables safe rollback | Single unified path rewrite |
