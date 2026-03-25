use dojo::model::{Model, ModelStorage, ModelStorageTest};
use dojo::sharding::compute_dojo_field_slot;
use dojo::sharding::slot::compute_dojo_packed_slot;
use dojo::utils::{entity_id_from_keys, combine_key};
use dojo::world::{
    IShardingSettlementDispatcher, IShardingSettlementDispatcherTrait, IWorldDispatcherTrait,
};
use starknet::ContractAddress;

use crate::tests::helpers::{Foo, PackedPair, deploy_world_and_foo, deploy_world_with_packed_pair};

const SLOT_KIND_DETERMINISTIC: felt252 = 1;
const SLOT_KIND_DYNAMIC_LOCK: felt252 = 2;

/// Build computation_keys span from entity_ids and member_selectors.
/// For struct fields (member != 0): comp_key = combine_key(entity, member).
/// For packed (member == 0): comp_key = entity_id.
fn build_comp_keys(entity_ids: Span<felt252>, member_selectors: Span<felt252>) -> Span<felt252> {
    let mut keys: Array<felt252> = ArrayTrait::new();
    let mut i: u32 = 0;
    while i < entity_ids.len() {
        let eid = *entity_ids[i];
        let member = *member_selectors[i];
        if member != 0 {
            keys.append(combine_key(eid, member));
        } else {
            keys.append(eid);
        };
        i += 1;
    };
    keys.span()
}

fn build_slot_kinds(count: u32, kind: felt252) -> Span<felt252> {
    let mut kinds: Array<felt252> = ArrayTrait::new();
    let mut i: u32 = 0;
    while i < count {
        kinds.append(kind);
        i += 1;
    };
    kinds.span()
}

/// `felt252` max value (`FIELD_PRIME - 1`).
const FELT_MAX: felt252 = 0x800000000000011000000000000000000000000000000000000000000000000;

fn foo_field_selectors() -> (felt252, felt252) {
    let layout = Model::<Foo>::layout();
    if let dojo::meta::Layout::Struct(fields) = layout {
        ((*fields[0]).selector, (*fields[1]).selector)
    } else {
        panic!("expected struct layout")
    }
}

/// Deploy world + Foo, write initial data, compute slots.
/// Returns (world, world_address, entity_id, slot_a, slot_b, model_selector).
fn setup_foo_shard() -> (
    dojo::world::WorldStorage,
    ContractAddress,
    felt252,
    felt252,
    felt252,
    felt252,
) {
    // Make test_address() the world owner (constructor uses account_contract_address).
    snforge_std::start_cheat_account_contract_address_global(snforge_std::test_address());
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let (sel_a, sel_b) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let slot_b = compute_dojo_field_slot(model_selector, entity_id, sel_b);

    (world, world_address, entity_id, slot_a, slot_b, model_selector)
}

/// Helper: settle as world owner with Set CRDT (no Add, no StorageCommitment).
fn settle_as_owner(
    world_address: ContractAddress,
    shard_id: felt252,
    changed_keys: Span<felt252>,
    changed_values: Span<felt252>,
    slot_model_selectors: Span<felt252>,
    slot_entity_ids: Span<felt252>,
    slot_member_selectors: Span<felt252>,
    slot_initial_values: Span<felt252>,
) {
    let comp_keys = build_comp_keys(slot_entity_ids, slot_member_selectors);
    let mut offsets: Array<u32> = ArrayTrait::new();
    let mut k: u32 = 0;
    while k < changed_keys.len() {
        offsets.append(0);
        k += 1;
    };

    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement
        .settle(
            shard_id,
            changed_keys,
            changed_values,
            0,
            0,
            slot_model_selectors,
            slot_entity_ids,
            comp_keys,
            build_slot_kinds(changed_keys.len(), SLOT_KIND_DETERMINISTIC),
            slot_member_selectors,
            offsets.span(),
            slot_initial_values,
            [].span(), // entity_model_selectors (not needed in unit tests)
            [].span(), // entity_keys_flat (not needed in unit tests)
        );
    snforge_std::stop_cheat_caller_address(world_address);
}

// ── Entity Locking ──────────────────────────────────────────────────────

#[test]
fn test_request_shard_locks_entities() {
    let (world, _, entity_id, _, _, _) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());
}

#[test]
#[should_panic(expected: ('Shard: entity already sharded',))]
fn test_request_shard_rejects_already_locked_entity() {
    let (world, _, entity_id, _, _, _) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());
    // Second request for same entity should fail.
    world.dispatcher.request_sharding([entity_id].span(), [].span());
}

#[test]
#[should_panic(expected: ('Shard: no entities',))]
fn test_request_shard_rejects_empty_entities() {
    let (world, _, _, _, _, _) = setup_foo_shard();
    world.dispatcher.request_sharding([].span(), [].span());
}

// ── Entity Lock Enforcement (write path) ────────────────────────────────

#[test]
#[should_panic(expected: ('Shard: entity locked',))]
fn test_write_model_blocked_while_entity_sharded() {
    let (mut world, _, entity_id, _, _, _) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    // write_model_test goes through set_entity_internal which checks entity lock.
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 999, b: 888 });
}

#[test]
#[should_panic(expected: ('Shard: entity locked',))]
fn test_delete_model_blocked_while_entity_sharded() {
    let (mut world, _, entity_id, _, _, _) =
        setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    // delete_entity_test goes through delete_entity_internal which checks entity lock.
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.erase_model_test(@Foo { caller: bob, a: 0, b: 0 });
}

#[test]
fn test_write_model_allowed_after_settlement_unlocks() {
    let (mut world, world_address, entity_id, slot_a, slot_b, model_selector) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let (sel_a, sel_b) = foo_field_selectors();
    settle_as_owner(
        world_address, 1,
        [slot_a, slot_b].span(), [500, 300].span(),
        [model_selector, model_selector].span(), [entity_id, entity_id].span(),
        [sel_a, sel_b].span(), [0, 0].span(),
    );

    // Entity unlocked — write should succeed.
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 999, b: 888 });
    let result: Foo = world.read_model(bob);
    assert(result.a == 999, 'write after settle ok');
}

#[test]
fn test_write_model_allowed_after_cancel_unlocks() {
    let (mut world, world_address, entity_id, _, _, _) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement.cancel_shard(1);
    snforge_std::stop_cheat_caller_address(world_address);

    // Entity unlocked — write should succeed.
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 999, b: 888 });
    let result: Foo = world.read_model(bob);
    assert(result.a == 999, 'write after cancel ok');
}

// ── Settlement: Set CRDT ────────────────────────────────────────────────

#[test]
fn test_settle_set_writes_values() {
    let (mut world, world_address, entity_id, slot_a, slot_b, model_selector) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let (sel_a, sel_b) = foo_field_selectors();
    settle_as_owner(
        world_address, 1,
        [slot_a, slot_b].span(), [999, 777].span(),
        [model_selector, model_selector].span(), [entity_id, entity_id].span(),
        [sel_a, sel_b].span(), [0, 0].span(),
    );

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let result: Foo = world.read_model(bob);
    assert(result.a == 999, 'a should be overwritten');
    assert(result.b == 777, 'b should be overwritten');
}

#[test]
fn test_settle_partial_changes() {
    let (mut world, world_address, entity_id, slot_a, slot_b, model_selector) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let (_sel_a, sel_b) = foo_field_selectors();
    // Only change slot_b.
    settle_as_owner(
        world_address, 1,
        [slot_b].span(), [777].span(),
        [model_selector].span(), [entity_id].span(),
        [sel_b].span(), [0].span(),
    );

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let result: Foo = world.read_model(bob);
    assert(result.a == 100, 'a should be unchanged');
    assert(result.b == 777, 'b should be overwritten');
}

// ── Settlement: Add CRDT ────────────────────────────────────────────────

#[test]
fn test_settle_add_crdt_computes_delta() {
    let (mut world, world_address, entity_id, slot_a, slot_b, model_selector) = setup_foo_shard();

    // Register policy: field `a` is Add CRDT.
    let (sel_a, _sel_b) = foo_field_selectors();
    world
        .dispatcher
        .register_shard_policy(
            model_selector,
            dojo::sharding::request::CRDVariant::Set,
            [dojo::sharding::request::ShardField {
                selector: sel_a,
                crdt: dojo::sharding::request::CRDVariant::Add,
                max_elements: 0,
            }]
                .span(),
        );

    world.dispatcher.request_sharding([entity_id].span(), [].span());

    // Settle with Add: shard produced 150, initial was 100 → delta = 50.
    settle_as_owner(
        world_address, 1,
        [slot_a].span(), [150].span(),
        [model_selector].span(), [entity_id].span(),
        [sel_a].span(), [100].span(), // initial_value = 100
    );

    // Expected: current(100) + (shard(150) - initial(100)) = 150.
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let result: Foo = world.read_model(bob);
    assert(result.a == 150, 'Add: 100 + (150-100) = 150');
    assert(result.b == 200, 'b unchanged');
}

#[test]
fn test_settle_mixed_add_and_set() {
    let (mut world, world_address, entity_id, slot_a, slot_b, model_selector) = setup_foo_shard();

    let (sel_a, sel_b) = foo_field_selectors();
    // Register policy: field `a` is Add, default (field `b`) is Set.
    world
        .dispatcher
        .register_shard_policy(
            model_selector,
            dojo::sharding::request::CRDVariant::Set,
            [dojo::sharding::request::ShardField {
                selector: sel_a,
                crdt: dojo::sharding::request::CRDVariant::Add,
                max_elements: 0,
            }]
                .span(),
        );

    world.dispatcher.request_sharding([entity_id].span(), [].span());

    settle_as_owner(
        world_address, 1,
        [slot_a, slot_b].span(), [150, 999].span(),
        [model_selector, model_selector].span(), [entity_id, entity_id].span(),
        [sel_a, sel_b].span(), [100, 0].span(), // initial_value=100 for Add slot_a, 0 for Set slot_b
    );

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let result: Foo = world.read_model(bob);
    // Add: current(100) + (shard(150) - initial(100)) = 150.
    assert(result.a == 150, 'Add: 100+(150-100)=150');
    assert(result.b == 999, 'Set: overwrite to 999');
}

// ── Settlement: Error Cases ─────────────────────────────────────────────

#[test]
#[should_panic(expected: ('Shard: unauthorized caller',))]
fn test_settle_rejects_unauthorized_caller() {
    let (world, world_address, entity_id, slot_a, slot_b, _) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    // Call settle as non-owner.
    let not_owner: ContractAddress = 0xdead.try_into().unwrap();
    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    let changed_keys = [slot_a, slot_b].span();
    snforge_std::start_cheat_caller_address(world_address, not_owner);
    settlement
        .settle(
            1, changed_keys, [999, 777].span(), 0, 0,
            [].span(), [].span(), [].span(), [].span(), [].span(), [].span(), [].span(),
            [].span(), [].span(),
        );
}

#[test]
#[should_panic(expected: ('Shard: keys/values len',))]
fn test_settle_rejects_keys_values_length_mismatch() {
    let (world, world_address, entity_id, slot_a, slot_b, _) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    // 2 changed keys but only 1 value.
    let changed_keys = [slot_a, slot_b].span();
    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement
        .settle(
            1, changed_keys, [999].span(), 0, 0,
            [].span(), [].span(), [].span(), [].span(), [].span(), [].span(), [].span(),
            [].span(), [].span(),
        );
}

#[test]
#[should_panic(expected: ('Shard: add delta underflow',))]
fn test_settle_add_underflow_rejected() {
    let (mut world, world_address, entity_id, slot_a, _slot_b, model_selector) = setup_foo_shard();

    let (sel_a, _sel_b) = foo_field_selectors();
    world
        .dispatcher
        .register_shard_policy(
            model_selector,
            dojo::sharding::request::CRDVariant::Set,
            [dojo::sharding::request::ShardField {
                selector: sel_a,
                crdt: dojo::sharding::request::CRDVariant::Add,
                max_elements: 0,
            }]
                .span(),
        );

    world.dispatcher.request_sharding([entity_id].span(), [].span());

    // Shard value (90) < initial (100) → underflow.
    settle_as_owner(
        world_address, 1,
        [slot_a].span(), [90].span(),
        [model_selector].span(), [entity_id].span(),
        [sel_a].span(), [100].span(),
    );
}

#[test]
#[should_panic(expected: ('Shard: arithmetic overflow',))]
fn test_settle_add_overflow_rejected() {
    snforge_std::start_cheat_account_contract_address_global(snforge_std::test_address());
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    // Set value near felt max BEFORE sharding (so entity lock doesn't block).
    world.write_model_test(@Foo { caller: bob, a: FELT_MAX, b: 200 });

    let (sel_a, _sel_b) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);

    // Register Add policy for field `a`.
    world
        .dispatcher
        .register_shard_policy(
            model_selector,
            dojo::sharding::request::CRDVariant::Set,
            [dojo::sharding::request::ShardField {
                selector: sel_a,
                crdt: dojo::sharding::request::CRDVariant::Add,
                max_elements: 0,
            }]
                .span(),
        );

    world.dispatcher.request_sharding([entity_id].span(), [].span());

    // delta = 1 - 0 = 1, current(FELT_MAX) + delta(1) overflows.
    settle_as_owner(
        world_address, 1,
        [slot_a].span(), [1].span(),
        [model_selector].span(), [entity_id].span(),
        [sel_a].span(), [0].span(),
    );
}

// ── Cancel ──────────────────────────────────────────────────────────────

#[test]
fn test_cancel_shard_unlocks_entities() {
    let (mut world, world_address, entity_id, _, _, _) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement.cancel_shard(1);
    snforge_std::stop_cheat_caller_address(world_address);

    // Entity should be unlocked — can request sharding again.
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let result: Foo = world.read_model(bob);
    assert(result.a == 100, 'cancel: value unchanged');
    assert(result.b == 200, 'cancel: value unchanged');
}

#[test]
#[should_panic(expected: ('Shard: unauthorized caller',))]
fn test_cancel_shard_rejects_unauthorized_caller() {
    let (world, world_address, entity_id, _, _, _) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    // Call cancel as non-owner.
    let not_owner: ContractAddress = 0xdead.try_into().unwrap();
    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, not_owner);
    settlement.cancel_shard(1);
}

// ── Settle Unlocks + Cleanup ────────────────────────────────────────────

#[test]
fn test_settle_unlocks_entities_for_resharding() {
    let (mut world, world_address, entity_id, slot_a, slot_b, model_selector) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let (sel_a, sel_b) = foo_field_selectors();
    settle_as_owner(
        world_address, 1,
        [slot_a, slot_b].span(), [999, 777].span(),
        [model_selector, model_selector].span(), [entity_id, entity_id].span(),
        [sel_a, sel_b].span(), [0, 0].span(),
    );

    // After settlement, entity should be unlocked — can request again.
    world.dispatcher.request_sharding([entity_id].span(), [].span());
}

#[test]
fn test_sequential_shards_increment_shard_id() {
    let (mut world, world_address, entity_id, slot_a, slot_b, model_selector) = setup_foo_shard();
    let (sel_a, sel_b) = foo_field_selectors();

    // Shard 1.
    world.dispatcher.request_sharding([entity_id].span(), [].span());
    settle_as_owner(
        world_address, 1,
        [slot_a, slot_b].span(), [500, 300].span(),
        [model_selector, model_selector].span(), [entity_id, entity_id].span(),
        [sel_a, sel_b].span(), [0, 0].span(),
    );

    // Shard 2 — entity now unlocked, new shard_id = 2.
    world.dispatcher.request_sharding([entity_id].span(), [].span());
    settle_as_owner(
        world_address, 2,
        [slot_a, slot_b].span(), [600, 400].span(),
        [model_selector, model_selector].span(), [entity_id, entity_id].span(),
        [sel_a, sel_b].span(), [0, 0].span(),
    );

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let result: Foo = world.read_model(bob);
    assert(result.a == 600, 'shard 2: a=600');
    assert(result.b == 400, 'shard 2: b=400');
}

// ── Shard Policy Registration Tests ──────────────────────────────────

#[test]
fn test_register_shard_policy_and_read_back() {
    let (_world, world_address, _entity_id, _slot_a, _slot_b, model_selector) = setup_foo_shard();
    let (sel_a, _sel_b) = foo_field_selectors();

    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    let world_disp = dojo::world::IWorldDispatcher { contract_address: world_address };

    world_disp.register_shard_policy(
        model_selector,
        dojo::sharding::request::CRDVariant::Set,
        [dojo::sharding::request::ShardField {
            selector: sel_a,
            crdt: dojo::sharding::request::CRDVariant::Add, max_elements: 0
        }].span(),
    );

    let (default_encoded, field_overrides) = world_disp.get_shard_policy(model_selector);
    // Set = 1
    assert(default_encoded == 1, 'default should be Set(1)');
    assert(field_overrides.len() == 1, 'should have 1 override');
    let field = *field_overrides[0];
    assert(field.selector == sel_a, 'override selector mismatch');
    assert(field.crdt == dojo::sharding::request::CRDVariant::Add, 'override crdt should be Add');

    snforge_std::stop_cheat_caller_address(world_address);
}

#[test]
fn test_register_shard_policy_update_clears_old_overrides() {
    let (_world, world_address, _entity_id, _slot_a, _slot_b, model_selector) = setup_foo_shard();
    let (sel_a, sel_b) = foo_field_selectors();

    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    let world_disp = dojo::world::IWorldDispatcher { contract_address: world_address };

    // Register with 2 field overrides.
    world_disp.register_shard_policy(
        model_selector,
        dojo::sharding::request::CRDVariant::Set,
        [
            dojo::sharding::request::ShardField { selector: sel_a, crdt: dojo::sharding::request::CRDVariant::Add, max_elements: 0 },
            dojo::sharding::request::ShardField { selector: sel_b, crdt: dojo::sharding::request::CRDVariant::Lock, max_elements: 0 },
        ].span(),
    );

    let (_, overrides1) = world_disp.get_shard_policy(model_selector);
    assert(overrides1.len() == 2, 'should have 2 overrides');

    // Re-register with 0 overrides and different default.
    world_disp.register_shard_policy(
        model_selector,
        dojo::sharding::request::CRDVariant::Add,
        [].span(),
    );

    let (default_encoded, overrides2) = world_disp.get_shard_policy(model_selector);
    assert(default_encoded == 2, 'default should be Add(2)');
    assert(overrides2.len() == 0, 'overrides should be cleared');

    snforge_std::stop_cheat_caller_address(world_address);
}

#[test]
fn test_get_shard_policy_unregistered_returns_zero() {
    let (_world, world_address, _entity_id, _slot_a, _slot_b, _model_selector) = setup_foo_shard();

    let world_disp = dojo::world::IWorldDispatcher { contract_address: world_address };
    let (default_encoded, overrides) = world_disp.get_shard_policy(0x12345);
    assert(default_encoded == 0, 'unregistered default is 0');
    assert(overrides.len() == 0, 'unregistered has no overrides');
}

#[test]
#[should_panic(expected: 'Shard: model selector is zero')]
fn test_register_shard_policy_rejects_zero_model() {
    let (_world, world_address, _entity_id, _slot_a, _slot_b, _model_selector) = setup_foo_shard();

    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    let world_disp = dojo::world::IWorldDispatcher { contract_address: world_address };

    world_disp.register_shard_policy(
        0, // zero model selector
        dojo::sharding::request::CRDVariant::Set,
        [].span(),
    );
}

// ── Slot Ownership Verification Tests ───────────────────────────────

#[test]
#[should_panic(expected: ('Shard: slot ownership mismatch',))]
fn test_settle_rejects_wrong_model_selector() {
    let (world, world_address, entity_id, slot_a, slot_b, model_selector) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let (sel_a, sel_b) = foo_field_selectors();
    let changed_keys = [slot_a, slot_b].span();
    let comp_keys = build_comp_keys([entity_id, entity_id].span(), [sel_a, sel_b].span());

    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    // Pass wrong model_selector (0xdead) for first slot — hash won't match slot_a.
    settlement
        .settle(
            1, changed_keys, [999, 777].span(), 0, 0,
            [0xdead, model_selector].span(), [entity_id, entity_id].span(),
            comp_keys, build_slot_kinds(2, SLOT_KIND_DETERMINISTIC), [sel_a, sel_b].span(),
            [0, 0].span(), [0, 0].span(),
            [].span(), [].span(),
        );
}

#[test]
#[should_panic(expected: ('Shard: entity not in shard',))]
fn test_settle_rejects_unlocked_entity() {
    let (world, world_address, entity_id, slot_a, _slot_b, model_selector) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    // Compute a slot for a DIFFERENT entity that is NOT locked.
    let other_entity: felt252 = 0x999;
    let (sel_a, _sel_b) = foo_field_selectors();
    let other_slot = dojo::sharding::compute_dojo_field_slot(model_selector, other_entity, sel_a);

    let changed_keys = [other_slot].span();
    let comp_keys = build_comp_keys([other_entity].span(), [sel_a].span());

    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    // Try to settle slot belonging to unlocked entity.
    settlement
        .settle(
            1, changed_keys, [42].span(), 0, 0,
            [model_selector].span(), [other_entity].span(),
            comp_keys, build_slot_kinds(1, SLOT_KIND_DETERMINISTIC), [sel_a].span(),
            [0].span(), [0].span(),
            [].span(), [].span(),
        );
}

#[test]
#[should_panic(expected: ('Shard: metadata len mismatch',))]
fn test_settle_rejects_metadata_length_mismatch() {
    let (world, world_address, entity_id, slot_a, slot_b, model_selector) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let (sel_a, _sel_b) = foo_field_selectors();
    let changed_keys = [slot_a, slot_b].span();

    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    // 2 changed slots but only 1 metadata entry.
    settlement
        .settle(
            1, changed_keys, [999, 777].span(), 0, 0,
            [model_selector].span(), [entity_id].span(),
            build_comp_keys([entity_id].span(), [sel_a].span()),
            build_slot_kinds(1, SLOT_KIND_DETERMINISTIC), [sel_a].span(), [0].span(), [0].span(),
            [].span(), [].span(),
        );
}

#[test]
#[should_panic(expected: ('Shard: bad det slot',))]
fn test_settle_rejects_zero_comp_key_for_deterministic_slot() {
    let (world, world_address, entity_id, slot_a, _slot_b, model_selector) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let (sel_a, _sel_b) = foo_field_selectors();
    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement
        .settle(
            1,
            [slot_a].span(),
            [42].span(),
            0,
            0,
            [model_selector].span(),
            [entity_id].span(),
            [0].span(),
            build_slot_kinds(1, SLOT_KIND_DETERMINISTIC),
            [sel_a].span(),
            [0].span(),
            [0].span(),
            [].span(),
            [].span(),
        );
}

#[test]
#[should_panic(expected: ('Shard: entity not in shard',))]
fn test_settle_rejects_packed_offset_bypass_for_unlocked_entity() {
    snforge_std::start_cheat_account_contract_address_global(snforge_std::test_address());
    let (mut world, model_selector) = deploy_world_with_packed_pair();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let alice: ContractAddress = 0xa11ce.try_into().unwrap();
    world.write_model_test(@PackedPair { player: bob, left: 10, right: 20 });
    world.write_model_test(@PackedPair { player: alice, left: 30, right: 40 });

    let bob_eid = entity_id_from_keys(@bob);
    let alice_eid = entity_id_from_keys(@alice);
    world.dispatcher.request_sharding([bob_eid].span(), [].span());

    let forged_key = compute_dojo_packed_slot(model_selector, alice_eid) + 1;
    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement
        .settle(
            1,
            [forged_key].span(),
            [999].span(),
            0,
            0,
            [model_selector].span(),
            [alice_eid].span(),
            [alice_eid].span(),
            build_slot_kinds(1, SLOT_KIND_DETERMINISTIC),
            [0].span(),
            [1].span(),
            [0].span(),
            [].span(),
            [].span(),
        );
}
