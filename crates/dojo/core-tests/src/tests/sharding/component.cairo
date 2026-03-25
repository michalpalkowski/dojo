use dojo::model::{Model, ModelStorage, ModelStorageTest};
use dojo::sharding::compute_dojo_field_slot;
use dojo::sharding::slot::compute_dojo_packed_slot;
use dojo::sharding::request::{SlotEntry, SlotVerification, DeterministicProof};
use dojo::utils::entity_id_from_keys;
use dojo::world::{
    IShardingSettlementDispatcher, IShardingSettlementDispatcherTrait,
    IShardingSettlementDevDispatcher, IShardingSettlementDevDispatcherTrait,
    IWorldDispatcherTrait,
};
use starknet::ContractAddress;

use crate::tests::helpers::{Foo, PackedPair, deploy_world_and_foo, deploy_world_with_packed_pair};

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

/// Build a SlotEntry for a struct-layout field (Deterministic with packed_offset=0).
fn make_field_slot(
    key: felt252,
    value: felt252,
    model_selector: felt252,
    entity_id: felt252,
    member_selector: felt252,
    initial_value: felt252,
) -> SlotEntry {
    SlotEntry {
        key,
        value,
        model_selector,
        entity_id,
        member_selector,
        initial_value,
        verification: SlotVerification::Deterministic(
            DeterministicProof {
                key_derivation_chain: [member_selector].span(),
                packed_offset: 0,
            },
        ),
    }
}

/// Helper: settle as world owner using settle_dev (no StorageCommitment proof).
fn settle_as_owner(
    world_address: ContractAddress,
    shard_id: felt252,
    slots: Span<SlotEntry>,
) {
    let settlement = IShardingSettlementDevDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement
        .settle_dev(
            shard_id,
            0, // end_block_number
            slots,
            [].span(), // entity_model_selectors
            [].span(), // entity_keys_flat
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
        [
            make_field_slot(slot_a, 500, model_selector, entity_id, sel_a, 0),
            make_field_slot(slot_b, 300, model_selector, entity_id, sel_b, 0),
        ].span(),
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
        [
            make_field_slot(slot_a, 999, model_selector, entity_id, sel_a, 0),
            make_field_slot(slot_b, 777, model_selector, entity_id, sel_b, 0),
        ].span(),
    );

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let result: Foo = world.read_model(bob);
    assert(result.a == 999, 'a should be overwritten');
    assert(result.b == 777, 'b should be overwritten');
}

#[test]
fn test_settle_partial_changes() {
    let (mut world, world_address, entity_id, _slot_a, slot_b, model_selector) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let (_sel_a, sel_b) = foo_field_selectors();
    // Only change slot_b.
    settle_as_owner(
        world_address, 1,
        [
            make_field_slot(slot_b, 777, model_selector, entity_id, sel_b, 0),
        ].span(),
    );

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let result: Foo = world.read_model(bob);
    assert(result.a == 100, 'a should be unchanged');
    assert(result.b == 777, 'b should be overwritten');
}

// ── Settlement: Add CRDT ────────────────────────────────────────────────

#[test]
fn test_settle_add_crdt_computes_delta() {
    let (mut world, world_address, entity_id, slot_a, _slot_b, model_selector) = setup_foo_shard();

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

    // Settle with Add: shard produced 150, initial was 100 -> delta = 50.
    settle_as_owner(
        world_address, 1,
        [
            make_field_slot(slot_a, 150, model_selector, entity_id, sel_a, 100),
        ].span(),
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
        [
            make_field_slot(slot_a, 150, model_selector, entity_id, sel_a, 100),
            make_field_slot(slot_b, 999, model_selector, entity_id, sel_b, 0),
        ].span(),
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
fn test_settle_dev_rejects_unauthorized_caller() {
    let (world, world_address, entity_id, slot_a, slot_b, model_selector) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let (sel_a, sel_b) = foo_field_selectors();

    // Call settle_dev as non-owner.
    let not_owner: ContractAddress = 0xdead.try_into().unwrap();
    let settlement = IShardingSettlementDevDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, not_owner);
    settlement
        .settle_dev(
            1,
            0, // end_block_number
            [
                make_field_slot(slot_a, 999, model_selector, entity_id, sel_a, 0),
                make_field_slot(slot_b, 777, model_selector, entity_id, sel_b, 0),
            ].span(),
            [].span(),
            [].span(),
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

    // Shard value (90) < initial (100) -> underflow.
    settle_as_owner(
        world_address, 1,
        [
            make_field_slot(slot_a, 90, model_selector, entity_id, sel_a, 100),
        ].span(),
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
        [
            make_field_slot(slot_a, 1, model_selector, entity_id, sel_a, 0),
        ].span(),
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
        [
            make_field_slot(slot_a, 999, model_selector, entity_id, sel_a, 0),
            make_field_slot(slot_b, 777, model_selector, entity_id, sel_b, 0),
        ].span(),
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
        [
            make_field_slot(slot_a, 500, model_selector, entity_id, sel_a, 0),
            make_field_slot(slot_b, 300, model_selector, entity_id, sel_b, 0),
        ].span(),
    );

    // Shard 2 — entity now unlocked, new shard_id = 2.
    world.dispatcher.request_sharding([entity_id].span(), [].span());
    settle_as_owner(
        world_address, 2,
        [
            make_field_slot(slot_a, 600, model_selector, entity_id, sel_a, 0),
            make_field_slot(slot_b, 400, model_selector, entity_id, sel_b, 0),
        ].span(),
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

    let settlement = IShardingSettlementDevDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    // Pass wrong model_selector (0xdead) for first slot — hash won't match slot_a.
    settlement
        .settle_dev(
            1,
            0, // end_block_number
            [
                SlotEntry {
                    key: slot_a,
                    value: 999,
                    model_selector: 0xdead, // wrong!
                    entity_id,
                    member_selector: sel_a,
                    initial_value: 0,
                    verification: SlotVerification::Deterministic(
                        DeterministicProof {
                            key_derivation_chain: [sel_a].span(),
                            packed_offset: 0,
                        },
                    ),
                },
                make_field_slot(slot_b, 777, model_selector, entity_id, sel_b, 0),
            ].span(),
            [].span(),
            [].span(),
        );
}

#[test]
#[should_panic(expected: ('Shard: entity not in shard',))]
fn test_settle_rejects_unlocked_entity() {
    let (world, world_address, entity_id, _slot_a, _slot_b, model_selector) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    // Compute a slot for a DIFFERENT entity that is NOT locked.
    let other_entity: felt252 = 0x999;
    let (sel_a, _sel_b) = foo_field_selectors();
    let other_slot = dojo::sharding::compute_dojo_field_slot(model_selector, other_entity, sel_a);

    let settlement = IShardingSettlementDevDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    // Try to settle slot belonging to unlocked entity.
    settlement
        .settle_dev(
            1,
            0, // end_block_number
            [
                make_field_slot(other_slot, 42, model_selector, other_entity, sel_a, 0),
            ].span(),
            [].span(),
            [].span(),
        );
}

#[test]
#[should_panic(expected: ('Shard: slot ownership mismatch',))]
fn test_settle_rejects_zero_comp_key_for_deterministic_slot() {
    let (world, world_address, entity_id, slot_a, _slot_b, model_selector) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let (sel_a, _sel_b) = foo_field_selectors();
    let settlement = IShardingSettlementDevDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    // Pass empty key_derivation_chain — derives entity_id, not combine_key(entity_id, sel_a),
    // so the recomputed slot won't match slot_a.
    settlement
        .settle_dev(
            1,
            0, // end_block_number
            [
                SlotEntry {
                    key: slot_a,
                    value: 42,
                    model_selector,
                    entity_id,
                    member_selector: sel_a,
                    initial_value: 0,
                    verification: SlotVerification::Deterministic(
                        DeterministicProof {
                            key_derivation_chain: [].span(), // empty — hash won't match
                            packed_offset: 0,
                        },
                    ),
                },
            ].span(),
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
    let settlement = IShardingSettlementDevDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement
        .settle_dev(
            1,
            0, // end_block_number
            [
                SlotEntry {
                    key: forged_key,
                    value: 999,
                    model_selector,
                    entity_id: alice_eid,
                    member_selector: 0,
                    initial_value: 0,
                    verification: SlotVerification::Deterministic(
                        DeterministicProof {
                            key_derivation_chain: [].span(),
                            packed_offset: 1,
                        },
                    ),
                },
            ].span(),
            [].span(),
            [].span(),
        );
}

// ── member_selector binding tests ───────────────────────────────────

#[test]
#[should_panic(expected: ('Shard: member_sel mismatch',))]
fn test_settle_rejects_empty_chain_with_nonzero_member_selector() {
    let (world, world_address, entity_id, _, _, model_selector) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let (sel_a, _sel_b) = foo_field_selectors();
    let packed_slot = compute_dojo_packed_slot(model_selector, entity_id);
    let settlement = IShardingSettlementDevDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement
        .settle_dev(
            1, 0,
            [
                SlotEntry {
                    key: packed_slot,
                    value: 42,
                    model_selector,
                    entity_id,
                    member_selector: sel_a, // non-zero but chain is empty
                    initial_value: 0,
                    verification: SlotVerification::Deterministic(
                        DeterministicProof {
                            key_derivation_chain: [].span(),
                            packed_offset: 0,
                        },
                    ),
                },
            ].span(),
            [].span(),
            [].span(),
        );
}

#[test]
#[should_panic(expected: ('Shard: member_sel mismatch',))]
fn test_settle_rejects_member_selector_not_matching_chain() {
    let (world, world_address, entity_id, slot_a, _, model_selector) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let (sel_a, sel_b) = foo_field_selectors();
    let settlement = IShardingSettlementDevDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement
        .settle_dev(
            1, 0,
            [
                SlotEntry {
                    key: slot_a,
                    value: 42,
                    model_selector,
                    entity_id,
                    member_selector: sel_b, // wrong! chain[0] = sel_a
                    initial_value: 0,
                    verification: SlotVerification::Deterministic(
                        DeterministicProof {
                            key_derivation_chain: [sel_a].span(),
                            packed_offset: 0,
                        },
                    ),
                },
            ].span(),
            [].span(),
            [].span(),
        );
}

// ── DynamicLock wrong entity ────────────────────────────────────────

#[test]
#[should_panic(expected: ('Shard: slot ownership mismatch',))]
fn test_settle_rejects_dynamic_lock_with_wrong_entity() {
    let (world, world_address, entity_id, _, _, model_selector) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let (sel_a, _sel_b) = foo_field_selectors();
    let wrong_entity: felt252 = 0xBAD;
    let wrong_lock_slot = dojo::sharding::slot::compute_dynamic_member_lock_slot(
        model_selector, wrong_entity, sel_a,
    );

    let settlement = IShardingSettlementDevDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement
        .settle_dev(
            1, 0,
            [
                SlotEntry {
                    key: wrong_lock_slot,
                    value: 1,
                    model_selector,
                    entity_id,
                    member_selector: sel_a,
                    initial_value: 0,
                    verification: SlotVerification::DynamicLock,
                },
            ].span(),
            [].span(),
            [].span(),
        );
}

// ── cancel_shard creator auth ───────────────────────────────────────

#[test]
fn test_cancel_shard_by_creator_succeeds() {
    let (mut world, world_address, entity_id, _, _, _) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement.cancel_shard(1);
    snforge_std::stop_cheat_caller_address(world_address);

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 42, b: 42 });
}

// ── register_shard_policy blocked during active shard ───────────────

#[test]
#[should_panic(expected: ('Shard: active shards exist',))]
fn test_register_policy_rejected_while_shard_active() {
    let (world, world_address, entity_id, _, _, model_selector) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    let world_disp = dojo::world::IWorldDispatcher { contract_address: world_address };
    world_disp.register_shard_policy(
        model_selector,
        dojo::sharding::request::CRDVariant::Add,
        [].span(),
    );
}

// ── Double settlement blocked ───────────────────────────────────────

#[test]
#[should_panic(expected: ('Shard: not found',))]
fn test_settle_same_shard_twice_rejected() {
    let (world, world_address, entity_id, slot_a, slot_b, model_selector) = setup_foo_shard();
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let (sel_a, sel_b) = foo_field_selectors();
    settle_as_owner(
        world_address, 1,
        [
            make_field_slot(slot_a, 500, model_selector, entity_id, sel_a, 0),
            make_field_slot(slot_b, 300, model_selector, entity_id, sel_b, 0),
        ].span(),
    );

    // Second settle for same shard_id → cleared, should fail.
    settle_as_owner(
        world_address, 1,
        [
            make_field_slot(slot_a, 999, model_selector, entity_id, sel_a, 0),
        ].span(),
    );
}

// ── Cancel nonexistent shard ────────────────────────────────────────

#[test]
#[should_panic(expected: ('Shard: not found',))]
fn test_cancel_nonexistent_shard_rejected() {
    let (_world, world_address, _, _, _, _) = setup_foo_shard();

    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement.cancel_shard(999);
}
