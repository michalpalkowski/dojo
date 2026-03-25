/// Integration tests for the sharding request → settle pipeline.
///
/// These tests exercise the full lifecycle across different scenarios:
/// multi-entity shards, re-sharding after settlement, and cancellation.

use dojo::model::{Model, ModelStorage, ModelStorageTest};
use dojo::sharding::compute_dojo_field_slot;
use dojo::sharding::request::{SlotEntry, SlotVerification, DeterministicProof};
use dojo::utils::{entity_id_from_keys, combine_key};
use dojo::world::{
    IShardingSettlementDispatcher, IShardingSettlementDispatcherTrait, IWorldDispatcherTrait,
};
use starknet::ContractAddress;

use crate::tests::helpers::{Foo, deploy_world_and_foo};

fn cheat_world_owner() {
    snforge_std::start_cheat_account_contract_address_global(snforge_std::test_address());
}

fn foo_field_selectors() -> (felt252, felt252) {
    let layout = Model::<Foo>::layout();
    if let dojo::meta::Layout::Struct(fields) = layout {
        ((*fields[0]).selector, (*fields[1]).selector)
    } else {
        panic!("expected struct layout")
    }
}

/// Build a deterministic SlotEntry for a struct-layout field.
fn field_slot_entry(
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
                computation_key: combine_key(entity_id, member_selector),
                packed_offset: 0,
            },
        ),
    }
}

/// Helper: settle as world owner.
fn do_settle(
    world_address: ContractAddress,
    shard_id: felt252,
    slots: Span<SlotEntry>,
) {
    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement
        .settle(
            shard_id,
            0, // global_state_root
            0, // end_block_number
            slots,
            [].span(), // entity_model_selectors
            [].span(), // entity_keys_flat
        );
    snforge_std::stop_cheat_caller_address(world_address);
}

// ── Multi-Entity Shard ──────────────────────────────────────────────────

#[test]
fn test_multi_entity_shard_settles_both() {
    cheat_world_owner();
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let alice: ContractAddress = 0xa11ce.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });
    world.write_model_test(@Foo { caller: alice, a: 10, b: 20 });

    let (sel_a, sel_b) = foo_field_selectors();

    let bob_eid = entity_id_from_keys(@bob);
    let alice_eid = entity_id_from_keys(@alice);

    // Lock both entities in one shard.
    world.dispatcher.request_sharding([bob_eid, alice_eid].span(), [].span());

    let bob_slot_a = compute_dojo_field_slot(model_selector, bob_eid, sel_a);
    let bob_slot_b = compute_dojo_field_slot(model_selector, bob_eid, sel_b);
    let alice_slot_a = compute_dojo_field_slot(model_selector, alice_eid, sel_a);
    let alice_slot_b = compute_dojo_field_slot(model_selector, alice_eid, sel_b);

    do_settle(
        world_address, 1,
        [
            field_slot_entry(bob_slot_a, 500, model_selector, bob_eid, sel_a, 0),
            field_slot_entry(bob_slot_b, 300, model_selector, bob_eid, sel_b, 0),
            field_slot_entry(alice_slot_a, 50, model_selector, alice_eid, sel_a, 0),
            field_slot_entry(alice_slot_b, 30, model_selector, alice_eid, sel_b, 0),
        ].span(),
    );

    let bob_result: Foo = world.read_model(bob);
    assert(bob_result.a == 500, 'bob.a = 500');
    assert(bob_result.b == 300, 'bob.b = 300');

    let alice_result: Foo = world.read_model(alice);
    assert(alice_result.a == 50, 'alice.a = 50');
    assert(alice_result.b == 30, 'alice.b = 30');
}

#[test]
fn test_multi_entity_cancel_unlocks_all() {
    cheat_world_owner();
    let (mut world, _) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let alice: ContractAddress = 0xa11ce.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });
    world.write_model_test(@Foo { caller: alice, a: 10, b: 20 });

    let bob_eid = entity_id_from_keys(@bob);
    let alice_eid = entity_id_from_keys(@alice);

    world.dispatcher.request_sharding([bob_eid, alice_eid].span(), [].span());

    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement.cancel_shard(1);
    snforge_std::stop_cheat_caller_address(world_address);

    // Both entities should be unlocked — can shard again.
    world.dispatcher.request_sharding([bob_eid, alice_eid].span(), [].span());

    // Values unchanged.
    let bob_result: Foo = world.read_model(bob);
    assert(bob_result.a == 100, 'bob unchanged');
    let alice_result: Foo = world.read_model(alice);
    assert(alice_result.a == 10, 'alice unchanged');
}

// ── Re-Sharding After Settlement ────────────────────────────────────────

#[test]
fn test_reshard_after_settlement_uses_new_values() {
    cheat_world_owner();
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let (sel_a, sel_b) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let slot_b = compute_dojo_field_slot(model_selector, entity_id, sel_b);

    // Shard 1: Set a=500, b=300.
    world.dispatcher.request_sharding([entity_id].span(), [].span());
    do_settle(
        world_address, 1,
        [
            field_slot_entry(slot_a, 500, model_selector, entity_id, sel_a, 0),
            field_slot_entry(slot_b, 300, model_selector, entity_id, sel_b, 0),
        ].span(),
    );

    // Shard 2: Add CRDT on slot_a (initial=500 from shard 1 result).
    world.dispatcher.request_sharding([entity_id].span(), [].span());
    // Shard sees initial=500, produces 600 → delta=100.
    do_settle(
        world_address, 2,
        [
            field_slot_entry(slot_a, 600, model_selector, entity_id, sel_a, 500),
        ].span(),
    );

    // Expected: current(500) + (shard(600) - initial(500)) = 600.
    let result: Foo = world.read_model(bob);
    assert(result.a == 600, 'reshard: a=600');
    assert(result.b == 300, 'reshard: b unchanged');
}

// ── No-Change Settlement ────────────────────────────────────────────────

#[test]
fn test_settle_with_no_changes() {
    cheat_world_owner();
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let (sel_a, sel_b) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let slot_b = compute_dojo_field_slot(model_selector, entity_id, sel_b);

    world.dispatcher.request_sharding([entity_id].span(), [].span());

    // Empty slots — nothing changed on shard.
    do_settle(world_address, 1, [].span());

    let result: Foo = world.read_model(bob);
    assert(result.a == 100, 'no change: a=100');
    assert(result.b == 200, 'no change: b=200');
}

// ── Cancel After Request ────────────────────────────────────────────────

#[test]
fn test_cancel_after_request_unlocks_entity() {
    cheat_world_owner();
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let (sel_a, sel_b) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let slot_b = compute_dojo_field_slot(model_selector, entity_id, sel_b);

    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    // Cancel instead of settling — should clean up shard state.
    settlement.cancel_shard(1);
    snforge_std::stop_cheat_caller_address(world_address);

    let result: Foo = world.read_model(bob);
    assert(result.a == 100, 'cancel: a unchanged');
    assert(result.b == 200, 'cancel: b unchanged');

    // Entity unlocked — can re-shard.
    world.dispatcher.request_sharding([entity_id].span(), [].span());
}
