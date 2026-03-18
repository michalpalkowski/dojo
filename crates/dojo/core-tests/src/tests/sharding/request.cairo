/// Integration tests for the sharding request → register → settle pipeline.
///
/// These tests exercise the full lifecycle across different scenarios:
/// multi-entity shards, re-sharding after settlement, and cancellation.

use core::poseidon::poseidon_hash_span;
use dojo::model::{Model, ModelStorage, ModelStorageTest};
use dojo::sharding::compute_dojo_field_slot;
use dojo::utils::{entity_id_from_keys, combine_key};
use dojo::world::{
    IShardingSettlementDispatcher, IShardingSettlementDispatcherTrait, IWorldDispatcherTrait,
};
use starknet::ContractAddress;

use crate::tests::helpers::{Foo, deploy_world_and_foo};

fn build_comp_keys(entity_ids: Span<felt252>, member_selectors: Span<felt252>) -> Span<felt252> {
    let mut keys: Array<felt252> = ArrayTrait::new();
    let mut i: u32 = 0;
    while i < entity_ids.len() {
        let eid = *entity_ids[i];
        let member = *member_selectors[i];
        if member != 0 { keys.append(combine_key(eid, member)); }
        else { keys.append(eid); };
        i += 1;
    };
    keys.span()
}

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
    world.dispatcher.request_sharding([bob_eid, alice_eid].span());

    let bob_slot_a = compute_dojo_field_slot(model_selector, bob_eid, sel_a);
    let bob_slot_b = compute_dojo_field_slot(model_selector, bob_eid, sel_b);
    let alice_slot_a = compute_dojo_field_slot(model_selector, alice_eid, sel_a);
    let alice_slot_b = compute_dojo_field_slot(model_selector, alice_eid, sel_b);

    let all_keys: Span<felt252> = [bob_slot_a, bob_slot_b, alice_slot_a, alice_slot_b].span();
    let commitment = poseidon_hash_span(all_keys);
    let state_diff_hash = poseidon_hash_span(all_keys);

    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement.register_commitment(1, commitment, 4, [].span(), [].span());
    settlement.settle(
        1, all_keys, [0, 1, 2, 3].span(), [500, 300, 50, 30].span(), state_diff_hash, 0, 0,
        [model_selector, model_selector, model_selector, model_selector].span(),
        [bob_eid, bob_eid, alice_eid, alice_eid].span(),
        build_comp_keys([bob_eid, bob_eid, alice_eid, alice_eid].span(), [sel_a, sel_b, sel_a, sel_b].span()),
        [sel_a, sel_b, sel_a, sel_b].span(),
        [0, 0, 0, 0].span());
    snforge_std::stop_cheat_caller_address(world_address);

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

    world.dispatcher.request_sharding([bob_eid, alice_eid].span());

    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement.cancel_shard(1);
    snforge_std::stop_cheat_caller_address(world_address);

    // Both entities should be unlocked — can shard again.
    world.dispatcher.request_sharding([bob_eid, alice_eid].span());

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
    let all_keys: Span<felt252> = [slot_a, slot_b].span();
    let commitment = poseidon_hash_span(all_keys);
    let state_diff_hash = poseidon_hash_span(all_keys);

    let settlement = IShardingSettlementDispatcher { contract_address: world_address };

    // Shard 1: Set a=500, b=300.
    world.dispatcher.request_sharding([entity_id].span());
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement.register_commitment(1, commitment, 2, [].span(), [].span());
    settlement.settle(1, all_keys, [0, 1].span(), [500, 300].span(), state_diff_hash, 0, 0,
        [model_selector, model_selector].span(), [entity_id, entity_id].span(), build_comp_keys([entity_id, entity_id].span(), [sel_a, sel_b].span()), [sel_a, sel_b].span(), [0, 0].span());
    snforge_std::stop_cheat_caller_address(world_address);

    // Shard 2: Add CRDT on slot_a (initial=500 from shard 1 result).
    world.dispatcher.request_sharding([entity_id].span());
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement.register_commitment(2, commitment, 2, [slot_a].span(), [500].span());
    // Shard sees initial=500, produces 600 → delta=100.
    let diff_hash_a = poseidon_hash_span([slot_a].span());
    settlement.settle(2, all_keys, [0].span(), [600].span(), diff_hash_a, 0, 0,
        [model_selector].span(), [entity_id].span(), build_comp_keys([entity_id].span(), [sel_a].span()), [sel_a].span(), [0].span());
    snforge_std::stop_cheat_caller_address(world_address);

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

    world.dispatcher.request_sharding([entity_id].span());
    let all_keys: Span<felt252> = [slot_a, slot_b].span();
    let commitment = poseidon_hash_span(all_keys);
    // Empty changed_indices — nothing changed on shard.
    let empty_diff_hash = poseidon_hash_span([].span());

    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement.register_commitment(1, commitment, 2, [].span(), [].span());
    settlement.settle(1, all_keys, [].span(), [].span(), empty_diff_hash, 0, 0,
        [].span(), [].span(), [].span(), [].span(), [].span());
    snforge_std::stop_cheat_caller_address(world_address);

    let result: Foo = world.read_model(bob);
    assert(result.a == 100, 'no change: a=100');
    assert(result.b == 200, 'no change: b=200');
}

// ── Cancel After Commitment ─────────────────────────────────────────────

#[test]
fn test_cancel_after_commitment_registered() {
    cheat_world_owner();
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let (sel_a, sel_b) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let slot_b = compute_dojo_field_slot(model_selector, entity_id, sel_b);

    world.dispatcher.request_sharding([entity_id].span());
    let all_keys: Span<felt252> = [slot_a, slot_b].span();
    let commitment = poseidon_hash_span(all_keys);

    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement.register_commitment(1, commitment, 2, [slot_a].span(), [100].span());
    // Cancel instead of settling — should clean up commitment + add state.
    settlement.cancel_shard(1);
    snforge_std::stop_cheat_caller_address(world_address);

    let result: Foo = world.read_model(bob);
    assert(result.a == 100, 'cancel: a unchanged');
    assert(result.b == 200, 'cancel: b unchanged');

    // Entity unlocked — can re-shard.
    world.dispatcher.request_sharding([entity_id].span());
}
