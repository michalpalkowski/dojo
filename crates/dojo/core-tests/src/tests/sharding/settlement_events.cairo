/// Settlement value-correctness tests across different model types.
///
/// These tests verify that the commitment-based settlement pipeline
/// correctly writes values to storage and that Dojo model reads return
/// the expected results. Tests cover: Foo (struct layout), Tile (multi-key),
/// Score (packed layout), and Add CRDT delta merging.

use dojo::model::{Model, ModelStorage, ModelStorageTest};
use dojo::sharding::compute_dojo_field_slot;
use dojo::sharding::slot::compute_dojo_packed_slot;
use dojo::sharding::request::{SlotEntry, SlotVerification, DeterministicProof};
use dojo::utils::{entity_id_from_keys, entity_id_from_serialized_keys};
use dojo::world::{
    IShardingSettlementDispatcher, IShardingSettlementDispatcherTrait,
    IShardingSettlementDevDispatcher, IShardingSettlementDevDispatcherTrait,
    IWorldDispatcherTrait,
};
use starknet::ContractAddress;

use crate::tests::helpers::{
    Foo, deploy_world_and_foo, Tile, deploy_world_with_tile, Score, deploy_world_with_score,
};

/// Make test_address() the world owner before deploying.
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
                key_derivation_chain: [member_selector].span(),
                packed_offset: 0,
            },
        ),
    }
}

/// Build a deterministic SlotEntry for a packed-layout model (member_selector=0).
fn packed_slot_entry(
    key: felt252,
    value: felt252,
    model_selector: felt252,
    entity_id: felt252,
    initial_value: felt252,
) -> SlotEntry {
    SlotEntry {
        key,
        value,
        model_selector,
        entity_id,
        member_selector: 0,
        initial_value,
        verification: SlotVerification::Deterministic(
            DeterministicProof { key_derivation_chain: [].span(), packed_offset: 0 },
        ),
    }
}

/// Helper: settle as world owner.
fn do_settle(
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
            [].span(), // entity_model_selectors (not needed in unit tests)
            [].span(), // entity_keys_flat (not needed in unit tests)
        );
    snforge_std::stop_cheat_caller_address(world_address);
}

// ── Foo (Struct Layout) ─────────────────────────────────────────────────

#[test]
fn test_settlement_set_updates_all_fields() {
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
    do_settle(
        world_address, 1,
        [
            field_slot_entry(slot_a, 999, model_selector, entity_id, sel_a, 0),
            field_slot_entry(slot_b, 777, model_selector, entity_id, sel_b, 0),
        ].span(),
    );

    let result: Foo = world.read_model(bob);
    assert(result.a == 999, 'a should be 999');
    assert(result.b == 777, 'b should be 777');
}

#[test]
fn test_settlement_add_crdt_produces_merged_value() {
    cheat_world_owner();
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    // Initial a=100 (entity locked after request, so current stays 100).
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let (sel_a, sel_b) = foo_field_selectors();
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

    // Shard produced 150 for slot_a (initial=100, delta=50).
    do_settle(
        world_address, 1,
        [
            field_slot_entry(slot_a, 150, model_selector, entity_id, sel_a, 100),
        ].span(),
    );

    // current(100) + (shard(150) - initial(100)) = 150.
    let result: Foo = world.read_model(bob);
    assert(result.a == 150, 'merged: 100+(150-100)=150');
    assert(result.b == 200, 'b unchanged');
}

#[test]
fn test_cancel_does_not_change_values() {
    cheat_world_owner();
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let entity_id = entity_id_from_keys(@bob);
    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement.cancel_shard(1);
    snforge_std::stop_cheat_caller_address(world_address);

    let result: Foo = world.read_model(bob);
    assert(result.a == 100, 'cancel: a unchanged');
    assert(result.b == 200, 'cancel: b unchanged');
}

// ── Tile (Multi-Key Struct Layout) ──────────────────────────────────────

#[test]
fn test_settlement_multi_key_model() {
    cheat_world_owner();
    let (mut world, model_selector) = deploy_world_with_tile();
    let world_address = world.dispatcher.contract_address;

    let tile = Tile { col: 5, row: 10, category: 3, entity_id: 42 };
    world.write_model_test(@tile);

    let keys: Span<felt252> = [5_felt252, 10_felt252].span();
    let entity_id = entity_id_from_serialized_keys(keys);

    let layout = Model::<Tile>::layout();
    let (sel_category, sel_entity_id) = if let dojo::meta::Layout::Struct(fields) = layout {
        ((*fields[0]).selector, (*fields[1]).selector)
    } else {
        panic!("expected struct layout")
    };
    let slot_cat = compute_dojo_field_slot(model_selector, entity_id, sel_category);
    let slot_eid = compute_dojo_field_slot(model_selector, entity_id, sel_entity_id);

    world.dispatcher.request_sharding([entity_id].span(), [].span());
    do_settle(
        world_address, 1,
        [
            field_slot_entry(slot_cat, 7, model_selector, entity_id, sel_category, 0),
            field_slot_entry(slot_eid, 99, model_selector, entity_id, sel_entity_id, 0),
        ].span(),
    );

    let result: Tile = world.read_model((5_u32, 10_u32));
    assert(result.category == 7, 'category should be 7');
    assert(result.entity_id == 99, 'entity_id should be 99');
}

#[test]
fn test_settlement_new_entity_created_on_shard() {
    cheat_world_owner();
    let (mut world, model_selector) = deploy_world_with_tile();
    let world_address = world.dispatcher.contract_address;

    // Do NOT write initial tile — entity created entirely on shard.
    let keys: Span<felt252> = [3_felt252, 7_felt252].span();
    let entity_id = entity_id_from_serialized_keys(keys);

    let layout = Model::<Tile>::layout();
    let (sel_category, sel_entity_id) = if let dojo::meta::Layout::Struct(fields) = layout {
        ((*fields[0]).selector, (*fields[1]).selector)
    } else {
        panic!("expected struct layout")
    };
    let slot_cat = compute_dojo_field_slot(model_selector, entity_id, sel_category);
    let slot_eid = compute_dojo_field_slot(model_selector, entity_id, sel_entity_id);

    world.dispatcher.request_sharding([entity_id].span(), [].span());
    do_settle(
        world_address, 1,
        [
            field_slot_entry(slot_cat, 2, model_selector, entity_id, sel_category, 0),
            field_slot_entry(slot_eid, 55, model_selector, entity_id, sel_entity_id, 0),
        ].span(),
    );

    let result: Tile = world.read_model((3_u32, 7_u32));
    assert(result.category == 2, 'category should be 2');
    assert(result.entity_id == 55, 'entity_id should be 55');
}

// ── Score (Packed Layout) ───────────────────────────────────────────────

#[test]
fn test_settlement_packed_model() {
    cheat_world_owner();
    let (mut world, model_selector) = deploy_world_with_score();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let score = Score { player: bob, points: 1000, level: 5 };
    world.write_model_test(@score);

    let entity_id = entity_id_from_keys(@bob);

    // Score: u128 (128 bits) + u32 (32 bits) = 160 bits → 1 packed slot.
    let packed_slot = compute_dojo_packed_slot(model_selector, entity_id);

    // Pack new values: points=2000, level=10.
    // u128 in bits 0..127, u32 in bits 128..159 → packed = 2000 + 10 * 2^128.
    let packed_value: felt252 = 2000 + 10 * 0x100000000000000000000000000000000;

    world.dispatcher.request_sharding([entity_id].span(), [].span());
    do_settle(
        world_address, 1,
        [
            packed_slot_entry(packed_slot, packed_value, model_selector, entity_id, 0),
        ].span(),
    );

    let result: Score = world.read_model(bob);
    assert(result.points == 2000, 'points should be 2000');
    assert(result.level == 10, 'level should be 10');
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

    // Shard 1: Set a=500, b=300.
    world.dispatcher.request_sharding([entity_id].span(), [].span());
    do_settle(
        world_address, 1,
        [
            field_slot_entry(slot_a, 500, model_selector, entity_id, sel_a, 100),
            field_slot_entry(slot_b, 300, model_selector, entity_id, sel_b, 0),
        ].span(),
    );

    // Verify shard 1 result: Add: current(100) + (500 - 100) = 500; Set: 300.
    let mid: Foo = world.read_model(bob);
    assert(mid.a == 500, 'shard1: a=500');

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

    let entity_id = entity_id_from_keys(@bob);

    world.dispatcher.request_sharding([entity_id].span(), [].span());
    // Empty slots — nothing changed on shard.
    do_settle(world_address, 1, [].span());

    let result: Foo = world.read_model(bob);
    assert(result.a == 100, 'no change: a=100');
    assert(result.b == 200, 'no change: b=200');
}

// ── Cancel After Request ─────────────────────────────────────────────

#[test]
fn test_cancel_after_request() {
    cheat_world_owner();
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let entity_id = entity_id_from_keys(@bob);

    world.dispatcher.request_sharding([entity_id].span(), [].span());

    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement.cancel_shard(1);
    snforge_std::stop_cheat_caller_address(world_address);

    let result: Foo = world.read_model(bob);
    assert(result.a == 100, 'cancel: a unchanged');
    assert(result.b == 200, 'cancel: b unchanged');

    // Entity unlocked — can re-shard.
    world.dispatcher.request_sharding([entity_id].span(), [].span());
}
