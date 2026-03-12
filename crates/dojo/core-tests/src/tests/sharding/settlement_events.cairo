use dojo::model::{Model, ModelStorage, ModelStorageTest};
use dojo::sharding::compute_dojo_field_slot;
use dojo::sharding::request::{IntoShardField, IntoShardModel, ShardModel};
use dojo::utils::entity_id_from_keys;
use dojo::world::{
    IShardingProxyDispatcher, IShardingProxyDispatcherTrait, IWorldDispatcherTrait,
    world as world_contract,
};
use dojo_snf_test::declare_and_deploy;
use snforge_std::{EventSpyAssertionsTrait, spy_events};
use starknet::ContractAddress;

use dojo::sharding::slot::compute_dojo_packed_slot;
use crate::tests::helpers::{
    Foo, deploy_world_and_foo, Tile, deploy_world_with_tile, Score, deploy_world_with_score,
};

fn foo_field_selectors() -> (felt252, felt252) {
    let layout = Model::<Foo>::layout();
    if let dojo::meta::Layout::Struct(fields) = layout {
        ((*fields[0]).selector, (*fields[1]).selector)
    } else {
        panic!("expected struct layout")
    }
}


/// Test: update_shard_state emits StoreSetRecord when entity keys are stored.
#[test]
fn test_settlement_emits_store_set_record() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let foo = Foo { caller: bob, a: 100, b: 200 };
    world.write_model_test(@foo);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let layout = Model::<Foo>::layout();

    let models = [(model_selector, layout).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (sel_a, sel_b) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let slot_b = compute_dojo_field_slot(model_selector, entity_id, sel_b);

    let mut spy = spy_events();

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.update_shard_state(array![(slot_a, 999), (slot_b, 777)]);
    snforge_std::stop_cheat_caller_address(world_address);

    // Check values by reading the model
    let result: Foo = world.read_model(bob);
    assert(result.a == 999, 'a should be updated');
    assert(result.b == 777, 'b should be updated');

    // Event assertion
    spy
        .assert_emitted(
            @array![
                (
                    world_address,
                    world_contract::Event::StoreSetRecord(
                        world_contract::StoreSetRecord {
                            selector: model_selector,
                            entity_id: entity_id,
                            keys: [bob.into()].span(),
                            values: [999, 777].span(),
                        },
                    ),
                ),
            ],
        );
}


/// Test: Add CRDT — emitted value is the delta-merged result, not raw shard value.
#[test]
fn test_settlement_add_crdt_emits_merged_value() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let foo = Foo { caller: bob, a: 100, b: 200 };
    world.write_model_test(@foo);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let layout = Model::<Foo>::layout();
    let models = [(model_selector, layout).shard_add([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    // Mainchain changes a from 100 → 120 while shard is active
    let foo_updated = Foo { caller: bob, a: 120, b: 200 };
    world.write_model_test(@foo_updated);

    let (sel_a, _) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);

    let mut spy = spy_events();

    // Shard saw initial=100, produced shard_value=150 (delta=50)
    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.update_shard_state(array![(slot_a, 150)]);
    snforge_std::stop_cheat_caller_address(world_address);

    // Expected merged value: current(120) + (shard(150) - initial(100)) = 170
    // StoreSetRecord emits full entity values [a, b] after CRDT merge
    spy
        .assert_emitted(
            @array![
                (
                    world_address,
                    world_contract::Event::StoreSetRecord(
                        world_contract::StoreSetRecord {
                            selector: model_selector,
                            entity_id: entity_id,
                            keys: [bob.into()].span(),
                            values: [170, 200].span(),
                        },
                    ),
                ),
            ],
        );
}

/// Test: cancel_shard_state cleans metadata and emits no StoreUpdateMember events.
#[test]
fn test_cancel_clears_metadata_no_events() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let foo = Foo { caller: bob, a: 100, b: 200 };
    world.write_model_test(@foo);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let layout = Model::<Foo>::layout();
    let models = [(model_selector, layout).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (sel_a, _) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);

    let mut spy = spy_events();

    // Cancel via IShardingProxy wrapper
    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.cancel_shard_state(array![slot_a].span());
    snforge_std::stop_cheat_caller_address(world_address);

    // No StoreUpdateMember should be emitted
    spy
        .assert_not_emitted(
            @array![
                (
                    world_address,
                    world_contract::Event::StoreUpdateMember(
                        world_contract::StoreUpdateMember {
                            selector: model_selector,
                            entity_id: entity_id,
                            member_selector: sel_a,
                            values: [100].span(),
                        },
                    ),
                ),
            ],
        );

    // Value should remain unchanged
    let result: Foo = world.read_model(bob);
    assert(result.a == 100, 'cancel should not change value');
}

/// Test: partial cancel keeps entity keys for remaining active slots.
///
/// Policy: partial cancel is allowed. If only part of an entity's active slots
/// are canceled, later settlement of the remaining slots must still emit
/// StoreSetRecord (with keys), not StoreUpdateMember.
#[test]
fn test_partial_cancel_keeps_keys_for_remaining_slots() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let foo = Foo { caller: bob, a: 100, b: 200 };
    world.write_model_test(@foo);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let layout = Model::<Foo>::layout();
    let models = [(model_selector, layout).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (sel_a, sel_b) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let slot_b = compute_dojo_field_slot(model_selector, entity_id, sel_b);

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };

    // Cancel only one slot.
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.cancel_shard_state(array![slot_a].span());
    snforge_std::stop_cheat_caller_address(world_address);

    let mut spy = spy_events();

    // Settle the remaining slot.
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.update_shard_state(array![(slot_b, 999)]);
    snforge_std::stop_cheat_caller_address(world_address);

    // Remaining settlement should still emit StoreSetRecord with keys.
    spy
        .assert_emitted(
            @array![
                (
                    world_address,
                    world_contract::Event::StoreSetRecord(
                        world_contract::StoreSetRecord {
                            selector: model_selector,
                            entity_id,
                            keys: [bob.into()].span(),
                            values: [100, 999].span(),
                        },
                    ),
                ),
            ],
        );

    // Explicitly ensure it did not degrade to member-only update.
    spy
        .assert_not_emitted(
            @array![
                (
                    world_address,
                    world_contract::Event::StoreUpdateMember(
                        world_contract::StoreUpdateMember {
                            selector: model_selector,
                            entity_id,
                            member_selector: sel_b,
                            values: [999].span(),
                        },
                    ),
                ),
            ],
        );
}

/// Test: per-field mixed CRDT — Add field emits merged value, Set field emits overwritten value.
#[test]
fn test_settlement_per_field_mixed_crdt_events() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let foo = Foo { caller: bob, a: 100, b: 200 };
    world.write_model_test(@foo);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    // Per-field: field a → Add, field b → Set
    let (sel_a, sel_b) = foo_field_selectors();
    let models = [
        ShardModel {
            selector: model_selector,
            keys: [bob.into()].span(),
            fields: [sel_a.as_add(), sel_b.as_set()].span(),
        },
    ]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);

    // Mainchain changes a from 100 → 120 while shard is active
    let foo_updated = Foo { caller: bob, a: 120, b: 200 };
    world.write_model_test(@foo_updated);

    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let slot_b = compute_dojo_field_slot(model_selector, entity_id, sel_b);

    let mut spy = spy_events();

    // Shard: a initial=100 → shard=150 (delta=50), b = 999 (Set overwrite)
    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.update_shard_state(array![(slot_a, 150), (slot_b, 999)]);
    snforge_std::stop_cheat_caller_address(world_address);

    // a: current(120) + (shard(150) - initial(100)) = 170
    // b: 999 (Set overwrite)
    // StoreSetRecord emitted with full entity values after CRDT merge
    spy
        .assert_emitted(
            @array![
                (
                    world_address,
                    world_contract::Event::StoreSetRecord(
                        world_contract::StoreSetRecord {
                            selector: model_selector,
                            entity_id: entity_id,
                            keys: [bob.into()].span(),
                            values: [170, 999].span(),
                        },
                    ),
                ),
            ],
        );
}

/// Test: Building-like multi-key model — StoreSetRecord includes composite keys.
/// Mimics a multi-key model with two composite keys (col, row),
/// fields stored per-slot (Layout::Struct). Verifies that:
/// 1. Entity keys (composite) are correctly stored and emitted
/// 2. Values are in Serde format (one felt252 per field)
/// 3. Torii can create the entity with proper indexed key columns
#[test]
fn test_settlement_building_like_multi_key() {
    let (mut world, model_selector) = deploy_world_with_tile();
    let world_address = world.dispatcher.contract_address;

    // Write an initial tile — simulating a building already on the map
    let tile = Tile { col: 5, row: 10, category: 3, entity_id: 42 };
    world.write_model_test(@tile);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    // Shard the tile with composite keys [col, row]
    let layout = Model::<Tile>::layout();
    let keys: Span<felt252> = [5_felt252, 10_felt252].span();
    let models = [(model_selector, layout).shard(keys)].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let entity_id = dojo::utils::entity_id_from_serialized_keys(keys);

    // Get field selectors from layout
    let (sel_category, sel_entity_id) = if let dojo::meta::Layout::Struct(fields) = layout {
        ((*fields[0]).selector, (*fields[1]).selector)
    } else {
        panic!("expected struct layout")
    };
    let slot_cat = compute_dojo_field_slot(model_selector, entity_id, sel_category);
    let slot_eid = compute_dojo_field_slot(model_selector, entity_id, sel_entity_id);

    let mut spy = spy_events();

    // Shard changes: category 3 → 7, entity_id 42 → 99
    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.update_shard_state(array![(slot_cat, 7), (slot_eid, 99)]);
    snforge_std::stop_cheat_caller_address(world_address);

    // StoreSetRecord must include composite keys [col=5, row=10]
    spy
        .assert_emitted(
            @array![
                (
                    world_address,
                    world_contract::Event::StoreSetRecord(
                        world_contract::StoreSetRecord {
                            selector: model_selector,
                            entity_id,
                            keys: [5, 10].span(),
                            values: [7, 99].span(),
                        },
                    ),
                ),
            ],
        );

    // Verify Dojo can read back via normal path
    let result: Tile = world.read_model((5_u32, 10_u32));
    assert(result.category == 7, 'category should be updated');
    assert(result.entity_id == 99, 'entity_id should be updated');
}

/// Test: Building creation on shard — entity didn't exist before sharding.
/// This is the critical case: building is created entirely on the shard, so the
/// main chain has no existing entity. StoreSetRecord must carry keys so Torii can
/// create the entity from scratch.
#[test]
fn test_settlement_new_entity_created_on_shard() {
    let (mut world, model_selector) = deploy_world_with_tile();
    let world_address = world.dispatcher.contract_address;

    // Do NOT write an initial tile — simulating a building created on the shard
    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let layout = Model::<Tile>::layout();
    let keys: Span<felt252> = [3_felt252, 7_felt252].span();
    let models = [(model_selector, layout).shard(keys)].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let entity_id = dojo::utils::entity_id_from_serialized_keys(keys);

    let (sel_category, sel_entity_id) = if let dojo::meta::Layout::Struct(fields) = layout {
        ((*fields[0]).selector, (*fields[1]).selector)
    } else {
        panic!("expected struct layout")
    };
    let slot_cat = compute_dojo_field_slot(model_selector, entity_id, sel_category);
    let slot_eid = compute_dojo_field_slot(model_selector, entity_id, sel_entity_id);

    let mut spy = spy_events();

    // Shard created a brand new building: category=2, entity_id=55
    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.update_shard_state(array![(slot_cat, 2), (slot_eid, 55)]);
    snforge_std::stop_cheat_caller_address(world_address);

    // Must emit StoreSetRecord with keys so Torii can create entity from scratch
    spy
        .assert_emitted(
            @array![
                (
                    world_address,
                    world_contract::Event::StoreSetRecord(
                        world_contract::StoreSetRecord {
                            selector: model_selector,
                            entity_id,
                            keys: [3, 7].span(),
                            values: [2, 55].span(),
                        },
                    ),
                ),
            ],
        );

    // Verify entity exists and is readable via Dojo
    let result: Tile = world.read_model((3_u32, 7_u32));
    assert(result.category == 2, 'category should be set');
    assert(result.entity_id == 55, 'entity_id should be set');
}

/// Test: Packed model settlement — values must be unpacked into Serde format.
/// Score uses IntrospectPacked (Layout::Fixed), so raw storage contains bit-packed
/// felt252 values. The emitted StoreSetRecord.values must be in unpacked Serde
/// format [points, level] — not the raw packed felt252.
#[test]
fn test_settlement_packed_model_unpacks_values() {
    let (mut world, model_selector) = deploy_world_with_score();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let score = Score { player: bob, points: 1000, level: 5 };
    world.write_model_test(@score);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let layout = Model::<Score>::layout();
    let models = [(model_selector, layout).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let entity_id = entity_id_from_keys(@bob);

    // For packed models, slots are based on compute_dojo_packed_slot + offset.
    // Score has u128 (128 bits) + u32 (32 bits) = 160 bits → 1 packed slot.
    let packed_slot = compute_dojo_packed_slot(model_selector, entity_id);

    // Pack the new values the same way Dojo would: points=2000, level=10
    // u128 in bits 0..127, u32 in bits 128..159 → packed = 2000 | (10 << 128)
    let packed_value: felt252 = 2000 + 10 * 0x100000000000000000000000000000000;

    let mut spy = spy_events();

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.update_shard_state(array![(packed_slot, packed_value)]);
    snforge_std::stop_cheat_caller_address(world_address);

    // StoreSetRecord values must be UNPACKED Serde values [points=2000, level=10],
    // not the raw packed felt252.
    spy
        .assert_emitted(
            @array![
                (
                    world_address,
                    world_contract::Event::StoreSetRecord(
                        world_contract::StoreSetRecord {
                            selector: model_selector,
                            entity_id,
                            keys: [bob.into()].span(),
                            values: [2000, 10].span(),
                        },
                    ),
                ),
            ],
        );

    // Verify Dojo can read back correctly
    let result: Score = world.read_model(bob);
    assert(result.points == 2000, 'points should be updated');
    assert(result.level == 10, 'level should be updated');
}
