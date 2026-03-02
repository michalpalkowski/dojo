use dojo::model::{Model, ModelStorage, ModelStorageTest};
use dojo::sharding::component::{IContractComponentDispatcher, IContractComponentDispatcherTrait};
use dojo::sharding::compute_dojo_field_slot;
use dojo::sharding::request::IntoShardModel;
use dojo::utils::entity_id_from_keys;
use dojo::world::IWorldDispatcherTrait;
use dojo_snf_test::declare_and_deploy;
use starknet::ContractAddress;

use crate::tests::helpers::{Foo, deploy_world_and_foo};

/// Helper: get the local layout field selectors for Foo (same compilation unit as write/read_model).
fn foo_field_selectors() -> (felt252, felt252) {
    let layout = Model::<Foo>::layout();
    if let dojo::meta::Layout::Struct(fields) = layout {
        ((*fields[0]).selector, (*fields[1]).selector)
    } else {
        panic!("expected struct layout")
    }
}

/// Test: request_sharding auto-computes slots for a Set CRDT model.
#[test]
fn test_request_sharding_set() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let foo = Foo { caller: bob, a: 100, b: 200 };
    world.write_model_test(@foo);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    // Use the new IWorld::request_sharding with ShardModel (includes layout).
    let layout = Model::<Foo>::layout();
    let models = [(model_selector, layout).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    // Compute slot_a using the same local field selectors.
    let (sel_a, _) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);

    let sharding = IContractComponentDispatcher { contract_address: world_address };

    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding.update_shard_state(array![(slot_a, 999)]);
    snforge_std::stop_cheat_caller_address(world_address);

    let result: Foo = world.read_model(bob);
    assert(result.a == 999, 'Set should overwrite');
    assert(result.b == 200, 'b should be unchanged');
}

/// Test: request_sharding with Add CRDT computes delta correctly.
#[test]
fn test_request_sharding_add_delta() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let foo = Foo { caller: bob, a: 100, b: 200 };
    world.write_model_test(@foo);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    // Use Add CRDT via shard_add.
    let layout = Model::<Foo>::layout();
    let models = [(model_selector, layout).shard_add([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    // Mainchain changes a to 120 while shard is active.
    let foo_updated = Foo { caller: bob, a: 120, b: 200 };
    world.write_model_test(@foo_updated);

    // Shard saw initial=100, produced shard_value=150 (delta=50).
    let (sel_a, _) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);

    let sharding = IContractComponentDispatcher { contract_address: world_address };

    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding.update_shard_state(array![(slot_a, 150)]);
    snforge_std::stop_cheat_caller_address(world_address);

    // Expected: current(120) + (shard(150) - initial(100)) = 170
    let result: Foo = world.read_model(bob);
    assert(result.a == 170, 'Add delta incorrect');
    assert(result.b == 200, 'b should be unchanged');
}

/// Test: request_sharding creates slots for ALL fields in the model.
#[test]
fn test_request_sharding_all_fields() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let foo = Foo { caller: bob, a: 42, b: 77 };
    world.write_model_test(@foo);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let layout = Model::<Foo>::layout();
    let models = [(model_selector, layout).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    // Update both fields via shard.
    let (sel_a, sel_b) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let slot_b = compute_dojo_field_slot(model_selector, entity_id, sel_b);

    let sharding = IContractComponentDispatcher { contract_address: world_address };

    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding.update_shard_state(array![(slot_a, 111), (slot_b, 222)]);
    snforge_std::stop_cheat_caller_address(world_address);

    let result: Foo = world.read_model(bob);
    assert(result.a == 111, 'a should be updated');
    assert(result.b == 222, 'b should be updated');
}

/// Test: shard_pn (PN-Counter) — both P and N fields are G-Counters (Add).
///
/// Scenario: Foo.a = P (additions), Foo.b = N (subtractions), balance = P - N.
/// On shard: P increases by 50 (mint), N increases by 30 (burn).
/// Main chain independently: P goes 100→120, N goes 50→60.
/// Expected merge: P_new = 120 + (150-100) = 170, N_new = 60 + (80-50) = 90.
/// Balance: 170 - 90 = 80.
#[test]
fn test_request_sharding_pn_counter() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    // P=100 (additions), N=50 (subtractions), balance = 50
    let foo = Foo { caller: bob, a: 100, b: 50 };
    world.write_model_test(@foo);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    // Use PNCounter CRDT via shard_pn — both fields become Add (G-Counter).
    let layout = Model::<Foo>::layout();
    let models = [(model_selector, layout).shard_pn([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    // Mainchain changes while shard is active: P 100→120, N 50→60 (balance 50→60).
    let foo_updated = Foo { caller: bob, a: 120, b: 60 };
    world.write_model_test(@foo_updated);

    // Shard: initial P=100, N=50. Shard adds 50 to P (mint), adds 30 to N (burn).
    // Shard final: P=150, N=80.
    let (sel_a, sel_b) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let slot_b = compute_dojo_field_slot(model_selector, entity_id, sel_b);

    let sharding = IContractComponentDispatcher { contract_address: world_address };

    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding.update_shard_state(array![(slot_a, 150), (slot_b, 80)]);
    snforge_std::stop_cheat_caller_address(world_address);

    // P: current(120) + (shard(150) - initial(100)) = 170
    // N: current(60) + (shard(80) - initial(50)) = 90
    // Balance: 170 - 90 = 80
    let result: Foo = world.read_model(bob);
    assert(result.a == 170, 'PN: P delta incorrect');
    assert(result.b == 90, 'PN: N delta incorrect');
}

/// Test: shard_pn with burn only (N increases, P unchanged) — simulates resource spending.
///
/// This is the key scenario that the old Add CRDT couldn't handle with a single balance field.
/// With PN-Counter, N is its own G-Counter so delta is always >= 0.
#[test]
fn test_request_sharding_pn_counter_burn_only() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    // P=1000 (total additions), N=200 (total subtractions), balance = 800
    let foo = Foo { caller: bob, a: 1000, b: 200 };
    world.write_model_test(@foo);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let layout = Model::<Foo>::layout();
    let models = [(model_selector, layout).shard_pn([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    // Shard: P unchanged (no minting), N increases by 300 (burning 300 resources).
    // Shard final: P=1000 (same), N=500.
    let (sel_a, sel_b) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let slot_b = compute_dojo_field_slot(model_selector, entity_id, sel_b);

    let sharding = IContractComponentDispatcher { contract_address: world_address };

    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding.update_shard_state(array![(slot_a, 1000), (slot_b, 500)]);
    snforge_std::stop_cheat_caller_address(world_address);

    // P: current(1000) + (shard(1000) - initial(1000)) = 1000 (no change)
    // N: current(200) + (shard(500) - initial(200)) = 500
    // Balance: 1000 - 500 = 500 (was 800, burned 300 on shard)
    let result: Foo = world.read_model(bob);
    assert(result.a == 1000, 'PN burn: P should be unchanged');
    assert(result.b == 500, 'PN burn: N delta incorrect');
}

/// Test: end_shard forwards to the proxy.
#[test]
fn test_end_shard() {
    let (mut world, model_selector) = deploy_world_and_foo();

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let foo = Foo { caller: bob, a: 100, b: 200 };
    world.write_model_test(@foo);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let layout = Model::<Foo>::layout();
    let models = [(model_selector, layout).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    // Should not panic — forwards to proxy.end_shard().
    world.dispatcher.end_shard();
}
