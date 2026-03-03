use dojo::model::{Model, ModelStorage, ModelStorageTest};
use dojo::sharding::component::{IContractComponentDispatcher, IContractComponentDispatcherTrait};
use dojo::sharding::compute_dojo_field_slot;
use dojo::sharding::request::IntoShardModel;
use dojo::utils::entity_id_from_keys;
use dojo::world::IWorldDispatcherTrait;
use dojo_snf_test::declare_and_deploy;
use starknet::ContractAddress;

use crate::tests::helpers::{Foo, deploy_world_and_foo};

#[starknet::contract]
pub mod mock_sharding_proxy {
    use dojo::sharding::crdt::CRDType;

    #[storage]
    struct Storage {}

    #[abi(embed_v0)]
    impl IShardingImpl of dojo::sharding::interface::ISharding<ContractState> {
        fn initialize_sharding(ref self: ContractState, storage_slots: Span<CRDType>) {}
        fn end_shard(ref self: ContractState) {}
    }
}

/// Helper: get the local layout field selectors for Foo.
fn foo_field_selectors() -> (felt252, felt252) {
    let layout = Model::<Foo>::layout();
    if let dojo::meta::Layout::Struct(fields) = layout {
        ((*fields[0]).selector, (*fields[1]).selector)
    } else {
        panic!("expected struct layout")
    }
}

#[test]
fn test_crdt_add_round_trip() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let foo = Foo { caller: bob, a: 100, b: 200 };
    world.write_model_test(@foo);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    // Initialize via request_sharding (IWorld path) with Add CRDT.
    let layout = Model::<Foo>::layout();
    let models = [(model_selector, layout).shard_add([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (sel_a, _) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot = compute_dojo_field_slot(model_selector, entity_id, sel_a);

    // Simulate mainchain change: Foo.a 100 → 120 while shard is active.
    let foo_updated = Foo { caller: bob, a: 120, b: 200 };
    world.write_model_test(@foo_updated);

    // Shard saw initial=100, produced shard_value=150 (delta=50).
    let sharding = IContractComponentDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding.update_shard_state(array![(slot, 150)]);
    snforge_std::stop_cheat_caller_address(world_address);

    // Expected: current(120) + (shard(150) - initial(100)) = 170
    let result: Foo = world.read_model(bob);
    assert(result.a == 170, 'CRDT Add delta incorrect');
    assert(result.b == 200, 'b should be unchanged');
}

#[test]
fn test_crdt_set_overwrites() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let foo = Foo { caller: bob, a: 100, b: 200 };
    world.write_model_test(@foo);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    // Initialize via request_sharding (IWorld path) with Set CRDT.
    let layout = Model::<Foo>::layout();
    let models = [(model_selector, layout).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (sel_a, _) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot = compute_dojo_field_slot(model_selector, entity_id, sel_a);

    let sharding = IContractComponentDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding.update_shard_state(array![(slot, 999)]);
    snforge_std::stop_cheat_caller_address(world_address);

    let result: Foo = world.read_model(bob);
    assert(result.a == 999, 'Set should overwrite');
    assert(result.b == 200, 'b should be unchanged');
}

#[test]
fn test_cancel_unlocks_slot() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let foo = Foo { caller: bob, a: 100, b: 200 };
    world.write_model_test(@foo);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    // Initialize via request_sharding (IWorld path) with Add CRDT.
    let layout = Model::<Foo>::layout();
    let models = [(model_selector, layout).shard_add([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (sel_a, _) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot = compute_dojo_field_slot(model_selector, entity_id, sel_a);

    let sharding = IContractComponentDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding.cancel_shard_state(array![slot].span());
    snforge_std::stop_cheat_caller_address(world_address);

    // Value should remain unchanged after cancel.
    let result: Foo = world.read_model(bob);
    assert(result.a == 100, 'cancel should not change value');
    assert(result.b == 200, 'b should be unchanged');
}
