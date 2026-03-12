use dojo::model::{Model, ModelStorage, ModelStorageTest};
use dojo::sharding::compute_dojo_field_slot;
use dojo::sharding::request::{CRDVariant, IntoShardModel, ShardFieldSelection};
use dojo::utils::entity_id_from_keys;
use dojo::world::{
    IShardingProxyDispatcher, IShardingProxyDispatcherTrait, IWorldDispatcherTrait,
};
use dojo_snf_test::declare_and_deploy;
use starknet::ContractAddress;

use crate::tests::helpers::{Foo, deploy_world_and_foo};

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

#[test]
#[should_panic(expected: ('Component: Bad metadata',))]
fn test_component_setlock_after_add_fails() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 10, b: 20 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let add_models = [(
        model_selector, Model::<Foo>::layout(),
    )
        .shard_with(
            [bob.into()].span(), CRDVariant::Add, ShardFieldSelection::AutoDeterministic,
        )]
        .span();
    world.dispatcher.request_sharding(proxy_address, add_models);

    let setlock_models = [
        (
            model_selector, Model::<Foo>::layout(),
        )
            .shard_with(
                [bob.into()].span(), CRDVariant::SetLock, ShardFieldSelection::AutoDeterministic,
            ),
    ]
        .span();
    world.dispatcher.request_sharding(proxy_address, setlock_models);
}

#[test]
#[should_panic(expected: ('Component: Type change active',))]
fn test_component_set_after_add_fails() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 10, b: 20 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let add_models = [(
        model_selector, Model::<Foo>::layout(),
    )
        .shard_with(
            [bob.into()].span(), CRDVariant::Add, ShardFieldSelection::AutoDeterministic,
        )]
        .span();
    world.dispatcher.request_sharding(proxy_address, add_models);

    let set_models = [(model_selector, Model::<Foo>::layout()).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, set_models);
}

#[test]
#[should_panic(expected: ('Component: Type change active',))]
fn test_component_add_after_set_fails() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 10, b: 20 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let set_models = [(model_selector, Model::<Foo>::layout()).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, set_models);

    let add_models = [(
        model_selector, Model::<Foo>::layout(),
    )
        .shard_with(
            [bob.into()].span(), CRDVariant::Add, ShardFieldSelection::AutoDeterministic,
        )]
        .span();
    world.dispatcher.request_sharding(proxy_address, add_models);
}

#[test]
#[should_panic(expected: ('Component: Slot locked by shard',))]
fn test_component_lock_after_lock_fails() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 10, b: 20 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let lock_models = [(
        model_selector, Model::<Foo>::layout(),
    )
        .shard_with(
            [bob.into()].span(), CRDVariant::Lock, ShardFieldSelection::AutoDeterministic,
        )]
        .span();
    world.dispatcher.request_sharding(proxy_address, lock_models);
    world.dispatcher.request_sharding(proxy_address, lock_models);
}

#[test]
#[should_panic(expected: ('Component: Unauthorized caller',))]
fn test_component_update_requires_proxy_caller() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let models = [(model_selector, Model::<Foo>::layout()).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (sel_a, _) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    sharding_proxy.settle_shard_changes(array![(slot_a, 999)], [].span(), [].span());
}

#[test]
#[should_panic(expected: ('Component: Unauthorized caller',))]
fn test_component_cancel_requires_proxy_caller() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let models = [(model_selector, Model::<Foo>::layout()).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (sel_a, _) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    sharding_proxy.cancel_shard_state(array![slot_a].span());
}

#[test]
#[should_panic(expected: ('Component: Duplicate slot',))]
fn test_component_rejects_duplicate_slots_in_settlement() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let models = [(model_selector, Model::<Foo>::layout()).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (sel_a, _) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(array![(slot_a, 111), (slot_a, 222)], [].span(), [].span());
}

#[test]
fn test_component_lock_settlement_does_not_write() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let models = [(
        model_selector, Model::<Foo>::layout(),
    )
        .shard_with(
            [bob.into()].span(), CRDVariant::Lock, ShardFieldSelection::AutoDeterministic,
        )]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (sel_a, sel_b) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let slot_b = compute_dojo_field_slot(model_selector, entity_id, sel_b);

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(array![(slot_a, 999), (slot_b, 777)], [].span(), [].span());
    snforge_std::stop_cheat_caller_address(world_address);

    let result: Foo = world.read_model(bob);
    assert(result.a == 100, 'Lock should not write value');
    assert(result.b == 200, 'Other field unchanged');
}

#[test]
#[should_panic(expected: ('Component: Add delta underflow',))]
fn test_component_add_underflow_rejected() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let models = [(
        model_selector, Model::<Foo>::layout(),
    )
        .shard_with(
            [bob.into()].span(), CRDVariant::Add, ShardFieldSelection::AutoDeterministic,
        )]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (sel_a, _) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(array![(slot_a, 90)], [].span(), [].span());
}

#[test]
#[should_panic(expected: ('Component: Arithmetic overflow',))]
fn test_component_add_overflow_rejected() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 0, b: 200 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let models = [(
        model_selector, Model::<Foo>::layout(),
    )
        .shard_with(
            [bob.into()].span(), CRDVariant::Add, ShardFieldSelection::AutoDeterministic,
        )]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);

    // Mainchain updates current value near felt max while shard is active.
    world.write_model_test(@Foo { caller: bob, a: FELT_MAX, b: 200 });

    let (sel_a, _) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    // Delta = 1 - 0 = 1, current + delta = FIELD_PRIME -> cannot fit into felt252.
    sharding_proxy.settle_shard_changes(array![(slot_a, 1)], [].span(), [].span());
}

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
