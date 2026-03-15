use core::array::ArrayTrait;
use dojo::model::{Model, ModelIndex, ModelStorage, ModelStorageTest};
use dojo::sharding::{
    compute_dojo_field_slot, compute_dynamic_member_changed_hash_slot,
    compute_dynamic_member_changed_len_slot, compute_dynamic_member_changed_slots_hash,
    compute_dynamic_member_lock_slot,
};
use dojo::sharding::slot::{PACKED_SLOT_BASE, compute_dojo_packed_slot};
use dojo::sharding::request::{
    CRDVariant, IntoShardField, IntoShardModel, ShardCoverage, ShardFieldSelection, ShardModel,
};
use dojo::utils::{combine_key, entity_id_from_keys};
use dojo::world::{
    IShardingMetadataDispatcher, IShardingMetadataDispatcherTrait, IShardingProxyDispatcher,
    IShardingProxyDispatcherTrait, IWorldDispatcherTrait, ShardDynamicMemberChanges,
};
use dojo_snf_test::declare_and_deploy;
use starknet::ContractAddress;

use crate::tests::helpers::{
    Balance256, Foo, MixedDynamic, NestedFixed, NestedStats, NotCopiable, PackedPair,
    TupleArrayOption, deploy_world_and_foo, deploy_world_with_balance256,
    deploy_world_with_mixed_dynamic, deploy_world_with_nested_fixed,
    deploy_world_with_not_copiable, deploy_world_with_packed_pair, deploy_world_with_score,
    deploy_world_with_tuple_array_option,
};

/// Helper: get the local layout field selectors for Foo (same compilation unit as write/read_model).
fn foo_field_selectors() -> (felt252, felt252) {
    let layout = Model::<Foo>::layout();
    if let dojo::meta::Layout::Struct(fields) = layout {
        ((*fields[0]).selector, (*fields[1]).selector)
    } else {
        panic!("expected struct layout")
    }
}

fn balance256_field_selector() -> felt252 {
    let layout = Model::<Balance256>::layout();
    if let dojo::meta::Layout::Struct(fields) = layout {
        (*fields[0]).selector
    } else {
        panic!("expected struct layout")
    }
}

fn mixed_dynamic_fixed_selector() -> felt252 {
    let layout = Model::<MixedDynamic>::layout();
    if let dojo::meta::Layout::Struct(fields) = layout {
        (*fields[0]).selector
    } else {
        panic!("expected struct layout")
    }
}

fn mixed_dynamic_selectors() -> (felt252, felt252) {
    let layout = Model::<MixedDynamic>::layout();
    if let dojo::meta::Layout::Struct(fields) = layout {
        ((*fields[0]).selector, (*fields[1]).selector)
    } else {
        panic!("expected struct layout")
    }
}

fn not_copiable_selectors() -> (felt252, felt252) {
    let layout = Model::<NotCopiable>::layout();
    if let dojo::meta::Layout::Struct(fields) = layout {
        ((*fields[0]).selector, (*fields[1]).selector)
    } else {
        panic!("expected struct layout")
    }
}

fn dynamic_member_slots(
    world_address: ContractAddress,
    model_selector: felt252,
    entity_id: felt252,
    member_selector: felt252,
) -> Span<felt252> {
    let metadata = IShardingMetadataDispatcher { contract_address: world_address };
    metadata.dynamic_member_storage_slots(model_selector, entity_id, member_selector)
}

fn build_dynamic_member_payload(
    world_address: ContractAddress,
    model_selector: felt252,
    entity_id: felt252,
    member_selector: felt252,
    values: Span<felt252>,
) -> (Array<ShardDynamicMemberChanges>, Array<(felt252, felt252)>) {
    let slots = dynamic_member_slots(world_address, model_selector, entity_id, member_selector);
    assert(slots.len() == values.len(), 'payload len');

    let mut dynamic_changes: Array<(felt252, felt252)> = ArrayTrait::new();
    let mut i: u32 = 0;
    while i < slots.len() {
        dynamic_changes.append((*slots[i], *values[i]));
        i += 1;
    };

    let mut dynamic_members: Array<ShardDynamicMemberChanges> = ArrayTrait::new();
    dynamic_members.append(
        ShardDynamicMemberChanges {
            model_selector,
            entity_id,
            member_selector,
            changes_offset: 0,
            changes_len: values.len(),
        },
    );

    (dynamic_members, dynamic_changes)
}

fn build_dynamic_member_tracking_proofs(
    model_selector: felt252,
    entity_id: felt252,
    member_selector: felt252,
    changed_slots: Span<felt252>,
) -> Array<(felt252, felt252)> {
    let lock_slot = compute_dynamic_member_lock_slot(model_selector, entity_id, member_selector);
    array![
        (
            compute_dynamic_member_changed_len_slot(lock_slot),
            changed_slots.len().into(),
        ),
        (
            compute_dynamic_member_changed_hash_slot(lock_slot),
            compute_dynamic_member_changed_slots_hash(changed_slots),
        ),
    ]
}

fn nested_fixed_selectors() -> (felt252, felt252, felt252) {
    let layout = Model::<NestedFixed>::layout();
    if let dojo::meta::Layout::Struct(fields) = layout {
        let stats_selector = (*fields[0]).selector;
        let gold_selector = (*fields[1]).selector;
        if let dojo::meta::Layout::Struct(stat_fields) = *fields[0].layout {
            let hp_selector = (*stat_fields[0]).selector;
            (stats_selector, hp_selector, gold_selector)
        } else {
            panic!("expected nested struct layout")
        }
    } else {
        panic!("expected struct layout")
    }
}

fn tuple_array_option_selectors() -> (felt252, felt252, felt252) {
    let layout = Model::<TupleArrayOption>::layout();
    if let dojo::meta::Layout::Struct(fields) = layout {
        let pair_selector = selector!("pair");
        let samples_selector = selector!("samples");
        let status_selector = selector!("status");
        assert(dojo::utils::find_field_layout(status_selector, fields).is_some(), 'missing status');
        (pair_selector, samples_selector, status_selector)
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

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };

    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![(slot_a, 999)], [].span(), [].span(), [].span());
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

    // Use Add CRDT via generic shard_with(...).
    let layout = Model::<Foo>::layout();
    let models = [(
        model_selector, layout,
    )
        .shard_with(
            [bob.into()].span(), CRDVariant::Add, ShardFieldSelection::AutoDeterministic,
        )]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);

    // Mainchain changes a to 120 while shard is active.
    let foo_updated = Foo { caller: bob, a: 120, b: 200 };
    world.write_model_test(@foo_updated);

    // Shard saw initial=100, produced shard_value=150 (delta=50).
    let (sel_a, _) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };

    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![(slot_a, 150)], [].span(), [].span(), [].span());
    snforge_std::stop_cheat_caller_address(world_address);

    // Expected: current(120) + (shard(150) - initial(100)) = 170
    let result: Foo = world.read_model(bob);
    assert(result.a == 170, 'Add delta incorrect');
    assert(result.b == 200, 'b should be unchanged');
}

/// Test: SetLock blocks regular main-chain writes while shard is active.
#[test]
#[should_panic]
fn test_request_sharding_set_lock_blocks_main_write() {
    let (mut world, model_selector) = deploy_world_and_foo();

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let foo = Foo { caller: bob, a: 100, b: 200 };
    world.write_model_test(@foo);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let layout = Model::<Foo>::layout();
    let models = [(
        model_selector, layout,
    )
        .shard_with(
            [bob.into()].span(), CRDVariant::SetLock, ShardFieldSelection::AutoDeterministic,
        )]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);

    // Regular world write must fail while SetLock slots are active.
    let foo_updated = Foo { caller: bob, a: 120, b: 250 };
    world.write_model_test(@foo_updated);
}

/// Regression: model-level writes must not bypass SetLock with forged layouts.
#[test]
#[should_panic]
fn test_request_sharding_set_lock_blocks_forged_model_layout_write() {
    let (mut world, model_selector) = deploy_world_and_foo();

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let models = [(
        model_selector, Model::<Foo>::layout(),
    )
        .shard_with(
            [bob.into()].span(), CRDVariant::SetLock, ShardFieldSelection::AutoDeterministic,
        )]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);

    let entity_id = entity_id_from_keys(@bob);
    let forged_layout = dojo::meta::Layout::Struct(
        [dojo::meta::FieldLayout {
            selector: 0xDEAD_BEEF,
            layout: dojo::meta::Layout::Fixed([8].span()),
        }]
            .span(),
    );
    world
        .dispatcher
        .set_entity(model_selector, ModelIndex::Id(entity_id), [1].span(), forged_layout);
}

/// Test: settle must provide all exclusive slots from the same model/entity group.
#[test]
#[should_panic]
fn test_settle_shard_changes_rejects_partial_exclusive_group() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let models = [(
        model_selector, Model::<Foo>::layout(),
    )
        .shard_with(
            [bob.into()].span(), CRDVariant::SetLock, ShardFieldSelection::AutoDeterministic,
        )]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (sel_a, _) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };

    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![(slot_a, 999)], [].span(), [].span(), [].span());
}

/// Test: cancel must provide all exclusive slots from the same model/entity group.
#[test]
#[should_panic]
fn test_cancel_shard_state_rejects_partial_exclusive_group() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let models = [(
        model_selector, Model::<Foo>::layout(),
    )
        .shard_with(
            [bob.into()].span(), CRDVariant::SetLock, ShardFieldSelection::AutoDeterministic,
        )]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (sel_a, _) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };

    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.cancel_shard_state(1, array![slot_a].span());
}

/// Regression: duplicate slots must not bypass full-coverage check for exclusive groups.
#[test]
#[should_panic]
fn test_cancel_shard_state_rejects_duplicate_slot_coverage_bypass() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let models = [(
        model_selector, Model::<Foo>::layout(),
    )
        .shard_with(
            [bob.into()].span(), CRDVariant::SetLock, ShardFieldSelection::AutoDeterministic,
        )]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (sel_a, _) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };

    // Buggy behavior accepts [A, A] as if it covered {A, B}.
    // Correct behavior must reject duplicate coverage input.
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.cancel_shard_state(1, array![slot_a, slot_a].span());
}

/// Test: settle must cover all exclusive slots in a single sharding request.
#[test]
#[should_panic]
fn test_settle_shard_changes_rejects_partial_exclusive_request() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let alice: ContractAddress = 0xa11ce.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });
    world.write_model_test(@Foo { caller: alice, a: 300, b: 400 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let models = [
        (
            model_selector, Model::<Foo>::layout(),
        )
            .shard_with(
                [bob.into()].span(), CRDVariant::SetLock, ShardFieldSelection::AutoDeterministic,
            ),
        (
            model_selector, Model::<Foo>::layout(),
        )
            .shard_with(
                [alice.into()].span(), CRDVariant::SetLock, ShardFieldSelection::AutoDeterministic,
            ),
    ]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (sel_a, sel_b) = foo_field_selectors();
    let bob_entity_id = entity_id_from_keys(@bob);
    let bob_slot_a = compute_dojo_field_slot(model_selector, bob_entity_id, sel_a);
    let bob_slot_b = compute_dojo_field_slot(model_selector, bob_entity_id, sel_b);
    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };

    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, 
        array![(bob_slot_a, 111), (bob_slot_b, 222)], [].span(), [].span(), [].span(),
    );
}

/// Test: cancel must cover all exclusive slots in a single sharding request.
#[test]
#[should_panic]
fn test_cancel_shard_state_rejects_partial_exclusive_request() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let alice: ContractAddress = 0xa11ce.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });
    world.write_model_test(@Foo { caller: alice, a: 300, b: 400 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let models = [
        (
            model_selector, Model::<Foo>::layout(),
        )
            .shard_with(
                [bob.into()].span(), CRDVariant::SetLock, ShardFieldSelection::AutoDeterministic,
            ),
        (
            model_selector, Model::<Foo>::layout(),
        )
            .shard_with(
                [alice.into()].span(), CRDVariant::SetLock, ShardFieldSelection::AutoDeterministic,
            ),
    ]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (sel_a, sel_b) = foo_field_selectors();
    let bob_entity_id = entity_id_from_keys(@bob);
    let bob_slot_a = compute_dojo_field_slot(model_selector, bob_entity_id, sel_a);
    let bob_slot_b = compute_dojo_field_slot(model_selector, bob_entity_id, sel_b);
    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };

    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.cancel_shard_state(1, array![bob_slot_a, bob_slot_b].span());
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

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };

    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![(slot_a, 111), (slot_b, 222)], [].span(), [].span(), [].span());
    snforge_std::stop_cheat_caller_address(world_address);

    let result: Foo = world.read_model(bob);
    assert(result.a == 111, 'a should be updated');
    assert(result.b == 222, 'b should be updated');
}

/// Test: full coverage policy rejects partial field selection.
#[test]
#[should_panic]
fn test_request_sharding_full_coverage_rejects_partial_selection() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let (sel_a, _) = foo_field_selectors();
    let models = [
        ShardModel {
            selector: model_selector,
            keys: [bob.into()].span(),
            fields: [sel_a.as_set()].span(),
            coverage: ShardCoverage::Full,
        },
    ]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);
}

/// Test: deterministic subset policy allows explicit partial field selection.
#[test]
fn test_request_sharding_deterministic_subset_allows_partial_selection() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let (sel_a, _) = foo_field_selectors();
    let models = [
        ShardModel {
            selector: model_selector,
            keys: [bob.into()].span(),
            fields: [sel_a.as_set()].span(),
            coverage: ShardCoverage::DeterministicSubset,
        },
    ]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);

    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };

    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![(slot_a, 777)], [].span(), [].span(), [].span());
    snforge_std::stop_cheat_caller_address(world_address);

    let result: Foo = world.read_model(bob);
    assert(result.a == 777, 'a should be updated');
    assert(result.b == 200, 'b should be unchanged');
}

/// Test: PN-Counter pattern using generic Add selection — both P and N fields are G-Counters (Add).
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

    // Use PNCounter CRDT via shard_with(..., Add, ...) — both fields become Add (G-Counter).
    let layout = Model::<Foo>::layout();
    let models = [(
        model_selector, layout,
    )
        .shard_with(
            [bob.into()].span(), CRDVariant::Add, ShardFieldSelection::AutoDeterministic,
        )]
        .span();
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

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };

    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![(slot_a, 150), (slot_b, 80)], [].span(), [].span(), [].span());
    snforge_std::stop_cheat_caller_address(world_address);

    // P: current(120) + (shard(150) - initial(100)) = 170
    // N: current(60) + (shard(80) - initial(50)) = 90
    // Balance: 170 - 90 = 80
    let result: Foo = world.read_model(bob);
    assert(result.a == 170, 'PN: P delta incorrect');
    assert(result.b == 90, 'PN: N delta incorrect');
}

/// Test: Add mode with burn only (N increases, P unchanged) — simulates resource spending.
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
    let models = [(
        model_selector, layout,
    )
        .shard_with(
            [bob.into()].span(), CRDVariant::Add, ShardFieldSelection::AutoDeterministic,
        )]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);

    // Shard: P unchanged (no minting), N increases by 300 (burning 300 resources).
    // Shard final: P=1000 (same), N=500.
    let (sel_a, sel_b) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let slot_b = compute_dojo_field_slot(model_selector, entity_id, sel_b);

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };

    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![(slot_a, 1000), (slot_b, 500)], [].span(), [].span(), [].span());
    snforge_std::stop_cheat_caller_address(world_address);

    // P: current(1000) + (shard(1000) - initial(1000)) = 1000 (no change)
    // N: current(200) + (shard(500) - initial(200)) = 500
    // Balance: 1000 - 500 = 500 (was 800, burned 300 on shard)
    let result: Foo = world.read_model(bob);
    assert(result.a == 1000, 'PN burn: P should be unchanged');
    assert(result.b == 500, 'PN burn: N delta incorrect');
}

/// Test: per-field CRDT — field `a` as Add (delta merge), field `b` as Set (overwrite).
///
/// This tests the new per-field CRDT feature where different fields of the same model
/// can use different merge strategies.
#[test]
fn test_per_field_crdt_mixed() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let foo = Foo { caller: bob, a: 100, b: 200 };
    world.write_model_test(@foo);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    // Per-field: field a → Add, field b → Set.
    let (sel_a, sel_b) = foo_field_selectors();
    let models = [
        ShardModel {
            selector: model_selector,
            keys: [bob.into()].span(),
            fields: [sel_a.as_add(), sel_b.as_set()].span(),
            coverage: ShardCoverage::Full,
        },
    ]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);

    // Mainchain changes a from 100 → 120 while shard is active.
    let foo_updated = Foo { caller: bob, a: 120, b: 200 };
    world.write_model_test(@foo_updated);

    // Shard saw initial a=100, produced shard a=150 (delta=50).
    // Shard overwrites b=999 (Set).
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let slot_b = compute_dojo_field_slot(model_selector, entity_id, sel_b);

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };

    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![(slot_a, 150), (slot_b, 999)], [].span(), [].span(), [].span());
    snforge_std::stop_cheat_caller_address(world_address);

    // a: current(120) + (shard(150) - initial(100)) = 170 (Add delta)
    // b: 999 (Set overwrite)
    let result: Foo = world.read_model(bob);
    assert(result.a == 170, 'Add delta incorrect');
    assert(result.b == 999, 'Set should overwrite');
}

/// Test: field with fixed layout spanning 2 slots (u256) must update both slots.
#[test]
fn test_request_sharding_struct_fixed_multi_slot_field() {
    let (mut world, model_selector) = deploy_world_with_balance256();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let initial = Balance256 { player: bob, amount: u256 { low: 1_u128, high: 2_u128 } };
    world.write_model_test(@initial);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let layout = Model::<Balance256>::layout();
    let models = [(model_selector, layout).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let selector = balance256_field_selector();
    let entity_id = entity_id_from_keys(@bob);
    let base_slot = compute_dojo_field_slot(model_selector, entity_id, selector);
    let high_slot = base_slot + 1;

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![(base_slot, 55), (high_slot, 66)], [].span(), [].span(), [].span());
    snforge_std::stop_cheat_caller_address(world_address);

    let result: Balance256 = world.read_model(bob);
    assert(result.amount.low == 55_u128, 'u256 low slot not updated');
    assert(result.amount.high == 66_u128, 'u256 high slot not updated');
}

/// Test: request_sharding rejects empty model list.
#[test]
#[should_panic]
fn test_request_sharding_rejects_empty_models() {
    let (mut world, _) = deploy_world_and_foo();
    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let models: Span<ShardModel> = [].span();
    world.dispatcher.request_sharding(proxy_address, models);
}

/// Test: request_sharding rejects model entries with no fields.
#[test]
#[should_panic]
fn test_request_sharding_rejects_empty_fields() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let models = [
        ShardModel {
            selector: model_selector,
            keys: [bob.into()].span(),
            fields: [].span(),
            coverage: ShardCoverage::Full,
        },
    ]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);
}

/// Test: fixed-layout models must use packed pseudo selectors.
#[test]
#[should_panic]
fn test_request_sharding_fixed_model_requires_packed_selector() {
    let (mut world, model_selector) = deploy_world_with_score();
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let models = [
        ShardModel {
            selector: model_selector,
            keys: [bob.into()].span(),
            fields: [selector!("points").as_set()].span(),
            coverage: ShardCoverage::Full,
        },
    ]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);
}

/// Test: packed selector offsets outside packed size are rejected.
#[test]
#[should_panic]
fn test_request_sharding_rejects_packed_selector_out_of_range() {
    let (mut world, model_selector) = deploy_world_with_score();
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    // Score has packed_size = 1, so offset 1 is out of range.
    let models = [
        ShardModel {
            selector: model_selector,
            keys: [bob.into()].span(),
            fields: [(PACKED_SLOT_BASE + 1).as_set()].span(),
            coverage: ShardCoverage::Full,
        },
    ]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);
}

/// Regression: partial settlement of packed models must not panic on the second chunk.
#[test]
fn test_request_sharding_packed_two_slot_partial_settlements_do_not_panic() {
    let (mut world, model_selector) = deploy_world_with_packed_pair();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@PackedPair { player: bob, left: 10, right: 20 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let layout = Model::<PackedPair>::layout();

    let packed_size = if let dojo::meta::Layout::Fixed(sizes) = layout {
        let mut sizes = sizes;
        dojo::storage::packing::calculate_packed_size(ref sizes)
    } else {
        panic!("expected fixed layout")
    };
    assert(packed_size == 2, 'packed pair should use 2 slots');

    let models = [(model_selector, layout).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let entity_id = entity_id_from_keys(@bob);
    let slot0 = compute_dojo_packed_slot(model_selector, entity_id);
    let slot1 = slot0 + 1;

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };

    // Settle first packed slot.
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![(slot0, 111)], [].span(), [].span(), [].span());
    snforge_std::stop_cheat_caller_address(world_address);

    // Settle second packed slot in a separate call.
    // Before the fix this panics due to premature key cleanup and fixed-layout member lookup.
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![(slot1, 222)], [].span(), [].span(), [].span());
    snforge_std::stop_cheat_caller_address(world_address);
}

/// Test: manual per-field sharding must reject unknown struct field selectors.
#[test]
#[should_panic]
fn test_request_sharding_rejects_unknown_field_selector() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let bob: ContractAddress = 0xb0b.try_into().unwrap();

    let foo = Foo { caller: bob, a: 10, b: 20 };
    world.write_model_test(@foo);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let models = [
        ShardModel {
            selector: model_selector,
            keys: [bob.into()].span(),
            fields: [0xDEADBEEF.as_set()].span(),
            coverage: ShardCoverage::Full,
        },
    ]
        .span();

    world.dispatcher.request_sharding(proxy_address, models);
}

/// Test: struct fields with non-fixed layouts are not supported for sharding.
#[test]
#[should_panic]
fn test_request_sharding_rejects_non_fixed_struct_field_layout() {
    let (world, model_selector) = deploy_world_with_not_copiable();
    let bob: ContractAddress = 0xb0b.try_into().unwrap();

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let layout = Model::<NotCopiable>::layout();
    let models = [(model_selector, layout).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);
}

/// Test: explicit deterministic-only policy skips dynamic fields and keeps fixed ones.
#[test]
fn test_request_sharding_deterministic_only_skips_dynamic_fields() {
    let (mut world, model_selector) = deploy_world_with_mixed_dynamic();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let value = MixedDynamic { player: bob, fixed_value: 10, note: "hello" };
    world.write_model_test(@value);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let layout = Model::<MixedDynamic>::layout();
    let models = [(model_selector, layout).shard_deterministic([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let fixed_selector = mixed_dynamic_fixed_selector();
    let entity_id = entity_id_from_keys(@bob);
    let slot = compute_dojo_field_slot(model_selector, entity_id, fixed_selector);

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![(slot, 77)], [].span(), [].span(), [].span());
    snforge_std::stop_cheat_caller_address(world_address);

    let result: MixedDynamic = world.read_model(bob);
    assert(result.fixed_value == 77, 'fixed field should be updated');
}

/// Test: dynamic members can be requested with explicit SetLock dynamic policy
/// and settled via `settle_shard_changes` raw segment writes.
#[test]
fn test_request_sharding_set_lock_with_dynamic_member() {
    let (mut world, model_selector) = deploy_world_with_mixed_dynamic();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let value = MixedDynamic { player: bob, fixed_value: 10, note: "hello" };
    world.write_model_test(@value);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let layout = Model::<MixedDynamic>::layout();
    let models = [(model_selector, layout).shard_dynamic([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (fixed_selector, note_selector) = mixed_dynamic_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let fixed_slot = compute_dojo_field_slot(model_selector, entity_id, fixed_selector);

    // Empty ByteArray serialized as [data_len, pending_word, pending_word_len].
    let (dynamic_members, dynamic_changes) = build_dynamic_member_payload(
        world_address, model_selector, entity_id, note_selector, [0, 0, 0].span(),
    );
    let tracking_proofs = build_dynamic_member_tracking_proofs(
        model_selector,
        entity_id,
        note_selector,
        dynamic_member_slots(world_address, model_selector, entity_id, note_selector),
    );

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, 
        array![(fixed_slot, 77)],
        dynamic_members.span(),
        dynamic_changes.span(),
        tracking_proofs.span(),
    );
    snforge_std::stop_cheat_caller_address(world_address);

    let result: MixedDynamic = world.read_model(bob);
    assert(result.fixed_value == 77, 'fixed member should be settled');
    assert(result.note == "", 'dynamic note mismatch');
}

/// Regression: dynamic member lock must also block forged deterministic member writes.
#[test]
#[should_panic]
fn test_request_sharding_dynamic_lock_blocks_forged_member_layout_write() {
    let (mut world, model_selector) = deploy_world_with_mixed_dynamic();

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let value = MixedDynamic { player: bob, fixed_value: 10, note: "hello" };
    world.write_model_test(@value);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let layout = Model::<MixedDynamic>::layout();
    let models = [(model_selector, layout).shard_dynamic([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (_, note_selector) = mixed_dynamic_selectors();
    let entity_id = entity_id_from_keys(@bob);

    // Attack attempt: same dynamic member selector with forged fixed layout.
    // Expected behavior: still blocked by dynamic member lock.
    world
        .dispatcher
        .set_entity(
            model_selector,
            ModelIndex::MemberId((entity_id, note_selector)),
            [1].span(),
            dojo::meta::Layout::Fixed([8].span()),
        );
}

/// Regression: unknown member selector writes must be rejected when sharding is active.
#[test]
#[should_panic]
fn test_request_sharding_rejects_unknown_member_selector_write() {
    let (mut world, model_selector) = deploy_world_with_mixed_dynamic();

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@MixedDynamic { player: bob, fixed_value: 10, note: "hello" });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let layout = Model::<MixedDynamic>::layout();
    let models = [(model_selector, layout).shard_dynamic([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let entity_id = entity_id_from_keys(@bob);
    world
        .dispatcher
        .set_entity(
            model_selector,
            ModelIndex::MemberId((entity_id, 0xDEAD_BEEF)),
            [1].span(),
            dojo::meta::Layout::Fixed([8].span()),
        );
}

/// Test: settle_shard_changes rejects empty payload (no slots, no members).
#[test]
#[should_panic]
fn test_settle_shard_changes_rejects_empty_payload() {
    let (mut world, model_selector) = deploy_world_with_mixed_dynamic();
    let world_address = world.dispatcher.contract_address;
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@MixedDynamic { player: bob, fixed_value: 10, note: "hello" });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let layout = Model::<MixedDynamic>::layout();
    let models = [(model_selector, layout).shard_dynamic([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![], [].span(), [].span(), [].span());
}

/// Test: settle_shard_changes rejects member writes with zero-length value slices.
#[test]
#[should_panic]
fn test_settle_shard_changes_rejects_empty_member_write_values() {
    let (mut world, model_selector) = deploy_world_with_mixed_dynamic();
    let world_address = world.dispatcher.contract_address;
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@MixedDynamic { player: bob, fixed_value: 10, note: "hello" });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let layout = Model::<MixedDynamic>::layout();
    let models = [(model_selector, layout).shard_dynamic([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (_, note_selector) = mixed_dynamic_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let dynamic_members = [
        ShardDynamicMemberChanges {
            model_selector,
            entity_id,
            member_selector: note_selector,
            changes_offset: 0,
            changes_len: 0,
        },
    ]
        .span();

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![], dynamic_members, [].span(), [].span());
}

/// Test: settle_shard_changes checks dynamic change slice bounds.
#[test]
#[should_panic]
fn test_settle_shard_changes_rejects_member_write_values_out_of_range() {
    let (mut world, model_selector) = deploy_world_with_mixed_dynamic();
    let world_address = world.dispatcher.contract_address;
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@MixedDynamic { player: bob, fixed_value: 10, note: "hello" });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let layout = Model::<MixedDynamic>::layout();
    let models = [(model_selector, layout).shard_dynamic([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (_, note_selector) = mixed_dynamic_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let dynamic_members = [
        ShardDynamicMemberChanges {
            model_selector,
            entity_id,
            member_selector: note_selector,
            changes_offset: 1,
            changes_len: 3,
        },
    ]
        .span();
    let dynamic_slots = dynamic_member_slots(world_address, model_selector, entity_id, note_selector);
    let dynamic_changes: Array<(felt252, felt252)> = array![(*dynamic_slots[0], 0), (*dynamic_slots[1], 0)];

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![], dynamic_members, dynamic_changes.span(), [].span());
}

/// Test: dynamic member changes require an active dynamic lock slot.
#[test]
#[should_panic]
fn test_settle_shard_changes_rejects_member_write_without_dynamic_lock() {
    let (mut world, model_selector) = deploy_world_with_mixed_dynamic();
    let world_address = world.dispatcher.contract_address;
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@MixedDynamic { player: bob, fixed_value: 10, note: "hello" });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let layout = Model::<MixedDynamic>::layout();
    // Deterministic-only policy excludes dynamic field lock.
    let models = [(model_selector, layout).shard_deterministic([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (_, note_selector) = mixed_dynamic_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let (dynamic_members, dynamic_changes) = build_dynamic_member_payload(
        world_address, model_selector, entity_id, note_selector, [0, 0, 0].span(),
    );

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![], dynamic_members.span(), dynamic_changes.span(), [].span());
}

/// Test: duplicate dynamic member segments for the same lock are rejected.
#[test]
#[should_panic]
fn test_settle_shard_changes_rejects_duplicate_member_writes() {
    let (mut world, model_selector) = deploy_world_with_mixed_dynamic();
    let world_address = world.dispatcher.contract_address;
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@MixedDynamic { player: bob, fixed_value: 10, note: "hello" });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let layout = Model::<MixedDynamic>::layout();
    let models = [(model_selector, layout).shard_dynamic([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (_, note_selector) = mixed_dynamic_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let dynamic_slots = dynamic_member_slots(world_address, model_selector, entity_id, note_selector);
    let dynamic_members = [
        ShardDynamicMemberChanges {
            model_selector,
            entity_id,
            member_selector: note_selector,
            changes_offset: 0,
            changes_len: 3,
        },
        ShardDynamicMemberChanges {
            model_selector,
            entity_id,
            member_selector: note_selector,
            changes_offset: 3,
            changes_len: 3,
        },
    ]
        .span();
    let dynamic_changes: Array<(felt252, felt252)> = array![
        (*dynamic_slots[0], 0), (*dynamic_slots[1], 0), (*dynamic_slots[2], 0),
        (*dynamic_slots[0], 0), (*dynamic_slots[1], 0), (*dynamic_slots[2], 0),
    ];

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![], dynamic_members, dynamic_changes.span(), [].span());
}

/// Regression: settle must reject trailing/unused dynamic change payload.
#[test]
#[should_panic]
fn test_settle_shard_changes_rejects_unused_member_write_values() {
    let (mut world, model_selector) = deploy_world_with_mixed_dynamic();
    let world_address = world.dispatcher.contract_address;
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@MixedDynamic { player: bob, fixed_value: 10, note: "hello" });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let layout = Model::<MixedDynamic>::layout();
    let models = [(model_selector, layout).shard_dynamic([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (fixed_selector, note_selector) = mixed_dynamic_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let fixed_slot = compute_dojo_field_slot(model_selector, entity_id, fixed_selector);
    let dynamic_members = [
        ShardDynamicMemberChanges {
            model_selector,
            entity_id,
            member_selector: note_selector,
            changes_offset: 0,
            changes_len: 3,
        },
    ]
        .span();
    let dynamic_slots = dynamic_member_slots(world_address, model_selector, entity_id, note_selector);
    let dynamic_changes: Array<(felt252, felt252)> = array![
        (*dynamic_slots[0], 0), (*dynamic_slots[1], 0), (*dynamic_slots[2], 0), (123, 456),
    ];

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![(fixed_slot, 77)], dynamic_members, dynamic_changes.span(), [].span());
}

/// Regression: dynamic change payload must be contiguous/non-overlapping by ranges.
#[test]
#[should_panic]
fn test_settle_shard_changes_rejects_overlapping_member_write_values_ranges() {
    let (world, model_selector) = deploy_world_with_not_copiable();
    let world_address = world.dispatcher.contract_address;
    let bob: ContractAddress = 0xb0b.try_into().unwrap();

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let layout = Model::<NotCopiable>::layout();
    let models = [(model_selector, layout).shard_dynamic([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (array_selector, byte_array_selector) = not_copiable_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let array_slots = dynamic_member_slots(world_address, model_selector, entity_id, array_selector);
    let byte_array_slots =
        dynamic_member_slots(world_address, model_selector, entity_id, byte_array_selector);
    let dynamic_members = [
        ShardDynamicMemberChanges {
            model_selector,
            entity_id,
            member_selector: array_selector,
            changes_offset: 0,
            changes_len: 1,
        },
        ShardDynamicMemberChanges {
            model_selector,
            entity_id,
            member_selector: byte_array_selector,
            changes_offset: 0,
            changes_len: 3,
        },
    ]
        .span();
    let dynamic_changes: Array<(felt252, felt252)> = array![
        (*array_slots[0], 0), (*byte_array_slots[0], 0), (*byte_array_slots[1], 0),
    ];

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![], dynamic_members, dynamic_changes.span(), [].span());
}

/// Test: dynamic fields must be requested with SetLock CRDT.
#[test]
#[should_panic]
fn test_request_sharding_dynamic_requires_set_lock() {
    let (world, model_selector) = deploy_world_with_mixed_dynamic();
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let (_, note_selector) = mixed_dynamic_selectors();

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let models = [
        ShardModel {
            selector: model_selector,
            keys: [bob.into()].span(),
            fields: [note_selector.as_set()].span(),
            coverage: ShardCoverage::Full,
        },
    ]
        .span();
    world.dispatcher.request_sharding(proxy_address, models);
}

/// Test: nested fixed layouts are expanded into concrete slots.
#[test]
fn test_request_sharding_nested_fixed_layout() {
    let (mut world, model_selector) = deploy_world_with_nested_fixed();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let value = NestedFixed {
        player: bob, stats: NestedStats { hp: 10, mana: 20 }, gold: 5,
    };
    world.write_model_test(@value);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let layout = Model::<NestedFixed>::layout();
    let models = [(model_selector, layout).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (stats_selector, hp_selector, gold_selector) = nested_fixed_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let stats_key = combine_key(entity_id, stats_selector);
    let hp_key = combine_key(stats_key, hp_selector);
    let hp_slot = compute_dojo_packed_slot(model_selector, hp_key);
    let gold_slot = compute_dojo_field_slot(model_selector, entity_id, gold_selector);

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, array![(hp_slot, 99), (gold_slot, 777)], [].span(), [].span(), [].span());
    snforge_std::stop_cheat_caller_address(world_address);

    let result: NestedFixed = world.read_model(bob);
    assert(result.stats.hp == 99, 'nested hp should be updated');
    assert(result.stats.mana == 20, 'nested mana should be unchanged');
    assert(result.gold == 777, 'gold should be updated');
}

/// Test: tuple/fixed-array/enum deterministic branches are expanded into concrete slots.
#[test]
fn test_request_sharding_tuple_fixedarray_enum_layout() {
    let (mut world, model_selector) = deploy_world_with_tuple_array_option();
    let world_address = world.dispatcher.contract_address;

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    let value = TupleArrayOption {
        player: bob,
        pair: (10_u32, 20_u64),
        samples: [1_u16, 2_u16, 3_u16],
        status: Option::Some(9_u32),
    };
    world.write_model_test(@value);

    let proxy_address = declare_and_deploy("mock_sharding_proxy");

    let layout = Model::<TupleArrayOption>::layout();
    let models = [(model_selector, layout).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    let (pair_selector, samples_selector, status_selector) = tuple_array_option_selectors();
    let entity_id = entity_id_from_keys(@bob);

    let pair_key = combine_key(entity_id, pair_selector);
    let pair_item0_key = combine_key(pair_key, 0);
    let pair_item0_slot = compute_dojo_packed_slot(model_selector, pair_item0_key);

    let samples_key = combine_key(entity_id, samples_selector);
    let samples_item1_key = combine_key(samples_key, 1);
    let samples_item1_slot = compute_dojo_packed_slot(model_selector, samples_item1_key);

    let status_key = combine_key(entity_id, status_selector);
    let status_discriminator_slot = compute_dojo_packed_slot(model_selector, status_key);
    let status_discriminator_address: starknet::storage_access::StorageAddress =
        status_discriminator_slot.try_into().unwrap();
    let current_discriminator = starknet::syscalls::storage_read_syscall(
        0, status_discriminator_address,
    )
        .unwrap();

    let sharding_proxy = IShardingProxyDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, proxy_address);
    sharding_proxy.settle_shard_changes(1, 
        array![
            (pair_item0_slot, 55),
            (samples_item1_slot, 66),
            (status_discriminator_slot, current_discriminator),
        ],
        [].span(),
        [].span(),
        [].span(),
    );
    snforge_std::stop_cheat_caller_address(world_address);

    let result: TupleArrayOption = world.read_model(bob);
    assert(result.pair == (55_u32, 20_u64), 'tuple item update failed');
    assert(result.samples == [1_u16, 66_u16, 3_u16], 'fixed array item update failed');
    let status_discriminator_after = starknet::syscalls::storage_read_syscall(
        0, status_discriminator_address,
    )
        .unwrap();
    assert(status_discriminator_after == current_discriminator, 'enum discr mismatch');
}

/// Test: default translator is strict and rejects mixed layouts with dynamic fields.
#[test]
#[should_panic]
fn test_request_sharding_default_rejects_mixed_dynamic() {
    let (world, model_selector) = deploy_world_with_mixed_dynamic();
    let bob: ContractAddress = 0xb0b.try_into().unwrap();

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let layout = Model::<MixedDynamic>::layout();
    let models = [(model_selector, layout).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);
}

/// Test: end_shard can be called by the same initiator that opened the shard.
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

    // Same caller that requested the shard can finish it.
    world.dispatcher.end_shard();
}

/// Regression: random non-owner, non-initiator callers cannot finish the shard.
#[test]
#[should_panic]
fn test_end_shard_rejects_non_initiator_non_owner_caller() {
    let (mut world, model_selector) = deploy_world_and_foo();
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let proxy_address = declare_and_deploy("mock_sharding_proxy");
    let layout = Model::<Foo>::layout();
    let models = [(model_selector, layout).shard([bob.into()].span())].span();
    world.dispatcher.request_sharding(proxy_address, models);

    snforge_std::start_cheat_caller_address(
        world.dispatcher.contract_address, 0xCAFE.try_into().unwrap(),
    );
    world.dispatcher.end_shard();
    snforge_std::stop_cheat_caller_address(world.dispatcher.contract_address);
}
