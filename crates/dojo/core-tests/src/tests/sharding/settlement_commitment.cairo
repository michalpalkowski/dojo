/// Production `settle()` (storage commitment) edge cases. Happy-path settlement with the
/// mock verifier lives in `component.cairo` via `helpers::settle_as_owner`.

use dojo::model::ModelStorageTest;
use dojo::sharding::compute_dojo_field_slot;
use dojo::utils::entity_id_from_keys;
use dojo::world::IWorldDispatcherTrait;
use starknet::ContractAddress;

use crate::tests::helpers::{Foo, deploy_world_and_foo};
use crate::tests::sharding::helpers::{foo_field_selectors, make_field_slot, settle_as_owner};

#[test]
#[should_panic(expected: ('Shard: registry not configured',))]
fn test_settle_panics_without_commitment_registry() {
    snforge_std::start_cheat_account_contract_address_global(snforge_std::test_address());
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;
    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let (sel_a, sel_b) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let slot_b = compute_dojo_field_slot(model_selector, entity_id, sel_b);

    world.dispatcher.request_sharding([entity_id].span(), [].span(), [].span());

    settle_as_owner(
        world_address,
        1,
        [
            make_field_slot(slot_a, 999, model_selector, entity_id, sel_a, 0),
            make_field_slot(slot_b, 777, model_selector, entity_id, sel_b, 0),
        ].span(),
    );
}
