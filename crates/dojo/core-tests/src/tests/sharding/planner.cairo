use dojo::model::Model;
use dojo::sharding::planner::{collect_shardable_slots, is_dynamic_layout, plan_model_slots};
use dojo::sharding::request::{CRDVariant, IntoShardField};
use dojo::sharding::slot::{
    PACKED_SLOT_BASE, compute_dojo_field_slot, compute_dojo_packed_slot, compute_dynamic_member_lock_slot,
};
use starknet::ContractAddress;

use crate::tests::helpers::{Foo, MixedDynamic, Score};

fn foo_selectors() -> (felt252, felt252) {
    let layout = Model::<Foo>::layout();
    if let dojo::meta::Layout::Struct(fields) = layout {
        ((*fields[0]).selector, (*fields[1]).selector)
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

#[test]
fn test_planner_struct_fixed_member_slot() {
    let model_selector = Model::<Foo>::selector(crate::tests::helpers::DOJO_NSH);
    let player: ContractAddress = 0xb0b.try_into().unwrap();
    let entity_id = dojo::utils::entity_id_from_keys(@player);
    let (a_selector, _) = foo_selectors();
    let layout = Model::<Foo>::layout();

    let planned_slots = plan_model_slots(
        model_selector, entity_id, layout, [a_selector.as_set()].span(),
    );
    assert(planned_slots.len() == 1, 'one slot');

    let planned = *planned_slots[0];
    let expected_slot = compute_dojo_field_slot(model_selector, entity_id, a_selector);
    assert(planned.slot == expected_slot, 'slot mismatch');
    assert(planned.member_selector == a_selector, 'member sel mismatch');
    assert(planned.crdt == CRDVariant::Set, 'crdt mismatch');
}

#[test]
fn test_planner_dynamic_member_setlock_uses_dynamic_lock_slot() {
    let model_selector = Model::<MixedDynamic>::selector(crate::tests::helpers::DOJO_NSH);
    let player: ContractAddress = 0xb0b.try_into().unwrap();
    let entity_id = dojo::utils::entity_id_from_keys(@player);
    let (_, note_selector) = mixed_dynamic_selectors();
    let layout = Model::<MixedDynamic>::layout();

    let planned_slots = plan_model_slots(
        model_selector, entity_id, layout, [note_selector.as_set_lock()].span(),
    );
    assert(planned_slots.len() == 1, 'one dynamic lock');

    let planned = *planned_slots[0];
    let expected_lock_slot = compute_dynamic_member_lock_slot(model_selector, entity_id, note_selector);
    assert(planned.slot == expected_lock_slot, 'dynamic lock slot mismatch');
    assert(planned.member_selector == note_selector, 'member sel mismatch');
    assert(planned.crdt == CRDVariant::SetLock, 'crdt mismatch');
}

#[test]
#[should_panic]
fn test_planner_dynamic_member_requires_setlock() {
    let model_selector = Model::<MixedDynamic>::selector(crate::tests::helpers::DOJO_NSH);
    let player: ContractAddress = 0xb0b.try_into().unwrap();
    let entity_id = dojo::utils::entity_id_from_keys(@player);
    let (_, note_selector) = mixed_dynamic_selectors();
    let layout = Model::<MixedDynamic>::layout();

    let _ = plan_model_slots(model_selector, entity_id, layout, [note_selector.as_set()].span());
}

#[test]
fn test_planner_fixed_model_packed_selector() {
    let model_selector = Model::<Score>::selector(crate::tests::helpers::DOJO_NSH);
    let player: ContractAddress = 0xb0b.try_into().unwrap();
    let entity_id = dojo::utils::entity_id_from_keys(@player);
    let layout = Model::<Score>::layout();

    let planned_slots = plan_model_slots(
        model_selector, entity_id, layout, [PACKED_SLOT_BASE.as_add()].span(),
    );
    assert(planned_slots.len() == 1, 'one packed slot');

    let planned = *planned_slots[0];
    let expected_slot = compute_dojo_packed_slot(model_selector, entity_id);
    assert(planned.slot == expected_slot, 'packed slot mismatch');
    assert(planned.member_selector == PACKED_SLOT_BASE, 'member sel mismatch');
    assert(planned.crdt == CRDVariant::Add, 'crdt mismatch');
}

#[test]
#[should_panic]
fn test_planner_fixed_model_packed_selector_out_of_range() {
    let model_selector = Model::<Score>::selector(crate::tests::helpers::DOJO_NSH);
    let player: ContractAddress = 0xb0b.try_into().unwrap();
    let entity_id = dojo::utils::entity_id_from_keys(@player);
    let layout = Model::<Score>::layout();

    let _ = plan_model_slots(
        model_selector, entity_id, layout, [(PACKED_SLOT_BASE + 1).as_set()].span(),
    );
}

#[test]
fn test_planner_collect_shardable_slots_reports_dynamic_branch() {
    let model_selector = Model::<MixedDynamic>::selector(crate::tests::helpers::DOJO_NSH);
    let player: ContractAddress = 0xb0b.try_into().unwrap();
    let entity_id = dojo::utils::entity_id_from_keys(@player);
    let (fixed_selector, _) = mixed_dynamic_selectors();
    let layout = Model::<MixedDynamic>::layout();

    let mut slots: Array<felt252> = ArrayTrait::new();
    let deterministic = collect_shardable_slots(ref slots, model_selector, entity_id, layout);
    assert(!deterministic, 'mixed dyn non-det');

    let expected_fixed_slot = compute_dojo_field_slot(model_selector, entity_id, fixed_selector);
    assert(slots.len() == 1, 'only fixed slot');
    assert(*slots[0] == expected_fixed_slot, 'fixed slot mismatch');
}

#[test]
fn test_planner_is_dynamic_layout() {
    assert(!is_dynamic_layout(Model::<Foo>::layout()), 'foo deterministic');
    assert(is_dynamic_layout(Model::<MixedDynamic>::layout()), 'mixed dynamic');
}
