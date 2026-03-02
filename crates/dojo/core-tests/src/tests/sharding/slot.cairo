use core::poseidon::poseidon_hash_span;
use dojo::sharding::compute_dojo_field_slot;
use dojo::storage::database;
use dojo::utils::combine_key;
use starknet::storage_access::storage_base_address_from_felt252;
use starknet::syscalls::storage_read_syscall;

/// Verifies that `compute_dojo_field_slot` produces the same storage address
/// that `database::set` writes to. We write via the database layer, then read
/// via `storage_read_syscall` at the address our function computes.
#[test]
fn test_slot_matches_database_write() {
    let model_selector: felt252 = 'test_model';
    let entity_id: felt252 = 'player_1';
    let field_selector: felt252 = selector!("score");

    let field_key = combine_key(entity_id, field_selector);

    let value: felt252 = 42;
    database::set(model_selector, field_key, [value].span(), 0, [251].span());

    let slot = compute_dojo_field_slot(model_selector, entity_id, field_selector);

    let base = storage_base_address_from_felt252(slot);
    let read_value = storage_read_syscall(0, starknet::storage_access::storage_address_from_base(base))
        .unwrap();
    assert(read_value == value, 'slot value mismatch');
}

/// Same formula computed manually vs via the helper — must match.
#[test]
fn test_slot_formula_manual() {
    let model_selector: felt252 = 0x1234;
    let entity_id: felt252 = 0xABCD;
    let field_selector: felt252 = 0x5678;

    let expected = poseidon_hash_span(
        [database::DOJO_STORAGE, model_selector, combine_key(entity_id, field_selector)].span(),
    );

    let result = compute_dojo_field_slot(model_selector, entity_id, field_selector);
    assert(result == expected, 'formula mismatch');
}

/// Different entity ids must produce different slots for the same field.
#[test]
fn test_different_entities_different_slots() {
    let model_selector: felt252 = 'model';
    let field_selector: felt252 = selector!("balance");

    let slot_a = compute_dojo_field_slot(model_selector, 'entity_a', field_selector);
    let slot_b = compute_dojo_field_slot(model_selector, 'entity_b', field_selector);
    assert(slot_a != slot_b, 'same slot for diff entities');
}

/// Different field selectors must produce different slots for the same entity.
#[test]
fn test_different_fields_different_slots() {
    let model_selector: felt252 = 'model';
    let entity_id: felt252 = 'entity';

    let slot_a = compute_dojo_field_slot(model_selector, entity_id, selector!("field_a"));
    let slot_b = compute_dojo_field_slot(model_selector, entity_id, selector!("field_b"));
    assert(slot_a != slot_b, 'same slot for diff fields');
}
