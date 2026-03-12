use dojo::storage::database::DOJO_STORAGE;
use dojo::utils::combine_key;
use core::poseidon::poseidon_hash_span;

/// Sentinel range for packed model slot offsets. Real selectors (Poseidon hashes) never collide.
pub const PACKED_SLOT_BASE: felt252 = 'dojo_packed_slot';
pub const DYNAMIC_MEMBER_LOCK_DOMAIN: felt252 = 'dojo_dynamic_member_lock';

#[inline(always)]
pub fn compute_dojo_field_slot(
    model_selector: felt252,
    entity_id: felt252,
    field_selector: felt252,
) -> felt252 {
    let combined_key = combine_key(entity_id, field_selector);
    poseidon_hash_span([DOJO_STORAGE, model_selector, combined_key].span())
}

#[inline(always)]
pub fn compute_dojo_packed_slot(
    model_selector: felt252,
    entity_id: felt252,
) -> felt252 {
    poseidon_hash_span([DOJO_STORAGE, model_selector, entity_id].span())
}

#[inline(always)]
pub fn compute_dynamic_member_lock_slot(
    model_selector: felt252,
    entity_id: felt252,
    member_selector: felt252,
) -> felt252 {
    poseidon_hash_span(
        [DYNAMIC_MEMBER_LOCK_DOMAIN, model_selector, entity_id, member_selector].span(),
    )
}

#[inline(always)]
pub fn is_packed_selector(selector: felt252) -> bool {
    let s: u256 = selector.into();
    let base: u256 = PACKED_SLOT_BASE.into();
    s >= base && s < base + 256
}
