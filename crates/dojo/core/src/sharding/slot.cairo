use dojo::storage::database::DOJO_STORAGE;
use dojo::utils::combine_key;
use core::poseidon::poseidon_hash_span;

/// Sentinel range for packed model slot offsets. Real selectors (Poseidon hashes) never collide.
pub const PACKED_SLOT_BASE: felt252 = 'dojo_packed_slot';
pub const DYNAMIC_MEMBER_LOCK_DOMAIN: felt252 = 'dojo_dynamic_member_lock';
pub const DYNAMIC_MEMBER_CHANGED_LEN_DOMAIN: felt252 = 'dojo_dynamic_changed_len';
pub const DYNAMIC_MEMBER_CHANGED_HASH_DOMAIN: felt252 = 'dojo_dynamic_changed_hash';
pub const DYNAMIC_MEMBER_CHANGED_DATA_DOMAIN: felt252 = 'dojo_dynamic_changed_data';
pub const DYNAMIC_MEMBER_CHANGED_SEEN_DOMAIN: felt252 = 'dojo_dynamic_changed_seen';
pub const DYNAMIC_MEMBER_CHANGED_HASH_SEED: felt252 = 'dojo_dynamic_changed_seed';

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
pub fn compute_dynamic_member_changed_len_slot(lock_slot: felt252) -> felt252 {
    poseidon_hash_span([DYNAMIC_MEMBER_CHANGED_LEN_DOMAIN, lock_slot].span())
}

#[inline(always)]
pub fn compute_dynamic_member_changed_hash_slot(lock_slot: felt252) -> felt252 {
    poseidon_hash_span([DYNAMIC_MEMBER_CHANGED_HASH_DOMAIN, lock_slot].span())
}

#[inline(always)]
pub fn compute_dynamic_member_changed_data_slot(lock_slot: felt252, index: u32) -> felt252 {
    poseidon_hash_span([DYNAMIC_MEMBER_CHANGED_DATA_DOMAIN, lock_slot, index.into()].span())
}

#[inline(always)]
pub fn compute_dynamic_member_changed_seen_slot(lock_slot: felt252, slot: felt252) -> felt252 {
    poseidon_hash_span([DYNAMIC_MEMBER_CHANGED_SEEN_DOMAIN, lock_slot, slot].span())
}

#[inline(always)]
pub fn fold_dynamic_member_changed_hash(current_hash: felt252, slot: felt252) -> felt252 {
    poseidon_hash_span([current_hash, slot].span())
}

pub fn compute_dynamic_member_changed_slots_hash(slots: Span<felt252>) -> felt252 {
    let mut hash = DYNAMIC_MEMBER_CHANGED_HASH_SEED;
    for slot in slots {
        hash = fold_dynamic_member_changed_hash(hash, *slot);
    };
    hash
}

#[inline(always)]
pub fn is_packed_selector(selector: felt252) -> bool {
    let s: u256 = selector.into();
    let base: u256 = PACKED_SLOT_BASE.into();
    s >= base && s < base + 256
}
