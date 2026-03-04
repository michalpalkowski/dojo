use dojo::storage::database::DOJO_STORAGE;
use dojo::utils::combine_key;
use core::poseidon::poseidon_hash_span;

/// Sentinel base for packed model slot selectors.
///
/// ShardField selectors in the range `[PACKED_SLOT_BASE, PACKED_SLOT_BASE + 255]`
/// are treated as packed model slot offsets by `request_sharding`.
/// Real field selectors (Poseidon hashes) are astronomically larger and never collide.
pub const PACKED_SLOT_BASE: felt252 = 'dojo_packed_slot';

/// Computes the raw Starknet storage address for a single Dojo model field.
///
/// The returned felt252 is `poseidon_hash(['dojo_storage', model_selector,
/// combine_key(entity_id, field_selector)])` — the same base address that
/// `storage::set` / `storage::get` resolve to via `storage_base_address_from_felt252`.
///
/// # Arguments
///   * `model_selector` – `Model::<M>::selector(namespace_hash)`
///   * `entity_id`      – `entity_id_from_keys(@keys)` or `entity_id_from_serialized_keys(keys)`
///   * `field_selector`  – `FieldLayout.selector` from the model's `Introspect::layout()`
#[inline(always)]
pub fn compute_dojo_field_slot(
    model_selector: felt252,
    entity_id: felt252,
    field_selector: felt252,
) -> felt252 {
    let combined_key = combine_key(entity_id, field_selector);
    poseidon_hash_span([DOJO_STORAGE, model_selector, combined_key].span())
}

/// Computes the base storage slot for a packed (Layout::Fixed) model entity.
///
/// Packed models store all data starting from `poseidon_hash(['dojo_storage',
/// model_selector, entity_id])`. Multi-slot packed data occupies sequential
/// addresses: base, base+1, base+2, etc.
#[inline(always)]
pub fn compute_dojo_packed_slot(
    model_selector: felt252,
    entity_id: felt252,
) -> felt252 {
    poseidon_hash_span([DOJO_STORAGE, model_selector, entity_id].span())
}

/// Returns true if the selector is a packed slot sentinel (offset from PACKED_SLOT_BASE).
#[inline(always)]
pub fn is_packed_selector(selector: felt252) -> bool {
    let s: u256 = selector.into();
    let base: u256 = PACKED_SLOT_BASE.into();
    s >= base && s < base + 256
}
