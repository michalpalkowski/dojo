use dojo::storage::database::DOJO_STORAGE;
use dojo::utils::combine_key;
use core::poseidon::poseidon_hash_span;

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
