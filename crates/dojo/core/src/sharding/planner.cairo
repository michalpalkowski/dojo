use dojo::meta::Layout;
use dojo::sharding::crdt::CRDType;
use dojo::sharding::request::{CRDVariant, ShardField};
use dojo::sharding::slot::{
    PACKED_SLOT_BASE, compute_dojo_packed_slot, compute_dynamic_member_lock_slot, is_packed_selector,
};
use dojo::utils::combine_key;
use starknet::ContractAddress;

#[derive(Copy, Drop)]
pub struct PlannedSlot {
    pub slot: felt252,
    pub member_selector: felt252,
    pub crdt: CRDVariant,
}

/// Collect deterministic storage slots for a layout rooted at `key`.
///
/// Returns `true` only when the full layout is deterministic. Dynamic
/// branches (`Array` and `ByteArray`) return `false` and are skipped.
pub fn collect_shardable_slots(
    ref slots: Array<felt252>, model_selector: felt252, key: felt252, layout: Layout,
) -> bool {
    match layout {
        Layout::Fixed(bits_layout) => {
            let mut bits_layout = bits_layout;
            let packed_size = dojo::storage::packing::calculate_packed_size(ref bits_layout);
            let packed_base = compute_dojo_packed_slot(model_selector, key);
            let mut i: usize = 0;
            while i < packed_size {
                slots.append(packed_base + i.into());
                i += 1;
            };
            true
        },
        Layout::Struct(fields) => {
            let mut deterministic = true;
            for field_layout in fields {
                let field_key = combine_key(key, *field_layout.selector);
                if !collect_shardable_slots(ref slots, model_selector, field_key, *field_layout.layout) {
                    deterministic = false;
                }
            };
            deterministic
        },
        Layout::Tuple(item_layouts) => {
            let mut deterministic = true;
            for (i, item_layout) in item_layouts.into_iter().enumerate() {
                let item_key = combine_key(key, i.into());
                if !collect_shardable_slots(ref slots, model_selector, item_key, *item_layout) {
                    deterministic = false;
                }
            };
            deterministic
        },
        Layout::FixedArray(fixed_array_layout) => {
            let (item_layouts, array_len): (Span<Layout>, u32) = fixed_array_layout;
            if item_layouts.len() == 0 {
                return false;
            }
            let item_layout = *item_layouts[0];
            let mut deterministic = true;
            let mut i: u32 = 0;
            while i < array_len {
                let item_key = combine_key(key, i.into());
                if !collect_shardable_slots(ref slots, model_selector, item_key, item_layout) {
                    deterministic = false;
                }
                i += 1;
            };
            deterministic
        },
        Layout::Enum(variant_layouts) => {
            // Variant discriminator is always stored at the root key.
            slots.append(compute_dojo_packed_slot(model_selector, key));

            let mut deterministic = true;
            for variant_layout in variant_layouts {
                let variant_key = combine_key(key, *variant_layout.selector);
                if !collect_shardable_slots(
                    ref slots, model_selector, variant_key, *variant_layout.layout,
                ) {
                    deterministic = false;
                }
            };
            deterministic
        },
        Layout::Array(_) => false,
        Layout::ByteArray => false,
    }
}

pub fn is_dynamic_layout(layout: Layout) -> bool {
    match layout {
        Layout::Fixed(_) => false,
        Layout::Struct(fields) => {
            for field in fields {
                if is_dynamic_layout(*field.layout) {
                    return true;
                }
            };
            false
        },
        Layout::Tuple(item_layouts) => {
            for item_layout in item_layouts {
                if is_dynamic_layout(*item_layout) {
                    return true;
                }
            };
            false
        },
        Layout::FixedArray(fixed_array_layout) => {
            let (item_layouts, _): (Span<Layout>, u32) = fixed_array_layout;
            if item_layouts.len() == 0 {
                return true;
            }
            is_dynamic_layout(*item_layouts[0])
        },
        Layout::Enum(variant_layouts) => {
            for variant_layout in variant_layouts {
                if is_dynamic_layout(*variant_layout.layout) {
                    return true;
                }
            };
            false
        },
        Layout::Array(_) => true,
        Layout::ByteArray => true,
    }
}

pub fn collect_dynamic_member_locks(
    ref slots: Array<felt252>, model_selector: felt252, entity_id: felt252, model_layout: Layout,
) {
    match model_layout {
        Layout::Struct(fields) => {
            for field in fields {
                if is_dynamic_layout(*field.layout) {
                    slots.append(
                        compute_dynamic_member_lock_slot(model_selector, entity_id, *field.selector),
                    );
                }
            };
        },
        _ => {},
    }
}

#[inline(always)]
pub fn shard_crdt_type(crdt: CRDVariant, world_addr: ContractAddress, slot: felt252) -> CRDType {
    match crdt {
        CRDVariant::Set => CRDType::Set((world_addr, slot)),
        CRDVariant::Add => CRDType::Add((world_addr, slot)),
        CRDVariant::Lock => CRDType::Lock((world_addr, slot)),
        CRDVariant::SetLock => CRDType::SetLock((world_addr, slot)),
    }
}

pub fn plan_shard_field(
    model_selector: felt252, entity_id: felt252, model_layout: Layout, shard_field: ShardField,
) -> Array<PlannedSlot> {
    match model_layout {
        Layout::Struct(field_layouts) => {
            let field_layout = match dojo::utils::find_field_layout(shard_field.selector, field_layouts) {
                Option::Some(layout) => layout,
                Option::None => panic!("request_sharding: unknown field selector"),
            };

            if is_dynamic_layout(field_layout) {
                assert(shard_field.crdt == CRDVariant::SetLock, 'Shard: dynamic SetLock');
                let lock_slot = compute_dynamic_member_lock_slot(
                    model_selector, entity_id, shard_field.selector,
                );
                let mut dynamic_slots: Array<PlannedSlot> = ArrayTrait::new();
                dynamic_slots.append(
                    PlannedSlot {
                        slot: lock_slot,
                        member_selector: shard_field.selector,
                        crdt: shard_field.crdt,
                    },
                );
                dynamic_slots
            } else {
                // Struct members may be nested; expand recursively to deterministic slots.
                let field_key = combine_key(entity_id, shard_field.selector);
                let mut field_slots: Array<felt252> = ArrayTrait::new();
                let deterministic = collect_shardable_slots(
                    ref field_slots, model_selector, field_key, field_layout,
                );
                assert(deterministic, 'Shard: deterministic only');
                assert(field_slots.len() != 0, 'Shard: no slots');

                let mut planned_slots: Array<PlannedSlot> = ArrayTrait::new();
                for slot in field_slots.span() {
                    planned_slots.append(
                        PlannedSlot {
                            slot: *slot,
                            member_selector: shard_field.selector,
                            crdt: shard_field.crdt,
                        },
                    );
                };
                planned_slots
            }
        },
        Layout::Fixed(model_fixed_layout) => {
            assert(is_packed_selector(shard_field.selector), 'Shard: packed selector');

            let mut model_fixed_layout = model_fixed_layout;
            let packed_size = dojo::storage::packing::calculate_packed_size(ref model_fixed_layout);
            let offset = shard_field.selector - PACKED_SLOT_BASE;
            let offset_index: usize = offset.try_into().unwrap();
            assert(offset_index < packed_size, 'Shard: packed range');

            let mut planned_slots: Array<PlannedSlot> = ArrayTrait::new();
            planned_slots.append(
                PlannedSlot {
                    slot: compute_dojo_packed_slot(model_selector, entity_id) + offset,
                    member_selector: shard_field.selector,
                    crdt: shard_field.crdt,
                },
            );
            planned_slots
        },
        _ => panic!("request_sharding: unsupported model layout"),
    }
}

pub fn plan_model_slots(
    model_selector: felt252,
    entity_id: felt252,
    model_layout: Layout,
    fields: Span<ShardField>,
) -> Array<PlannedSlot> {
    let mut planned_slots: Array<PlannedSlot> = ArrayTrait::new();
    for shard_field in fields {
        let field_slots = plan_shard_field(model_selector, entity_id, model_layout, *shard_field);
        for planned in field_slots.span() {
            planned_slots.append(*planned);
        };
    };
    planned_slots
}
