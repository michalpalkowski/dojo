use dojo::meta::Layout;
use dojo::sharding::slot::PACKED_SLOT_BASE;

#[derive(Drop, Serde, Copy, Debug, PartialEq)]
pub enum CRDVariant {
    #[default]
    Set,
    Add,
    Lock,
    SetLock,
}

#[derive(Drop, Serde, Copy, Debug, PartialEq)]
pub struct ShardField {
    pub selector: felt252,
    pub crdt: CRDVariant,
}

pub trait IntoShardField {
    fn as_set(self: felt252) -> ShardField;
    fn as_add(self: felt252) -> ShardField;
    fn as_lock(self: felt252) -> ShardField;
    fn as_set_lock(self: felt252) -> ShardField;
}

impl Felt252IntoShardField of IntoShardField {
    fn as_set(self: felt252) -> ShardField {
        ShardField { selector: self, crdt: CRDVariant::Set }
    }

    fn as_add(self: felt252) -> ShardField {
        ShardField { selector: self, crdt: CRDVariant::Add }
    }

    fn as_lock(self: felt252) -> ShardField {
        ShardField { selector: self, crdt: CRDVariant::Lock }
    }

    fn as_set_lock(self: felt252) -> ShardField {
        ShardField { selector: self, crdt: CRDVariant::SetLock }
    }
}

#[derive(Drop, Serde, Copy)]
pub struct ShardModel {
    pub selector: felt252,
    pub keys: Span<felt252>,
    pub fields: Span<ShardField>,
}

#[derive(Copy, Drop)]
enum ShardSelectionPolicy {
    AutoSubset,
    StrictAll,
}

/// Layout is deterministic when all of its storage slots can be derived
/// without reading runtime lengths (no `Array` / `ByteArray` branches).
fn is_deterministic_layout(layout: Layout) -> bool {
    match layout {
        Layout::Fixed(_) => true,
        Layout::Struct(fields) => {
            for field in fields {
                if !is_deterministic_layout(*field.layout) {
                    return false;
                }
            };
            true
        },
        Layout::Tuple(items) => {
            for item in items {
                if !is_deterministic_layout(*item) {
                    return false;
                }
            };
            true
        },
        Layout::FixedArray(fixed_array_layout) => {
            let (item_layouts, _): (Span<Layout>, u32) = fixed_array_layout;
            if item_layouts.len() == 0 {
                return false;
            }
            is_deterministic_layout(*item_layouts[0])
        },
        Layout::Enum(variants) => {
            for variant in variants {
                if !is_deterministic_layout(*variant.layout) {
                    return false;
                }
            };
            true
        },
        Layout::Array(_) => false,
        Layout::ByteArray => false,
    }
}

/// Translate `Layout::Struct` into top-level `ShardField`s.
///
/// We keep member selectors top-level for protocol compatibility
/// (`StoreUpdateMember` emission), while world-side request handling expands
/// each selected member recursively into concrete deterministic slots.
fn translate_struct_fields(
    fields: Span<dojo::meta::FieldLayout>, crdt: CRDVariant, policy: ShardSelectionPolicy,
) -> Span<ShardField> {
    let mut result: Array<ShardField> = ArrayTrait::new();
    for field in fields {
        if is_deterministic_layout(*field.layout) {
            result.append(ShardField { selector: (*field).selector, crdt });
        } else {
            // Auto policy intentionally skips dynamic members.
            // Strict policy fails fast on the first unsupported member.
            if let ShardSelectionPolicy::StrictAll = policy {
                panic!("ShardModel: unsupported field layout");
            }
        }
    };
    assert(result.len() != 0, 'ShardModel: no shardable fields');
    result.span()
}

fn translate_layout(
    layout: Layout, crdt: CRDVariant, policy: ShardSelectionPolicy,
) -> Span<ShardField> {
    match layout {
        Layout::Struct(fields) => translate_struct_fields(fields, crdt, policy),
        Layout::Fixed(sizes) => {
            let mut sizes = sizes;
            let num_slots = dojo::storage::packing::calculate_packed_size(ref sizes);
            assert(num_slots <= 256, 'ShardModel: packed too large');
            let mut result: Array<ShardField> = ArrayTrait::new();
            let mut i: u32 = 0;
            while i < num_slots {
                result.append(ShardField { selector: PACKED_SLOT_BASE + i.into(), crdt });
                i += 1;
            };
            result.span()
        },
        _ => panic!("ShardModel: unsupported layout type"),
    }
}

/// Apply a single CRDT to all fields of a model. For per-field control,
/// construct `ShardModel` directly.
pub trait IntoShardModel {
    fn shard(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel;
    fn shard_add(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel;
    fn shard_lock(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel;
    fn shard_set_lock(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel;
    fn shard_strict(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel;
    fn shard_add_strict(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel;
    fn shard_lock_strict(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel;
    fn shard_set_lock_strict(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel;
}

impl SelectorLayoutIntoShardModel of IntoShardModel {
    fn shard(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        let (selector, layout) = self;
        ShardModel {
            selector, keys, fields: translate_layout(layout, CRDVariant::Set, ShardSelectionPolicy::AutoSubset),
        }
    }

    fn shard_add(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        let (selector, layout) = self;
        ShardModel {
            selector, keys, fields: translate_layout(layout, CRDVariant::Add, ShardSelectionPolicy::AutoSubset),
        }
    }

    fn shard_lock(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        let (selector, layout) = self;
        ShardModel {
            selector, keys, fields: translate_layout(layout, CRDVariant::Lock, ShardSelectionPolicy::AutoSubset),
        }
    }

    fn shard_set_lock(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        let (selector, layout) = self;
        ShardModel {
            selector, keys: keys, fields: translate_layout(layout, CRDVariant::SetLock, ShardSelectionPolicy::AutoSubset),
        }
    }

    fn shard_strict(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        let (selector, layout) = self;
        ShardModel {
            selector, keys, fields: translate_layout(layout, CRDVariant::Set, ShardSelectionPolicy::StrictAll),
        }
    }

    fn shard_add_strict(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        let (selector, layout) = self;
        ShardModel {
            selector, keys, fields: translate_layout(layout, CRDVariant::Add, ShardSelectionPolicy::StrictAll),
        }
    }

    fn shard_lock_strict(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        let (selector, layout) = self;
        ShardModel {
            selector, keys, fields: translate_layout(layout, CRDVariant::Lock, ShardSelectionPolicy::StrictAll),
        }
    }

    fn shard_set_lock_strict(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        let (selector, layout) = self;
        ShardModel {
            selector, keys, fields: translate_layout(layout, CRDVariant::SetLock, ShardSelectionPolicy::StrictAll),
        }
    }
}
