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
    /// Maximum number of array elements to pre-allocate (0 = not a dynamic array).
    pub max_elements: u32,
}

pub trait IntoShardField {
    fn as_set(self: felt252) -> ShardField;
    fn as_add(self: felt252) -> ShardField;
    fn as_lock(self: felt252) -> ShardField;
    fn as_set_lock(self: felt252) -> ShardField;
}

impl Felt252IntoShardField of IntoShardField {
    fn as_set(self: felt252) -> ShardField {
        ShardField { selector: self, crdt: CRDVariant::Set, max_elements: 0 }
    }

    fn as_add(self: felt252) -> ShardField {
        ShardField { selector: self, crdt: CRDVariant::Add, max_elements: 0 }
    }

    fn as_lock(self: felt252) -> ShardField {
        ShardField { selector: self, crdt: CRDVariant::Lock, max_elements: 0 }
    }

    fn as_set_lock(self: felt252) -> ShardField {
        ShardField { selector: self, crdt: CRDVariant::SetLock, max_elements: 0 }
    }
}

#[derive(Drop, Serde, Copy)]
pub struct ShardModel {
    pub selector: felt252,
    pub keys: Span<felt252>,
    pub fields: Span<ShardField>,
    pub coverage: ShardCoverage,
}

#[derive(Drop, Serde, Copy, Debug, PartialEq)]
pub enum ShardCoverage {
    #[default]
    Full,
    DeterministicSubset,
}

#[derive(Copy, Drop)]
pub enum ShardFieldSelection {
    AutoDeterministic,
    AutoIncludeDynamic,
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
    fields: Span<dojo::meta::FieldLayout>, crdt: CRDVariant, selection: ShardFieldSelection,
) -> Span<ShardField> {
    let mut result: Array<ShardField> = ArrayTrait::new();
    let include_dynamic = should_include_dynamic(selection);
    let strict_all = is_strict_all(selection);

    for field in fields {
        if is_deterministic_layout(*field.layout) {
            result.append(ShardField { selector: (*field).selector, crdt, max_elements: 0 });
            continue;
        }

        if include_dynamic {
            result.append(ShardField { selector: (*field).selector, crdt, max_elements: 0 });
            continue;
        }

        // Deterministic-only policy intentionally skips dynamic members.
        // Strict policy fails fast on the first unsupported member.
        if strict_all {
            panic!("ShardModel: unsupported field layout");
        }
    };
    assert(result.len() != 0, 'ShardModel: no shardable fields');
    result.span()
}

#[inline(always)]
fn should_include_dynamic(selection: ShardFieldSelection) -> bool {
    match selection {
        ShardFieldSelection::AutoIncludeDynamic => true,
        _ => false,
    }
}

#[inline(always)]
fn is_strict_all(selection: ShardFieldSelection) -> bool {
    match selection {
        ShardFieldSelection::StrictAll => true,
        _ => false,
    }
}

fn translate_layout(
    layout: Layout, crdt: CRDVariant, selection: ShardFieldSelection,
) -> Span<ShardField> {
    match layout {
        Layout::Struct(fields) => translate_struct_fields(fields, crdt, selection),
        Layout::Fixed(sizes) => {
            let mut sizes = sizes;
            let num_slots = dojo::storage::packing::calculate_packed_size(ref sizes);
            assert(num_slots <= 256, 'ShardModel: packed too large');
            let mut result: Array<ShardField> = ArrayTrait::new();
            let mut i: u32 = 0;
            while i < num_slots {
                result.append(ShardField { selector: PACKED_SLOT_BASE + i.into(), crdt, max_elements: 0 });
                i += 1;
            };
            result.span()
        },
        _ => panic!("ShardModel: unsupported layout type"),
    }
}

/// Apply a single CRDT to all fields of a model. For per-field control,
/// construct `ShardModel` directly and set `coverage` explicitly.
///
/// `shard()` is strict-by-default and rejects dynamic members. Use
/// `shard_deterministic()` only when partial deterministic coverage is an
/// explicit, intentional choice.
pub trait IntoShardModel {
    fn shard_with(
        self: (felt252, Layout),
        keys: Span<felt252>,
        crdt: CRDVariant,
        selection: ShardFieldSelection,
    ) -> ShardModel;

    fn shard_deterministic(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel;
    fn shard(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel;
    fn shard_dynamic(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel;
}

impl SelectorLayoutIntoShardModel of IntoShardModel {
    fn shard_with(
        self: (felt252, Layout),
        keys: Span<felt252>,
        crdt: CRDVariant,
        selection: ShardFieldSelection,
    ) -> ShardModel {
        let (selector, layout) = self;
        let coverage = match selection {
            ShardFieldSelection::AutoDeterministic => ShardCoverage::DeterministicSubset,
            _ => ShardCoverage::Full,
        };
        ShardModel { selector, keys, fields: translate_layout(layout, crdt, selection), coverage }
    }

    fn shard_deterministic(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        self.shard_with(keys, CRDVariant::Set, ShardFieldSelection::AutoDeterministic)
    }

    fn shard(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        self.shard_with(keys, CRDVariant::Set, ShardFieldSelection::StrictAll)
    }

    fn shard_dynamic(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        self.shard_with(keys, CRDVariant::SetLock, ShardFieldSelection::AutoIncludeDynamic)
    }
}
