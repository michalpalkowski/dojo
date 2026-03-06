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

fn expand_layout(layout: Layout, crdt: CRDVariant) -> Span<ShardField> {
    match layout {
        Layout::Struct(fields) => {
            let mut result: Array<ShardField> = ArrayTrait::new();
            for field in fields {
                result.append(ShardField { selector: (*field).selector, crdt });
            };
            result.span()
        },
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
}

impl SelectorLayoutIntoShardModel of IntoShardModel {
    fn shard(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        let (selector, layout) = self;
        ShardModel { selector, keys, fields: expand_layout(layout, CRDVariant::Set) }
    }

    fn shard_add(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        let (selector, layout) = self;
        ShardModel { selector, keys, fields: expand_layout(layout, CRDVariant::Add) }
    }

    fn shard_lock(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        let (selector, layout) = self;
        ShardModel { selector, keys, fields: expand_layout(layout, CRDVariant::Lock) }
    }

    fn shard_set_lock(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        let (selector, layout) = self;
        ShardModel { selector, keys, fields: expand_layout(layout, CRDVariant::SetLock) }
    }
}
