use dojo::meta::Layout;

/// Lightweight CRDT variant selector (no address/slot — just the type).
///
/// Used in `ShardField` to specify the CRDT strategy per field.
/// World expands this into full `CRDType` values with computed slots.
#[derive(Drop, Serde, Copy, Debug, PartialEq)]
pub enum CRDVariant {
    #[default]
    Set,
    Add,
    Lock,
    SetLock,
}

/// Per-field CRDT configuration for sharding.
///
/// Each `ShardField` maps a model field (by its layout selector) to a CRDT type.
/// This allows different fields within the same model to use different merge strategies.
///
/// # Example
/// ```cairo
/// use dojo::sharding::request::IntoShardField;
///
/// let fields = [
///     selector!("stone_balance").as_add(),   // delta merge
///     selector!("wood_balance").as_add(),     // delta merge
///     selector!("owner").as_lock(),           // exclusive reservation
/// ].span();
/// ```
#[derive(Drop, Serde, Copy, Debug, PartialEq)]
pub struct ShardField {
    pub selector: felt252,
    pub crdt: CRDVariant,
}

/// Ergonomic constructors for `ShardField` from a field selector.
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

/// Describes a model to include in a sharding request.
///
/// The game contract creates these and passes them to `world.request_sharding()`.
/// Each field specifies its own CRDT strategy via `ShardField`.
///
/// # Two usage modes
///
/// **Whole-model** — apply one CRDT to all fields (use `IntoShardModel` helpers):
/// ```cairo
/// let layout = Model::<Resource>::layout();
/// (resource_sel, layout).shard_add(keys)   // all fields → Add
/// ```
///
/// **Per-field** — different CRDT per field (construct directly):
/// ```cairo
/// ShardModel {
///     selector: resource_sel, keys,
///     fields: [
///         selector!("stone_balance").as_add(),
///         selector!("owner").as_lock(),
///     ].span(),
/// }
/// ```
#[derive(Drop, Serde, Copy)]
pub struct ShardModel {
    pub selector: felt252,
    pub keys: Span<felt252>,
    pub fields: Span<ShardField>,
}

/// Expands a model `Layout` into `Span<ShardField>` with a uniform CRDT for all fields.
///
/// Only supports `Layout::Struct` — panics on other layout types.
/// Used internally by `IntoShardModel` helpers.
fn expand_layout(layout: Layout, crdt: CRDVariant) -> Span<ShardField> {
    if let Layout::Struct(fields) = layout {
        let mut result: Array<ShardField> = ArrayTrait::new();
        for field in fields {
            result.append(ShardField { selector: (*field).selector, crdt });
        };
        result.span()
    } else {
        panic!("ShardModel: expected Layout::Struct")
    }
}

/// Ergonomic constructors for `ShardModel` via (selector, layout) tuples.
///
/// These apply a single CRDT to ALL fields of the model. For per-field control,
/// construct `ShardModel` directly with a `fields` array.
///
/// # Example
/// ```cairo
/// use dojo::sharding::request::IntoShardModel;
///
/// let resource_selector = Model::<Resource>::selector(namespace_hash);
/// let resource_layout = Model::<Resource>::layout();
/// let keys = [player_id].span();
///
/// world.request_sharding(proxy, [
///     (resource_selector, resource_layout).shard_add(keys),   // Add CRDT
///     (army_selector, army_layout).shard(keys),               // Set CRDT (default)
/// ].span());
/// ```
pub trait IntoShardModel {
    fn shard(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel;
    fn shard_add(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel;
    fn shard_lock(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel;
    fn shard_set_lock(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel;
    /// PN-Counter: paired G-Counters for fields that can be both added to and subtracted from.
    ///
    /// The model should have fields in (additions, subtractions) pairs.
    /// Both are treated as G-Counters (grow-only Add CRDT).
    /// Balance = additions - subtractions.
    fn shard_pn(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel;
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

    fn shard_pn(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        let (selector, layout) = self;
        ShardModel { selector, keys, fields: expand_layout(layout, CRDVariant::Add) }
    }
}
