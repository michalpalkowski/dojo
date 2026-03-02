use dojo::meta::Layout;

/// Lightweight CRDT variant selector (no address/slot — just the type).
///
/// Used in `ShardModel` to specify the CRDT strategy per model.
/// World expands this into full `CRDType` values with computed slots.
#[derive(Drop, Serde, Copy, Debug, PartialEq)]
pub enum CRDVariant {
    #[default]
    Set,
    Add,
    Lock,
    SetLock,
    /// PN-Counter: a pair of G-Counters (additions + subtractions).
    ///
    /// All fields of the model are treated as Add (G-Counter, grow-only).
    /// The game model should expose paired fields: additions (P) and subtractions (N).
    /// Balance = P - N. Both P and N only grow, ensuring non-negative deltas at merge.
    ///
    /// This is correct per CRDT theory: PN-Counter = two independent G-Counters
    /// that converge regardless of merge order, even with concurrent shards.
    PNCounter,
}

/// Describes a model to include in a sharding request.
///
/// The game contract creates these and passes them to `world.request_sharding()`.
/// World uses the provided layout to auto-compute storage slots.
///
/// The `layout` field MUST come from `Model::<M>::layout()` called locally in the game
/// contract. Do NOT use a cross-contract `IStoredResource::layout()` call, as compiled
/// field selectors may differ between contract classes.
#[derive(Drop, Serde, Copy)]
pub struct ShardModel {
    pub selector: felt252,
    pub keys: Span<felt252>,
    pub crdt: CRDVariant,
    pub layout: Layout,
}

/// Ergonomic constructors for `ShardModel` via (selector, layout) tuples.
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
        ShardModel { selector, keys, crdt: CRDVariant::Set, layout }
    }

    fn shard_add(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        let (selector, layout) = self;
        ShardModel { selector, keys, crdt: CRDVariant::Add, layout }
    }

    fn shard_lock(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        let (selector, layout) = self;
        ShardModel { selector, keys, crdt: CRDVariant::Lock, layout }
    }

    fn shard_set_lock(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        let (selector, layout) = self;
        ShardModel { selector, keys, crdt: CRDVariant::SetLock, layout }
    }

    fn shard_pn(self: (felt252, Layout), keys: Span<felt252>) -> ShardModel {
        let (selector, layout) = self;
        ShardModel { selector, keys, crdt: CRDVariant::PNCounter, layout }
    }
}
