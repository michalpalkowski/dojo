//! Sharding-specific types and interfaces for the Dojo world contract.
//!
//! These are only relevant when the `sharding` feature is enabled.
//! The types define the ABI surface for the sharding proxy (settlement)
//! and metadata queries used by the operator.

/// Describes a dynamic member change submitted during shard settlement.
#[derive(Drop, Serde, Copy)]
pub struct ShardDynamicMemberChanges {
    pub model_selector: felt252,
    pub entity_id: felt252,
    pub member_selector: felt252,
    pub changes_offset: u32,
    pub changes_len: u32,
}

/// Describes a single storage slot registered for sharding, with metadata
/// about which model/entity/member it belongs to.
#[derive(Drop, Serde, Copy)]
pub struct ShardSlotDescriptor {
    pub slot: felt252,
    pub model_selector: felt252,
    pub entity_id: felt252,
    pub member_selector: felt252,
    pub is_dynamic_lock: bool,
}

/// Methods exposed for the sharding proxy (operator's contract).
/// Called by the proxy after settlement to apply or cancel storage changes.
/// Access control is enforced inside the sharding component:
/// caller must equal the `sharding_contract_address` set during `request_sharding`.
#[starknet::interface]
pub trait IShardingProxy<T> {
    fn settle_shard_changes(
        ref self: T,
        shard_id: felt252,
        slot_changes: Array<(felt252, felt252)>,
        dynamic_members: Span<ShardDynamicMemberChanges>,
        dynamic_changes: Span<(felt252, felt252)>,
        dynamic_tracking_proofs: Span<(felt252, felt252)>,
    );
    fn cancel_shard_state(ref self: T, shard_id: felt252, slots: Span<felt252>);
}

/// Read-only metadata queries for the sharding operator.
/// Used to introspect slot descriptors and dynamic member tracking state.
#[starknet::interface]
pub trait IShardingMetadata<T> {
    fn describe_shard_slots(self: @T, slots: Span<felt252>) -> Span<ShardSlotDescriptor>;
    fn dynamic_member_storage_slots(
        self: @T, model_selector: felt252, entity_id: felt252, member_selector: felt252,
    ) -> Span<felt252>;
    fn dynamic_member_changed_slots(
        self: @T, model_selector: felt252, entity_id: felt252, member_selector: felt252,
    ) -> Span<felt252>;
    fn dynamic_member_tracking_proof_slots(
        self: @T, model_selector: felt252, entity_id: felt252, member_selector: felt252,
    ) -> Span<felt252>;
}
