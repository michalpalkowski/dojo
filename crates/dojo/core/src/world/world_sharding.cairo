//! Sharding-specific types and interfaces for the Dojo world contract.
//!
//! These are only relevant when the `sharding` feature is enabled.
//! The types define the ABI surface for sharding settlement
//! and metadata queries used by the operator.

/// Sharding settlement methods on the world contract.
/// Called by the world owner (operator) to settle or cancel shards.
#[starknet::interface]
pub trait IShardingSettlement<T> {
    /// Settle changed slots with StorageCommitment verification.
    /// Per-slot metadata enables on-chain ownership verification and Torii notification.
    /// `slot_initial_values` provides Add CRDT initial values for delta computation.
    /// `entity_model_selectors` + `entity_keys_flat` provide entity keys for Torii
    /// StoreSetRecord emission (one entry per unique entity).
    fn settle(
        ref self: T,
        shard_id: felt252,
        changed_keys: Span<felt252>,
        changed_values: Span<felt252>,
        state_diff_hash: felt252,
        global_state_root: felt252,
        end_block_number: u64,
        slot_model_selectors: Span<felt252>,
        slot_entity_ids: Span<felt252>,
        slot_computation_keys: Span<felt252>,
        slot_member_selectors: Span<felt252>,
        slot_packed_offsets: Span<u32>,
        slot_initial_values: Span<felt252>,
        entity_model_selectors: Span<felt252>,
        entity_keys_flat: Span<felt252>,
    );

    /// Configure StorageCommitment verifier contract. One-shot.
    fn set_storage_commitment_registry(ref self: T, registry: starknet::ContractAddress);

    /// Cancel shard: unlock entities without applying changes.
    fn cancel_shard(ref self: T, shard_id: felt252);

    /// Return the stored commitment hash for a shard (0 if not set).
    fn get_shard_commitment(self: @T, shard_id: felt252) -> felt252;

    /// Configure the sharding proxy address (event bus). One-shot.
    fn set_sharding_proxy(ref self: T, proxy: starknet::ContractAddress);
}

/// Dev-only settlement interface (compiled only with `--features dev`).
/// Applies CRDT state changes WITHOUT requiring TEE attestation or SP1 proof.
/// Shard activity verification is still enforced.
/// Owner-only access. Never available in production builds.
#[cfg(feature: 'dev')]
#[starknet::interface]
pub trait IShardingSettlementDev<T> {
    fn settle_dev(
        ref self: T,
        shard_id: felt252,
        changed_keys: Span<felt252>,
        changed_values: Span<felt252>,
        end_block_number: u64,
        slot_model_selectors: Span<felt252>,
        slot_entity_ids: Span<felt252>,
        slot_computation_keys: Span<felt252>,
        slot_member_selectors: Span<felt252>,
        slot_packed_offsets: Span<u32>,
        slot_initial_values: Span<felt252>,
        entity_model_selectors: Span<felt252>,
        entity_keys_flat: Span<felt252>,
    );
}
