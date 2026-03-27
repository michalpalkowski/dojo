//! Sharding-specific types and interfaces for the Dojo world contract.
//!
//! These are only relevant when the `sharding` feature is enabled.
//! The types define the ABI surface for sharding settlement
//! and metadata queries used by the operator.

use dojo::sharding::request::SlotEntry;

/// Sharding settlement methods on the world contract.
/// Called by the world owner (operator) to settle or cancel shards.
#[starknet::interface]
pub trait IShardingSettlement<T> {
    /// Settle changed slots with StorageCommitment verification.
    /// `entity_model_selectors` + `entity_keys_flat` provide entity keys for Torii
    /// StoreSetRecord emission (one entry per unique entity).
    fn settle(
        ref self: T,
        shard_id: felt252,
        global_state_root: felt252,
        end_block_number: u64,
        slots: Span<SlotEntry>,
        entity_model_selectors: Span<felt252>,
        entity_keys_flat: Span<felt252>,
        initial_proof: dojo::sharding::request::InitialProof,
    );

    /// Configure StorageCommitment verifier contract. One-shot.
    fn set_storage_commitment_registry(ref self: T, registry: starknet::ContractAddress);

    /// Cancel shard: unlock entities without applying changes.
    fn cancel_shard(ref self: T, shard_id: felt252);

    /// Return the request-time block number bound to this shard.
    fn get_shard_attestation_fork_block_number(self: @T, shard_id: felt252) -> u64;

    /// Return the configured sharding proxy address (0 if unset).
    fn get_sharding_proxy(self: @T) -> starknet::ContractAddress;

    /// Configure the sharding proxy address (event bus). One-shot.
    fn set_sharding_proxy(ref self: T, proxy: starknet::ContractAddress);

    /// Enable shard fork mode: skip entity_lock checks in write/delete paths.
    /// Requires world owner + at least one active shard. One-shot (cannot re-enable).
    /// Auto-resets on next `request_shard` call, preventing permanent bypass.
    /// On main chain: equivalent to cancel_shard on all active shards (world owner
    /// already has this power). On fork: enables gameplay without mass storage loading.
    fn enable_shard_fork_mode(ref self: T);

    /// Query entity lock: returns shard_id that holds this entity (0 = unlocked).
    fn get_entity_shard(self: @T, entity_id: felt252) -> felt252;

    /// List all active shard IDs (those with locked entities).
    fn get_active_shards(self: @T) -> Array<felt252>;
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
        end_block_number: u64,
        slots: Span<SlotEntry>,
        entity_model_selectors: Span<felt252>,
        entity_keys_flat: Span<felt252>,
    );
}
