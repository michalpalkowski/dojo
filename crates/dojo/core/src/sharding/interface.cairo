use starknet::ContractAddress;

/// Base interface that every sharding-enabled game contract should implement.
/// Provides the common operations that the operator and deployment tooling
/// can rely on regardless of game-specific sharding logic.
///
/// Games extend this with their own `request_shard` variants
/// (e.g. `request_shard_realm` in Eternum) since entity selection
/// is inherently game-specific.
#[starknet::interface]
pub trait IShardingGame<T> {
    /// Register CRDT policies for all shardable models. Called once at deploy.
    fn register_policies(ref self: T);
    /// Lock entities and allocate a shard. Returns shard_id.
    ///
    /// Arguments match the sharding component's `request_shard` so the operator
    /// can call any game contract with the same generic signature.
    ///
    /// * `entities` — Dojo entity IDs for exclusive lock (SetLock/Lock).
    /// * `shared_entities` — Dojo entity IDs for concurrent lock (Set/Add).
    /// * `entity_keys_flat` — Length-prefixed key groups: [n_keys_0, key0..., n_keys_1, key1...].
    fn request_shard(
        ref self: T,
        entities: Span<felt252>,
        shared_entities: Span<felt252>,
        entity_keys_flat: Span<felt252>,
    ) -> felt252;
    /// Signal that the shard has finished gameplay (triggers settlement flow).
    fn end_shard(ref self: T, shard_id: felt252);
}

/// Interface for the StorageCommitment contract that verifies
/// SP1-proven storage commitments on-chain.
#[starknet::interface]
pub trait IStorageCommitmentVerifier<T> {
    /// Returns (verified, event_game_contract, event_shard_id).
    fn verify(
        ref self: T,
        storage_commitment: felt252,
        contract_address: ContractAddress,
        global_state_root: felt252,
        end_block_number: u64,
    ) -> (bool, felt252, felt252);
}

