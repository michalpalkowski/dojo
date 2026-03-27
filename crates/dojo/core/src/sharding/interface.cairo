use starknet::ContractAddress;

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

