use super::crdt::CRDType;
use starknet::ContractAddress;

/// Minimal proxy interface — only the methods that the contract_component
/// calls on the sharding proxy (sharding.cairo in sharding_operator).
#[starknet::interface]
pub trait ISharding<TContractState> {
    fn initialize_sharding(ref self: TContractState, storage_slots: Span<CRDType>);
    fn get_shard_id(self: @TContractState, contract_address: ContractAddress) -> felt252;
    fn end_shard(ref self: TContractState);
}
