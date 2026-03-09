//! Keeps only the mock sharding proxy contract used by sharding tests.
//! Behavior tests live in `request.cairo` and `settlement_events.cairo`.

#[starknet::contract]
pub mod mock_sharding_proxy {
    use dojo::sharding::crdt::CRDType;

    #[storage]
    struct Storage {}

    #[abi(embed_v0)]
    impl IShardingImpl of dojo::sharding::interface::ISharding<ContractState> {
        fn initialize_sharding(ref self: ContractState, storage_slots: Span<CRDType>) {}
        fn end_shard(ref self: ContractState) {}
    }
}
