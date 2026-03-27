use core::poseidon::poseidon_hash_span;
use dojo::model::{Model, ModelStorageTest};
use dojo::sharding::compute_dojo_field_slot;
use dojo::sharding::request::{InitialProof, SlotEntry, SlotVerification, DeterministicProof};
use dojo::utils::entity_id_from_keys;
use dojo::world::{
    IShardingSettlementDispatcher, IShardingSettlementDispatcherTrait,
};
use dojo_snf_test::declare_and_deploy;
use starknet::ContractAddress;

use crate::tests::helpers::{Foo, deploy_world_and_foo};

// ── Mock StorageCommitment Verifier ─────────────────────────────────

/// Always-approve verifier for unit tests. Deployed once per test setup
/// and registered via `set_storage_commitment_registry`.
#[starknet::contract]
pub mod mock_storage_commitment_verifier {
    use starknet::ContractAddress;

    #[storage]
    struct Storage {}

    #[abi(embed_v0)]
    impl MockVerifier of dojo::sharding::IStorageCommitmentVerifier<ContractState> {
        fn verify(
            ref self: ContractState,
            storage_commitment: felt252,
            contract_address: ContractAddress,
            global_state_root: felt252,
            end_block_number: u64,
        ) -> (bool, felt252, felt252) {
            (true, 0, 0)
        }
    }
}

/// Minimal sharding proxy for tests.
/// Captures forwarded request payload in an emitted event.
#[starknet::contract]
pub mod mock_sharding_proxy {
    #[starknet::interface]
    pub trait IProxyAbi<T> {
        fn notify_shard_requested(
            ref self: T,
            shard_id: felt252,
            entities: Span<felt252>,
            entity_keys_flat: Span<felt252>,
        );
        fn end_shard(ref self: T, shard_id: felt252);
    }

    #[event]
    #[derive(Drop, starknet::Event)]
    pub enum Event {
        ProxyNotified: ProxyNotified,
        ProxyEnded: ProxyEnded,
    }

    #[derive(Drop, starknet::Event)]
    pub struct ProxyNotified {
        #[key]
        pub shard_id: felt252,
        pub entities: Span<felt252>,
        pub entity_keys_flat: Span<felt252>,
    }

    #[derive(Drop, starknet::Event)]
    pub struct ProxyEnded {
        #[key]
        pub shard_id: felt252,
    }

    #[storage]
    struct Storage {}

    #[abi(embed_v0)]
    impl MockProxy of IProxyAbi<ContractState> {
        fn notify_shard_requested(
            ref self: ContractState,
            shard_id: felt252,
            entities: Span<felt252>,
            entity_keys_flat: Span<felt252>,
        ) {
            self.emit(ProxyNotified { shard_id, entities, entity_keys_flat });
        }

        fn end_shard(ref self: ContractState, shard_id: felt252) {
            self.emit(ProxyEnded { shard_id });
        }
    }
}

// ── Field selector helpers ──────────────────────────────────────────

pub fn foo_field_selectors() -> (felt252, felt252) {
    let layout = Model::<Foo>::layout();
    if let dojo::meta::Layout::Struct(fields) = layout {
        ((*fields[0]).selector, (*fields[1]).selector)
    } else {
        panic!("expected struct layout")
    }
}

// ── Setup helpers ─────────────────────────────────────────────────

/// Deploy mock verifier and register it on the world (world owner caller).
pub fn register_mock_storage_commitment_verifier(
    world_address: ContractAddress,
) {
    let mock_verifier = declare_and_deploy("mock_storage_commitment_verifier");
    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement.set_storage_commitment_registry(mock_verifier);
    snforge_std::stop_cheat_caller_address(world_address);
}

/// Deploy mock sharding proxy and register it on world.
pub fn register_mock_sharding_proxy(world_address: ContractAddress) -> ContractAddress {
    let proxy = declare_and_deploy("mock_sharding_proxy");
    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement.set_sharding_proxy(proxy);
    snforge_std::stop_cheat_caller_address(world_address);
    proxy
}

/// Deploy world + Foo + mock verifier, write initial data, compute slots.
/// Returns (world, world_address, entity_id, slot_a, slot_b, model_selector).
pub fn setup_foo_shard() -> (
    dojo::world::WorldStorage,
    ContractAddress,
    felt252,
    felt252,
    felt252,
    felt252,
) {
    snforge_std::start_cheat_account_contract_address_global(snforge_std::test_address());
    let (mut world, model_selector) = deploy_world_and_foo();
    let world_address = world.dispatcher.contract_address;
    register_mock_storage_commitment_verifier(world_address);

    let bob: ContractAddress = 0xb0b.try_into().unwrap();
    world.write_model_test(@Foo { caller: bob, a: 100, b: 200 });

    let (sel_a, sel_b) = foo_field_selectors();
    let entity_id = entity_id_from_keys(@bob);
    let slot_a = compute_dojo_field_slot(model_selector, entity_id, sel_a);
    let slot_b = compute_dojo_field_slot(model_selector, entity_id, sel_b);

    (world, world_address, entity_id, slot_a, slot_b, model_selector)
}

// ── SlotEntry builders ────────────────────────────────────────────

/// Build a SlotEntry for a struct-layout field (Deterministic, packed_offset=0).
pub fn make_field_slot(
    key: felt252,
    value: felt252,
    model_selector: felt252,
    entity_id: felt252,
    member_selector: felt252,
    initial_value: felt252,
) -> SlotEntry {
    SlotEntry {
        key,
        value,
        model_selector,
        entity_id,
        member_selector,
        initial_value,
        verification: SlotVerification::Deterministic(
            DeterministicProof {
                key_derivation_chain: [member_selector].span(),
                packed_offset: 0,
            },
        ),
    }
}

/// Settle via production `settle()` (StorageCommitment path). Requires a registered verifier
/// (e.g. `register_mock_storage_commitment_verifier`).
pub fn settle_as_owner(
    world_address: ContractAddress,
    shard_id: felt252,
    slots: Span<SlotEntry>,
) {
    settle_with_caller(world_address, snforge_std::test_address(), shard_id, slots);
}

/// Same as [`settle_as_owner`] but with an explicit caller (e.g. non-owner panic tests).
///
/// Automatically builds initial proof data from slots that have non-zero `initial_value`,
/// so Add CRDT tests work without manually constructing initial proofs.
pub fn settle_with_caller(
    world_address: ContractAddress,
    caller: ContractAddress,
    shard_id: felt252,
    slots: Span<SlotEntry>,
) {
    // Include all slots in the initial proof so Add CRDT verification works
    // even when initial_value is zero (valid fork-time state).
    let mut initial_keys: Array<felt252> = ArrayTrait::new();
    let mut initial_values: Array<felt252> = ArrayTrait::new();
    for entry in slots {
        initial_keys.append(*entry.key);
        initial_values.append(*entry.initial_value);
    };

    // Build InitialProof from collected Add CRDT slots.
    let initial_proof: InitialProof = if initial_keys.len() > 0 {
        let mut commitment_data: Array<felt252> = ArrayTrait::new();
        for k in initial_keys.span() {
            commitment_data.append(*k);
        };
        for v in initial_values.span() {
            commitment_data.append(*v);
        };
        InitialProof {
            keys: initial_keys.span(),
            values: initial_values.span(),
            commitment: poseidon_hash_span(commitment_data.span()),
            fork_state_root: 0x1,
        }
    } else {
        InitialProof {
            keys: [].span(),
            values: [].span(),
            commitment: 0,
            fork_state_root: 0,
        }
    };

    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, caller);
    settlement
        .settle(
            shard_id,
            0x1, // global_state_root (mock verifier accepts any)
            1, // end_block_number — must be non-zero for production settle
            slots,
            [].span(),
            [].span(),
            initial_proof,
        );
    snforge_std::stop_cheat_caller_address(world_address);
}
