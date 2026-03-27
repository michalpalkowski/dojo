use dojo::model::{Model, ModelStorageTest};
use dojo::sharding::compute_dojo_field_slot;
use dojo::sharding::request::{SlotEntry, SlotVerification, DeterministicProof};
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
pub fn register_mock_storage_commitment_verifier(world_address: ContractAddress) {
    let mock_verifier = declare_and_deploy("mock_storage_commitment_verifier");
    let settlement = IShardingSettlementDispatcher { contract_address: world_address };
    snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
    settlement.set_storage_commitment_registry(mock_verifier);
    snforge_std::stop_cheat_caller_address(world_address);
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
pub fn settle_with_caller(
    world_address: ContractAddress,
    caller: ContractAddress,
    shard_id: felt252,
    slots: Span<SlotEntry>,
) {
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
        );
    snforge_std::stop_cheat_caller_address(world_address);
}
