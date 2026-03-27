#[starknet::interface]
pub trait IShardingSystems<T> {
    fn register_policies(ref self: T);
    fn request_shard(ref self: T, player_ids: Span<felt252>);
    fn end_shard(ref self: T, shard_id: felt252);
}

#[dojo::contract]
pub mod sharding_systems {
    use dojo::model::Model;
    use dojo::sharding::request::{CRDVariant, ShardField};
    use dojo::world::{IWorldDispatcherTrait, WorldStorage};
    use dojo_examples::models::{MockToken, Moves, PlayerConfig, Position};

    fn register_policy(
        ref world: WorldStorage,
        ns_hash: felt252,
        model_selector: felt252,
        default_crdt: CRDVariant,
        field_overrides: Span<ShardField>,
    ) {
        world.dispatcher.register_shard_policy(model_selector, default_crdt, field_overrides);
    }

    #[abi(embed_v0)]
    impl ShardingSystemsImpl of super::IShardingSystems<ContractState> {
        fn register_policies(ref self: ContractState) {
            let mut world = self.world(@"ns");
            let ns_hash = dojo::utils::bytearray_hash(@"ns");

            // ── Entity-locked models → SetLock (exclusive overwrite) ──

            // Moves: player state during gameplay session
            register_policy(
                ref world,
                ns_hash,
                Model::<Moves>::selector(ns_hash),
                CRDVariant::SetLock,
                [].span(),
            );

            // Position: player coordinates, packed struct
            register_policy(
                ref world,
                ns_hash,
                Model::<Position>::selector(ns_hash),
                CRDVariant::SetLock,
                [].span(),
            );

            // MockToken: per-account balance
            register_policy(
                ref world,
                ns_hash,
                Model::<MockToken>::selector(ns_hash),
                CRDVariant::SetLock,
                [].span(),
            );

            // PlayerConfig: entity-locked, but `items` is a dynamic array
            register_policy(
                ref world,
                ns_hash,
                Model::<PlayerConfig>::selector(ns_hash),
                CRDVariant::SetLock,
                [
                    ShardField {
                        selector: selector!("items"),
                        crdt: CRDVariant::SetLock,
                        max_elements: 10,
                    },
                ]
                    .span(),
            );
        }

        fn request_shard(ref self: ContractState, player_ids: Span<felt252>) {
            let world = self.world(@"ns");
            let mut dojo_entities: Array<felt252> = ArrayTrait::new();
            let mut entity_keys_flat: Array<felt252> = ArrayTrait::new();

            for id in player_ids {
                dojo_entities
                    .append(dojo::utils::entity_id_from_serialized_keys([*id].span()));
                entity_keys_flat.append(1); // 1 key per entity
                entity_keys_flat.append(*id);
            };

            world
                .dispatcher
                .request_sharding(dojo_entities.span(), [].span(), entity_keys_flat.span());
        }

        fn end_shard(ref self: ContractState, shard_id: felt252) {
            let world = self.world(@"ns");
            world.dispatcher.end_shard(shard_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use dojo::model::{Model, ModelStorage, ModelStorageTest};
    use dojo::sharding::compute_dojo_field_slot;
    use dojo::sharding::slot::{compute_dojo_packed_slot, compute_dynamic_member_lock_slot};
    use dojo::sharding::request::{CRDVariant, ShardField, SlotEntry, SlotVerification, DeterministicProof};
    use dojo::utils::entity_id_from_keys;
    use dojo::world::{
        IShardingSettlementDispatcher, IShardingSettlementDispatcherTrait,
        IWorldDispatcherTrait, WorldStorageTrait,
    };
    use dojo_examples::actions::{IActionsDispatcher, IActionsDispatcherTrait};
    use dojo_examples::models::{Direction, Moves, PlayerConfig, PlayerItem, Position, Vec2};
    use dojo_snf_test::{
        ContractDef, ContractDefTrait, NamespaceDef, TestResource, WorldStorageTestTrait,
        declare_and_deploy, set_caller_address, spawn_test_world,
    };
    use starknet::ContractAddress;
    use super::{IShardingSystemsDispatcher, IShardingSystemsDispatcherTrait};

    // ── Mock StorageCommitment Verifier (always approves) ──────────────

    #[starknet::interface]
    trait IMockVerifierConfig<T> {
        fn set_shard_id(ref self: T, shard_id: felt252);
    }

    #[starknet::contract]
    mod mock_storage_commitment_verifier {
        use starknet::ContractAddress;
        use starknet::storage::{StoragePointerReadAccess, StoragePointerWriteAccess};

        #[storage]
        struct Storage {
            expected_shard_id: felt252,
        }

        #[abi(embed_v0)]
        impl MockVerifier of dojo::sharding::IStorageCommitmentVerifier<ContractState> {
            fn verify(
                ref self: ContractState,
                storage_commitment: felt252,
                contract_address: ContractAddress,
                global_state_root: felt252,
                end_block_number: u64,
            ) -> (bool, felt252, felt252) {
                (true, contract_address.into(), self.expected_shard_id.read())
            }
        }

        #[abi(embed_v0)]
        impl MockConfig of super::IMockVerifierConfig<ContractState> {
            fn set_shard_id(ref self: ContractState, shard_id: felt252) {
                self.expected_shard_id.write(shard_id);
            }
        }
    }

    // ── Namespace + contract definitions ───────────────────────────────

    fn namespace_def() -> NamespaceDef {
        NamespaceDef {
            namespace: "ns",
            resources: [
                TestResource::Model("Position"),
                TestResource::Model("Moves"),
                TestResource::Model("MockToken"),
                TestResource::Model("PlayerConfig"),
                TestResource::Event("Moved"),
                TestResource::Contract("actions"),
                TestResource::Contract("sharding_systems"),
                TestResource::Library(("simple_math", "0_1_0")),
            ]
                .span(),
        }
    }

    fn contract_defs() -> Span<ContractDef> {
        [
            ContractDefTrait::new(@"ns", @"actions")
                .with_writer_of([dojo::utils::bytearray_hash(@"ns")].span()),
            ContractDefTrait::new(@"ns", @"sharding_systems")
                .with_writer_of([0].span()),
        ]
            .span()
    }

    // ── Setup helpers ─────────────────────────────────────────────────

    fn setup_world() -> dojo::world::WorldStorage {
        snforge_std::start_cheat_account_contract_address_global(snforge_std::test_address());
        let ndef = namespace_def();
        let mut world = spawn_test_world([ndef].span());
        world.sync_perms_and_inits(contract_defs());
        world
    }

    fn register_mock_verifier(world_address: ContractAddress) -> ContractAddress {
        let mock_verifier = declare_and_deploy("mock_storage_commitment_verifier");
        let settlement = IShardingSettlementDispatcher { contract_address: world_address };
        snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
        settlement.set_storage_commitment_registry(mock_verifier);
        snforge_std::stop_cheat_caller_address(world_address);
        mock_verifier
    }

    fn moves_field_selectors() -> (felt252, felt252) {
        let layout = Model::<Moves>::layout();
        if let dojo::meta::Layout::Struct(fields) = layout {
            ((*fields[0]).selector, (*fields[1]).selector)
        } else {
            panic!("expected struct layout for Moves")
        }
    }

    fn make_field_slot(
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

    fn settle_as_owner(
        world_address: ContractAddress,
        mock_verifier: ContractAddress,
        shard_id: felt252,
        slots: Span<SlotEntry>,
    ) {
        // Configure mock verifier to return matching shard_id
        let mock_config = IMockVerifierConfigDispatcher { contract_address: mock_verifier };
        mock_config.set_shard_id(shard_id);

        // Build InitialProof from all settlement slots.
        // This keeps Add semantics correct even when initial_value == 0.
        // The mock verifier accepts any commitment, so we only need a consistent keys/values dict.
        let mut init_keys: Array<felt252> = ArrayTrait::new();
        let mut init_values: Array<felt252> = ArrayTrait::new();
        for entry in slots {
            init_keys.append(*entry.key);
            init_values.append(*entry.initial_value);
        };
        let has_initials = init_keys.len() > 0;
        let initial_commitment = if has_initials {
            // Compute commitment so the on-chain check passes (mock verifier accepts any).
            let mut data: Array<felt252> = ArrayTrait::new();
            for k in init_keys.span() {
                data.append(*k);
            };
            for v in init_values.span() {
                data.append(*v);
            };
            core::poseidon::poseidon_hash_span(data.span())
        } else {
            0
        };

        let settlement = IShardingSettlementDispatcher { contract_address: world_address };
        snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
        settlement
            .settle(
                shard_id,
                0x1, // global_state_root (mock verifier accepts any)
                1, // end_block_number (must be non-zero)
                slots,
                [].span(), // entity_model_selectors (for Torii events, empty OK in test)
                [].span(), // entity_keys_flat
                dojo::sharding::request::InitialProof {
                    keys: init_keys.span(),
                    values: init_values.span(),
                    commitment: initial_commitment,
                    fork_state_root: if has_initials { 0x2 } else { 0 },
                },
            );
        snforge_std::stop_cheat_caller_address(world_address);
    }

    // ══════════════════════════════════════════════════════════════════
    //  HAPPY PATH: basic lifecycle
    // ══════════════════════════════════════════════════════════════════

    #[test]

    fn test_register_policies_and_request_shard() {
        let mut world = setup_world();
        let caller = dojo_snf_test::get_default_caller_address();

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        sharding.register_policies();
        actions.spawn();

        let position: Position = world.read_model(caller);
        assert(position.vec.x == 10 && position.vec.y == 10, 'initial position');
        let moves: Moves = world.read_model(caller);
        assert(moves.remaining == 99, 'initial moves');

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());
        sharding.end_shard(1);
    }

    #[test]

    fn test_active_shards_tracked() {
        let world = setup_world();
        let caller = dojo_snf_test::get_default_caller_address();

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        sharding.register_policies();
        actions.spawn();

        let settlement = IShardingSettlementDispatcher {
            contract_address: world.dispatcher.contract_address,
        };

        let active = settlement.get_active_shards();
        assert(active.len() == 0, 'no shards initially');

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());
        let active = settlement.get_active_shards();
        assert(active.len() == 1, 'one shard active');
    }

    // ══════════════════════════════════════════════════════════════════
    //  ENTITY LOCKING: all models for entity are blocked
    // ══════════════════════════════════════════════════════════════════

    #[test]

    #[should_panic(expected: ('Shard: entity locked',))]
    fn test_move_blocked_while_sharded() {
        let mut world = setup_world();
        let caller = dojo_snf_test::get_default_caller_address();

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        sharding.register_policies();
        actions.spawn();

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());

        // move() writes to both Moves and Position — should panic
        actions.move(Direction::Right);
    }

    #[test]

    #[should_panic(expected: ('Shard: entity locked',))]
    fn test_write_model_blocked_while_sharded() {
        let mut world = setup_world();
        let caller = dojo_snf_test::get_default_caller_address();

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        sharding.register_policies();
        actions.spawn();

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());

        // Direct model write should also be blocked
        world.write_model_test(@Moves { player: caller, remaining: 50, last_direction: Direction::None });
    }

    #[test]

    fn test_other_player_not_affected_by_shard() {
        let mut world = setup_world();
        let alice = dojo_snf_test::get_default_caller_address();
        let bob: ContractAddress = 0xb0b.try_into().unwrap();

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        // Spawn Alice and write initial state for Bob
        set_caller_address(alice);
        sharding.register_policies();
        actions.spawn();
        world.write_model_test(
            @Moves { player: bob, remaining: 50, last_direction: Direction::None },
        );

        // Shard only Alice
        let alice_felt: felt252 = alice.into();
        sharding.request_shard([alice_felt].span());

        // Bob's entities are NOT locked — write should succeed
        world.write_model_test(
            @Moves { player: bob, remaining: 42, last_direction: Direction::Right },
        );
        let bob_moves: Moves = world.read_model(bob);
        assert(bob_moves.remaining == 42, 'bob write ok');
    }

    // ══════════════════════════════════════════════════════════════════
    //  FULL SETTLEMENT: shard → lock → settle → unlock + verify values
    // ══════════════════════════════════════════════════════════════════

    #[test]

    fn test_full_settlement_set_crdt() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let caller = dojo_snf_test::get_default_caller_address();
        let mock_verifier = register_mock_verifier(world_address);

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        sharding.register_policies();
        actions.spawn();

        // Record initial state
        let initial_moves: Moves = world.read_model(caller);
        assert(initial_moves.remaining == 99, 'pre: 99 moves');

        // Request shard (locks entity)
        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());

        // --- Simulate shard gameplay result ---
        // On the shard Katana, the player used 10 moves. Remaining: 89.
        // Settlement carries this value back to mainnet via Set CRDT.

        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let moves_selector = Model::<Moves>::selector(ns_hash);
        let entity_id = entity_id_from_keys(@caller);
        let (sel_remaining, _sel_direction) = moves_field_selectors();
        let slot_remaining = compute_dojo_field_slot(moves_selector, entity_id, sel_remaining);

        // Settle: write remaining=89 via Set CRDT
        settle_as_owner(
            world_address,
            mock_verifier,
            1,
            [
                make_field_slot(slot_remaining, 89, moves_selector, entity_id, sel_remaining, 0),
            ]
                .span(),
        );

        // Verify: entity unlocked and value updated
        let settled_moves: Moves = world.read_model(caller);
        assert(settled_moves.remaining == 89, 'post: 89 moves');

        // Entity should be unlocked — can write again
        world.write_model_test(
            @Moves { player: caller, remaining: 50, last_direction: Direction::None },
        );
        let after: Moves = world.read_model(caller);
        assert(after.remaining == 50, 'unlocked: write ok');
    }

    #[test]

    fn test_settlement_partial_only_changed_fields() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let caller = dojo_snf_test::get_default_caller_address();
        let mock_verifier = register_mock_verifier(world_address);

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        sharding.register_policies();
        actions.spawn();
        // Move right first so last_direction = Right
        actions.move(Direction::Right);
        let pre: Moves = world.read_model(caller);
        assert(pre.remaining == 98, 'pre: 98 moves');

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());

        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let moves_selector = Model::<Moves>::selector(ns_hash);
        let entity_id = entity_id_from_keys(@caller);
        let (sel_remaining, _) = moves_field_selectors();
        let slot_remaining = compute_dojo_field_slot(moves_selector, entity_id, sel_remaining);

        // Settle only `remaining` — `last_direction` untouched
        settle_as_owner(
            world_address,
            mock_verifier,
            1,
            [
                make_field_slot(slot_remaining, 50, moves_selector, entity_id, sel_remaining, 0),
            ]
                .span(),
        );

        let after: Moves = world.read_model(caller);
        assert(after.remaining == 50, 'remaining settled');
        // last_direction should be unchanged (Right from the pre-shard move)
        let right: felt252 = Direction::Right.into();
        assert(after.last_direction.into() == right, 'direction unchanged');
    }

    #[test]
    fn test_entity_locked_before_cancel_then_unlocked_after() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let caller = dojo_snf_test::get_default_caller_address();

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        sharding.register_policies();
        actions.spawn();

        let player_felt: felt252 = caller.into();
        let entity_id = dojo::utils::entity_id_from_serialized_keys([player_felt].span());
        sharding.request_shard([player_felt].span());

        // Verify entity IS locked (shard_id = 1)
        let settlement = IShardingSettlementDispatcher { contract_address: world_address };
        assert(settlement.get_entity_shard(entity_id) == 1, 'locked: shard 1');

        // Cancel
        snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
        settlement.cancel_shard(1);
        snforge_std::stop_cheat_caller_address(world_address);

        // Verify entity is UNLOCKED (shard_id = 0)
        assert(settlement.get_entity_shard(entity_id) == 0, 'unlocked: shard 0');

        // Original values preserved
        let moves: Moves = world.read_model(caller);
        assert(moves.remaining == 99, 'cancel: original');

        // Can write again
        set_caller_address(caller);
        actions.move(Direction::Left);
        let after: Moves = world.read_model(caller);
        assert(after.remaining == 98, 'cancel: can move');
    }

    // ══════════════════════════════════════════════════════════════════
    //  ADD CRDT: delta merge
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_add_crdt_computes_delta() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let caller = dojo_snf_test::get_default_caller_address();
        let mock_verifier = register_mock_verifier(world_address);

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        actions.spawn(); // remaining = 99

        // Override: register Moves.remaining as Add CRDT (instead of default Set)
        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let moves_selector = Model::<Moves>::selector(ns_hash);
        let (sel_remaining, _) = moves_field_selectors();
        world
            .dispatcher
            .register_shard_policy(
                moves_selector,
                CRDVariant::SetLock,
                [ShardField { selector: sel_remaining, crdt: CRDVariant::Add, max_elements: 0 }]
                    .span(),
            );

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());

        let entity_id = entity_id_from_keys(@caller);
        let slot_remaining = compute_dojo_field_slot(moves_selector, entity_id, sel_remaining);

        // Shard gameplay: remaining went from 99 to 109 (gained 10 moves)
        // Delta = 109 - 99 = 10
        // Expected on-chain: current(99) + delta(10) = 109
        settle_as_owner(
            world_address,
            mock_verifier,
            1,
            [make_field_slot(slot_remaining, 109, moves_selector, entity_id, sel_remaining, 99)]
                .span(),
        );

        let result: Moves = world.read_model(caller);
        assert(result.remaining == 109, 'Add: 99+(109-99)=109');
    }

    #[test]
    fn test_mixed_set_and_add_on_same_model() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let caller = dojo_snf_test::get_default_caller_address();
        let mock_verifier = register_mock_verifier(world_address);

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        actions.spawn(); // remaining=99
        // Move right so direction=Right before sharding
        actions.move(Direction::Right); // remaining=98

        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let moves_selector = Model::<Moves>::selector(ns_hash);
        let (sel_remaining, _) = moves_field_selectors();

        // Override: remaining=Add, rest stays Set (default)
        world
            .dispatcher
            .register_shard_policy(
                moves_selector,
                CRDVariant::SetLock,
                [ShardField { selector: sel_remaining, crdt: CRDVariant::Add, max_elements: 0 }]
                    .span(),
            );

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());

        let entity_id = entity_id_from_keys(@caller);
        let slot_remaining = compute_dojo_field_slot(moves_selector, entity_id, sel_remaining);

        // Settle only remaining via Add: 98→108, delta=10
        // Direction field not settled → stays Right (unchanged)
        settle_as_owner(
            world_address,
            mock_verifier,
            1,
            [
                make_field_slot(slot_remaining, 108, moves_selector, entity_id, sel_remaining, 98),
            ]
                .span(),
        );

        let result: Moves = world.read_model(caller);
        assert(result.remaining == 108, 'Add: 98+(108-98)=108');
        let right: felt252 = Direction::Right.into();
        assert(result.last_direction.into() == right, 'direction unchanged');
    }

    #[test]
    #[should_panic(expected: ('Shard: add delta underflow',))]
    fn test_add_crdt_underflow_rejected() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let caller = dojo_snf_test::get_default_caller_address();
        let mock_verifier = register_mock_verifier(world_address);

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        actions.spawn(); // remaining = 99

        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let moves_selector = Model::<Moves>::selector(ns_hash);
        let (sel_remaining, _) = moves_field_selectors();
        world
            .dispatcher
            .register_shard_policy(
                moves_selector,
                CRDVariant::SetLock,
                [ShardField { selector: sel_remaining, crdt: CRDVariant::Add, max_elements: 0 }]
                    .span(),
            );

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());

        let entity_id = entity_id_from_keys(@caller);
        let slot_remaining = compute_dojo_field_slot(moves_selector, entity_id, sel_remaining);

        // shard_value(50) < initial(99) → negative delta → underflow panic
        settle_as_owner(
            world_address,
            mock_verifier,
            1,
            [make_field_slot(slot_remaining, 50, moves_selector, entity_id, sel_remaining, 99)]
                .span(),
        );
    }

    // ══════════════════════════════════════════════════════════════════
    //  NEGATIVE SCENARIOS
    // ══════════════════════════════════════════════════════════════════

    #[test]

    #[should_panic(expected: ('Shard: entity already sharded',))]
    fn test_double_shard_same_entity_rejected() {
        let world = setup_world();
        let caller = dojo_snf_test::get_default_caller_address();

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        sharding.register_policies();
        actions.spawn();

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());
        // Second shard for same entity → panic
        sharding.request_shard([player_felt].span());
    }

    #[test]

    #[should_panic(expected: ('Shard: no entities',))]
    fn test_request_shard_empty_entities_rejected() {
        let world = setup_world();
        let caller = dojo_snf_test::get_default_caller_address();

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };

        set_caller_address(caller);
        sharding.register_policies();
        // Empty entity list → panic
        sharding.request_shard([].span());
    }

    #[test]

    #[should_panic(expected: ('Shard: slot ownership mismatch',))]
    fn test_settle_wrong_slot_key_rejected() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let caller = dojo_snf_test::get_default_caller_address();
        let mock_verifier = register_mock_verifier(world_address);

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        sharding.register_policies();
        actions.spawn();

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());

        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let moves_selector = Model::<Moves>::selector(ns_hash);
        let entity_id = entity_id_from_keys(@caller);
        let (sel_remaining, _) = moves_field_selectors();

        // Fabricate a wrong slot key (doesn't match derivation chain)
        let fake_key: felt252 = 0xdeadbeef;

        settle_as_owner(
            world_address,
            mock_verifier,
            1,
            [
                make_field_slot(fake_key, 50, moves_selector, entity_id, sel_remaining, 0),
            ]
                .span(),
        );
    }

    #[test]

    #[should_panic(expected: ('Shard: entity not in shard',))]
    fn test_settle_foreign_entity_rejected() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let caller = dojo_snf_test::get_default_caller_address();
        let eve: ContractAddress = 0xe0e.try_into().unwrap();
        let mock_verifier = register_mock_verifier(world_address);

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        sharding.register_policies();
        actions.spawn();
        // Write some data for Eve (not sharded)
        world.write_model_test(@Moves { player: eve, remaining: 10, last_direction: Direction::None });

        // Shard only the caller
        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());

        // Try to settle a slot belonging to Eve's entity — not in shard
        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let moves_selector = Model::<Moves>::selector(ns_hash);
        let eve_entity_id = entity_id_from_keys(@eve);
        let (sel_remaining, _) = moves_field_selectors();
        let eve_slot = compute_dojo_field_slot(moves_selector, eve_entity_id, sel_remaining);

        settle_as_owner(
            world_address,
            mock_verifier,
            1,
            [
                make_field_slot(
                    eve_slot, 999, moves_selector, eve_entity_id, sel_remaining, 0,
                ),
            ]
                .span(),
        );
    }

    // ══════════════════════════════════════════════════════════════════
    //  CONCURRENT SHARDS: two separate shards coexist and settle
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_two_shards_coexist_and_settle_independently() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let alice = dojo_snf_test::get_default_caller_address();
        let bob: ContractAddress = 0xb0b.try_into().unwrap();
        let mock_verifier = register_mock_verifier(world_address);

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        // Spawn Alice (remaining=99) and Bob (remaining=50)
        set_caller_address(alice);
        sharding.register_policies();
        actions.spawn();
        world.write_model_test(
            @Moves { player: bob, remaining: 50, last_direction: Direction::None },
        );

        // Shard Alice → shard_id=1
        let alice_felt: felt252 = alice.into();
        sharding.request_shard([alice_felt].span());

        // Shard Bob → shard_id=2  (different entity, coexists with shard 1)
        let bob_felt: felt252 = bob.into();
        sharding.request_shard([bob_felt].span());

        // Both active
        let settlement = IShardingSettlementDispatcher { contract_address: world_address };
        let active = settlement.get_active_shards();
        assert(active.len() == 2, 'two shards active');

        // Settle shard 2 (Bob) first — order doesn't matter
        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let moves_selector = Model::<Moves>::selector(ns_hash);
        let bob_entity = entity_id_from_keys(@bob);
        let (sel_remaining, _) = moves_field_selectors();
        let bob_slot = compute_dojo_field_slot(moves_selector, bob_entity, sel_remaining);

        settle_as_owner(
            world_address,
            mock_verifier,
            2, // shard_id=2
            [make_field_slot(bob_slot, 30, moves_selector, bob_entity, sel_remaining, 0)].span(),
        );

        // Bob settled, Alice still locked
        let bob_result: Moves = world.read_model(bob);
        assert(bob_result.remaining == 30, 'bob settled to 30');
        assert(settlement.get_entity_shard(entity_id_from_keys(@alice)) == 1, 'alice still locked');
        assert(settlement.get_entity_shard(bob_entity) == 0, 'bob unlocked');

        // Now settle shard 1 (Alice)
        let alice_entity = entity_id_from_keys(@alice);
        let alice_slot = compute_dojo_field_slot(moves_selector, alice_entity, sel_remaining);

        settle_as_owner(
            world_address,
            mock_verifier,
            1, // shard_id=1
            [make_field_slot(alice_slot, 80, moves_selector, alice_entity, sel_remaining, 0)]
                .span(),
        );

        let alice_result: Moves = world.read_model(alice);
        assert(alice_result.remaining == 80, 'alice settled to 80');
        assert(settlement.get_entity_shard(alice_entity) == 0, 'alice unlocked');

        // Both unlocked — no active shards
        let active = settlement.get_active_shards();
        assert(active.len() == 0, 'no shards left');
    }

    /// BUG/LIMITATION: entity_lock is exclusive regardless of CRDT type.
    /// Even with Add CRDT (designed for concurrent delta merge on global models),
    /// the same entity cannot be in two shards simultaneously.
    /// This means Add on a locked entity degrades to Set behavior:
    ///   new = current + (shard - initial) = initial + (shard - initial) = shard
    /// because no mainnet writes can happen while entity is locked.
    /// TODO: verify_slot_ownership should exempt Add entities from entity_lock
    /// to enable true concurrent global counter access across shards.
    #[test]
    #[should_panic(expected: ('Shard: entity already sharded',))]
    fn test_add_entity_still_exclusively_locked() {
        let mut world = setup_world();
        let caller = dojo_snf_test::get_default_caller_address();

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        actions.spawn();

        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let moves_selector = Model::<Moves>::selector(ns_hash);
        let (sel_remaining, _) = moves_field_selectors();
        world
            .dispatcher
            .register_shard_policy(
                moves_selector,
                CRDVariant::SetLock,
                [ShardField { selector: sel_remaining, crdt: CRDVariant::Add, max_elements: 0 }]
                    .span(),
            );

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());
        // Panics — entity_lock is exclusive even for Add CRDT entities.
        sharding.request_shard([player_felt].span());
    }

    /// CRDT policies cannot be changed while any shard is active.
    /// register_shard_policy asserts active_shard_count == 0.
    #[test]
    #[should_panic(expected: ('Shard: active shards exist',))]
    fn test_cannot_change_crdt_policy_during_active_shard() {
        let mut world = setup_world();
        let caller = dojo_snf_test::get_default_caller_address();

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        sharding.register_policies(); // Set policies initially
        actions.spawn();

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span()); // now active_shard_count > 0

        // Try to change CRDT policy while shard is active → panics
        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let moves_selector = Model::<Moves>::selector(ns_hash);
        world
            .dispatcher
            .register_shard_policy(moves_selector, CRDVariant::Add, [].span());
    }

    // ══════════════════════════════════════════════════════════════════
    //  LAYOUT COVERAGE: packed models, dynamic arrays
    // ══════════════════════════════════════════════════════════════════

    /// Position uses IntrospectPacked → Fixed layout.
    /// Slot = Poseidon(DOJO_STORAGE, model_selector, entity_id) + packed_offset.
    /// Verification: empty key_derivation_chain, member_selector=0.
    #[test]
    fn test_packed_model_settlement() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let caller = dojo_snf_test::get_default_caller_address();
        let mock_verifier = register_mock_verifier(world_address);

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        sharding.register_policies();
        actions.spawn(); // Position { vec: Vec2 { x: 10, y: 10 } }

        let pre: Position = world.read_model(caller);
        assert(pre.vec.x == 10 && pre.vec.y == 10, 'pre: (10,10)');

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());

        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let position_selector = Model::<Position>::selector(ns_hash);
        let entity_id = entity_id_from_keys(@caller);

        // Packed slot: Poseidon(DOJO_STORAGE, model, entity_id) + 0
        let packed_slot = compute_dojo_packed_slot(position_selector, entity_id);

        // Vec2 { x: 20, y: 30 } packed as: x (low 32 bits) | y << 32 (high 32 bits)
        // Dojo packs fields left-to-right into ascending bit positions.
        let packed_value: felt252 = (20_u64 + 30_u64 * 0x100000000_u64).into();

        // Packed models: empty key_derivation_chain, member_selector=0, packed_offset=0
        let slot_entry = SlotEntry {
            key: packed_slot,
            value: packed_value,
            model_selector: position_selector,
            entity_id,
            member_selector: 0,
            initial_value: 0,
            verification: SlotVerification::Deterministic(
                DeterministicProof {
                    key_derivation_chain: [].span(),
                    packed_offset: 0,
                },
            ),
        };

        settle_as_owner(world_address, mock_verifier, 1, [slot_entry].span());

        let post: Position = world.read_model(caller);
        assert(post.vec.x == 20, 'packed: x=20');
        assert(post.vec.y == 30, 'packed: y=30');
    }

    /// PlayerConfig.items is Array<PlayerItem> registered as SetLock with max_elements=10.
    /// Dynamic arrays use SlotVerification::DynamicLock with domain 'dojo_dynamic_member_lock'.
    #[test]
    fn test_dynamic_array_lock_slot_settlement() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let caller = dojo_snf_test::get_default_caller_address();
        let mock_verifier = register_mock_verifier(world_address);

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };

        set_caller_address(caller);
        sharding.register_policies(); // registers items with SetLock, max_elements=10

        // Write initial PlayerConfig with items
        world
            .write_model_test(
                @PlayerConfig {
                    player: caller,
                    name: "alice",
                    items: array![PlayerItem { item_id: 1, quantity: 5, score: 10 }],
                    favorite_item: Option::Some(1),
                },
            );

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());

        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let config_selector = Model::<PlayerConfig>::selector(ns_hash);
        let entity_id = entity_id_from_keys(@caller);
        let items_selector = selector!("items");

        // DynamicLock slot: Poseidon('dojo_dynamic_member_lock', model, entity, member)
        let dynamic_slot = compute_dynamic_member_lock_slot(
            config_selector, entity_id, items_selector,
        );

        // DynamicLock verification — writes value to the lock tracking slot
        let slot_entry = SlotEntry {
            key: dynamic_slot,
            value: 42, // arbitrary value written to the dynamic lock slot
            model_selector: config_selector,
            entity_id,
            member_selector: items_selector,
            initial_value: 0,
            verification: SlotVerification::DynamicLock,
        };

        // Settlement accepts DynamicLock slots for SetLock-registered fields
        settle_as_owner(world_address, mock_verifier, 1, [slot_entry].span());

        // Entity unlocked after settlement
        let settlement = IShardingSettlementDispatcher { contract_address: world_address };
        assert(settlement.get_entity_shard(entity_id) == 0, 'unlocked after settle');
    }

    /// Packed model with wrong packed_offset should fail slot ownership check.
    #[test]
    #[should_panic(expected: ('Shard: slot ownership mismatch',))]
    fn test_packed_wrong_offset_rejected() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let caller = dojo_snf_test::get_default_caller_address();
        let mock_verifier = register_mock_verifier(world_address);

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        sharding.register_policies();
        actions.spawn();

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());

        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let position_selector = Model::<Position>::selector(ns_hash);
        let entity_id = entity_id_from_keys(@caller);

        // Correct base slot but WRONG packed_offset (99 instead of 0)
        // The recomputed key won't match → slot ownership mismatch
        let packed_slot = compute_dojo_packed_slot(position_selector, entity_id);
        let slot_entry = SlotEntry {
            key: packed_slot, // key was computed with offset=0
            value: 0,
            model_selector: position_selector,
            entity_id,
            member_selector: 0,
            initial_value: 0,
            verification: SlotVerification::Deterministic(
                DeterministicProof {
                    key_derivation_chain: [].span(),
                    packed_offset: 99, // WRONG — expected_key = base + 99 ≠ base + 0
                },
            ),
        };

        settle_as_owner(world_address, mock_verifier, 1, [slot_entry].span());
    }

    // ══════════════════════════════════════════════════════════════════
    //  EDGE CASES: multi-entity, empty settle, double settle, end_shard
    // ══════════════════════════════════════════════════════════════════

    /// Multiple entities locked in a single shard — the realistic "battle" scenario.
    /// Both entities are locked, both must be settled to unlock.
    #[test]
    fn test_multi_entity_shard_locks_and_settles_all() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let alice = dojo_snf_test::get_default_caller_address();
        let bob: ContractAddress = 0xb0b.try_into().unwrap();
        let mock_verifier = register_mock_verifier(world_address);

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(alice);
        sharding.register_policies();
        actions.spawn(); // Alice: remaining=99
        world.write_model_test(
            @Moves { player: bob, remaining: 50, last_direction: Direction::None },
        );

        // Lock BOTH entities in one shard
        let alice_felt: felt252 = alice.into();
        let bob_felt: felt252 = bob.into();
        sharding.request_shard([alice_felt, bob_felt].span());

        let settlement = IShardingSettlementDispatcher { contract_address: world_address };
        let alice_eid = entity_id_from_keys(@alice);
        let bob_eid = entity_id_from_keys(@bob);

        // Both locked in shard 1
        assert(settlement.get_entity_shard(alice_eid) == 1, 'alice locked');
        assert(settlement.get_entity_shard(bob_eid) == 1, 'bob locked');

        // Both writes blocked
        // (can't use should_panic for two checks, so verify via get_entity_shard)

        // Settle both entities
        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let moves_sel = Model::<Moves>::selector(ns_hash);
        let (sel_remaining, _) = moves_field_selectors();
        let alice_slot = compute_dojo_field_slot(moves_sel, alice_eid, sel_remaining);
        let bob_slot = compute_dojo_field_slot(moves_sel, bob_eid, sel_remaining);

        settle_as_owner(
            world_address,
            mock_verifier,
            1,
            [
                make_field_slot(alice_slot, 80, moves_sel, alice_eid, sel_remaining, 0),
                make_field_slot(bob_slot, 30, moves_sel, bob_eid, sel_remaining, 0),
            ]
                .span(),
        );

        // Both unlocked and values updated
        assert(settlement.get_entity_shard(alice_eid) == 0, 'alice unlocked');
        assert(settlement.get_entity_shard(bob_eid) == 0, 'bob unlocked');
        let alice_moves: Moves = world.read_model(alice);
        let bob_moves: Moves = world.read_model(bob);
        assert(alice_moves.remaining == 80, 'alice settled 80');
        assert(bob_moves.remaining == 30, 'bob settled 30');
    }

    /// Settlement with zero slots still unlocks entities.
    /// Scenario: shard gameplay resulted in no state changes.
    #[test]
    fn test_empty_settlement_unlocks_entities() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let caller = dojo_snf_test::get_default_caller_address();
        let mock_verifier = register_mock_verifier(world_address);

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        sharding.register_policies();
        actions.spawn();

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());

        // Settle with ZERO slots — no state changes
        settle_as_owner(world_address, mock_verifier, 1, [].span());

        // Entity should still be unlocked
        let settlement = IShardingSettlementDispatcher { contract_address: world_address };
        let entity_id = entity_id_from_keys(@caller);
        assert(settlement.get_entity_shard(entity_id) == 0, 'unlocked after empty');

        // Original values preserved
        let moves: Moves = world.read_model(caller);
        assert(moves.remaining == 99, 'values unchanged');
    }

    /// Double-settling the same shard should fail — shard state is cleared after first settle.
    #[test]
    #[should_panic(expected: ('Shard: not found',))]
    fn test_double_settle_same_shard_rejected() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let caller = dojo_snf_test::get_default_caller_address();
        let mock_verifier = register_mock_verifier(world_address);

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        sharding.register_policies();
        actions.spawn();

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());

        // First settle succeeds
        settle_as_owner(world_address, mock_verifier, 1, [].span());

        // Second settle on same shard_id — shard state already cleared
        settle_as_owner(world_address, mock_verifier, 1, [].span());
    }

    /// end_shard does NOT unlock entities — they stay locked until settle or cancel.
    /// end_shard only emits ShardFinished event to signal the operator.
    #[test]
    #[should_panic(expected: ('Shard: entity locked',))]
    fn test_end_shard_does_not_unlock() {
        let mut world = setup_world();
        let caller = dojo_snf_test::get_default_caller_address();

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        sharding.register_policies();
        actions.spawn();

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());

        // end_shard signals the operator but does NOT unlock
        sharding.end_shard(1);

        // Entity is STILL locked — write panics
        world.write_model_test(
            @Moves { player: caller, remaining: 50, last_direction: Direction::None },
        );
    }

    /// Read always works — even for locked entities.
    #[test]
    fn test_read_works_during_shard() {
        let mut world = setup_world();
        let caller = dojo_snf_test::get_default_caller_address();

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        sharding.register_policies();
        actions.spawn();

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());

        // Read should work fine — only writes are blocked
        let moves: Moves = world.read_model(caller);
        assert(moves.remaining == 99, 'read works');
        let pos: Position = world.read_model(caller);
        assert(pos.vec.x == 10, 'read pos works');
    }
}
