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

            // ── Entity-locked models → Set (direct overwrite) ──

            // Moves: player state during gameplay session
            register_policy(
                ref world,
                ns_hash,
                Model::<Moves>::selector(ns_hash),
                CRDVariant::Set,
                [].span(),
            );

            // Position: player coordinates, packed struct
            register_policy(
                ref world,
                ns_hash,
                Model::<Position>::selector(ns_hash),
                CRDVariant::Set,
                [].span(),
            );

            // MockToken: per-account balance
            register_policy(
                ref world,
                ns_hash,
                Model::<MockToken>::selector(ns_hash),
                CRDVariant::Set,
                [].span(),
            );

            // PlayerConfig: entity-locked, but `items` is a dynamic array
            register_policy(
                ref world,
                ns_hash,
                Model::<PlayerConfig>::selector(ns_hash),
                CRDVariant::Set,
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
                .request_sharding(dojo_entities.span(), entity_keys_flat.span());
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
    use dojo::sharding::request::{SlotEntry, SlotVerification, DeterministicProof};
    use dojo::utils::entity_id_from_keys;
    use dojo::sharding::request::{CRDVariant, ShardField};
    use dojo::world::{
        IShardingSettlementDispatcher, IShardingSettlementDispatcherTrait, WorldStorageTrait,
    };
    use dojo_examples::actions::{IActionsDispatcher, IActionsDispatcherTrait};
    use dojo_examples::models::{Direction, Moves, Position};
    use dojo_snf_test::{
        ContractDef, ContractDefTrait, NamespaceDef, TestResource, WorldStorageTestTrait,
        declare_and_deploy, set_caller_address, spawn_test_world,
    };
    use starknet::ContractAddress;
    use super::{IShardingSystemsDispatcher, IShardingSystemsDispatcherTrait};

    // ── Mock StorageCommitment Verifier (always approves) ──────────────

    #[starknet::contract]
    mod mock_storage_commitment_verifier {
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

    fn register_mock_verifier(world_address: ContractAddress) {
        let mock_verifier = declare_and_deploy("mock_storage_commitment_verifier");
        let settlement = IShardingSettlementDispatcher { contract_address: world_address };
        snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
        settlement.set_storage_commitment_registry(mock_verifier);
        snforge_std::stop_cheat_caller_address(world_address);
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
        shard_id: felt252,
        slots: Span<SlotEntry>,
    ) {
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
        register_mock_verifier(world_address);

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
        register_mock_verifier(world_address);

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
        register_mock_verifier(world_address);

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
                CRDVariant::Set,
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
            1,
            [make_field_slot(slot_remaining, 109, moves_selector, entity_id, sel_remaining, 99)]
                .span(),
        );

        let result: Moves = world.read_model(caller);
        assert(result.remaining == 109, 'Add: 99+(109-99)=109');
    }

    #[test]
    fn test_mixed_set_and_add_settlement() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let caller = dojo_snf_test::get_default_caller_address();
        register_mock_verifier(world_address);

        let (sharding_addr, _) = world.dns(@"sharding_systems").unwrap();
        let sharding = IShardingSystemsDispatcher { contract_address: sharding_addr };
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        actions.spawn(); // remaining=99

        // Register MockToken as Add CRDT
        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let moves_selector = Model::<Moves>::selector(ns_hash);
        let (sel_remaining, sel_direction) = moves_field_selectors();

        // remaining=Add (counter), last_direction=Set (overwrite)
        world
            .dispatcher
            .register_shard_policy(
                moves_selector,
                CRDVariant::Set,
                [ShardField { selector: sel_remaining, crdt: CRDVariant::Add, max_elements: 0 }]
                    .span(),
            );

        let player_felt: felt252 = caller.into();
        sharding.request_shard([player_felt].span());

        let entity_id = entity_id_from_keys(@caller);
        let slot_remaining = compute_dojo_field_slot(moves_selector, entity_id, sel_remaining);
        let slot_direction = compute_dojo_field_slot(moves_selector, entity_id, sel_direction);

        // Settle: remaining via Add (99→109, delta=10), direction via Set (overwrite to Up=3)
        settle_as_owner(
            world_address,
            1,
            [
                make_field_slot(slot_remaining, 109, moves_selector, entity_id, sel_remaining, 99),
                make_field_slot(slot_direction, 3, moves_selector, entity_id, sel_direction, 0),
            ]
                .span(),
        );

        let result: Moves = world.read_model(caller);
        assert(result.remaining == 109, 'Add: 99+(109-99)=109');
        assert(result.last_direction.into() == 3, 'Set: overwrite to Up');
    }

    #[test]
    #[should_panic(expected: ('Shard: add delta underflow',))]
    fn test_add_crdt_underflow_rejected() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let caller = dojo_snf_test::get_default_caller_address();
        register_mock_verifier(world_address);

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
                CRDVariant::Set,
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
        register_mock_verifier(world_address);

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
        register_mock_verifier(world_address);

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
            1,
            [
                make_field_slot(
                    eve_slot, 999, moves_selector, eve_entity_id, sel_remaining, 0,
                ),
            ]
                .span(),
        );
    }
}
