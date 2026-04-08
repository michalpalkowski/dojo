#[dojo::contract]
pub mod sharding_systems {
    use dojo::model::Model;
    use dojo::sharding::request::{CRDVariant, ShardField};
    use dojo::sharding::IShardingGame;
    use dojo::world::WorldStorage;
    use dojo::world::world_sharding::{
        IShardingSettlementDispatcher, IShardingSettlementDispatcherTrait,
    };
    use dojo_examples::models::{MockToken, Moves, PlayerConfig, Position};

    fn sharding_disp(world: @WorldStorage) -> IShardingSettlementDispatcher {
        IShardingSettlementDispatcher { contract_address: *world.dispatcher.contract_address }
    }

    fn register_policy(
        ref world: WorldStorage,
        ns_hash: felt252,
        model_selector: felt252,
        default_crdt: CRDVariant,
        field_overrides: Span<ShardField>,
    ) {
        sharding_disp(@world).register_shard_policy(model_selector, default_crdt, field_overrides);
    }

    #[abi(embed_v0)]
    impl ShardingGameImpl of IShardingGame<ContractState> {
        fn register_policies(ref self: ContractState) {
            let mut world = self.world(@"ns");
            let ns_hash = dojo::utils::bytearray_hash(@"ns");

            // ── Exclusive player state → SetLock ──

            register_policy(
                ref world,
                ns_hash,
                Model::<Moves>::selector(ns_hash),
                CRDVariant::SetLock,
                [].span(),
            );

            register_policy(
                ref world,
                ns_hash,
                Model::<Position>::selector(ns_hash),
                CRDVariant::SetLock,
                [].span(),
            );

            // PlayerConfig: exclusive lock, `items` is a dynamic array
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

            // ── Shared global state → Add (concurrent shard access) ──

            register_policy(
                ref world,
                ns_hash,
                Model::<MockToken>::selector(ns_hash),
                CRDVariant::Add,
                [].span(),
            );
        }

        fn request_shard(
            ref self: ContractState,
            entities: Span<felt252>,
            shared_entities: Span<felt252>,
            entity_keys_flat: Span<felt252>,
        ) -> felt252 {
            let world = self.world(@"ns");
            sharding_disp(@world)
                .request_sharding(entities, shared_entities, entity_keys_flat)
        }

        fn end_shard(ref self: ContractState, shard_id: felt252) {
            let world = self.world(@"ns");
            sharding_disp(@world).end_shard(shard_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use dojo::model::{Model, ModelStorage, ModelStorageTest};
    use dojo::sharding::compute_dojo_field_slot;
    use dojo::sharding::request::{
        InitialProof, SlotEntry, SlotVerification, DeterministicProof,
    };
    use dojo::utils::entity_id_from_keys;
    use dojo::world::{
        IShardingSettlementDispatcher, IShardingSettlementDispatcherTrait, WorldStorageTrait,
    };
    use dojo_examples::actions::{IActionsDispatcher, IActionsDispatcherTrait};
    use dojo_examples::models::{Direction, MockToken, Moves, Position};
    use dojo_snf_test::{
        ContractDef, ContractDefTrait, NamespaceDef, TestResource, WorldStorageTestTrait,
        declare_and_deploy, set_caller_address, spawn_test_world,
    };
    use dojo::sharding::{IShardingGameDispatcher, IShardingGameDispatcherTrait};
    use dojo::utils::entity_id_from_serialized_keys;
    use starknet::ContractAddress;

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
                (true, contract_address.into(), 0)
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

    fn sharding_game(world: @dojo::world::WorldStorage) -> IShardingGameDispatcher {
        let (addr, _) = world.dns(@"sharding_systems").unwrap();
        IShardingGameDispatcher { contract_address: addr }
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

    fn mock_token_field_selector() -> felt252 {
        let layout = Model::<MockToken>::layout();
        if let dojo::meta::Layout::Struct(fields) = layout {
            (*fields[0]).selector
        } else {
            panic!("expected struct layout for MockToken")
        }
    }

    /// Test helper: wraps raw player felt252s into the 3-arg request_shard format.
    fn request_shard_for(
        game: IShardingGameDispatcher,
        exclusive_ids: Span<felt252>,
        shared_ids: Span<felt252>,
    ) {
        let mut entities: Array<felt252> = ArrayTrait::new();
        let mut shared_entities: Array<felt252> = ArrayTrait::new();
        let mut keys_flat: Array<felt252> = ArrayTrait::new();
        for id in exclusive_ids {
            entities.append(entity_id_from_serialized_keys([*id].span()));
            keys_flat.append(1);
            keys_flat.append(*id);
        };
        for id in shared_ids {
            shared_entities.append(entity_id_from_serialized_keys([*id].span()));
            keys_flat.append(1);
            keys_flat.append(*id);
        };
        game.request_shard(entities.span(), shared_entities.span(), keys_flat.span());
    }

    fn empty_initial_proof() -> InitialProof {
        InitialProof {
            keys: [].span(),
            values: [].span(),
            commitment: 0,
            fork_state_root: 0,
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
        initial_proof: InitialProof,
    ) {
        let settlement = IShardingSettlementDispatcher { contract_address: world_address };
        snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
        settlement
            .settle(
                shard_id,
                0x1,
                1,
                slots,
                [].span(),
                [].span(),
                initial_proof,
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

        let game = sharding_game(@world);
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        game.register_policies();
        actions.spawn();

        let position: Position = world.read_model(caller);
        assert(position.vec.x == 10 && position.vec.y == 10, 'initial position');
        let moves: Moves = world.read_model(caller);
        assert(moves.remaining == 99, 'initial moves');

        let player_felt: felt252 = caller.into();
        request_shard_for(game,[player_felt].span(), [].span());
        game.end_shard(1);
    }

    #[test]

    fn test_active_shards_tracked() {
        let world = setup_world();
        let caller = dojo_snf_test::get_default_caller_address();

        let game = sharding_game(@world);
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        game.register_policies();
        actions.spawn();

        let settlement = IShardingSettlementDispatcher {
            contract_address: world.dispatcher.contract_address,
        };

        let active = settlement.get_active_shards();
        assert(active.len() == 0, 'no shards initially');

        let player_felt: felt252 = caller.into();
        request_shard_for(game,[player_felt].span(), [].span());
        let active = settlement.get_active_shards();
        assert(active.len() == 1, 'one shard active');
    }

    // ══════════════════════════════════════════════════════════════════
    //  ENTITY LOCKING: exclusive entities are blocked
    // ══════════════════════════════════════════════════════════════════

    #[test]

    #[should_panic(expected: ('Shard: entity locked',))]
    fn test_move_blocked_while_sharded() {
        let mut world = setup_world();
        let caller = dojo_snf_test::get_default_caller_address();

        let game = sharding_game(@world);
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        game.register_policies();
        actions.spawn();

        let player_felt: felt252 = caller.into();
        request_shard_for(game,[player_felt].span(), [].span());

        // move() writes to both Moves and Position — should panic
        actions.move(Direction::Right);
    }

    #[test]

    #[should_panic(expected: ('Shard: entity locked',))]
    fn test_write_model_blocked_while_sharded() {
        let mut world = setup_world();
        let caller = dojo_snf_test::get_default_caller_address();

        let game = sharding_game(@world);
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        game.register_policies();
        actions.spawn();

        let player_felt: felt252 = caller.into();
        request_shard_for(game,[player_felt].span(), [].span());

        // Direct model write should also be blocked
        world.write_model_test(@Moves { player: caller, remaining: 50, last_direction: Direction::None });
    }

    #[test]

    fn test_other_player_not_affected_by_shard() {
        let mut world = setup_world();
        let alice = dojo_snf_test::get_default_caller_address();
        let bob: ContractAddress = 0xb0b.try_into().unwrap();

        let game = sharding_game(@world);
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        // Spawn Alice and write initial state for Bob
        set_caller_address(alice);
        game.register_policies();
        actions.spawn();
        world.write_model_test(
            @Moves { player: bob, remaining: 50, last_direction: Direction::None },
        );

        // Shard only Alice
        let alice_felt: felt252 = alice.into();
        request_shard_for(game,[alice_felt].span(), [].span());

        // Bob's entities are NOT locked — write should succeed
        world.write_model_test(
            @Moves { player: bob, remaining: 42, last_direction: Direction::Right },
        );
        let bob_moves: Moves = world.read_model(bob);
        assert(bob_moves.remaining == 42, 'bob write ok');
    }

    // ══════════════════════════════════════════════════════════════════
    //  SHARED ENTITIES: mainnet writes allowed for Add CRDT
    // ══════════════════════════════════════════════════════════════════

    #[test]

    fn test_shared_entity_allows_mainnet_write() {
        let mut world = setup_world();
        let caller = dojo_snf_test::get_default_caller_address();
        let token_account: ContractAddress = 0xacc.try_into().unwrap();

        let game = sharding_game(@world);
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        game.register_policies();
        actions.spawn();

        // Initialize token balance
        world.write_model_test(@MockToken { account: token_account, amount: 100 });

        // Shard player exclusively, token entity as shared
        let player_felt: felt252 = caller.into();
        let token_felt: felt252 = token_account.into();
        request_shard_for(game,[player_felt].span(), [token_felt].span());

        // Shared entity (MockToken) should still be writable on mainnet
        world.write_model_test(@MockToken { account: token_account, amount: 150 });
        let token: MockToken = world.read_model(token_account);
        assert(token.amount == 150, 'shared write ok');
    }

    // ══════════════════════════════════════════════════════════════════
    //  FULL SETTLEMENT: SetLock CRDT (exclusive overwrite)
    // ══════════════════════════════════════════════════════════════════

    #[test]

    fn test_full_settlement_set_crdt() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let caller = dojo_snf_test::get_default_caller_address();
        register_mock_verifier(world_address);

        let game = sharding_game(@world);
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        game.register_policies();
        actions.spawn();

        // Record initial state
        let initial_moves: Moves = world.read_model(caller);
        assert(initial_moves.remaining == 99, 'pre: 99 moves');

        // Request shard (locks entity)
        let player_felt: felt252 = caller.into();
        request_shard_for(game,[player_felt].span(), [].span());

        // --- Simulate shard gameplay result ---
        // On the shard Katana, the player used 10 moves. Remaining: 89.
        // Settlement carries this value back to mainnet via SetLock CRDT.

        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let moves_selector = Model::<Moves>::selector(ns_hash);
        let entity_id = entity_id_from_keys(@caller);
        let (sel_remaining, _sel_direction) = moves_field_selectors();
        let slot_remaining = compute_dojo_field_slot(moves_selector, entity_id, sel_remaining);

        // Settle: write remaining=89 via SetLock CRDT
        settle_as_owner(
            world_address,
            1,
            [
                make_field_slot(slot_remaining, 89, moves_selector, entity_id, sel_remaining, 0),
            ]
                .span(),
            empty_initial_proof(),
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

        let game = sharding_game(@world);
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        game.register_policies();
        actions.spawn();
        // Move right first so last_direction = Right
        actions.move(Direction::Right);
        let pre: Moves = world.read_model(caller);
        assert(pre.remaining == 98, 'pre: 98 moves');

        let player_felt: felt252 = caller.into();
        request_shard_for(game,[player_felt].span(), [].span());

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
            empty_initial_proof(),
        );

        let after: Moves = world.read_model(caller);
        assert(after.remaining == 50, 'remaining settled');
        // last_direction should be unchanged (Right from the pre-shard move)
        let right: felt252 = Direction::Right.into();
        assert(after.last_direction.into() == right, 'direction unchanged');
    }

    #[test]

    fn test_cancel_shard_unlocks_entity() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let caller = dojo_snf_test::get_default_caller_address();

        let game = sharding_game(@world);
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        game.register_policies();
        actions.spawn();

        let player_felt: felt252 = caller.into();
        request_shard_for(game,[player_felt].span(), [].span());

        // Cancel instead of settle
        let settlement = IShardingSettlementDispatcher { contract_address: world_address };
        snforge_std::start_cheat_caller_address(world_address, snforge_std::test_address());
        settlement.cancel_shard(1);
        snforge_std::stop_cheat_caller_address(world_address);

        // Entity should be unlocked — original values preserved
        let moves: Moves = world.read_model(caller);
        assert(moves.remaining == 99, 'cancel: original');

        // Can write again
        set_caller_address(caller);
        actions.move(Direction::Left);
        let after: Moves = world.read_model(caller);
        assert(after.remaining == 98, 'cancel: can move');
    }

    // ══════════════════════════════════════════════════════════════════
    //  ADD CRDT: shared MockToken balance accumulates deltas
    // ══════════════════════════════════════════════════════════════════

    #[test]

    fn test_settlement_add_crdt_accumulates_delta() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let caller = dojo_snf_test::get_default_caller_address();
        let token_account: ContractAddress = 0xacc.try_into().unwrap();
        register_mock_verifier(world_address);

        let game = sharding_game(@world);
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        game.register_policies();
        actions.spawn();

        // Initialize token balance: 100
        world.write_model_test(@MockToken { account: token_account, amount: 100 });

        // Shard player exclusively, token entity as shared (Add CRDT)
        let player_felt: felt252 = caller.into();
        let token_felt: felt252 = token_account.into();
        request_shard_for(game,[player_felt].span(), [token_felt].span());

        // Simulate mainnet activity while shard is active:
        // someone adds 50 tokens on mainnet (shared entity allows writes)
        world.write_model_test(@MockToken { account: token_account, amount: 150 });

        // --- Simulate shard gameplay result ---
        // On the shard, the player earned 30 tokens.
        // Shard forked with initial_value=100, shard final value=130.
        // Add CRDT: delta = 130 - 100 = 30
        // Mainnet result: current(150) + delta(30) = 180

        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let token_selector = Model::<MockToken>::selector(ns_hash);
        let token_entity_id = entity_id_from_keys(@token_account);
        let sel_amount = mock_token_field_selector();
        let slot_amount = compute_dojo_field_slot(token_selector, token_entity_id, sel_amount);

        let initial_proof = InitialProof {
            keys: [slot_amount].span(),
            values: [100].span(),
            commitment: core::poseidon::poseidon_hash_span([slot_amount, 100].span()),
            fork_state_root: 0x42,
        };

        settle_as_owner(
            world_address,
            1,
            [
                make_field_slot(
                    slot_amount,
                    130,             // shard final value
                    token_selector,
                    token_entity_id,
                    sel_amount,
                    100,             // initial value at fork time
                ),
            ]
                .span(),
            initial_proof,
        );

        // Verify: 150 (mainnet current) + (130 - 100) (shard delta) = 180
        let token: MockToken = world.read_model(token_account);
        assert(token.amount == 180, 'add crdt: 150+30=180');
    }

    // ══════════════════════════════════════════════════════════════════
    //  NEGATIVE SCENARIOS
    // ══════════════════════════════════════════════════════════════════

    #[test]

    #[should_panic(expected: ('Shard: entity already sharded',))]
    fn test_double_shard_same_entity_rejected() {
        let world = setup_world();
        let caller = dojo_snf_test::get_default_caller_address();

        let game = sharding_game(@world);
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        game.register_policies();
        actions.spawn();

        let player_felt: felt252 = caller.into();
        request_shard_for(game,[player_felt].span(), [].span());
        // Second shard for same entity -> panic
        request_shard_for(game,[player_felt].span(), [].span());
    }

    #[test]

    #[should_panic(expected: ('Shard: no entities',))]
    fn test_request_shard_empty_entities_rejected() {
        let world = setup_world();
        let caller = dojo_snf_test::get_default_caller_address();

        let game = sharding_game(@world);

        set_caller_address(caller);
        game.register_policies();
        // Empty entity list -> panic
        request_shard_for(game,[].span(), [].span());
    }

    #[test]

    #[should_panic(expected: ('Shard: slot ownership mismatch',))]
    fn test_settle_wrong_slot_key_rejected() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let caller = dojo_snf_test::get_default_caller_address();
        register_mock_verifier(world_address);

        let game = sharding_game(@world);
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        game.register_policies();
        actions.spawn();

        let player_felt: felt252 = caller.into();
        request_shard_for(game,[player_felt].span(), [].span());

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
            empty_initial_proof(),
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

        let game = sharding_game(@world);
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(caller);
        game.register_policies();
        actions.spawn();
        // Write some data for Eve (not sharded)
        world.write_model_test(@Moves { player: eve, remaining: 10, last_direction: Direction::None });

        // Shard only the caller
        let player_felt: felt252 = caller.into();
        request_shard_for(game,[player_felt].span(), [].span());

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
            empty_initial_proof(),
        );
    }

    // ══════════════════════════════════════════════════════════════════
    //  DEMO: Full-game shard — all player state goes to shard
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_demo_full_game_shard() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let alice = dojo_snf_test::get_default_caller_address();
        register_mock_verifier(world_address);

        let game = sharding_game(@world);
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(alice);
        game.register_policies();
        actions.spawn();

        // Initialize token balance
        world.write_model_test(@MockToken { account: alice, amount: 1000 });

        // ── SHARD ENTIRE GAME ──
        // Alice's entity is exclusively locked for all models (Moves, Position).
        // Alice's token entity is also exclusively locked (single-player shard).
        let alice_felt: felt252 = alice.into();
        request_shard_for(game,[alice_felt].span(), [].span());

        // Verify: all writes blocked on mainnet
        let settlement = IShardingSettlementDispatcher { contract_address: world_address };
        assert(settlement.get_entity_shard(
            dojo::utils::entity_id_from_serialized_keys([alice_felt].span())
        ) != 0, 'entity is sharded');

        // ── SIMULATE SHARD GAMEPLAY ──
        // On the shard Katana, the player moved 20 times (remaining: 99 -> 79).
        // Settlement carries this value back to mainnet via SetLock CRDT.
        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let moves_selector = Model::<Moves>::selector(ns_hash);
        let entity_id = entity_id_from_keys(@alice);

        let (sel_remaining, _sel_direction) = moves_field_selectors();
        let slot_remaining = compute_dojo_field_slot(moves_selector, entity_id, sel_remaining);

        settle_as_owner(
            world_address,
            1,
            [
                make_field_slot(slot_remaining, 79, moves_selector, entity_id, sel_remaining, 0),
            ]
                .span(),
            empty_initial_proof(),
        );

        // ── VERIFY SETTLEMENT ──
        let moves: Moves = world.read_model(alice);
        assert(moves.remaining == 79, 'full: 79 moves');

        // Entity unlocked — game can continue on mainnet
        actions.move(Direction::Right);
        let after: Moves = world.read_model(alice);
        assert(after.remaining == 78, 'full: can move again');
    }

    // ══════════════════════════════════════════════════════════════════
    //  DEMO: Partial shard — only player movement, tokens shared
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_demo_partial_shard_with_shared_tokens() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let alice = dojo_snf_test::get_default_caller_address();
        let treasury: ContractAddress = 0x7ea5.try_into().unwrap();
        register_mock_verifier(world_address);

        let game = sharding_game(@world);
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(alice);
        game.register_policies();
        actions.spawn();

        // Global treasury token balance (shared across shards)
        world.write_model_test(@MockToken { account: treasury, amount: 500 });

        // ── PARTIAL SHARD ──
        // Alice's movement is exclusive (locked), treasury is shared (Add CRDT).
        // This allows other shards or mainnet to also modify the treasury concurrently.
        let alice_felt: felt252 = alice.into();
        let treasury_felt: felt252 = treasury.into();
        request_shard_for(game,
            [alice_felt].span(),          // exclusive: player movement
            [treasury_felt].span(),       // shared: global treasury
        );

        // ── MAINNET ACTIVITY WHILE SHARD RUNS ──
        // Another system deposits 200 tokens into the treasury on mainnet.
        // This is allowed because treasury is a shared entity.
        world.write_model_test(@MockToken { account: treasury, amount: 700 });

        // ── SHARD SETTLEMENT ──
        // On the shard: Alice used 5 moves (remaining 94), and the shard
        // awarded 100 tokens to the treasury (fork initial 500 -> shard final 600).
        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let moves_selector = Model::<Moves>::selector(ns_hash);
        let token_selector = Model::<MockToken>::selector(ns_hash);
        let alice_entity = entity_id_from_keys(@alice);
        let treasury_entity = entity_id_from_keys(@treasury);

        let (sel_remaining, _) = moves_field_selectors();
        let slot_remaining = compute_dojo_field_slot(moves_selector, alice_entity, sel_remaining);

        let sel_amount = mock_token_field_selector();
        let slot_amount = compute_dojo_field_slot(token_selector, treasury_entity, sel_amount);

        let initial_proof = InitialProof {
            keys: [slot_amount].span(),
            values: [500].span(),
            commitment: core::poseidon::poseidon_hash_span([slot_amount, 500].span()),
            fork_state_root: 0x42,
        };

        settle_as_owner(
            world_address,
            1,
            [
                // Alice's moves: SetLock overwrite
                make_field_slot(slot_remaining, 94, moves_selector, alice_entity, sel_remaining, 0),
                // Treasury tokens: Add CRDT delta (600 - 500 = +100)
                make_field_slot(slot_amount, 600, token_selector, treasury_entity, sel_amount, 500),
            ]
                .span(),
            initial_proof,
        );

        // ── VERIFY ──
        let moves: Moves = world.read_model(alice);
        assert(moves.remaining == 94, 'partial: 94 moves');

        // Treasury: mainnet had 700 + shard delta (+100) = 800
        let token: MockToken = world.read_model(treasury);
        assert(token.amount == 800, 'partial: 700+100=800');
    }

    // ══════════════════════════════════════════════════════════════════
    //  DEMO: Two concurrent shards sharing global state
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_demo_two_shards_concurrent_shared_token() {
        let mut world = setup_world();
        let world_address = world.dispatcher.contract_address;
        let alice = dojo_snf_test::get_default_caller_address();
        let bob: ContractAddress = 0xb0b.try_into().unwrap();
        let pool: ContractAddress = 0x9001.try_into().unwrap();
        register_mock_verifier(world_address);

        let game = sharding_game(@world);
        let (actions_addr, _) = world.dns(@"actions").unwrap();
        let actions = IActionsDispatcher { contract_address: actions_addr };

        set_caller_address(alice);
        game.register_policies();
        actions.spawn();

        // Bob's state
        world.write_model_test(@Moves { player: bob, remaining: 99, last_direction: Direction::None });
        world.write_model_test(@Position { player: bob, vec: dojo_examples::models::Vec2 { x: 10, y: 10 } });

        // Shared reward pool
        world.write_model_test(@MockToken { account: pool, amount: 1000 });

        // ── TWO CONCURRENT SHARDS ──
        // Shard 1: Alice (exclusive) + reward pool (shared)
        let alice_felt: felt252 = alice.into();
        let pool_felt: felt252 = pool.into();
        request_shard_for(game,[alice_felt].span(), [pool_felt].span());

        // Shard 2: Bob (exclusive) + same reward pool (shared)
        let bob_felt: felt252 = bob.into();
        request_shard_for(game,[bob_felt].span(), [pool_felt].span());

        let settlement = IShardingSettlementDispatcher { contract_address: world_address };
        let active = settlement.get_active_shards();
        assert(active.len() == 2, 'two shards active');

        // ── SETTLE SHARD 1 (Alice) ──
        // Alice used 10 moves, shard awarded 50 tokens to the pool
        let ns_hash = dojo::utils::bytearray_hash(@"ns");
        let moves_selector = Model::<Moves>::selector(ns_hash);
        let token_selector = Model::<MockToken>::selector(ns_hash);
        let alice_entity = entity_id_from_keys(@alice);
        let pool_entity = entity_id_from_keys(@pool);

        let (sel_remaining, _) = moves_field_selectors();
        let sel_amount = mock_token_field_selector();

        let alice_slot_remaining = compute_dojo_field_slot(moves_selector, alice_entity, sel_remaining);
        let pool_slot_amount = compute_dojo_field_slot(token_selector, pool_entity, sel_amount);

        let initial_proof_1 = InitialProof {
            keys: [pool_slot_amount].span(),
            values: [1000].span(),
            commitment: core::poseidon::poseidon_hash_span([pool_slot_amount, 1000].span()),
            fork_state_root: 0x42,
        };

        settle_as_owner(
            world_address,
            1, // shard_id = 1
            [
                make_field_slot(alice_slot_remaining, 89, moves_selector, alice_entity, sel_remaining, 0),
                make_field_slot(pool_slot_amount, 1050, token_selector, pool_entity, sel_amount, 1000),
            ]
                .span(),
            initial_proof_1,
        );

        // After shard 1: pool = 1000 + (1050 - 1000) = 1050
        let pool_after_1: MockToken = world.read_model(pool);
        assert(pool_after_1.amount == 1050, 'shard1: 1000+50=1050');

        // ── SETTLE SHARD 2 (Bob) ──
        // Bob used 15 moves, shard awarded 75 tokens to the pool
        let bob_entity = entity_id_from_keys(@bob);
        let bob_slot_remaining = compute_dojo_field_slot(moves_selector, bob_entity, sel_remaining);

        let initial_proof_2 = InitialProof {
            keys: [pool_slot_amount].span(),
            values: [1000].span(),
            commitment: core::poseidon::poseidon_hash_span([pool_slot_amount, 1000].span()),
            fork_state_root: 0x42,
        };

        settle_as_owner(
            world_address,
            2, // shard_id = 2
            [
                make_field_slot(bob_slot_remaining, 84, moves_selector, bob_entity, sel_remaining, 0),
                make_field_slot(pool_slot_amount, 1075, token_selector, pool_entity, sel_amount, 1000),
            ]
                .span(),
            initial_proof_2,
        );

        // ── VERIFY FINAL STATE ──
        // Alice: 89 moves remaining
        let alice_moves: Moves = world.read_model(alice);
        assert(alice_moves.remaining == 89, 'alice: 89 moves');

        // Bob: 84 moves remaining
        let bob_moves: Moves = world.read_model(bob);
        assert(bob_moves.remaining == 84, 'bob: 84 moves');

        // Pool: 1050 (after shard 1) + (1075 - 1000) (shard 2 delta) = 1125
        // Both shards' token rewards accumulated correctly via Add CRDT!
        let pool_final: MockToken = world.read_model(pool);
        assert(pool_final.amount == 1125, 'pool: 1050+75=1125');

        // All shards done
        let active = settlement.get_active_shards();
        assert(active.len() == 0, 'all shards settled');

        // Both players can play again on mainnet
        set_caller_address(alice);
        actions.move(Direction::Up);
        let after: Moves = world.read_model(alice);
        assert(after.remaining == 88, 'alice: can move');
    }
}
