use starknet::ContractAddress;
use super::crdt::{CRDType, SlotKey, SlotValue};

#[starknet::interface]
pub trait IContractComponent<TContractState> {
    fn initialize_shard(
        ref self: TContractState,
        sharding_contract_address: ContractAddress,
        contract_slots_changes: Span<CRDType>,
    );
    fn update_shard_state(ref self: TContractState, storage_changes: Array<(SlotKey, SlotValue)>);
    fn cancel_shard_state(ref self: TContractState, slots: Span<felt252>);
    fn end_shard(ref self: TContractState);
}

#[starknet::component]
pub mod sharding_component {
    use core::num::traits::Zero;
    use starknet::SyscallResultTrait;
    use starknet::storage::{StoragePointerReadAccess, StoragePointerWriteAccess};
    use dojo::sharding::interface::{IShardingDispatcher, IShardingDispatcherTrait};
    use dojo::sharding::crdt::{CRDType, CRDTypeTrait, safe_increment, SlotValue};
    use starknet::storage::{Map, StorageMapReadAccess, StorageMapWriteAccess};
    use starknet::storage_access::StorageAddress;
    use starknet::syscalls::{storage_read_syscall, storage_write_syscall};
    use starknet::{ContractAddress, get_caller_address, get_contract_address};
    use core::dict::{Felt252Dict, Felt252DictTrait};

    type InitCount = felt252;

    #[storage]
    pub struct Storage {
        slots: Map<SlotValue, (CRDType, InitCount)>,
        sharding_contract_address: ContractAddress,
        /// Add CRDT snapshots for delta computation: delta = shard_value - initial.
        initial_add_values: Map<SlotValue, felt252>,
        /// Per-slot metadata (model_selector, entity_id, member_selector) for Torii events.
        slot_model_selector: Map<felt252, felt252>,
        slot_entity_id: Map<felt252, felt252>,
        slot_member_selector: Map<felt252, felt252>,
        /// Entity keys for StoreSetRecord emission (Torii needs keys for new entities).
        entity_keys_len: Map<felt252, u32>,
        entity_keys_data: Map<felt252, felt252>,
    }

    pub mod Errors {
        pub const NOT_INITIALIZED: felt252 = 'Component: Not initialized';
        pub const STORAGE_UNLOCKED: felt252 = 'Component: Storage is unlocked';
        pub const NO_STORAGE_CHANGES: felt252 = 'Component: No storage changes';
        pub const UNAUTHORIZED_CALLER: felt252 = 'Component: Unauthorized caller';
        pub const ALREADY_INITIALIZED: felt252 = 'Component: Already initialized';
        pub const SLOT_LOCKED: felt252 = 'Component: Slot locked by shard';
        pub const TYPE_CHANGE_WHILE_ACTIVE: felt252 = 'Component: Type change active';
        pub const ADD_DELTA_UNDERFLOW: felt252 = 'Component: Add delta underflow';
        pub const ARITHMETIC_OVERFLOW: felt252 = 'Component: Arithmetic overflow';
        pub const SHARDING_PROXY_MISMATCH: felt252 = 'Component: Proxy mismatch';
        pub const DUPLICATE_SLOT: felt252 = 'Component: Duplicate slot';
    }

    #[embeddable_as(ContractComponentImpl)]
    impl ContractImpl<
        TContractState, +HasComponent<TContractState>,
    > of super::IContractComponent<ComponentState<TContractState>> {
        fn initialize_shard(
            ref self: ComponentState<TContractState>,
            sharding_contract_address: ContractAddress,
            contract_slots_changes: Span<CRDType>,
        ) {
            let current_proxy = self.sharding_contract_address.read();
            if !current_proxy.is_zero() {
                assert(
                    current_proxy == sharding_contract_address, Errors::SHARDING_PROXY_MISMATCH,
                );
            }
            self.sharding_contract_address.write(sharding_contract_address);

            for crd_type in contract_slots_changes {
                let crd_type = *crd_type;

                let (prev_crd_type, init_count) = self.slots.read(crd_type.slot());

                if init_count != 0 {
                    assert(!prev_crd_type.is_exclusive(), Errors::SLOT_LOCKED);
                    assert(
                        prev_crd_type.is_same_variant(crd_type), Errors::TYPE_CHANGE_WHILE_ACTIVE,
                    );
                } else {
                    prev_crd_type.assert_is_base_set();
                }

                let new_init_count = safe_increment(init_count, 'Init count overflow');
                self.slots.write(crd_type.slot(), (crd_type, new_init_count));

                // For Add CRDTs, snapshot the current value ONLY on first lock (0→1).
                // When multiple shards stack on the same Add slot, the initial snapshot
                // must remain from the first shard — otherwise subsequent initializations
                // would overwrite it and corrupt delta computation at settlement time.
                if let CRDType::Add(_) = crd_type {
                    if init_count == 0 {
                        let storage_address: StorageAddress = crd_type.slot().try_into().unwrap();
                        let current = storage_read_syscall(0, storage_address).unwrap_syscall();
                        self.initial_add_values.write(crd_type.slot(), current);
                    }
                }
            }

            // Forward to proxy — emits ShardingRequested event.
            // The proxy is the single source of truth for shard_id.
            let sharding_dispatcher = IShardingDispatcher {
                contract_address: sharding_contract_address,
            };
            sharding_dispatcher.initialize_sharding(contract_slots_changes);
        }

        fn update_shard_state(
            ref self: ComponentState<TContractState>, storage_changes: Array<(felt252, felt252)>,
        ) {
            let caller = get_caller_address();
            assert(caller == self.sharding_contract_address.read(), Errors::UNAUTHORIZED_CALLER);

            assert(storage_changes.len() != 0, Errors::NO_STORAGE_CHANGES);

            let contract_address = get_contract_address();

            // Filter to locked slots only. Unregistered slots are silently ignored —
            // the settlement proof may contain slots from other contracts.
            let mut locked_changes: Array<(felt252, felt252)> = ArrayTrait::new();
            let mut seen_keys: Felt252Dict<felt252> = Default::default();
            for slot_entry in storage_changes.span() {
                let (storage_key, storage_value) = *slot_entry;
                let (_, init_count) = self.slots.read(storage_key);
                if init_count != 0 {
                    assert(Felt252DictTrait::get(ref seen_keys, storage_key) == 0, Errors::DUPLICATE_SLOT);
                    Felt252DictTrait::insert(ref seen_keys, storage_key, 1);
                    locked_changes.append((storage_key, storage_value));
                }
            };

            assert(locked_changes.len() != 0, Errors::NO_STORAGE_CHANGES);

            self.update_shard(locked_changes.clone(), contract_address);

            for slot_entry in locked_changes.span() {
                let (storage_key, _) = *slot_entry;
                self.unlock_slot(storage_key, contract_address);
            }
        }

        fn cancel_shard_state(ref self: ComponentState<TContractState>, slots: Span<felt252>) {
            let caller = get_caller_address();
            assert(caller == self.sharding_contract_address.read(), Errors::UNAUTHORIZED_CALLER);

            let contract_address = get_contract_address();

            for slot_key in slots {
                let slot_key = *slot_key;
                let (_, init_count) = self.slots.read(slot_key);
                if init_count == 0 {
                    continue;
                }
                self.unlock_slot(slot_key, contract_address);
            }
        }

        fn end_shard(ref self: ComponentState<TContractState>) {
            let sharding_address = self.sharding_contract_address.read();
            assert(!sharding_address.is_zero(), Errors::NOT_INITIALIZED);
            let sharding_dispatcher = IShardingDispatcher { contract_address: sharding_address };
            sharding_dispatcher.end_shard();
        }
    }

    #[generate_trait]
    pub impl MetadataImpl<
        TContractState, +HasComponent<TContractState>,
    > of MetadataTrait<TContractState> {
        fn store_slot_metadata(
            ref self: ComponentState<TContractState>,
            slot: felt252,
            model_selector: felt252,
            entity_id: felt252,
            member_selector: felt252,
        ) {
            self.slot_model_selector.write(slot, model_selector);
            self.slot_entity_id.write(slot, entity_id);
            self.slot_member_selector.write(slot, member_selector);
        }

        fn read_slot_metadata(
            self: @ComponentState<TContractState>, slot: felt252,
        ) -> (felt252, felt252, felt252) {
            (
                self.slot_model_selector.read(slot),
                self.slot_entity_id.read(slot),
                self.slot_member_selector.read(slot),
            )
        }

        /// Returns true when `slot` is currently active under an exclusive CRDT
        /// (`SetLock` or `Lock`). These slots must reject regular world writes
        /// while shard gameplay is active on main chain.
        fn is_slot_exclusive_locked(
            self: @ComponentState<TContractState>, slot: felt252,
        ) -> bool {
            let (crd_type, init_count) = self.slots.read(slot);
            init_count != 0 && crd_type.is_exclusive()
        }

        fn clear_slot_metadata(ref self: ComponentState<TContractState>, slot: felt252) {
            self.slot_model_selector.write(slot, 0);
            self.slot_entity_id.write(slot, 0);
            self.slot_member_selector.write(slot, 0);
        }

        /// Idempotent — skips if already stored for this entity_id.
        fn store_entity_keys(
            ref self: ComponentState<TContractState>,
            entity_id: felt252,
            keys: Span<felt252>,
        ) {
            if self.entity_keys_len.read(entity_id) != 0 {
                return;
            }
            let len: u32 = keys.len();
            self.entity_keys_len.write(entity_id, len);
            let mut i: u32 = 0;
            while i < len {
                let data_key = dojo::utils::combine_key(entity_id, i.into());
                self.entity_keys_data.write(data_key, *keys[i]);
                i += 1;
            }
        }

        fn read_entity_keys(
            self: @ComponentState<TContractState>,
            entity_id: felt252,
        ) -> Span<felt252> {
            let len = self.entity_keys_len.read(entity_id);
            if len == 0 {
                return [].span();
            }
            let mut keys: Array<felt252> = ArrayTrait::new();
            let mut i: u32 = 0;
            while i < len {
                let data_key = dojo::utils::combine_key(entity_id, i.into());
                keys.append(self.entity_keys_data.read(data_key));
                i += 1;
            };
            keys.span()
        }

        fn clear_entity_keys(
            ref self: ComponentState<TContractState>,
            entity_id: felt252,
        ) {
            let len = self.entity_keys_len.read(entity_id);
            if len == 0 {
                return;
            }
            let mut i: u32 = 0;
            while i < len {
                let data_key = dojo::utils::combine_key(entity_id, i.into());
                self.entity_keys_data.write(data_key, 0);
                i += 1;
            };
            self.entity_keys_len.write(entity_id, 0);
        }
    }

    #[generate_trait]
    pub impl InternalImpl<
        TContractState, +HasComponent<TContractState>,
    > of InternalTrait<TContractState> {
        /// Decrement init_count and reset slot to base Set when fully unlocked.
        /// Lock/SetLock are exclusive (init_count can only be 1), so they always fully reset.
        fn unlock_slot(
            ref self: ComponentState<TContractState>,
            slot_key: felt252,
            contract_address: ContractAddress,
        ) {
            let (crd_type, init_count) = self.slots.read(slot_key);
            let base_set = CRDType::Set((contract_address, slot_key));

            if crd_type.is_lock() || init_count - 1 == 0 {
                self.slots.write(slot_key, (base_set, 0));
                if let CRDType::Add(_) = crd_type {
                    self.initial_add_values.write(slot_key, 0);
                }
            } else {
                self.slots.write(slot_key, (crd_type, init_count - 1));
            }
        }

        fn update_shard(
            ref self: ComponentState<TContractState>,
            storage_changes: Array<(felt252, felt252)>,
            contract_address: ContractAddress,
        ) {
            for storage_change in storage_changes.span() {
                let (key, value) = *storage_change;
                let storage_address: StorageAddress = key.try_into().unwrap();

                let (crd_type, _) = self.slots.read(key);

                match crd_type {
                    CRDType::SetLock(_) |
                    CRDType::Set(_) => {
                        storage_write_syscall(0, storage_address, value).unwrap_syscall();
                    },
                    CRDType::Add(_) => {
                        let current_value = storage_read_syscall(0, storage_address)
                            .unwrap_syscall();
                        let initial_value = self.initial_add_values.read(key);
                        let current_u256: u256 = current_value.into();
                        let shard_u256: u256 = value.into();
                        let initial_u256: u256 = initial_value.into();
                        assert(shard_u256 >= initial_u256, Errors::ADD_DELTA_UNDERFLOW);
                        let delta = shard_u256 - initial_u256;
                        let sum = current_u256 + delta;
                        let new_value: felt252 = sum.try_into().expect(Errors::ARITHMETIC_OVERFLOW);
                        storage_write_syscall(0, storage_address, new_value).unwrap_syscall();
                    },
                    CRDType::Lock(_) => {},
                }
            }
        }
    }
}
