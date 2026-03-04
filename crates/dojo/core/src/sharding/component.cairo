use starknet::ContractAddress;
use super::crdt::{CRDType, slot_key, slot_value};

#[starknet::interface]
pub trait IContractComponent<TContractState> {
    fn initialize_shard(
        ref self: TContractState,
        sharding_contract_address: ContractAddress,
        contract_slots_changes: Span<CRDType>,
    );
    /// Apply storage changes from a settled shard and unlock the slots.
    /// Caller must be the registered sharding contract (proxy).
    /// The proxy already verifies shard_id — the game contract trusts the proxy.
    fn update_shard_state(ref self: TContractState, storage_changes: Array<(slot_key, slot_value)>);
    /// Cancel (unlock) slots without applying shard values.
    /// Caller must be the registered sharding contract (proxy).
    fn cancel_shard_state(ref self: TContractState, slots: Span<felt252>);
    /// Signal end of shard to the sharding proxy.
    /// The proxy emits `ShardFinished` which the operator watches.
    fn end_shard(ref self: TContractState);
}

#[starknet::component]
pub mod sharding_component {
    use core::num::traits::Zero;
    use starknet::SyscallResultTrait;
    use starknet::storage::{StoragePointerReadAccess, StoragePointerWriteAccess};
    use dojo::sharding::interface::{IShardingDispatcher, IShardingDispatcherTrait};
    use dojo::sharding::crdt::{CRDType, CRDTypeTrait, safe_increment, slot_value};
    use starknet::storage::{Map, StorageMapReadAccess, StorageMapWriteAccess};
    use starknet::storage_access::StorageAddress;
    use starknet::syscalls::{storage_read_syscall, storage_write_syscall};
    use starknet::{ContractAddress, get_caller_address, get_contract_address};

    type init_count = felt252;

    #[storage]
    pub struct Storage {
        slots: Map<slot_value, (CRDType, init_count)>,
        sharding_contract_address: ContractAddress,
        /// Snapshot of Add slot values at initialization time.
        /// Used to compute delta = (shard_value - initial) during settlement.
        initial_add_values: Map<slot_value, felt252>,
    }

    #[event]
    #[derive(Drop, starknet::Event)]
    pub enum Event {
        ContractSlotUpdated: ContractSlotUpdated,
        ContractComponentUpdated: ContractComponentUpdated,
    }

    #[derive(Drop, starknet::Event, Clone)]
    pub struct ContractSlotUpdated {
        pub contract_address: ContractAddress,
        pub slots_to_change: Array<(felt252, felt252)>,
    }

    #[derive(Drop, starknet::Event)]
    pub struct ContractComponentUpdated {
        pub storage_changes: Array<(felt252, felt252)>,
    }

    pub mod Errors {
        pub const NOT_INITIALIZED: felt252 = 'Component: Not initialized';
        pub const STORAGE_UNLOCKED: felt252 = 'Component: Storage is unlocked';
        pub const NO_CONTRACTS_SUBMITTED: felt252 = 'Component: No contracts';
        pub const UNAUTHORIZED_CALLER: felt252 = 'Component: Unauthorized caller';
        pub const ALREADY_INITIALIZED: felt252 = 'Component: Already initialized';
        pub const SLOT_LOCKED: felt252 = 'Component: Slot locked by shard';
        pub const TYPE_CHANGE_WHILE_ACTIVE: felt252 = 'Component: Type change active';
        pub const ADD_DELTA_UNDERFLOW: felt252 = 'Component: Add delta underflow';
        pub const ARITHMETIC_OVERFLOW: felt252 = 'Component: Arithmetic overflow';
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
            self.sharding_contract_address.write(sharding_contract_address);

            // Validate and lock slots
            for crd_type in contract_slots_changes {
                let crd_type = *crd_type;

                let (prev_crd_type, init_count) = self.slots.read(crd_type.slot());

                if init_count != 0 {
                    // Slot is active — SetLock and Lock are exclusive (no stacking)
                    let is_locking = match prev_crd_type {
                        CRDType::SetLock(_) | CRDType::Lock(_) => true,
                        _ => false,
                    };
                    assert(!is_locking, Errors::SLOT_LOCKED);
                    // Set and Add allow same-type stacking only
                    assert(
                        prev_crd_type.is_same_variant(crd_type), Errors::TYPE_CHANGE_WHILE_ACTIVE,
                    );
                } else {
                    // Slot is free (init_count == 0) — check type transition from Set
                    prev_crd_type.verify_crd_type(crd_type);
                }

                let new_init_count = safe_increment(init_count, 'Init count overflow');
                self.slots.write(crd_type.slot(), (crd_type, new_init_count));

                // For Add CRDTs, snapshot the current value so we can compute
                // delta = (shard_value - initial) during settlement.
                if let CRDType::Add(_) = crd_type {
                    let storage_address: StorageAddress = crd_type.slot().try_into().unwrap();
                    let current = storage_read_syscall(0, storage_address).unwrap_syscall();
                    self.initial_add_values.write(crd_type.slot(), current);
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

            assert(storage_changes.len() != 0, Errors::NO_CONTRACTS_SUBMITTED);

            let contract_address = get_contract_address();

            // Filter to only locked slots (init_count > 0).
            // The settlement proof may contain slots from multiple contracts, but each
            // contract_component only applies changes to its own registered slots.
            // Unregistered slots are silently ignored — this is by design, not an error.
            let mut locked_changes: Array<(felt252, felt252)> = ArrayTrait::new();
            for slot_entry in storage_changes.span() {
                let (storage_key, storage_value) = *slot_entry;
                let (_, init_count) = self.slots.read(storage_key);
                if init_count != 0 {
                    locked_changes.append((storage_key, storage_value));
                }
            }

            assert(locked_changes.len() != 0, Errors::NO_CONTRACTS_SUBMITTED);

            // Apply storage changes via CRDT logic (only locked slots)
            self.update_shard(locked_changes.clone(), contract_address);

            // Unlock slots: decrement init_count, reset Lock types to Set
            for slot_entry in locked_changes.span() {
                let (storage_key, _) = *slot_entry;

                let (crd_type, init_count) = self.slots.read(storage_key);

                // Lock slots reserve the storage key during shard execution
                // but always discard the shard value — reset fully on unlock.
                let is_lock = match crd_type {
                    CRDType::Lock(_) => true,
                    _ => false,
                };

                if is_lock {
                    self
                        .slots
                        .write(storage_key, (CRDType::Set((contract_address, storage_key)), 0));
                } else {
                    let new_init_count = init_count - 1;
                    if new_init_count == 0 {
                        self
                            .slots
                            .write(storage_key, (CRDType::Set((contract_address, storage_key)), 0));
                        if let CRDType::Add(_) = crd_type {
                            self.initial_add_values.write(storage_key, 0);
                        }
                    } else {
                        // Other shards still active on this slot — just decrement
                        self.slots.write(storage_key, (crd_type, new_init_count));
                    }
                }
            }

            // self.emit(ContractSlotUpdated { contract_address, slots_to_change: locked_changes });
        }

        fn cancel_shard_state(ref self: ComponentState<TContractState>, slots: Span<felt252>) {
            let caller = get_caller_address();
            assert(caller == self.sharding_contract_address.read(), Errors::UNAUTHORIZED_CALLER);

            let contract_address = get_contract_address();

            for slot_key in slots {
                let slot_key = *slot_key;

                let (crd_type, init_count) = self.slots.read(slot_key);
                if init_count == 0 {
                    continue;
                }

                let new_init_count = init_count - 1;
                if new_init_count == 0 {
                    // Fully unlocked — reset to base Set type
                    self.slots.write(slot_key, (CRDType::Set((contract_address, slot_key)), 0));
                    if let CRDType::Add(_) = crd_type {
                        self.initial_add_values.write(slot_key, 0);
                    }
                } else {
                    // Other shards still active on this slot — just decrement
                    self.slots.write(slot_key, (crd_type, new_init_count));
                }
            }
        }

        fn end_shard(ref self: ComponentState<TContractState>) {
            let sharding_address = self.sharding_contract_address.read();
            if sharding_address.is_zero() {
                return;
            }
            let sharding_dispatcher = IShardingDispatcher { contract_address: sharding_address };
            sharding_dispatcher.end_shard();
        }
    }

    #[generate_trait]
    pub impl InternalImpl<
        TContractState, +HasComponent<TContractState>,
    > of InternalTrait<TContractState> {
        /// Apply CRDT-based storage changes from a settled shard.
        ///
        /// Each slot is updated according to its registered CRDType:
        /// - **Set/SetLock**: direct overwrite — `storage[key] = value`
        /// - **Add**: delta merge — `delta = shard_value - initial_snapshot`,
        ///   then `storage[key] = current + delta` (prevents double-counting)
        /// - **Lock**: slot was reserved but value is discarded (unlock happens in caller)
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
                        // value is the absolute shard state; compute delta vs fork snapshot
                        let current_u256: u256 = current_value.into();
                        let shard_u256: u256 = value.into();
                        let initial_u256: u256 = initial_value.into();
                        assert(shard_u256 >= initial_u256, Errors::ADD_DELTA_UNDERFLOW);
                        let delta = shard_u256 - initial_u256;
                        let sum = current_u256 + delta;
                        let new_value: felt252 = sum.try_into().expect(Errors::ARITHMETIC_OVERFLOW);
                        storage_write_syscall(0, storage_address, new_value).unwrap_syscall();
                    },
                    // Lock reserves the slot during shard execution but discards
                    // the shard's value; the slot is unlocked in update_shard_state.
                    CRDType::Lock(_) => {},
                }
            }
            // self.emit(ContractComponentUpdated { storage_changes });
        }
    }
}
