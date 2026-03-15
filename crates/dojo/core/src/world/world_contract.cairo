use core::fmt::{Display, Error, Formatter};

#[derive(Copy, Drop, PartialEq)]
pub enum Permission {
    Writer,
    Owner,
}

impl PermissionDisplay of Display<Permission> {
    fn fmt(self: @Permission, ref f: Formatter) -> Result<(), Error> {
        let str = match self {
            Permission::Writer => @"WRITER",
            Permission::Owner => @"OWNER",
        };
        f.buffer.append(str);
        Result::Ok(())
    }
}

#[starknet::contract]
pub mod world {
    use core::array::{ArrayTrait, SpanTrait};
    use core::box::BoxTrait;
    use core::num::traits::Zero;
    use core::poseidon::poseidon_hash_span;
    use core::panic_with_felt252;
    use core::panics::panic_with_byte_array;
    use core::serde::Serde;
    use core::traits::Into;
    use dojo::contract::components::upgradeable::{
        IUpgradeableDispatcher, IUpgradeableDispatcherTrait,
    };
    use dojo::meta::{
        IDeployedResourceDispatcher, IDeployedResourceDispatcherTrait,
        IDeployedResourceLibraryDispatcher, IStoredResourceDispatcher,
        IStoredResourceDispatcherTrait, Layout, LayoutCompareTrait, TyCompareTrait,
    };
    use dojo::model::{Model, ModelIndex, ResourceMetadata, metadata};
    use dojo::storage;
    use dojo::storage::database::DOJO_STORAGE;
    use dojo::storage::packing;
    use dojo::utils::{
        bytearray_hash, combine_key, default_address, default_class_hash,
        entity_id_from_serialized_keys,
        selector_from_namespace_and_name,
    };
    use dojo::world::{IUpgradeableWorld, IWorld, Resource, ResourceIsNoneTrait, errors};
    use starknet::storage::Map;
    pub use starknet::storage::{
        StorageMapReadAccess, StorageMapWriteAccess, StoragePointerReadAccess,
        StoragePointerWriteAccess,
    };
    use starknet::syscalls::{
        call_contract_syscall, deploy_syscall, get_class_hash_at_syscall, replace_class_syscall,
        storage_read_syscall, storage_write_syscall,
    };
    use core::dict::{Felt252Dict, Felt252DictTrait};
    use starknet::{ClassHash, ContractAddress, SyscallResultTrait, get_caller_address, get_tx_info};
    use starknet::storage_access::{
        StorageAddress, storage_address_from_base, storage_address_from_base_and_offset,
        storage_base_address_from_felt252,
    };
    use super::Permission;
    use dojo::world::world_sharding::{ShardDynamicMemberChanges, ShardSlotDescriptor};

    // ── Sharding imports ────────────────────────────────────────────────
    use dojo::sharding::component::sharding_component as sharding_cpt;
    use sharding_cpt::InternalTrait as ShardingInternalTrait;
    use dojo::sharding::crdt::CRDType;
    use dojo::sharding::interface::{IShardingDispatcher, IShardingDispatcherTrait};
    use dojo::sharding::planner::{
        collect_dynamic_member_locks, collect_shardable_slots, is_dynamic_layout, plan_model_slots,
        shard_crdt_type,
    };
    use dojo::sharding::request::{CRDVariant, ShardModel};
    use dojo::sharding::slot::{
        DYNAMIC_MEMBER_CHANGED_HASH_SEED, compute_dynamic_member_changed_data_slot,
        compute_dynamic_member_changed_hash_slot, compute_dynamic_member_changed_len_slot,
        compute_dynamic_member_changed_seen_slot, compute_dynamic_member_lock_slot,
        fold_dynamic_member_changed_hash,
    };

    component!(path: sharding_cpt, storage: sharding, event: ShardingEvent);

    pub const WORLD: felt252 = 0;
    pub const DOJO_INIT_SELECTOR: felt252 = selector!("dojo_init");
    pub const WORLD_VERSION: felt252 = '1.8.0';

    #[derive(Copy, Drop)]
    struct SlotMeta {
        slot: felt252,
        model_selector: felt252,
        entity_id: felt252,
        member_selector: felt252,
    }

    #[derive(Copy, Drop)]
    struct RequestedSlotMeta {
        slot: felt252,
        model_selector: felt252,
        entity_id: felt252,
        member_selector: felt252,
        group_id: felt252,
    }

    #[event]
    #[derive(Drop, starknet::Event)]
    pub enum Event {
        WorldSpawned: WorldSpawned,
        WorldUpgraded: WorldUpgraded,
        NamespaceRegistered: NamespaceRegistered,
        ModelRegistered: ModelRegistered,
        EventRegistered: EventRegistered,
        ContractRegistered: ContractRegistered,
        ExternalContractRegistered: ExternalContractRegistered,
        ExternalContractUpgraded: ExternalContractUpgraded,
        ModelUpgraded: ModelUpgraded,
        EventUpgraded: EventUpgraded,
        ContractUpgraded: ContractUpgraded,
        ContractInitialized: ContractInitialized,
        LibraryRegistered: LibraryRegistered,
        EventEmitted: EventEmitted,
        MetadataUpdate: MetadataUpdate,
        StoreSetRecord: StoreSetRecord,
        StoreUpdateRecord: StoreUpdateRecord,
        StoreUpdateMember: StoreUpdateMember,
        StoreDelRecord: StoreDelRecord,
        WriterUpdated: WriterUpdated,
        OwnerUpdated: OwnerUpdated,
        #[flat]
        ShardingEvent: sharding_cpt::Event,
    }

    #[derive(Drop, starknet::Event)]
    pub struct WorldSpawned {
        pub creator: ContractAddress,
        pub class_hash: ClassHash,
    }

    #[derive(Drop, starknet::Event)]
    pub struct WorldUpgraded {
        pub class_hash: ClassHash,
    }

    #[derive(Drop, starknet::Event)]
    pub struct ContractRegistered {
        #[key]
        pub name: ByteArray,
        #[key]
        pub namespace: ByteArray,
        pub address: ContractAddress,
        pub class_hash: ClassHash,
        pub salt: felt252,
    }

    #[derive(Drop, starknet::Event)]
    pub struct ContractUpgraded {
        #[key]
        pub selector: felt252,
        pub class_hash: ClassHash,
    }

    #[derive(Drop, starknet::Event)]
    pub struct ExternalContractRegistered {
        #[key]
        pub namespace: ByteArray,
        #[key]
        pub contract_name: ByteArray,
        #[key]
        pub instance_name: ByteArray,
        #[key]
        pub contract_selector: felt252,
        pub class_hash: ClassHash,
        pub contract_address: ContractAddress,
        pub block_number: u64,
    }

    #[derive(Drop, starknet::Event)]
    pub struct ExternalContractUpgraded {
        #[key]
        pub namespace: ByteArray,
        #[key]
        pub instance_name: ByteArray,
        #[key]
        pub contract_selector: felt252,
        pub class_hash: ClassHash,
        pub contract_address: ContractAddress,
        pub block_number: u64,
    }

    #[derive(Drop, starknet::Event)]
    pub struct LibraryRegistered {
        #[key]
        pub name: ByteArray,
        #[key]
        pub namespace: ByteArray,
        pub class_hash: ClassHash,
    }

    #[derive(Drop, starknet::Event)]
    pub struct MetadataUpdate {
        #[key]
        pub resource: felt252,
        pub uri: ByteArray,
        pub hash: felt252,
    }

    #[derive(Drop, starknet::Event)]
    pub struct NamespaceRegistered {
        #[key]
        pub namespace: ByteArray,
        pub hash: felt252,
    }

    #[derive(Drop, starknet::Event)]
    pub struct ModelRegistered {
        #[key]
        pub name: ByteArray,
        #[key]
        pub namespace: ByteArray,
        pub class_hash: ClassHash,
        pub address: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct ModelUpgraded {
        #[key]
        pub selector: felt252,
        pub class_hash: ClassHash,
        pub address: ContractAddress,
        pub prev_address: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct EventRegistered {
        #[key]
        pub name: ByteArray,
        #[key]
        pub namespace: ByteArray,
        pub class_hash: ClassHash,
        pub address: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct EventUpgraded {
        #[key]
        pub selector: felt252,
        pub class_hash: ClassHash,
        pub address: ContractAddress,
        pub prev_address: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct StoreSetRecord {
        #[key]
        pub selector: felt252,
        #[key]
        pub entity_id: felt252,
        pub keys: Span<felt252>,
        pub values: Span<felt252>,
    }

    #[derive(Drop, starknet::Event)]
    pub struct StoreUpdateRecord {
        #[key]
        pub selector: felt252,
        #[key]
        pub entity_id: felt252,
        pub values: Span<felt252>,
    }

    #[derive(Drop, starknet::Event)]
    pub struct StoreUpdateMember {
        #[key]
        pub selector: felt252,
        #[key]
        pub entity_id: felt252,
        #[key]
        pub member_selector: felt252,
        pub values: Span<felt252>,
    }

    #[derive(Drop, starknet::Event)]
    pub struct StoreDelRecord {
        #[key]
        pub selector: felt252,
        #[key]
        pub entity_id: felt252,
    }

    #[derive(Drop, starknet::Event)]
    pub struct WriterUpdated {
        #[key]
        pub resource: felt252,
        #[key]
        pub contract: ContractAddress,
        pub value: bool,
    }

    #[derive(Drop, starknet::Event)]
    pub struct OwnerUpdated {
        #[key]
        pub resource: felt252,
        #[key]
        pub contract: ContractAddress,
        pub value: bool,
    }

    #[derive(Drop, starknet::Event)]
    pub struct ContractInitialized {
        #[key]
        pub selector: felt252,
        pub init_calldata: Span<felt252>,
    }

    #[derive(Drop, starknet::Event)]
    pub struct EventEmitted {
        #[key]
        pub selector: felt252,
        #[key]
        pub system_address: ContractAddress,
        pub keys: Span<felt252>,
        pub values: Span<felt252>,
    }

    #[storage]
    struct Storage {
        nonce: usize,
        models_salt: usize,
        events_salt: usize,
        resources: Map<felt252, Resource>,
        owners: Map<(felt252, ContractAddress), bool>,
        writers: Map<(felt252, ContractAddress), bool>,
        owner_count: Map<felt252, u64>,
        initialized_contracts: Map<felt252, bool>,
        #[substorage(v0)]
        sharding: sharding_cpt::Storage,
    }

    /// Constructor for the world contract.
    ///
    /// The account originating the transaction is the creator of the world.
    #[constructor]
    fn constructor(ref self: ContractState) {
        let creator = starknet::get_tx_info().unbox().account_contract_address;

        let (internal_ns, internal_ns_hash) = self.world_internal_namespace();

        self.resources.write(internal_ns_hash, Resource::Namespace(internal_ns));
        self.write_ownership(internal_ns_hash, creator, true);

        self.resources.write(WORLD, Resource::World);
        self.write_ownership(WORLD, creator, true);

        // This model doesn't need to have the class hash or the contract address
        // set since they are manually controlled by the world contract.
        self
            .resources
            .write(
                metadata::resource_metadata_selector(internal_ns_hash),
                Resource::Model((default_address(), default_class_hash().into())),
            );

        // It is valid to call the syscall to get the class hash of the deploying contract,
        // even in the constructor.
        let world_class_hash = starknet::syscalls::get_class_hash_at_syscall(
            starknet::get_contract_address(),
        )
            .unwrap();

        self.emit(WorldSpawned { creator, class_hash: world_class_hash });
    }

    #[cfg(target: "test")]
    #[abi(embed_v0)]
    impl WorldTestImpl of dojo::world::IWorldTest<ContractState> {
        fn set_entity_test(
            ref self: ContractState,
            model_selector: felt252,
            index: ModelIndex,
            values: Span<felt252>,
            layout: Layout,
        ) {
            self.set_entity_internal(model_selector, index, values, layout);
        }

        fn delete_entity_test(
            ref self: ContractState, model_selector: felt252, index: ModelIndex, layout: Layout,
        ) {
            self.delete_entity_internal(model_selector, index, layout);
        }

        fn emit_event_test(
            ref self: ContractState,
            event_selector: felt252,
            keys: Span<felt252>,
            values: Span<felt252>,
        ) {
            self
                .emit(
                    EventEmitted {
                        selector: event_selector,
                        system_address: get_caller_address(),
                        keys,
                        values,
                    },
                );
        }

        fn dojo_contract_address(
            self: @ContractState, contract_selector: felt252,
        ) -> ContractAddress {
            match self.resources.read(contract_selector) {
                Resource::Contract((a, _)) => a,
                Resource::ExternalContract((a, _)) => a,
                _ => core::panics::panic_with_byte_array(
                    @format!("Contract/ExternalContract not registered: {}", contract_selector),
                ),
            }
        }
    }

    #[abi(embed_v0)]
    impl World of IWorld<ContractState> {
        fn world_version(self: @ContractState) -> felt252 {
            WORLD_VERSION
        }

        fn metadata(self: @ContractState, resource_selector: felt252) -> ResourceMetadata {
            let (_, internal_ns_hash) = self.world_internal_namespace();

            let mut values = storage::entity_model::read_model_entity(
                metadata::resource_metadata_selector(internal_ns_hash),
                entity_id_from_serialized_keys([resource_selector].span()),
                Model::<ResourceMetadata>::layout(),
            );

            let mut keys = [resource_selector].span();

            match Model::<ResourceMetadata>::from_serialized(keys, values) {
                Option::Some(x) => x,
                Option::None => panic!("Model `ResourceMetadata`: deserialization failed."),
            }
        }

        fn set_metadata(ref self: ContractState, metadata: ResourceMetadata) {
            self.assert_caller_permissions(metadata.resource_id, Permission::Owner);

            let (_, internal_ns_hash) = self.world_internal_namespace();

            storage::entity_model::write_model_entity(
                metadata::resource_metadata_selector(internal_ns_hash),
                entity_id_from_serialized_keys([metadata.resource_id].span()),
                metadata.serialized_values(),
                Model::<ResourceMetadata>::layout(),
            );

            self
                .emit(
                    MetadataUpdate {
                        resource: metadata.resource_id,
                        uri: metadata.metadata_uri,
                        hash: metadata.metadata_hash,
                    },
                );
        }

        fn is_owner(self: @ContractState, resource: felt252, address: ContractAddress) -> bool {
            self.owners.read((resource, address))
        }

        fn grant_owner(ref self: ContractState, resource: felt252, address: ContractAddress) {
            if self.resources.read(resource).is_unregistered() {
                panic_with_byte_array(@errors::resource_not_registered(resource));
            }

            self.assert_caller_permissions(resource, Permission::Owner);

            self.write_ownership(resource, address, true);

            self.emit(OwnerUpdated { contract: address, resource, value: true });
        }

        fn revoke_owner(ref self: ContractState, resource: felt252, address: ContractAddress) {
            if self.resources.read(resource).is_unregistered() {
                panic_with_byte_array(@errors::resource_not_registered(resource));
            }

            self.assert_caller_permissions(resource, Permission::Owner);

            self.write_ownership(resource, address, false);

            self.emit(OwnerUpdated { contract: address, resource, value: false });
        }

        fn owners_count(self: @ContractState, resource: felt252) -> u64 {
            self.owner_count.read(resource)
        }

        fn is_writer(self: @ContractState, resource: felt252, contract: ContractAddress) -> bool {
            self.writers.read((resource, contract))
        }

        fn grant_writer(ref self: ContractState, resource: felt252, contract: ContractAddress) {
            if self.resources.read(resource).is_unregistered() {
                panic_with_byte_array(@errors::resource_not_registered(resource));
            }

            self.assert_caller_permissions(resource, Permission::Owner);

            self.writers.write((resource, contract), true);

            self.emit(WriterUpdated { resource, contract, value: true });
        }

        fn revoke_writer(ref self: ContractState, resource: felt252, contract: ContractAddress) {
            if self.resources.read(resource).is_unregistered() {
                panic_with_byte_array(@errors::resource_not_registered(resource));
            }

            self.assert_caller_permissions(resource, Permission::Owner);

            self.writers.write((resource, contract), false);

            self.emit(WriterUpdated { resource, contract, value: false });
        }

        fn register_event(ref self: ContractState, namespace: ByteArray, class_hash: ClassHash) {
            let caller = get_caller_address();
            let salt = self.events_salt.read();

            let namespace_hash = bytearray_hash(@namespace);

            let (contract_address, _) = starknet::syscalls::deploy_syscall(
                class_hash, salt.into(), [].span(), false,
            )
                .unwrap_syscall();
            self.events_salt.write(salt + 1);

            let event = IDeployedResourceDispatcher { contract_address };
            let event_name = event.dojo_name();

            self.assert_name(@event_name);

            let event_selector = selector_from_namespace_and_name(namespace_hash, @event_name);

            if !self.is_namespace_registered(namespace_hash) {
                panic_with_byte_array(@errors::namespace_not_registered(@namespace));
            }

            self.assert_caller_permissions(namespace_hash, Permission::Owner);

            let maybe_existing_event = self.resources.read(event_selector);
            if !maybe_existing_event.is_unregistered() {
                panic_with_byte_array(@errors::event_already_registered(@namespace, @event_name));
            }

            self
                .resources
                .write(event_selector, Resource::Event((contract_address, namespace_hash)));
            self.write_ownership(event_selector, caller, true);

            self
                .emit(
                    EventRegistered {
                        name: event_name.clone(),
                        namespace: namespace.clone(),
                        address: contract_address,
                        class_hash,
                    },
                );
        }

        fn upgrade_event(ref self: ContractState, namespace: ByteArray, class_hash: ClassHash) {
            let salt = self.events_salt.read();

            let (new_contract_address, _) = starknet::syscalls::deploy_syscall(
                class_hash, salt.into(), [].span(), false,
            )
                .unwrap_syscall();

            self.events_salt.write(salt + 1);

            let namespace_hash = bytearray_hash(@namespace);

            let event = IDeployedResourceDispatcher { contract_address: new_contract_address };
            let event_name = event.dojo_name();
            let event_selector = selector_from_namespace_and_name(namespace_hash, @event_name);

            if !self.is_namespace_registered(namespace_hash) {
                panic_with_byte_array(@errors::namespace_not_registered(@namespace));
            }

            self.assert_caller_permissions(event_selector, Permission::Owner);

            let mut prev_address = core::num::traits::Zero::<ContractAddress>::zero();

            // If the namespace or name of the event have been changed, the selector
            // will be different, hence not upgradeable.
            match self.resources.read(event_selector) {
                Resource::Event((model_address, _)) => { prev_address = model_address; },
                Resource::Unregistered => {
                    panic_with_byte_array(
                        @errors::resource_not_registered_details(@namespace, @event_name),
                    )
                },
                _ => panic_with_byte_array(
                    @errors::resource_conflict(
                        @format!("{}-{}", @namespace, @event_name), @"event",
                    ),
                ),
            }

            self
                .assert_resource_upgradability(
                    @namespace, @event_name, prev_address, new_contract_address,
                );

            self
                .resources
                .write(event_selector, Resource::Event((new_contract_address, namespace_hash)));

            self
                .emit(
                    EventUpgraded {
                        selector: event_selector,
                        prev_address,
                        address: new_contract_address,
                        class_hash,
                    },
                );
        }

        fn register_model(ref self: ContractState, namespace: ByteArray, class_hash: ClassHash) {
            let caller = get_caller_address();
            let salt = self.models_salt.read();

            let namespace_hash = bytearray_hash(@namespace);

            let (contract_address, _) = starknet::syscalls::deploy_syscall(
                class_hash, salt.into(), [].span(), false,
            )
                .unwrap_syscall();
            self.models_salt.write(salt + 1);

            let model = IDeployedResourceDispatcher { contract_address };
            let model_name = model.dojo_name();

            self.assert_name(@model_name);

            let model_selector = selector_from_namespace_and_name(namespace_hash, @model_name);

            if !self.is_namespace_registered(namespace_hash) {
                panic_with_byte_array(@errors::namespace_not_registered(@namespace));
            }

            self.assert_caller_permissions(namespace_hash, Permission::Owner);

            let maybe_existing_model = self.resources.read(model_selector);
            if !maybe_existing_model.is_unregistered() {
                panic_with_byte_array(@errors::model_already_registered(@namespace, @model_name));
            }

            self
                .resources
                .write(model_selector, Resource::Model((contract_address, namespace_hash)));
            self.write_ownership(model_selector, caller, true);

            self
                .emit(
                    ModelRegistered {
                        name: model_name.clone(),
                        namespace: namespace.clone(),
                        address: contract_address,
                        class_hash,
                    },
                );
        }

        fn upgrade_model(ref self: ContractState, namespace: ByteArray, class_hash: ClassHash) {
            let salt = self.models_salt.read();

            let (new_contract_address, _) = starknet::syscalls::deploy_syscall(
                class_hash, salt.into(), [].span(), false,
            )
                .unwrap_syscall();

            self.models_salt.write(salt + 1);

            let namespace_hash = bytearray_hash(@namespace);

            let model = IDeployedResourceDispatcher { contract_address: new_contract_address };
            let model_name = model.dojo_name();
            let model_selector = selector_from_namespace_and_name(namespace_hash, @model_name);

            if !self.is_namespace_registered(namespace_hash) {
                panic_with_byte_array(@errors::namespace_not_registered(@namespace));
            }

            self.assert_caller_permissions(model_selector, Permission::Owner);

            let mut prev_address = core::num::traits::Zero::<ContractAddress>::zero();

            // If the namespace or name of the model have been changed, the selector
            // will be different, hence detected as not registered as model.
            match self.resources.read(model_selector) {
                Resource::Model((model_address, _)) => { prev_address = model_address; },
                Resource::Unregistered => {
                    panic_with_byte_array(
                        @errors::resource_not_registered_details(@namespace, @model_name),
                    )
                },
                _ => panic_with_byte_array(
                    @errors::resource_conflict(
                        @format!("{}-{}", @namespace, @model_name), @"model",
                    ),
                ),
            }

            self
                .assert_resource_upgradability(
                    @namespace, @model_name, prev_address, new_contract_address,
                );

            self
                .resources
                .write(model_selector, Resource::Model((new_contract_address, namespace_hash)));

            self
                .emit(
                    ModelUpgraded {
                        selector: model_selector,
                        prev_address,
                        address: new_contract_address,
                        class_hash,
                    },
                );
        }

        fn register_namespace(ref self: ContractState, namespace: ByteArray) {
            self.assert_namespace(@namespace);

            let caller = get_caller_address();

            let hash = bytearray_hash(@namespace);

            match self.resources.read(hash) {
                Resource::Namespace(_) => panic_with_byte_array(
                    @errors::namespace_already_registered(@namespace),
                ),
                Resource::Unregistered => {
                    self.resources.write(hash, Resource::Namespace(namespace.clone()));
                    self.write_ownership(hash, caller, true);

                    self.emit(NamespaceRegistered { namespace, hash });
                },
                _ => {
                    panic_with_byte_array(@errors::resource_conflict(@namespace, @"namespace"));
                },
            };
        }

        fn register_contract(
            ref self: ContractState, salt: felt252, namespace: ByteArray, class_hash: ClassHash,
        ) -> ContractAddress {
            let caller = get_caller_address();

            let (contract_address, _) = deploy_syscall(class_hash, salt, [].span(), false)
                .unwrap_syscall();

            let namespace_hash = bytearray_hash(@namespace);

            let contract = IDeployedResourceDispatcher { contract_address };
            let contract_name = contract.dojo_name();
            let contract_selector = selector_from_namespace_and_name(
                namespace_hash, @contract_name,
            );

            self.assert_name(@contract_name);

            let maybe_existing_contract = self.resources.read(contract_selector);
            if !maybe_existing_contract.is_unregistered() {
                panic_with_byte_array(
                    @errors::contract_already_registered(@namespace, @contract_name),
                );
            }

            if !self.is_namespace_registered(namespace_hash) {
                panic_with_byte_array(@errors::namespace_not_registered(@namespace));
            }

            self.assert_caller_permissions(namespace_hash, Permission::Owner);

            self.write_ownership(contract_selector, caller, true);
            self
                .resources
                .write(contract_selector, Resource::Contract((contract_address, namespace_hash)));

            self
                .emit(
                    ContractRegistered {
                        salt, class_hash, address: contract_address, namespace, name: contract_name,
                    },
                );

            contract_address
        }

        fn upgrade_contract(
            ref self: ContractState, namespace: ByteArray, class_hash: ClassHash,
        ) -> ClassHash {
            // Only contracts use an external salt during registration. To ensure the
            // upgrade can also be done into a multicall, we combine the transaction hash
            // and the namespace hash since we can't have the same class hash registered more than
            // once in the same namespace.
            let salt = core::poseidon::poseidon_hash_span(
                [
                    starknet::get_tx_info().unbox().transaction_hash,
                    dojo::utils::bytearray_hash(@namespace),
                ]
                    .span(),
            );

            let (new_contract_address, _) = deploy_syscall(class_hash, salt, [].span(), false)
                .unwrap_syscall();

            let namespace_hash = bytearray_hash(@namespace);

            let contract = IDeployedResourceDispatcher { contract_address: new_contract_address };
            let contract_name = contract.dojo_name();
            let contract_selector = selector_from_namespace_and_name(
                namespace_hash, @contract_name,
            );

            // If namespace and name are the same, the contract is already registered and we
            // can upgrade it.
            match self.resources.read(contract_selector) {
                Resource::Contract((
                    contract_address, _,
                )) => {
                    self.assert_caller_permissions(contract_selector, Permission::Owner);

                    IUpgradeableDispatcher { contract_address }.upgrade(class_hash);
                    self.emit(ContractUpgraded { class_hash, selector: contract_selector });

                    class_hash
                },
                Resource::Unregistered => {
                    panic_with_byte_array(
                        @errors::resource_not_registered_details(@namespace, @contract_name),
                    )
                },
                _ => panic_with_byte_array(
                    @errors::resource_conflict(
                        @format!("{}-{}", @namespace, @contract_name), @"contract",
                    ),
                ),
            }
            // class_hash will be retrieved with get_class_hash_at_syscall, so no need to update
        // resource.
        }

        fn init_contract(ref self: ContractState, selector: felt252, init_calldata: Span<felt252>) {
            if let Resource::Contract((contract_address, _)) = self.resources.read(selector) {
                if self.initialized_contracts.read(selector) {
                    let dispatcher = IDeployedResourceDispatcher { contract_address };
                    panic_with_byte_array(
                        @errors::contract_already_initialized(@dispatcher.dojo_name()),
                    );
                } else {
                    self.assert_caller_permissions(selector, Permission::Owner);

                    // For the init, to ensure only the world can call the init function,
                    // the verification is done in the init function of the contract that is
                    // injected by the plugin.
                    // <crates/compiler/src/plugin/attribute_macros/contract.rs#L275>
                    call_contract_syscall(contract_address, DOJO_INIT_SELECTOR, init_calldata)
                        .unwrap_syscall();

                    self.initialized_contracts.write(selector, true);

                    self.emit(ContractInitialized { selector, init_calldata });
                }
            } else {
                panic_with_byte_array(
                    @errors::resource_conflict(@format!("{selector}"), @"contract"),
                );
            }
        }

        fn register_external_contract(
            ref self: ContractState,
            namespace: ByteArray,
            contract_name: ByteArray,
            instance_name: ByteArray,
            contract_address: ContractAddress,
            block_number: u64,
        ) {
            let caller = get_caller_address();
            let class_hash = get_class_hash_at_syscall(contract_address).unwrap_syscall();

            self.assert_name(@instance_name);

            let namespace_hash = bytearray_hash(@namespace);
            let contract_selector = selector_from_namespace_and_name(
                namespace_hash, @instance_name,
            );

            let maybe_existing_contract = self.resources.read(contract_selector);
            if !maybe_existing_contract.is_unregistered() {
                panic_with_byte_array(
                    @errors::external_contract_already_registered(
                        @namespace, @contract_name, @instance_name,
                    ),
                );
            }

            if !self.is_namespace_registered(namespace_hash) {
                panic_with_byte_array(@errors::namespace_not_registered(@namespace));
            }

            self.assert_caller_permissions(namespace_hash, Permission::Owner);

            self.write_ownership(contract_selector, caller, true);
            self
                .resources
                .write(
                    contract_selector,
                    Resource::ExternalContract((contract_address, namespace_hash)),
                );

            self
                .emit(
                    ExternalContractRegistered {
                        namespace,
                        contract_name,
                        instance_name,
                        contract_selector,
                        class_hash,
                        contract_address,
                        block_number,
                    },
                );
        }

        fn upgrade_external_contract(
            ref self: ContractState,
            namespace: ByteArray,
            instance_name: ByteArray,
            contract_address: ContractAddress,
            block_number: u64,
        ) {
            let class_hash = get_class_hash_at_syscall(contract_address).unwrap_syscall();

            let namespace_hash = bytearray_hash(@namespace);
            let contract_selector = selector_from_namespace_and_name(
                namespace_hash, @instance_name,
            );

            match self.resources.read(contract_selector) {
                Resource::ExternalContract(_) => {
                    self.assert_caller_permissions(contract_selector, Permission::Owner);

                    self
                        .resources
                        .write(
                            contract_selector,
                            Resource::ExternalContract((contract_address, namespace_hash)),
                        );

                    self
                        .emit(
                            ExternalContractUpgraded {
                                namespace,
                                instance_name,
                                contract_selector,
                                class_hash,
                                contract_address,
                                block_number,
                            },
                        );
                },
                Resource::Unregistered => panic_with_byte_array(
                    @errors::resource_not_registered_details(@namespace, @instance_name),
                ),
                _ => panic_with_byte_array(
                    @errors::resource_conflict(
                        @format!("{}-{}", @namespace, @instance_name), @"external contract",
                    ),
                ),
            }
        }

        fn register_library(
            ref self: ContractState,
            namespace: ByteArray,
            class_hash: ClassHash,
            name: ByteArray,
            version: ByteArray,
        ) -> ClassHash {
            let caller = get_caller_address();

            let namespace_hash = bytearray_hash(@namespace);

            let contract_name = format!("{}_v{}", name, version);
            self.assert_name(@contract_name);

            let contract_selector = selector_from_namespace_and_name(
                namespace_hash, @contract_name,
            );

            let maybe_existing_library = self.resources.read(contract_selector);
            if !maybe_existing_library.is_unregistered() {
                panic_with_byte_array(
                    @errors::library_already_registered(@namespace, @contract_name),
                );
            }

            if !self.is_namespace_registered(namespace_hash) {
                panic_with_byte_array(@errors::namespace_not_registered(@namespace));
            }

            self.assert_caller_permissions(namespace_hash, Permission::Owner);

            self.write_ownership(contract_selector, caller, true);
            self
                .resources
                .write(contract_selector, Resource::Library((class_hash, namespace_hash)));

            self.emit(LibraryRegistered { class_hash, namespace, name: contract_name });

            class_hash
        }

        fn uuid(ref self: ContractState) -> usize {
            let current = self.nonce.read();
            self.nonce.write(current + 1);
            current
        }

        fn emit_event(
            ref self: ContractState,
            event_selector: felt252,
            keys: Span<felt252>,
            values: Span<felt252>,
        ) {
            if let Resource::Event((_, _)) = self.resources.read(event_selector) {
                self.assert_caller_permissions(event_selector, Permission::Writer);

                self
                    .emit(
                        EventEmitted {
                            selector: event_selector,
                            system_address: get_caller_address(),
                            keys,
                            values,
                        },
                    );
            } else {
                panic_with_byte_array(
                    @errors::resource_conflict(@format!("{event_selector}"), @"event"),
                );
            }
        }

        fn emit_events(
            ref self: ContractState,
            event_selector: felt252,
            keys: Span<Span<felt252>>,
            values: Span<Span<felt252>>,
        ) {
            if let Resource::Event((_, _)) = self.resources.read(event_selector) {
                self.assert_caller_permissions(event_selector, Permission::Writer);

                if keys.len() != values.len() {
                    panic_with_byte_array(
                        @errors::lengths_mismatch(@"keys", @"values", @"emit_events"),
                    );
                }

                for i in 0..keys.len() {
                    self
                        .emit(
                            EventEmitted {
                                selector: event_selector,
                                system_address: get_caller_address(),
                                keys: *keys[i],
                                values: *values[i],
                            },
                        );
                }
            } else {
                panic_with_byte_array(
                    @errors::resource_conflict(@format!("{event_selector}"), @"event"),
                );
            }
        }

        fn entity(
            self: @ContractState, model_selector: felt252, index: ModelIndex, layout: Layout,
        ) -> Span<felt252> {
            self.get_entity_internal(model_selector, index, layout)
        }

        fn entities(
            self: @ContractState,
            model_selector: felt252,
            indexes: Span<ModelIndex>,
            layout: Layout,
        ) -> Span<Span<felt252>> {
            let mut models: Array<Span<felt252>> = array![];

            for i in indexes {
                models.append(self.get_entity_internal(model_selector, *i, layout));
            }

            models.span()
        }

        fn set_entity(
            ref self: ContractState,
            model_selector: felt252,
            index: ModelIndex,
            values: Span<felt252>,
            layout: Layout,
        ) {
            if let Resource::Model((_, _)) = self.resources.read(model_selector) {
                self.assert_caller_permissions(model_selector, Permission::Writer);
                self.set_entity_internal(model_selector, index, values, layout);
            } else {
                panic_with_byte_array(
                    @errors::resource_conflict(@format!("{model_selector}"), @"model"),
                );
            }
        }

        fn set_entities(
            ref self: ContractState,
            model_selector: felt252,
            indexes: Span<ModelIndex>,
            values: Span<Span<felt252>>,
            layout: Layout,
        ) {
            if indexes.len() != values.len() {
                panic_with_byte_array(
                    @errors::lengths_mismatch(@"indexes", @"values", @"set_entities"),
                );
            }

            if let Resource::Model((_, _)) = self.resources.read(model_selector) {
                self.assert_caller_permissions(model_selector, Permission::Writer);

                for i in 0..indexes.len() {
                    self.set_entity_internal(model_selector, *indexes[i], *values[i], layout);
                };
            } else {
                panic_with_byte_array(
                    @errors::resource_conflict(@format!("{model_selector}"), @"model"),
                );
            }
        }

        fn delete_entity(
            ref self: ContractState, model_selector: felt252, index: ModelIndex, layout: Layout,
        ) {
            if let Resource::Model((_, _)) = self.resources.read(model_selector) {
                self.assert_caller_permissions(model_selector, Permission::Writer);
                self.delete_entity_internal(model_selector, index, layout);
            } else {
                panic_with_byte_array(
                    @errors::resource_conflict(@format!("{model_selector}"), @"model"),
                );
            }
        }

        fn delete_entities(
            ref self: ContractState,
            model_selector: felt252,
            indexes: Span<ModelIndex>,
            layout: Layout,
        ) {
            if let Resource::Model((_, _)) = self.resources.read(model_selector) {
                self.assert_caller_permissions(model_selector, Permission::Writer);

                for i in indexes {
                    self.delete_entity_internal(model_selector, *i, layout);
                }
            } else {
                panic_with_byte_array(
                    @errors::resource_conflict(@format!("{model_selector}"), @"model"),
                );
            }
        }

        fn resource(self: @ContractState, selector: felt252) -> Resource {
            self.resources.read(selector)
        }

        fn request_sharding(
            ref self: ContractState, proxy: ContractAddress, models: Span<ShardModel>,
        ) {
            assert(models.len() != 0, 'request_sharding: empty models');

            let world_addr = starknet::get_contract_address();
            let mut all_slots: Array<CRDType> = ArrayTrait::new();
            let mut slot_metas: Array<RequestedSlotMeta> = ArrayTrait::new();
            let mut seen_slots: Felt252Dict<felt252> = Default::default();
            let mut cleared_entities: Felt252Dict<felt252> = Default::default();
            let request_exclusive_group_id = self.sharding.next_exclusive_group_id();

            for model in models {
                let model = *model;
                assert(model.fields.len() != 0, 'request_sharding: empty fields');

                // Caller must have writer permission for this model.
                self.assert_caller_permissions(model.selector, Permission::Writer);

                // Verify model is registered and read canonical onchain layout.
                let model_layout = self.read_model_layout_or_panic(model.selector);

                // Compute entity_id from keys.
                let entity_id = entity_id_from_serialized_keys(model.keys);
                self.sharding.store_entity_keys(entity_id, model.keys);

                let planned_slots = plan_model_slots(
                    model.selector, entity_id, model_layout, model.fields, model.coverage,
                );
                for planned in planned_slots.span() {
                    let planned = *planned;
                    assert(
                        Felt252DictTrait::get(ref seen_slots, planned.slot) == 0,
                        sharding_cpt::Errors::DUPLICATE_SLOT,
                    );
                    Felt252DictTrait::insert(ref seen_slots, planned.slot, 1);

                    let (prev_model, prev_entity, _) = self.sharding.read_slot_metadata(planned.slot);
                    if prev_model != 0 && !self.sharding.is_slot_active(planned.slot) {
                        self.sharding.clear_slot_metadata(planned.slot);
                        self.clear_entity_keys_if_inactive(prev_entity, ref cleared_entities);
                    }

                    if planned.crdt == CRDVariant::SetLock {
                        if let Layout::Struct(_) = model_layout {
                            let member_layout = self.read_member_layout_or_panic(
                                model.selector, planned.member_selector,
                            );
                            if is_dynamic_layout(member_layout) {
                                self.clear_dynamic_member_changed_slots(planned.slot);
                            }
                        }
                    }

                    let group_id = match planned.crdt {
                        CRDVariant::SetLock | CRDVariant::Lock => request_exclusive_group_id,
                        _ => 0,
                    };
                    slot_metas.append(
                        RequestedSlotMeta {
                            slot: planned.slot,
                            model_selector: model.selector,
                            entity_id,
                            member_selector: planned.member_selector,
                            group_id,
                        },
                    );
                    all_slots.append(shard_crdt_type(planned.crdt, world_addr, planned.slot));
                };
            };

            assert(all_slots.len() != 0, 'Shard: no slots');
            self.sharding.initialize_shard(proxy, all_slots.span());
            for slot_meta in slot_metas.span() {
                let slot_meta = *slot_meta;
                self.sharding.store_slot_metadata(
                    slot_meta.slot,
                    slot_meta.model_selector,
                    slot_meta.entity_id,
                    slot_meta.member_selector,
                    slot_meta.group_id,
                );
            };
        }

        fn end_shard(ref self: ContractState) {
            let caller = get_caller_address();
            let proxy = self.sharding.sharding_contract();
            assert(!proxy.is_zero(), sharding_cpt::Errors::NOT_INITIALIZED);
            let shard_id = IShardingDispatcher { contract_address: proxy }
                .get_shard_id(starknet::get_contract_address());
            let initiator = self.sharding.end_shard_initiator(shard_id);
            assert(
                caller == initiator || self.is_caller_world_owner(),
                sharding_cpt::Errors::UNAUTHORIZED_CALLER,
            );
            self.sharding.end_shard();
        }
    }

    #[abi(embed_v0)]
    impl UpgradeableWorld of IUpgradeableWorld<ContractState> {
        fn upgrade(ref self: ContractState, new_class_hash: ClassHash) {
            assert(new_class_hash.is_non_zero(), 'invalid class_hash');

            if !self.is_caller_world_owner() {
                panic_with_byte_array(@errors::not_owner_upgrade(get_caller_address(), WORLD));
            }

            replace_class_syscall(new_class_hash).unwrap();

            self.emit(WorldUpgraded { class_hash: new_class_hash });
        }
    }

    // Instantiate the component impls (not ABI-exposed) so self.sharding.xxx() works.
    impl ShardingComponentImpl = sharding_cpt::ContractComponentImpl<ContractState>;
    impl ShardingMetadataImpl = sharding_cpt::MetadataImpl<ContractState>;

    // ── Sharding ABI (feature-gated) ────────────────────────────────────
    //
    // The impls below are only exposed when the `sharding` feature is
    // enabled.  Non-sharding worlds never compile these entry points.

    /// Sharding component methods callable by the sharding proxy (operator's contract).
    /// Access control is enforced inside the component: caller must equal the
    /// `sharding_contract_address` set when `request_sharding` was first called.
    #[cfg(feature: 'sharding')]
    #[abi(embed_v0)]
    impl ShardingProxyImpl of dojo::world::world_sharding::IShardingProxy<ContractState> {
        fn settle_shard_changes(
            ref self: ContractState,
            shard_id: felt252,
            slot_changes: Array<(felt252, felt252)>,
            dynamic_members: Span<ShardDynamicMemberChanges>,
            dynamic_changes: Span<(felt252, felt252)>,
            dynamic_tracking_proofs: Span<(felt252, felt252)>,
        ) {
            assert(
                slot_changes.len() != 0 || dynamic_members.len() != 0,
                'Shard settle: empty',
            );
            assert(
                get_caller_address() == self.sharding.sharding_contract(),
                sharding_cpt::Errors::UNAUTHORIZED_CALLER,
            );
            self.assert_dynamic_changes_exact_coverage(dynamic_members, dynamic_changes);
            self.assert_dynamic_tracking_proofs_exact_coverage(
                dynamic_members, dynamic_tracking_proofs,
            );

            let slot_changes_span = slot_changes.span();

            let (requested_unlock_slots, mut slot_metas) = self
                .collect_settlement_unlock_slots_and_metas(
                    shard_id, slot_changes_span, dynamic_members,
                );
            self.sharding.assert_exclusive_group_full_coverage(requested_unlock_slots.span());

            if slot_changes.len() != 0 {
                self.sharding.settle_slot_changes(shard_id, slot_changes);
            }

            let dynamic_slots_to_unlock = self.apply_dynamic_member_changes(
                shard_id,
                slot_changes_span,
                dynamic_members,
                dynamic_changes,
                dynamic_tracking_proofs,
                ref slot_metas,
            );

            if dynamic_slots_to_unlock.len() != 0 {
                self.sharding.cancel_shard_state(shard_id, dynamic_slots_to_unlock.span());
            }

            self.emit_settlement_events_and_cleanup(slot_metas.span());
        }

        fn cancel_shard_state(ref self: ContractState, shard_id: felt252, slots: Span<felt252>) {
            assert(
                get_caller_address() == self.sharding.sharding_contract(),
                sharding_cpt::Errors::UNAUTHORIZED_CALLER,
            );

            // Pass 1 (validate): shard membership + dedup + group coverage
            let mut seen_slots: Felt252Dict<felt252> = Default::default();
            let mut seen_groups: Felt252Dict<felt252> = Default::default();
            let mut group_counts: Felt252Dict<felt252> = Default::default();
            let mut groups: Array<felt252> = ArrayTrait::new();

            for slot in slots {
                let slot = *slot;
                assert(
                    self.sharding.is_slot_active_for_shard(shard_id, slot),
                    'Shard cancel: slot not in shard',
                );
                assert(
                    Felt252DictTrait::get(ref seen_slots, slot) == 0,
                    sharding_cpt::Errors::DUPLICATE_SLOT,
                );
                Felt252DictTrait::insert(ref seen_slots, slot, 1);

                let group_id = self.sharding.read_slot_group_id(slot);
                if group_id != 0 {
                    assert(self.sharding.is_slot_active(slot), 'Shard grp: inactive');
                    if Felt252DictTrait::get(ref seen_groups, group_id) == 0 {
                        Felt252DictTrait::insert(ref seen_groups, group_id, 1);
                        groups.append(group_id);
                    }
                    let current = Felt252DictTrait::get(ref group_counts, group_id);
                    Felt252DictTrait::insert(ref group_counts, group_id, current + 1);
                }
            };

            for group_id in groups.span() {
                let group_id = *group_id;
                let expected: felt252 = self.sharding.read_group_active_slot_count(group_id).into();
                let provided = Felt252DictTrait::get(ref group_counts, group_id);
                assert(expected != 0, 'Shard grp: invalid');
                assert(provided == expected, 'Shard grp: partial');
            };

            // Pass 2 (execute): unlock + dynamic cleanup
            let contract_address = starknet::get_contract_address();
            for slot in slots {
                let slot = *slot;
                self.sharding.unlock_slot(shard_id, slot, contract_address);

                let (model_selector, entity_id, member_selector) = self.sharding.read_slot_metadata(slot);
                if model_selector != 0
                    && member_selector != 0
                    && slot
                        == compute_dynamic_member_lock_slot(
                            model_selector, entity_id, member_selector,
                        ) {
                    self.clear_dynamic_member_changed_slots(slot);
                }
            };
        }
    }

    #[cfg(feature: 'sharding')]
    #[abi(embed_v0)]
    impl ShardingMetadataViewImpl of dojo::world::world_sharding::IShardingMetadata<ContractState> {
        fn describe_shard_slots(
            self: @ContractState, slots: Span<felt252>,
        ) -> Span<ShardSlotDescriptor> {
            let mut descriptors: Array<ShardSlotDescriptor> = ArrayTrait::new();
            for slot in slots {
                descriptors.append(self.describe_shard_slot(*slot));
            };
            descriptors.span()
        }

        fn dynamic_member_storage_slots(
            self: @ContractState, model_selector: felt252, entity_id: felt252, member_selector: felt252,
        ) -> Span<felt252> {
            let member_layout = self.read_member_layout_or_panic(model_selector, member_selector);
            assert(is_dynamic_layout(member_layout), 'Shard meta: dynamic only');

            let mut slots: Array<felt252> = ArrayTrait::new();
            self.collect_storage_slots_for_layout(
                model_selector, combine_key(entity_id, member_selector), member_layout, ref slots,
            );
            slots.span()
        }

        fn dynamic_member_changed_slots(
            self: @ContractState, model_selector: felt252, entity_id: felt252, member_selector: felt252,
        ) -> Span<felt252> {
            let member_layout = self.read_member_layout_or_panic(model_selector, member_selector);
            assert(is_dynamic_layout(member_layout), 'Shard meta: dynamic only');

            let lock_slot = compute_dynamic_member_lock_slot(model_selector, entity_id, member_selector);
            self.read_dynamic_member_changed_slots_by_lock(lock_slot)
        }

        fn dynamic_member_tracking_proof_slots(
            self: @ContractState, model_selector: felt252, entity_id: felt252, member_selector: felt252,
        ) -> Span<felt252> {
            let member_layout = self.read_member_layout_or_panic(model_selector, member_selector);
            assert(is_dynamic_layout(member_layout), 'Shard meta: dynamic only');

            let lock_slot = compute_dynamic_member_lock_slot(model_selector, entity_id, member_selector);
            [
                compute_dynamic_member_changed_len_slot(lock_slot),
                compute_dynamic_member_changed_hash_slot(lock_slot),
            ]
                .span()
        }
    }

    // ── Core internal helpers ────────────────────────────────────────────

    #[generate_trait]
    impl SelfImpl of SelfTrait {
        /// Update the ownership status of a resource owner.
        ///
        /// # Arguments
        ///   * `resource` - The selector of the resource.
        ///   * `owner` - The owner address.
        ///   $ `is_owner` - true to set `owner` as a new resource owner,
        ///                  false to remove the `owner` from the resource owners.
        fn write_ownership(
            ref self: ContractState, resource: felt252, owner: ContractAddress, is_owner: bool,
        ) {
            let was_owner = self.owners.read((resource, owner));

            if was_owner != is_owner {
                let new_count = if is_owner {
                    self.owner_count.read(resource) + 1
                } else {
                    self.owner_count.read(resource) - 1
                };

                self.owner_count.write(resource, new_count);
                self.owners.write((resource, owner), is_owner);
            }
        }

        #[inline(always)]
        /// Indicates if the caller is the owner of the world.
        fn is_caller_world_owner(self: @ContractState) -> bool {
            self.is_owner(WORLD, get_caller_address())
        }

        /// Asserts the caller has the required permissions for a resource, following the
        /// permissions hierarchy:
        /// 1. World Owner
        /// 2. Namespace Owner
        /// 3. Resource Owner
        /// [if writer]
        /// 4. Namespace Writer
        /// 5. Resource Writer
        ///
        /// This function is expected to be called very often as it's used to check permissions
        /// for all the resource access in the system.
        /// For this reason, here are the following optimizations:
        ///     * Use several single `if` because it seems more efficient than a big one with
        ///       several conditions based on how cairo is lowered to sierra.
        ///     * Sort conditions by order of probability so once a condition is met, the function
        ///       returns.
        ///
        /// # Arguments
        ///   * `resource_selector` - the selector of the resource.
        ///   * `permission` - the required permission.
        fn assert_caller_permissions(
            self: @ContractState, resource_selector: felt252, permission: Permission,
        ) {
            let caller = get_caller_address();

            if permission == Permission::Writer {
                if self.is_writer(resource_selector, caller) {
                    return;
                }
            }

            if self.is_owner(resource_selector, caller) {
                return;
            }

            if self.is_caller_world_owner() {
                return;
            }

            // At this point, [`Resource::Contract`], [`Resource::ExternalContract`],
            // [`Resource::Model`] and [`Resource::Event`] require extra checks
            // by switching to the namespace hash being the resource selector.
            let namespace_hash = match self.resources.read(resource_selector) {
                Resource::Contract((_, namespace_hash)) => { namespace_hash },
                Resource::ExternalContract((_, namespace_hash)) => { namespace_hash },
                Resource::Model((_, namespace_hash)) => { namespace_hash },
                Resource::Event((_, namespace_hash)) => { namespace_hash },
                Resource::Unregistered => {
                    panic_with_byte_array(@errors::resource_not_registered(resource_selector))
                },
                _ => self.panic_with_details(caller, resource_selector, permission),
            };

            if permission == Permission::Writer {
                if self.is_writer(namespace_hash, caller) {
                    return;
                }
            }

            if self.is_owner(namespace_hash, caller) {
                return;
            }

            self.panic_with_details(caller, resource_selector, permission)
        }

        /// Asserts the name is valid according to the naming convention.
        fn assert_name(self: @ContractState, name: @ByteArray) {
            if !dojo::utils::is_name_valid(name) {
                panic_with_byte_array(@errors::invalid_naming("Name", name))
            }
        }

        /// Asserts the namespace is valid according to the naming convention.
        fn assert_namespace(self: @ContractState, namespace: @ByteArray) {
            if !dojo::utils::is_name_valid(namespace) {
                panic_with_byte_array(@errors::invalid_naming("Namespace", namespace))
            }
        }

        /// Panics if a resource is not upgradable.
        ///
        /// Upgradable means:
        /// - the layout type must remain the same (Struct or Fixed),
        /// - existing fields cannot be changed or moved inside the resource,
        /// - new fields can only be appended at the end of the resource.
        ///
        /// # Arguments
        ///   * `namespace` - the namespace of the resource.
        ///   * `name` - the name of the resource.
        ///   * `prev_address` - the address of the current resource.
        ///   * `new_address` - the address of the newly deployed resource.
        ///
        fn assert_resource_upgradability(
            self: @ContractState,
            namespace: @ByteArray,
            name: @ByteArray,
            prev_address: ContractAddress,
            new_address: ContractAddress,
        ) {
            let resource = IStoredResourceDispatcher { contract_address: prev_address };
            let old_layout = resource.layout();
            let old_schema = resource.schema();

            let new_resource = IStoredResourceDispatcher { contract_address: new_address };
            let new_layout = new_resource.layout();
            let new_schema = new_resource.schema();

            if !new_layout.is_same_type_of(@old_layout) {
                panic_with_byte_array(@errors::invalid_resource_layout_upgrade(namespace, name));
            }

            // The layout of a resource using packed layout must remain the same.
            // It is upgradeable since the class hash may have changed, but no fields have been
            // changed.
            if let Layout::Fixed(_) = new_layout {
                if new_layout != old_layout {
                    panic_with_byte_array(
                        @errors::packed_layout_cannot_be_upgraded(namespace, name),
                    );
                }
            }

            if !new_schema.is_an_upgrade_of(@old_schema) {
                panic_with_byte_array(@errors::invalid_resource_schema_upgrade(namespace, name));
            }
        }

        /// Panics with the caller details.
        ///
        /// # Arguments
        ///   * `caller` - the address of the caller.
        ///   * `resource_selector` - the selector of the resource.
        ///   * `permission` - the required permission.
        fn panic_with_details(
            self: @ContractState,
            caller: ContractAddress,
            resource_selector: felt252,
            permission: Permission,
        ) -> core::never {
            let resource_name = match self.resources.read(resource_selector) {
                Resource::Contract((
                    contract_address, _,
                )) => {
                    let d = IDeployedResourceDispatcher { contract_address };
                    format!("contract (or its namespace) `{}`", d.dojo_name())
                },
                Resource::ExternalContract((
                    contract_address, _,
                )) => { format!("external contract (at 0x{:x})", contract_address) },
                Resource::Event((
                    contract_address, _,
                )) => {
                    let d = IDeployedResourceDispatcher { contract_address };
                    format!("event (or its namespace) `{}`", d.dojo_name())
                },
                Resource::Model((
                    contract_address, _,
                )) => {
                    let d = IDeployedResourceDispatcher { contract_address };
                    format!("model (or its namespace) `{}`", d.dojo_name())
                },
                Resource::Namespace(ns) => { format!("namespace `{}`", ns) },
                Resource::World => { format!("world") },
                Resource::Unregistered => { panic!("Unreachable") },
                Resource::Library((
                    class_hash, _,
                )) => {
                    let d = IDeployedResourceLibraryDispatcher { class_hash };
                    format!("library (or its namespace) `{}`", d.dojo_name())
                },
            };

            let caller_name = if caller == get_tx_info().account_contract_address {
                format!("Account `0x{:x}`", caller)
            } else {
                match call_contract_syscall(caller, selector!("dojo_name"), [].span()) {
                    Result::Ok(mut serialized_name) => {
                        let name = Serde::<ByteArray>::deserialize(ref serialized_name).unwrap();
                        format!("Dojo Contract `{}`", name)
                    },
                    _ => format!("Contract `0x{:x}`", caller),
                }
            };

            panic_with_byte_array(
                @format!("{} does NOT have {} role on {}", caller_name, permission, resource_name),
            )
        }

        // ── Sharding internal helpers ────────────────────────────────────
        //
        // Everything below this marker is sharding-specific.  These
        // functions are only reachable at runtime when `shard_active` is
        // true (i.e. a sharding contract has been registered via
        // `request_sharding`).  When the `sharding` Scarb feature is off,
        // the ABI entry-points that call into these helpers are compiled
        // out, making them unreachable dead code.

        fn describe_shard_slot(self: @ContractState, slot: felt252) -> ShardSlotDescriptor {
            let (model_selector, entity_id, member_selector) = self.sharding.read_slot_metadata(slot);
            let is_dynamic_lock = if model_selector == 0 || member_selector == 0 {
                false
            } else {
                match self.read_model_layout_or_panic(model_selector) {
                    Layout::Struct(_) => {
                        let member_layout = self.read_member_layout_or_panic(
                            model_selector, member_selector,
                        );
                        is_dynamic_layout(member_layout)
                    },
                    Layout::Fixed(_) => false,
                    _ => false,
                }
            };

            ShardSlotDescriptor {
                slot,
                model_selector,
                entity_id,
                member_selector,
                is_dynamic_lock,
            }
        }

        fn read_member_layout_or_panic(
            self: @ContractState, model_selector: felt252, member_selector: felt252,
        ) -> Layout {
            let model_layout = self.read_model_layout_or_panic(model_selector);
            match dojo::utils::find_model_field_layout(model_layout, member_selector) {
                Option::Some(layout) => layout,
                Option::None => panic_with_byte_array(
                    @format!(
                        "Shard metadata: field layout not found for member {}",
                        member_selector,
                    ),
                ),
            }
        }

        fn dojo_storage_key(table: felt252, key: felt252) -> Span<felt252> {
            [DOJO_STORAGE, table, key].span()
        }

        fn append_storage_segment_slots(
            self: @ContractState, storage_key: Span<felt252>, raw_len: u32, ref slots: Array<felt252>,
        ) {
            if raw_len == 0 {
                return;
            }

            let base = storage_base_address_from_felt252(poseidon_hash_span(storage_key));
            let base_address = storage_address_from_base(base);
            let mut chunk: felt252 = 0;
            let mut chunk_base = base;
            let mut index_in_chunk = 0_u8;
            let mut produced: u32 = 0;

            loop {
                let storage_address: StorageAddress =
                    storage_address_from_base_and_offset(chunk_base, index_in_chunk);
                slots.append(storage_address.into());
                produced += 1;
                if produced == raw_len {
                    break;
                }

                dojo::storage::storage::next_index_in_chunk(
                    ref index_in_chunk, ref chunk, ref chunk_base, base_address,
                );
            };
        }

        fn collect_storage_slots_for_layout(
            self: @ContractState,
            model_selector: felt252,
            key: felt252,
            layout: Layout,
            ref slots: Array<felt252>,
        ) {
            match layout {
                Layout::Fixed(bits_layout) => {
                    let mut bits_layout = bits_layout;
                    let packed_size = packing::calculate_packed_size(ref bits_layout);
                    let raw_len: u32 = packed_size.try_into().unwrap();
                    self.append_storage_segment_slots(
                        Self::dojo_storage_key(model_selector, key), raw_len, ref slots,
                    );
                },
                Layout::Struct(field_layouts) => {
                    for field_layout in field_layouts {
                        let field_key = combine_key(key, *field_layout.selector);
                        self.collect_storage_slots_for_layout(
                            model_selector, field_key, *field_layout.layout, ref slots,
                        );
                    };
                },
                Layout::Tuple(item_layouts) => {
                    for (i, item_layout) in item_layouts.into_iter().enumerate() {
                        let item_key = combine_key(key, i.into());
                        self.collect_storage_slots_for_layout(
                            model_selector, item_key, *item_layout, ref slots,
                        );
                    };
                },
                Layout::FixedArray(fixed_array_layout) => {
                    let (item_layouts, array_len): (Span<Layout>, u32) = fixed_array_layout;
                    if item_layouts.len() == 0 {
                        return;
                    }

                    let item_layout = *item_layouts[0];
                    let mut i: u32 = 0;
                    while i < array_len {
                        let item_key = combine_key(key, i.into());
                        self.collect_storage_slots_for_layout(
                            model_selector, item_key, item_layout, ref slots,
                        );
                        i += 1;
                    };
                },
                Layout::Array(item_layouts) => {
                    self.append_storage_segment_slots(
                        Self::dojo_storage_key(model_selector, key), 1, ref slots,
                    );

                    let array_len = dojo::storage::database::get_single(model_selector, key);
                    assert(array_len.into() <= dojo::storage::database::MAX_ARRAY_LENGTH, 'invalid array length');
                    let item_layout = *item_layouts[0];
                    let array_len: u32 = array_len.try_into().unwrap();
                    let mut i: u32 = 0;
                    while i < array_len {
                        let item_key = combine_key(key, i.into());
                        self.collect_storage_slots_for_layout(
                            model_selector, item_key, item_layout, ref slots,
                        );
                        i += 1;
                    };
                },
                Layout::ByteArray => {
                    let data_len = dojo::storage::database::get_single(model_selector, key);
                    assert(
                        data_len.into()
                            <= (dojo::storage::database::MAX_ARRAY_LENGTH - 3_u256),
                        'invalid array length',
                    );
                    let raw_len: u32 = data_len.try_into().unwrap() + 3;
                    self.append_storage_segment_slots(
                        Self::dojo_storage_key(model_selector, key), raw_len, ref slots,
                    );
                },
                Layout::Enum(variant_layouts) => {
                    self.append_storage_segment_slots(
                        Self::dojo_storage_key(model_selector, key), 1, ref slots,
                    );

                    let variant = dojo::storage::database::get_single(model_selector, key);
                    assert(variant.into() < 256_u256, 'invalid variant value');
                    let variant_key = combine_key(key, variant);
                    match dojo::utils::find_field_layout(variant, variant_layouts) {
                        Option::Some(variant_layout) => {
                            self.collect_storage_slots_for_layout(
                                model_selector, variant_key, variant_layout, ref slots,
                            );
                        },
                        Option::None => {
                            if variant != 0 {
                                panic_with_byte_array(
                                    @format!(
                                        "Shard metadata: enum layout not found for variant {}",
                                        variant,
                                    ),
                                );
                            }
                        },
                    }
                },
            }
        }

        fn collect_written_slots_for_layout(
            self: @ContractState,
            model_selector: felt252,
            key: felt252,
            values: Span<felt252>,
            ref offset: u32,
            layout: Layout,
            ref slots: Array<felt252>,
        ) {
            match layout {
                Layout::Fixed(bits_layout) => {
                    let raw_values_len: u32 = bits_layout.len();
                    let mut bits_layout_calc = bits_layout;
                    let packed_size = packing::calculate_packed_size(ref bits_layout_calc);
                    let raw_len: u32 = packed_size.try_into().unwrap();
                    self.append_storage_segment_slots(
                        Self::dojo_storage_key(model_selector, key), raw_len, ref slots,
                    );
                    offset += raw_values_len;
                },
                Layout::Struct(field_layouts) => {
                    for field_layout in field_layouts {
                        let field_key = combine_key(key, *field_layout.selector);
                        self.collect_written_slots_for_layout(
                            model_selector,
                            field_key,
                            values,
                            ref offset,
                            *field_layout.layout,
                            ref slots,
                        );
                    };
                },
                Layout::Tuple(item_layouts) => {
                    for (i, item_layout) in item_layouts.into_iter().enumerate() {
                        let item_key = combine_key(key, i.into());
                        self.collect_written_slots_for_layout(
                            model_selector, item_key, values, ref offset, *item_layout, ref slots,
                        );
                    };
                },
                Layout::FixedArray(fixed_array_layout) => {
                    let (item_layouts, array_len): (Span<Layout>, u32) = fixed_array_layout;
                    if item_layouts.len() == 0 {
                        return;
                    }

                    let item_layout = *item_layouts[0];
                    let mut i: u32 = 0;
                    while i < array_len {
                        let item_key = combine_key(key, i.into());
                        self.collect_written_slots_for_layout(
                            model_selector, item_key, values, ref offset, item_layout, ref slots,
                        );
                        i += 1;
                    };
                },
                Layout::Array(item_layouts) => {
                    assert((values.len() - offset) > 0, 'Invalid values length');
                    self.append_storage_segment_slots(
                        Self::dojo_storage_key(model_selector, key), 1, ref slots,
                    );

                    let array_len = *values.at(offset);
                    assert(
                        array_len.into() <= dojo::storage::database::MAX_ARRAY_LENGTH,
                        'invalid array length',
                    );
                    offset += 1;

                    let item_layout = *item_layouts[0];
                    let array_len: u32 = array_len.try_into().unwrap();
                    let mut i: u32 = 0;
                    while i < array_len {
                        let item_key = combine_key(key, i.into());
                        self.collect_written_slots_for_layout(
                            model_selector, item_key, values, ref offset, item_layout, ref slots,
                        );
                        i += 1;
                    };
                },
                Layout::ByteArray => {
                    assert((values.len() - offset) >= 3, 'Invalid values length');
                    let data_len = *values.at(offset);
                    assert(
                        data_len.into()
                            <= (dojo::storage::database::MAX_ARRAY_LENGTH - 3_u256),
                        'invalid array length',
                    );
                    let raw_len: u32 = data_len.try_into().unwrap() + 3;
                    assert((values.len() - offset) >= raw_len, 'Invalid values length');
                    self.append_storage_segment_slots(
                        Self::dojo_storage_key(model_selector, key), raw_len, ref slots,
                    );
                    offset += raw_len;
                },
                Layout::Enum(variant_layouts) => {
                    assert(values.len() - offset > 0, 'enum values oob');
                    self.append_storage_segment_slots(
                        Self::dojo_storage_key(model_selector, key), 1, ref slots,
                    );

                    let variant = *values.at(offset);
                    assert(variant.into() < 256_u256, 'invalid variant value');
                    offset += 1;

                    let variant_key = combine_key(key, variant);
                    match dojo::utils::find_field_layout(variant, variant_layouts) {
                        Option::Some(variant_layout) => {
                            self.collect_written_slots_for_layout(
                                model_selector,
                                variant_key,
                                values,
                                ref offset,
                                variant_layout,
                                ref slots,
                            );
                        },
                        Option::None => {
                            if variant != 0 {
                                panic_with_byte_array(
                                    @format!(
                                        "Shard metadata: enum layout not found for variant {}",
                                        variant,
                                    ),
                                );
                            }
                        },
                    }
                },
            }
        }

        fn collect_deleted_slots_for_layout(
            self: @ContractState,
            model_selector: felt252,
            key: felt252,
            layout: Layout,
            ref slots: Array<felt252>,
        ) {
            match layout {
                Layout::Fixed(bits_layout) => {
                    let mut bits_layout_calc = bits_layout;
                    let packed_size = packing::calculate_packed_size(ref bits_layout_calc);
                    let raw_len: u32 = packed_size.try_into().unwrap();
                    self.append_storage_segment_slots(
                        Self::dojo_storage_key(model_selector, key), raw_len, ref slots,
                    );
                },
                Layout::Struct(field_layouts) => {
                    for field_layout in field_layouts {
                        self.collect_deleted_slots_for_layout(
                            model_selector,
                            combine_key(key, *field_layout.selector),
                            *field_layout.layout,
                            ref slots,
                        );
                    };
                },
                Layout::Tuple(item_layouts) => {
                    for (i, item_layout) in item_layouts.into_iter().enumerate() {
                        self.collect_deleted_slots_for_layout(
                            model_selector, combine_key(key, i.into()), *item_layout, ref slots,
                        );
                    };
                },
                Layout::FixedArray(fixed_array_layout) => {
                    let (item_layouts, array_len): (Span<Layout>, u32) = fixed_array_layout;
                    if item_layouts.len() == 0 {
                        return;
                    }

                    let item_layout = *item_layouts[0];
                    let mut i: u32 = 0;
                    while i < array_len {
                        self.collect_deleted_slots_for_layout(
                            model_selector, combine_key(key, i.into()), item_layout, ref slots,
                        );
                        i += 1;
                    };
                },
                Layout::Array(item_layouts) => {
                    self.append_storage_segment_slots(
                        Self::dojo_storage_key(model_selector, key), 1, ref slots,
                    );

                    let array_len = dojo::storage::database::get_single(model_selector, key);
                    assert(array_len.into() <= dojo::storage::database::MAX_ARRAY_LENGTH, 'invalid array length');
                    let item_layout = *item_layouts[0];
                    let array_len: u32 = array_len.try_into().unwrap();
                    let mut i: u32 = 0;
                    while i < array_len {
                        let item_key = combine_key(key, i.into());
                        self.collect_deleted_slots_for_layout(
                            model_selector, item_key, item_layout, ref slots,
                        );
                        i += 1;
                    };
                },
                Layout::ByteArray => {
                    let data_len = dojo::storage::database::get_single(model_selector, key);
                    assert(
                        data_len.into()
                            <= (dojo::storage::database::MAX_ARRAY_LENGTH - 3_u256),
                        'invalid array length',
                    );
                    let raw_len: u32 = data_len.try_into().unwrap() + 3;
                    self.append_storage_segment_slots(
                        Self::dojo_storage_key(model_selector, key), raw_len, ref slots,
                    );
                },
                Layout::Enum(variant_layouts) => {
                    self.append_storage_segment_slots(
                        Self::dojo_storage_key(model_selector, key), 1, ref slots,
                    );

                    let variant = dojo::storage::database::get_single(model_selector, key);
                    assert(variant.into() < 256_u256, 'invalid variant value');
                    let variant_key = combine_key(key, variant);
                    match dojo::utils::find_field_layout(variant, variant_layouts) {
                        Option::Some(variant_layout) => {
                            self.collect_deleted_slots_for_layout(
                                model_selector, variant_key, variant_layout, ref slots,
                            );
                        },
                        Option::None => {
                            if variant != 0 {
                                panic_with_byte_array(
                                    @format!(
                                        "Shard metadata: enum layout not found for variant {}",
                                        variant,
                                    ),
                                );
                            }
                        },
                    }
                },
            }
        }

        fn clear_entity_keys_if_inactive(
            ref self: ContractState,
            entity_id: felt252,
            ref cleared_entities: Felt252Dict<felt252>,
        ) {
            if entity_id == 0 {
                return;
            }
            if self.sharding.has_entity_active_slots(entity_id) {
                return;
            }
            if Felt252DictTrait::get(ref cleared_entities, entity_id) != 0 {
                return;
            }

            Felt252DictTrait::insert(ref cleared_entities, entity_id, 1);
            self.sharding.clear_entity_keys(entity_id);
        }

        /// Single pass over slot_changes (validate + read metadata) and dynamic_members
        /// (validate). Returns the full list of unlock slots and the slot metadata
        /// for Torii event emission.
        fn collect_settlement_unlock_slots_and_metas(
            self: @ContractState,
            shard_id: felt252,
            slot_changes: Span<(felt252, felt252)>,
            dynamic_members: Span<ShardDynamicMemberChanges>,
        ) -> (Array<felt252>, Array<SlotMeta>) {
            let mut slots: Array<felt252> = ArrayTrait::new();
            let mut seen_slots: Felt252Dict<felt252> = Default::default();
            let mut slot_metas: Array<SlotMeta> = ArrayTrait::new();

            for entry in slot_changes {
                let (slot, _) = *entry;
                assert(
                    self.sharding.is_slot_active_for_shard(shard_id, slot),
                    'Shard settle: slot not in shard',
                );
                assert(
                    Felt252DictTrait::get(ref seen_slots, slot) == 0,
                    sharding_cpt::Errors::DUPLICATE_SLOT,
                );
                Felt252DictTrait::insert(ref seen_slots, slot, 1);
                slots.append(slot);

                let (model_selector, entity_id, member_selector) = self
                    .sharding
                    .read_slot_metadata(slot);
                if model_selector != 0 {
                    slot_metas.append(
                        SlotMeta { slot, model_selector, entity_id, member_selector },
                    );
                }
            };

            for dynamic_member in dynamic_members {
                let dynamic_member = *dynamic_member;
                let lock_slot = compute_dynamic_member_lock_slot(
                    dynamic_member.model_selector,
                    dynamic_member.entity_id,
                    dynamic_member.member_selector,
                );
                assert(
                    self.sharding.is_slot_active_for_shard(shard_id, lock_slot),
                    'Shard settle: slot not in shard',
                );
                assert(
                    Felt252DictTrait::get(ref seen_slots, lock_slot) == 0,
                    sharding_cpt::Errors::DUPLICATE_SLOT,
                );
                Felt252DictTrait::insert(ref seen_slots, lock_slot, 1);
                slots.append(lock_slot);
            };

            (slots, slot_metas)
        }

        fn collect_dynamic_member_changes(
            self: @ContractState,
            dynamic_member: ShardDynamicMemberChanges,
            dynamic_changes: Span<(felt252, felt252)>,
        ) -> Array<(felt252, felt252)> {
            let changes_end = dynamic_member.changes_offset + dynamic_member.changes_len;
            assert(changes_end <= dynamic_changes.len(), 'Shard settle: changes range');

            let mut member_changes: Array<(felt252, felt252)> = ArrayTrait::new();
            let mut change_i: u32 = dynamic_member.changes_offset;
            while change_i < changes_end {
                member_changes.append(*dynamic_changes[change_i]);
                change_i += 1;
            };
            member_changes
        }

        #[inline(always)]
        fn read_raw_storage_slot(slot: felt252) -> felt252 {
            let storage_address: StorageAddress = slot.try_into().unwrap();
            storage_read_syscall(0, storage_address).unwrap_syscall()
        }

        #[inline(always)]
        fn write_raw_storage_slot(slot: felt252, value: felt252) {
            let storage_address: StorageAddress = slot.try_into().unwrap();
            storage_write_syscall(0, storage_address, value).unwrap_syscall();
        }

        fn read_dynamic_member_changed_slots_by_lock(
            self: @ContractState, lock_slot: felt252,
        ) -> Span<felt252> {
            let len_felt = Self::read_raw_storage_slot(
                compute_dynamic_member_changed_len_slot(lock_slot),
            );
            if len_felt == 0 {
                return [].span();
            }

            let len: u32 = len_felt.try_into().unwrap();
            let mut slots: Array<felt252> = ArrayTrait::new();
            let mut i: u32 = 0;
            while i < len {
                slots.append(
                    Self::read_raw_storage_slot(
                        compute_dynamic_member_changed_data_slot(lock_slot, i),
                    ),
                );
                i += 1;
            };
            slots.span()
        }

        fn clear_dynamic_member_changed_slots(ref self: ContractState, lock_slot: felt252) {
            let len_slot = compute_dynamic_member_changed_len_slot(lock_slot);
            let hash_slot = compute_dynamic_member_changed_hash_slot(lock_slot);
            let len_felt = Self::read_raw_storage_slot(len_slot);
            if len_felt != 0 {
                let len: u32 = len_felt.try_into().unwrap();
                let mut i: u32 = 0;
                while i < len {
                    let data_slot = compute_dynamic_member_changed_data_slot(lock_slot, i);
                    let changed_slot = Self::read_raw_storage_slot(data_slot);
                    if changed_slot != 0 {
                        Self::write_raw_storage_slot(
                            compute_dynamic_member_changed_seen_slot(lock_slot, changed_slot), 0,
                        );
                    }
                    Self::write_raw_storage_slot(data_slot, 0);
                    i += 1;
                };
            }

            Self::write_raw_storage_slot(len_slot, 0);
            Self::write_raw_storage_slot(hash_slot, DYNAMIC_MEMBER_CHANGED_HASH_SEED);
        }

        fn record_dynamic_member_changed_slot(
            ref self: ContractState, lock_slot: felt252, changed_slot: felt252,
        ) {
            let seen_slot = compute_dynamic_member_changed_seen_slot(lock_slot, changed_slot);
            if Self::read_raw_storage_slot(seen_slot) != 0 {
                return;
            }

            let len_slot = compute_dynamic_member_changed_len_slot(lock_slot);
            let hash_slot = compute_dynamic_member_changed_hash_slot(lock_slot);
            let len_felt = Self::read_raw_storage_slot(len_slot);
            let len: u32 = if len_felt == 0 { 0 } else { len_felt.try_into().unwrap() };
            let current_hash = Self::read_raw_storage_slot(hash_slot);
            let current_hash = if current_hash == 0 {
                DYNAMIC_MEMBER_CHANGED_HASH_SEED
            } else {
                current_hash
            };

            Self::write_raw_storage_slot(seen_slot, 1);
            Self::write_raw_storage_slot(
                compute_dynamic_member_changed_data_slot(lock_slot, len), changed_slot,
            );
            Self::write_raw_storage_slot(len_slot, (len + 1).into());
            Self::write_raw_storage_slot(
                hash_slot, fold_dynamic_member_changed_hash(current_hash, changed_slot),
            );
        }

        fn record_dynamic_member_changed_slots(
            ref self: ContractState, lock_slot: felt252, changed_slots: Span<felt252>,
        ) {
            for changed_slot in changed_slots {
                self.record_dynamic_member_changed_slot(lock_slot, *changed_slot);
            };
        }

        fn assert_dynamic_changes_exact_coverage(
            self: @ContractState,
            dynamic_members: Span<ShardDynamicMemberChanges>,
            dynamic_changes: Span<(felt252, felt252)>,
        ) {
            if dynamic_members.len() == 0 {
                assert(dynamic_changes.len() == 0, 'Shard settle: changes unused');
                return;
            }

            let mut covered: Felt252Dict<felt252> = Default::default();
            let mut covered_len: u32 = 0;
            for dynamic_member in dynamic_members {
                let dynamic_member = *dynamic_member;
                let changes_end = dynamic_member.changes_offset + dynamic_member.changes_len;
                assert(changes_end <= dynamic_changes.len(), 'Shard settle: changes range');

                let mut change_i: u32 = dynamic_member.changes_offset;
                while change_i < changes_end {
                    let changes_key: felt252 = change_i.into();
                    assert(
                        Felt252DictTrait::get(ref covered, changes_key) == 0,
                        'Shard settle: changes overlap',
                    );
                    Felt252DictTrait::insert(ref covered, changes_key, 1);
                    covered_len += 1;
                    change_i += 1;
                };
            };

            assert(covered_len == dynamic_changes.len(), 'Shard settle: changes unused');
        }

        fn assert_dynamic_tracking_proofs_exact_coverage(
            self: @ContractState,
            dynamic_members: Span<ShardDynamicMemberChanges>,
            dynamic_tracking_proofs: Span<(felt252, felt252)>,
        ) {
            if dynamic_members.len() == 0 {
                assert(dynamic_tracking_proofs.len() == 0, 'Shard settle: proof unused');
                return;
            }

            assert(
                dynamic_tracking_proofs.len() == dynamic_members.len() * 2,
                'Shard settle: proof unused',
            );

            let mut seen_proof_slots: Felt252Dict<felt252> = Default::default();
            for proof in dynamic_tracking_proofs {
                let (slot, _) = *proof;
                assert(
                    Felt252DictTrait::get(ref seen_proof_slots, slot) == 0,
                    sharding_cpt::Errors::DUPLICATE_SLOT,
                );
                Felt252DictTrait::insert(ref seen_proof_slots, slot, 1);
            };

            for dynamic_member in dynamic_members {
                let dynamic_member = *dynamic_member;
                let lock_slot = compute_dynamic_member_lock_slot(
                    dynamic_member.model_selector,
                    dynamic_member.entity_id,
                    dynamic_member.member_selector,
                );
                assert(
                    Felt252DictTrait::get(
                        ref seen_proof_slots, compute_dynamic_member_changed_len_slot(lock_slot),
                    ) != 0,
                    'Shard settle: proof missing',
                );
                assert(
                    Felt252DictTrait::get(
                        ref seen_proof_slots, compute_dynamic_member_changed_hash_slot(lock_slot),
                    ) != 0,
                    'Shard settle: proof missing',
                );
            };
        }

        fn read_model_layout_or_panic(self: @ContractState, model_selector: felt252) -> Layout {
            let model_addr = match self.resources.read(model_selector) {
                Resource::Model((addr, _)) => addr,
                _ => panic_with_byte_array(
                    @errors::resource_conflict(@format!("{}", model_selector), @"model"),
                ),
            };
            IStoredResourceDispatcher { contract_address: model_addr }.layout()
        }

        fn read_dynamic_tracking_proof_value(
            self: @ContractState,
            dynamic_tracking_proofs: Span<(felt252, felt252)>,
            expected_slot: felt252,
        ) -> felt252 {
            for proof in dynamic_tracking_proofs {
                let (slot, value) = *proof;
                if slot == expected_slot {
                    return value;
                }
            };

            panic_with_felt252('Shard settle: proof missing')
        }

        fn compute_dynamic_member_changes_hash(
            self: @ContractState, member_changes: Span<(felt252, felt252)>,
        ) -> felt252 {
            let mut hash = DYNAMIC_MEMBER_CHANGED_HASH_SEED;
            for member_change in member_changes {
                let (slot, _) = *member_change;
                hash = fold_dynamic_member_changed_hash(hash, slot);
            };
            hash
        }

        fn apply_dynamic_member_changes(
            ref self: ContractState,
            shard_id: felt252,
            slot_changes: Span<(felt252, felt252)>,
            dynamic_members: Span<ShardDynamicMemberChanges>,
            dynamic_changes: Span<(felt252, felt252)>,
            dynamic_tracking_proofs: Span<(felt252, felt252)>,
            ref slot_metas: Array<SlotMeta>,
        ) -> Array<felt252> {
            let mut dynamic_slots_to_unlock: Array<felt252> = ArrayTrait::new();
            let mut seen_dynamic_locks: Felt252Dict<felt252> = Default::default();
            let mut seen_settlement_slots: Felt252Dict<felt252> = Default::default();

            for slot_change in slot_changes {
                let (slot, _) = *slot_change;
                Felt252DictTrait::insert(ref seen_settlement_slots, slot, 1);
            };

            for dynamic_member in dynamic_members {
                let dynamic_member = *dynamic_member;
                let lock_slot = compute_dynamic_member_lock_slot(
                    dynamic_member.model_selector,
                    dynamic_member.entity_id,
                    dynamic_member.member_selector,
                );
                assert(
                    self.sharding.is_slot_active_for_shard(shard_id, lock_slot),
                    'Shard settle: slot not in shard',
                );
                assert(
                    Felt252DictTrait::get(ref seen_dynamic_locks, lock_slot) == 0,
                    sharding_cpt::Errors::DUPLICATE_SLOT,
                );
                Felt252DictTrait::insert(ref seen_dynamic_locks, lock_slot, 1);

                assert(
                    self.sharding.is_slot_exclusive_locked(lock_slot),
                    'Shard settle: lock missing',
                );
                let (meta_model, meta_entity, meta_member) = self.sharding.read_slot_metadata(lock_slot);
                assert(meta_model == dynamic_member.model_selector, sharding_cpt::Errors::SLOT_METADATA_MISMATCH);
                assert(meta_entity == dynamic_member.entity_id, sharding_cpt::Errors::SLOT_METADATA_MISMATCH);
                assert(meta_member == dynamic_member.member_selector, sharding_cpt::Errors::SLOT_METADATA_MISMATCH);

                let member_changes = self.collect_dynamic_member_changes(
                    dynamic_member, dynamic_changes,
                );
                let proved_len = self.read_dynamic_tracking_proof_value(
                    dynamic_tracking_proofs,
                    compute_dynamic_member_changed_len_slot(lock_slot),
                );
                assert(proved_len == dynamic_member.changes_len.into(), 'Shard settle: proof len');
                let proved_hash = self.read_dynamic_tracking_proof_value(
                    dynamic_tracking_proofs,
                    compute_dynamic_member_changed_hash_slot(lock_slot),
                );
                assert(
                    self.compute_dynamic_member_changes_hash(member_changes.span()) == proved_hash,
                    'Shard settle: proof hash',
                );

                for member_change in member_changes.span() {
                    let (slot, value) = *member_change;
                    assert(
                        Felt252DictTrait::get(ref seen_settlement_slots, slot) == 0,
                        sharding_cpt::Errors::DUPLICATE_SLOT,
                    );
                    Felt252DictTrait::insert(ref seen_settlement_slots, slot, 1);

                    let storage_address: StorageAddress = slot.try_into().unwrap();
                    storage_write_syscall(0, storage_address, value).unwrap_syscall();
                };

                self.clear_dynamic_member_changed_slots(lock_slot);
                dynamic_slots_to_unlock.append(lock_slot);
                slot_metas.append(
                    SlotMeta {
                        slot: lock_slot,
                        model_selector: dynamic_member.model_selector,
                        entity_id: dynamic_member.entity_id,
                        member_selector: dynamic_member.member_selector,
                    },
                );
            };

            dynamic_slots_to_unlock
        }

        fn get_cached_model_layout(
            self: @ContractState,
            model_selector: felt252,
            ref layout_dict: Felt252Dict<felt252>,
            ref layouts_store: Array<Layout>,
        ) -> Layout {
            let cached_idx = Felt252DictTrait::get(ref layout_dict, model_selector);
            if cached_idx != 0 {
                let idx: u32 = (cached_idx - 1).try_into().unwrap();
                return *layouts_store[idx];
            }

            let model_layout = self.read_model_layout_or_panic(model_selector);
            let store_idx: felt252 = (layouts_store.len() + 1).into();
            layouts_store.append(model_layout);
            Felt252DictTrait::insert(ref layout_dict, model_selector, store_idx);
            model_layout
        }

        fn emit_settlement_events_and_cleanup(
            ref self: ContractState, slot_metas: Span<SlotMeta>,
        ) {
            let mut emitted: Felt252Dict<felt252> = Default::default();
            let mut layout_dict: Felt252Dict<felt252> = Default::default();
            let mut layouts_store: Array<Layout> = ArrayTrait::new();

            for meta in slot_metas {
                let meta = *meta;

                // StoreSetRecord if keys stored (new entities), StoreUpdateMember otherwise.
                let keys = self.sharding.read_entity_keys(meta.entity_id);
                let has_keys = keys.len() > 0;
                let dedup_member = if has_keys { 0 } else { meta.member_selector };
                let dedup_key = core::poseidon::poseidon_hash_span(
                    [meta.model_selector, meta.entity_id, dedup_member].span(),
                );
                if Felt252DictTrait::get(ref emitted, dedup_key) != 0 {
                    continue;
                }
                Felt252DictTrait::insert(ref emitted, dedup_key, 1);

                let layout = self.get_cached_model_layout(
                    meta.model_selector, ref layout_dict, ref layouts_store,
                );

                if has_keys {
                    let values = storage::entity_model::read_model_entity(
                        meta.model_selector, meta.entity_id, layout,
                    );
                    self
                        .emit(
                            StoreSetRecord {
                                selector: meta.model_selector,
                                entity_id: meta.entity_id,
                                keys,
                                values,
                            },
                        );
                } else {
                    let values = match dojo::utils::find_model_field_layout(
                        layout, meta.member_selector,
                    ) {
                        Option::Some(member_layout) => {
                            storage::entity_model::read_model_member(
                                meta.model_selector, meta.entity_id, meta.member_selector, member_layout,
                            )
                        },
                        Option::None => panic_with_byte_array(
                            @format!(
                                "Shard settlement: field layout not found for member {}",
                                meta.member_selector,
                            ),
                        ),
                    };
                    self
                        .emit(
                            StoreUpdateMember {
                                selector: meta.model_selector,
                                entity_id: meta.entity_id,
                                member_selector: meta.member_selector,
                                values,
                            },
                        );
                }
            };
        }

        /// Indicates if the provided namespace is already registered
        ///
        /// # Arguments
        ///   * `namespace_hash` - the hash of the namespace.
        #[inline(always)]
        fn is_namespace_registered(self: @ContractState, namespace_hash: felt252) -> bool {
            match self.resources.read(namespace_hash) {
                Resource::Namespace(_) => true,
                _ => false,
            }
        }

        fn record_entity_dynamic_member_changes_if_locked(
            ref self: ContractState,
            model_selector: felt252,
            entity_id: felt252,
            values: Span<felt252>,
            layout: Layout,
        ) {
            let canonical_layout = self.read_model_layout_or_panic(model_selector);
            let canonical_fields = if let Layout::Struct(fields) = canonical_layout {
                fields
            } else {
                return;
            };
            let provided_fields = if let Layout::Struct(fields) = layout {
                fields
            } else {
                return;
            };

            let mut offset: u32 = 0;
            for field_layout in provided_fields {
                let field_selector = *field_layout.selector;
                let field_key = combine_key(entity_id, field_selector);
                let mut written_slots: Array<felt252> = ArrayTrait::new();
                self.collect_written_slots_for_layout(
                    model_selector,
                    field_key,
                    values,
                    ref offset,
                    *field_layout.layout,
                    ref written_slots,
                );

                let canonical_field_layout = if let Option::Some(found) =
                    dojo::utils::find_field_layout(field_selector, canonical_fields) {
                    found
                } else {
                    continue;
                };
                if !is_dynamic_layout(canonical_field_layout) {
                    continue;
                }

                let lock_slot = compute_dynamic_member_lock_slot(
                    model_selector, entity_id, field_selector,
                );
                if !self.sharding.is_slot_active(lock_slot) {
                    continue;
                }
                self.record_dynamic_member_changed_slots(lock_slot, written_slots.span());
            };
        }

        fn record_entity_dynamic_member_existing_slots_if_locked(
            ref self: ContractState, model_selector: felt252, entity_id: felt252, layout: Layout,
        ) {
            let canonical_layout = self.read_model_layout_or_panic(model_selector);
            let canonical_fields = if let Layout::Struct(fields) = canonical_layout {
                fields
            } else {
                return;
            };
            let provided_fields = if let Layout::Struct(fields) = layout {
                fields
            } else {
                return;
            };

            for field_layout in provided_fields {
                let field_selector = *field_layout.selector;
                let canonical_field_layout = if let Option::Some(found) =
                    dojo::utils::find_field_layout(field_selector, canonical_fields) {
                    found
                } else {
                    continue;
                };
                if !is_dynamic_layout(canonical_field_layout) {
                    continue;
                }

                let lock_slot = compute_dynamic_member_lock_slot(
                    model_selector, entity_id, field_selector,
                );
                if !self.sharding.is_slot_active(lock_slot) {
                    continue;
                }

                let mut existing_slots: Array<felt252> = ArrayTrait::new();
                self.collect_storage_slots_for_layout(
                    model_selector,
                    combine_key(entity_id, field_selector),
                    canonical_field_layout,
                    ref existing_slots,
                );
                self.record_dynamic_member_changed_slots(lock_slot, existing_slots.span());
            };
        }

        fn record_member_dynamic_changes_if_locked(
            ref self: ContractState,
            model_selector: felt252,
            entity_id: felt252,
            member_selector: felt252,
            values: Span<felt252>,
            layout: Layout,
        ) {
            let canonical_layout = self.read_member_layout_or_panic(model_selector, member_selector);
            if !is_dynamic_layout(canonical_layout) {
                return;
            }

            let lock_slot = compute_dynamic_member_lock_slot(
                model_selector, entity_id, member_selector,
            );
            if !self.sharding.is_slot_active(lock_slot) {
                return;
            }

            let mut written_slots: Array<felt252> = ArrayTrait::new();
            let mut offset: u32 = 0;
            self.collect_written_slots_for_layout(
                model_selector,
                combine_key(entity_id, member_selector),
                values,
                ref offset,
                layout,
                ref written_slots,
            );
            self.record_dynamic_member_changed_slots(lock_slot, written_slots.span());
        }

        fn record_member_dynamic_existing_slots_if_locked(
            ref self: ContractState, model_selector: felt252, entity_id: felt252, member_selector: felt252,
        ) {
            let canonical_layout = self.read_member_layout_or_panic(model_selector, member_selector);
            if !is_dynamic_layout(canonical_layout) {
                return;
            }

            let lock_slot = compute_dynamic_member_lock_slot(
                model_selector, entity_id, member_selector,
            );
            if !self.sharding.is_slot_active(lock_slot) {
                return;
            }

            let mut existing_slots: Array<felt252> = ArrayTrait::new();
            self.collect_storage_slots_for_layout(
                model_selector,
                combine_key(entity_id, member_selector),
                canonical_layout,
                ref existing_slots,
            );
            self.record_dynamic_member_changed_slots(lock_slot, existing_slots.span());
        }

        fn record_entity_dynamic_member_deletes_if_locked(
            ref self: ContractState, model_selector: felt252, entity_id: felt252, layout: Layout,
        ) {
            let canonical_layout = self.read_model_layout_or_panic(model_selector);
            let canonical_fields = if let Layout::Struct(fields) = canonical_layout {
                fields
            } else {
                return;
            };
            let provided_fields = if let Layout::Struct(fields) = layout {
                fields
            } else {
                return;
            };

            for field_layout in provided_fields {
                let field_selector = *field_layout.selector;
                let canonical_field_layout = if let Option::Some(found) =
                    dojo::utils::find_field_layout(field_selector, canonical_fields) {
                    found
                } else {
                    continue;
                };
                if !is_dynamic_layout(canonical_field_layout) {
                    continue;
                }

                let lock_slot = compute_dynamic_member_lock_slot(
                    model_selector, entity_id, field_selector,
                );
                if !self.sharding.is_slot_active(lock_slot) {
                    continue;
                }

                let mut deleted_slots: Array<felt252> = ArrayTrait::new();
                self.collect_deleted_slots_for_layout(
                    model_selector,
                    combine_key(entity_id, field_selector),
                    *field_layout.layout,
                    ref deleted_slots,
                );
                self.record_dynamic_member_changed_slots(lock_slot, deleted_slots.span());
            };
        }

        /// Sets the model value for a model record/entity/member.
        ///
        /// # Arguments
        ///
        /// * `model_selector` - The selector of the model to be set.
        /// * `index` - The index of the record/entity/member to write.
        /// * `values` - The value to be set, serialized using the model layout format.
        /// * `layout` - The memory layout of the model.
        // ── Entity write path ────────────────────────────────────────────
        //
        // The core write/delete/read operations.  Sharding hooks are
        // guarded by `shard_active` (a single storage read) so they add
        // zero overhead when no shard has been requested.

        fn set_entity_internal(
            ref self: ContractState,
            model_selector: felt252,
            index: ModelIndex,
            values: Span<felt252>,
            layout: Layout,
        ) {
            let shard_active = !self.sharding.sharding_contract().is_zero();
            match index {
                ModelIndex::Keys(keys) => {
                    let entity_id = entity_id_from_serialized_keys(keys);
                    if shard_active {
                        self.assert_model_write_unlocked(model_selector, entity_id);
                        self.record_entity_dynamic_member_existing_slots_if_locked(
                            model_selector, entity_id, layout,
                        );
                    }
                    storage::entity_model::write_model_entity(
                        model_selector, entity_id, values, layout,
                    );
                    if shard_active {
                        self.record_entity_dynamic_member_changes_if_locked(
                            model_selector, entity_id, values, layout,
                        );
                    }
                    self.emit(StoreSetRecord { selector: model_selector, keys, values, entity_id });
                },
                ModelIndex::Id(entity_id) => {
                    if shard_active {
                        self.assert_model_write_unlocked(model_selector, entity_id);
                        self.record_entity_dynamic_member_existing_slots_if_locked(
                            model_selector, entity_id, layout,
                        );
                    }
                    storage::entity_model::write_model_entity(
                        model_selector, entity_id, values, layout,
                    );
                    if shard_active {
                        self.record_entity_dynamic_member_changes_if_locked(
                            model_selector, entity_id, values, layout,
                        );
                    }
                    self.emit(StoreUpdateRecord { selector: model_selector, entity_id, values });
                },
                ModelIndex::MemberId((
                    entity_id, member_selector,
                )) => {
                    if shard_active {
                        self.assert_member_write_unlocked(
                            model_selector, entity_id, member_selector, layout,
                        );
                        self.record_member_dynamic_existing_slots_if_locked(
                            model_selector, entity_id, member_selector,
                        );
                    }
                    storage::entity_model::write_model_member(
                        model_selector, entity_id, member_selector, values, layout,
                    );
                    if shard_active {
                        self.record_member_dynamic_changes_if_locked(
                            model_selector, entity_id, member_selector, values, layout,
                        );
                    }
                    self
                        .emit(
                            StoreUpdateMember {
                                selector: model_selector, entity_id, member_selector, values,
                            },
                        );
                },
            }
        }

        /// Deletes an entity for the given model, setting all the values to 0 in the given layout.
        ///
        /// # Arguments
        ///
        /// * `model_selector` - The selector of the model to be deleted.
        /// * `index` - The index of the record/entity to delete.
        /// * `layout` - The memory layout of the model.
        fn delete_entity_internal(
            ref self: ContractState, model_selector: felt252, index: ModelIndex, layout: Layout,
        ) {
            let shard_active = !self.sharding.sharding_contract().is_zero();
            match index {
                ModelIndex::Keys(keys) => {
                    let entity_id = entity_id_from_serialized_keys(keys);
                    if shard_active {
                        self.assert_model_write_unlocked(model_selector, entity_id);
                        self.record_entity_dynamic_member_deletes_if_locked(
                            model_selector, entity_id, layout,
                        );
                    }
                    storage::entity_model::delete_model_entity(model_selector, entity_id, layout);
                    self.emit(StoreDelRecord { selector: model_selector, entity_id });
                },
                ModelIndex::Id(entity_id) => {
                    if shard_active {
                        self.assert_model_write_unlocked(model_selector, entity_id);
                        self.record_entity_dynamic_member_deletes_if_locked(
                            model_selector, entity_id, layout,
                        );
                    }
                    storage::entity_model::delete_model_entity(model_selector, entity_id, layout);
                    self.emit(StoreDelRecord { selector: model_selector, entity_id });
                },
                ModelIndex::MemberId(_) => { panic_with_felt252(errors::DELETE_ENTITY_MEMBER); },
            }
        }

        /// Gets the model values for the given entity.
        ///
        /// # Arguments
        ///
        /// * `model_selector` - The selector of the model to be retrieved.
        /// * `index` - The entity/member to read for the given model.
        /// * `layout` - The memory layout of the model.
        fn get_entity_internal(
            self: @ContractState, model_selector: felt252, index: ModelIndex, layout: Layout,
        ) -> Span<felt252> {
            match index {
                ModelIndex::Keys(keys) => {
                    let entity_id = entity_id_from_serialized_keys(keys);
                    storage::entity_model::read_model_entity(model_selector, entity_id, layout)
                },
                ModelIndex::Id(entity_id) => {
                    storage::entity_model::read_model_entity(model_selector, entity_id, layout)
                },
                ModelIndex::MemberId((
                    entity_id, member_id,
                )) => {
                    storage::entity_model::read_model_member(
                        model_selector, entity_id, member_id, layout,
                    )
                },
            }
        }

        /// Returns the hash of the internal namespace for a dojo world.
        fn world_internal_namespace(self: @ContractState) -> (ByteArray, felt252) {
            let name = "__DOJO__";
            let hash = bytearray_hash(@name);

            (name, hash)
        }

        /// Guard regular world writes against active exclusive shard locks.
        ///
        /// Settlement writes still flow through `settle_shard_changes` (proxy-only path),
        /// while gameplay writes to main chain must fail for `SetLock/Lock` slots.
        // ── Sharding write-path guards ──────────────────────────────────
        //
        // These assertions ensure that locked slots cannot be overwritten
        // while a shard is active.  They are only called when
        // `shard_active == true`.

        fn assert_model_write_unlocked(
            self: @ContractState, model_selector: felt252, entity_id: felt252,
        ) {
            // Use canonical on-chain model layout for lock checks.
            // Caller-provided layouts may be intentionally partial/legacy and must not
            // weaken sharding lock enforcement.
            let layout = self.read_model_layout_or_panic(model_selector);

            let mut deterministic_slots: Array<felt252> = ArrayTrait::new();
            let _ = collect_shardable_slots(ref deterministic_slots, model_selector, entity_id, layout);
            for slot in deterministic_slots.span() {
                self.assert_slot_writable(*slot);
            };

            let mut dynamic_locks: Array<felt252> = ArrayTrait::new();
            collect_dynamic_member_locks(ref dynamic_locks, model_selector, entity_id, layout);
            for slot in dynamic_locks.span() {
                self.assert_slot_writable(*slot);
            };
        }

        /// Member writes map to all deterministic slots under the member key.
        fn assert_member_write_unlocked(
            self: @ContractState,
            model_selector: felt252,
            entity_id: felt252,
            member_selector: felt252,
            member_layout: Layout,
        ) {
            let model_layout = self.read_model_layout_or_panic(model_selector);
            let canonical_member_layout = match dojo::utils::find_model_field_layout(
                model_layout, member_selector,
            ) {
                Option::Some(layout) => layout,
                Option::None => panic_with_byte_array(
                    @format!(
                        "set_entity: field layout not found for member {}",
                        member_selector,
                    ),
                ),
            };

            let member_key = combine_key(entity_id, member_selector);
            let mut slots: Array<felt252> = ArrayTrait::new();
            let deterministic = collect_shardable_slots(
                ref slots, model_selector, member_key, member_layout,
            );
            for slot in slots.span() {
                self.assert_slot_writable(*slot);
            };

            if !deterministic {
                let dynamic_slot = compute_dynamic_member_lock_slot(
                    model_selector, entity_id, member_selector,
                );
                self.assert_slot_writable(dynamic_slot);
                return;
            }

            // Defend against forged deterministic layouts for dynamic top-level members.
            // Even if caller-provided `member_layout` is deterministic, canonical model
            // layout may mark this member as dynamic and therefore protected by lock slot.
            if is_dynamic_layout(canonical_member_layout) {
                let dynamic_slot = compute_dynamic_member_lock_slot(
                    model_selector, entity_id, member_selector,
                );
                self.assert_slot_writable(dynamic_slot);
            }
        }

        #[inline(always)]
        fn assert_slot_writable(self: @ContractState, slot: felt252) {
            assert(!self.sharding.is_slot_exclusive_locked(slot), sharding_cpt::Errors::SLOT_LOCKED);
        }
    }
}
