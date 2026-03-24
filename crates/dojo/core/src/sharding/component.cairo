use starknet::ContractAddress;
use dojo::sharding::request::{CRDVariant, ShardField};

/// Minimal interface for notifying the sharding proxy (event bus).
#[starknet::interface]
pub trait IShardingProxy<T> {
    fn notify_shard_requested(ref self: T, shard_id: felt252, entities: Span<felt252>, entity_keys_flat: Span<felt252>);
    fn end_shard(ref self: T, shard_id: felt252);
}

#[starknet::interface]
pub trait IContractComponent<TContractState> {
    /// Lock entities and allocate shard. Returns shard_id.
    ///
    /// Commitment is computed atomically as `H(sorted(entities))` and stored
    /// on-chain — no separate registration step needed.
    ///
    /// `entities` must contain **Dojo entity_ids** (= `Poseidon(serialized_keys)`),
    /// NOT raw model keys. The world contract's write-protection computes
    /// `entity_id_from_serialized_keys(keys)` before checking `entity_lock`,
    /// so entity_lock must store hashed entity_ids to match.
    fn request_shard(
        ref self: TContractState,
        entities: Span<felt252>,
        entity_keys_flat: Span<felt252>,
    ) -> felt252;

    /// Register CRDT policy for a model. Called once per model at deploy time.
    /// `default_crdt` applies to all fields unless overridden.
    /// `field_overrides` provides per-field CRDT overrides.
    fn register_shard_policy(
        ref self: TContractState,
        model_selector: felt252,
        default_crdt: CRDVariant,
        field_overrides: Span<ShardField>,
    );

    /// Read the registered CRDT policy for a model.
    /// Returns (default_crdt_encoded, field_count, field_selectors, field_crdts).
    fn get_shard_policy(
        self: @TContractState,
        model_selector: felt252,
    ) -> (felt252, Span<ShardField>);

    /// Configure the sharding proxy address (event bus). One-shot.
    fn set_sharding_proxy(ref self: TContractState, proxy: ContractAddress);

    /// Settle changed slots with StorageCommitment verification.
    /// Per-slot metadata enables on-chain ownership verification and Torii notification.
    /// `slot_initial_values` provides Add CRDT initial values for delta computation.
    fn settle(
        ref self: TContractState,
        shard_id: felt252,
        changed_keys: Span<felt252>,
        changed_values: Span<felt252>,
        state_diff_hash: felt252,
        global_state_root: felt252,
        end_block_number: u64,
        slot_model_selectors: Span<felt252>,
        slot_entity_ids: Span<felt252>,
        slot_computation_keys: Span<felt252>,
        slot_member_selectors: Span<felt252>,
        slot_packed_offsets: Span<u32>,
        slot_initial_values: Span<felt252>,
    );

    /// Configure StorageCommitment verifier contract. One-shot.
    fn set_storage_commitment_registry(ref self: TContractState, registry: ContractAddress);

    /// Cancel shard: unlock entities without applying changes.
    fn cancel_shard(ref self: TContractState, shard_id: felt252);

    /// Signal that the shard has finished. Emits ShardFinished event.
    fn end_shard(ref self: TContractState, shard_id: felt252);

    /// Dev-only settlement: commitment + diff verification WITHOUT StorageCommitment proof.
    #[cfg(feature: 'dev')]
    fn settle_dev(
        ref self: TContractState,
        shard_id: felt252,
        changed_keys: Span<felt252>,
        changed_values: Span<felt252>,
        end_block_number: u64,
        slot_model_selectors: Span<felt252>,
        slot_entity_ids: Span<felt252>,
        slot_computation_keys: Span<felt252>,
        slot_member_selectors: Span<felt252>,
        slot_packed_offsets: Span<u32>,
        slot_initial_values: Span<felt252>,
    );
}

#[starknet::component]
pub mod sharding_component {
    use starknet::SyscallResultTrait;
    use starknet::storage::{StoragePointerReadAccess, StoragePointerWriteAccess};
    use starknet::storage::{Map, StorageMapReadAccess, StorageMapWriteAccess};
    use starknet::storage_access::StorageAddress;
    use starknet::syscalls::{storage_read_syscall, storage_write_syscall};
    use starknet::{ContractAddress, get_contract_address, get_execution_info};
    use core::poseidon::poseidon_hash_span;
    use dojo::sharding::interface::{
        IStorageCommitmentVerifierDispatcher, IStorageCommitmentVerifierDispatcherTrait,
    };
    use dojo::storage::database::DOJO_STORAGE;
    use super::{IShardingProxyDispatcher, IShardingProxyDispatcherTrait};

    #[storage]
    pub struct Storage {
        /// Entity-level lock: entity_id → shard_id (0 = unlocked).
        entity_lock: Map<felt252, felt252>,
        /// Shard entity tracking for unlock: (shard_id, index) → entity_id.
        shard_entity_count: Map<felt252, u32>,
        shard_entities: Map<(felt252, u32), felt252>,
        /// Commitment: shard_id → H(sorted(entities)). Set atomically in request_shard.
        shard_commitment: Map<felt252, felt252>,
        /// Request-time block number bound to the shard session.
        shard_fork_block_number: Map<felt252, u64>,
        /// Internal shard ID counter.
        next_shard_id: felt252,
        /// Address of StorageCommitment verifier (0 = not configured).
        storage_commitment_registry: ContractAddress,
        /// Address of the sharding proxy contract (event bus). Set once by owner.
        sharding_proxy: ContractAddress,
        /// Shard policy: model_selector → default CRDVariant encoded (0=unset, 1=Set, 2=Add, 3=Lock, 4=SetLock).
        model_policy_default: Map<felt252, felt252>,
        /// Shard policy per-field override: (model_selector, field_selector) → CRDVariant encoded (0=use default).
        model_policy_field: Map<(felt252, felt252), felt252>,
        /// Number of field overrides per model (for enumeration).
        model_policy_field_count: Map<felt252, u32>,
        /// Field override selectors for enumeration: (model_selector, index) → field_selector.
        model_policy_fields: Map<(felt252, u32), felt252>,
        /// Max array elements per field: (model_selector, field_selector) → max_elements (0 = not dynamic).
        model_policy_max_elements: Map<(felt252, felt252), u32>,
        /// Fork mode: when true, entity_lock checks are skipped in world contract writes.
        /// Not exposed as public entrypoint — set via katana_setStorageAt (dev-only RPC)
        /// which is unavailable on main chain sequencers.
        shard_fork_mode: bool,
        /// Active shard tracking: O(active) enumeration instead of O(total_ever_created).
        active_shard_count: u32,
        active_shard_list: Map<u32, felt252>,
    }

    /// Maximum entity_keys_flat felts per chunk event (stay well under Starknet's 300 data limit).
    const MAX_KEYS_PER_CHUNK: u32 = 250;

    #[event]
    #[derive(Drop, starknet::Event)]
    pub enum Event {
        ShardRequested: ShardRequested,
        ShardEntityKeysChunk: ShardEntityKeysChunk,
        ShardFinished: ShardFinished,
    }

    #[derive(Drop, starknet::Event)]
    pub struct ShardRequested {
        #[key]
        pub shard_id: felt252,
        pub entities: Span<felt252>,
        /// Number of `ShardEntityKeysChunk` events that follow (0 if no keys).
        pub entity_key_chunks: u32,
    }

    #[derive(Drop, starknet::Event)]
    pub struct ShardEntityKeysChunk {
        #[key]
        pub shard_id: felt252,
        #[key]
        pub chunk_index: u32,
        pub entity_keys_flat: Span<felt252>,
    }

    #[derive(Drop, starknet::Event)]
    pub struct ShardFinished {
        #[key]
        pub shard_id: felt252,
    }

    fn encode_crdt(crdt: super::CRDVariant) -> felt252 {
        match crdt {
            super::CRDVariant::Set => 1,
            super::CRDVariant::Add => 2,
            super::CRDVariant::Lock => 3,
            super::CRDVariant::SetLock => 4,
        }
    }

    fn decode_crdt(encoded: felt252) -> super::CRDVariant {
        if encoded == 1 {
            super::CRDVariant::Set
        } else if encoded == 2 {
            super::CRDVariant::Add
        } else if encoded == 3 {
            super::CRDVariant::Lock
        } else if encoded == 4 {
            super::CRDVariant::SetLock
        } else {
            super::CRDVariant::Set // default for unset (0)
        }
    }

    pub mod Errors {
        pub const ENTITY_ALREADY_SHARDED: felt252 = 'Shard: entity already sharded';
        pub const SHARD_NOT_FOUND: felt252 = 'Shard: not found';
        pub const STATE_DIFF_MISMATCH: felt252 = 'Shard: state diff mismatch';
        pub const CHANGED_KEYS_VALUES_LEN: felt252 = 'Shard: keys/values len';
        pub const ADD_DELTA_UNDERFLOW: felt252 = 'Shard: add delta underflow';
        pub const ARITHMETIC_OVERFLOW: felt252 = 'Shard: arithmetic overflow';
        pub const UNAUTHORIZED_CALLER: felt252 = 'Shard: unauthorized caller';
        pub const NO_ENTITIES: felt252 = 'Shard: no entities';
        pub const END_BLOCK_NOT_PROVEN: felt252 = 'Shard: end block not proven';
        pub const COMMITMENT_NOT_VERIFIED: felt252 = 'Shard: commitment not verified';
        pub const REGISTRY_ALREADY_SET: felt252 = 'Shard: registry already set';
        pub const REGISTRY_NOT_SET: felt252 = 'Shard: registry not configured';
        pub const POLICY_INVALID_CRDT: felt252 = 'Shard: invalid crdt variant';
        pub const POLICY_NO_MODEL: felt252 = 'Shard: model selector is zero';
        pub const SLOT_OWNERSHIP_MISMATCH: felt252 = 'Shard: slot ownership mismatch';
        pub const ENTITY_NOT_IN_SHARD: felt252 = 'Shard: entity not in shard';
        pub const METADATA_LEN_MISMATCH: felt252 = 'Shard: metadata len mismatch';
        pub const ENTITY_KEYS_MALFORMED: felt252 = 'Shard: entity_keys malformed';
    }

    #[embeddable_as(ContractComponentImpl)]
    impl ContractImpl<
        TContractState, +HasComponent<TContractState>,
    > of super::IContractComponent<ComponentState<TContractState>> {
        fn request_shard(
            ref self: ComponentState<TContractState>,
            entities: Span<felt252>,
            entity_keys_flat: Span<felt252>,
        ) -> felt252 {
            assert(entities.len() != 0, Errors::NO_ENTITIES);

            // Validate entity_keys_flat format: [n_keys_0, key0..., n_keys_1, key1...]
            // Must contain exactly one length-prefixed key group per entity.
            if entity_keys_flat.len() != 0 {
                let mut offset: u32 = 0;
                let mut parsed_count: u32 = 0;
                while offset < entity_keys_flat.len() {
                    let n_keys: u256 = (*entity_keys_flat[offset]).into();
                    let n: u32 = n_keys.try_into().expect(Errors::ENTITY_KEYS_MALFORMED);
                    offset += 1 + n;
                    parsed_count += 1;
                };
                assert(offset == entity_keys_flat.len(), Errors::ENTITY_KEYS_MALFORMED);
                assert(parsed_count == entities.len(), Errors::ENTITY_KEYS_MALFORMED);
            }

            for entity_id in entities {
                let entity_id = *entity_id;
                assert(self.entity_lock.read(entity_id) == 0, Errors::ENTITY_ALREADY_SHARDED);
            };

            let shard_id = self.next_shard_id.read() + 1;
            self.next_shard_id.write(shard_id);
            let fork_block_number = get_execution_info().block_info.block_number;
            self.shard_fork_block_number.write(shard_id, fork_block_number);

            // Track active shard for O(active) enumeration.
            let active_idx = self.active_shard_count.read();
            self.active_shard_list.write(active_idx, shard_id);
            self.active_shard_count.write(active_idx + 1);

            let entity_count: u32 = entities.len();
            self.shard_entity_count.write(shard_id, entity_count);
            let mut i: u32 = 0;
            for entity_id in entities {
                let entity_id = *entity_id;
                self.entity_lock.write(entity_id, shard_id);
                self.shard_entities.write((shard_id, i), entity_id);
                i += 1;
            };

            // Atomic commitment: H(sorted(entities)).
            // This locks the entity set at request time — no separate registration needed.
            let sorted = sort_felt_span(entities);
            let commitment = poseidon_hash_span(sorted);
            self.shard_commitment.write(shard_id, commitment);

            // Emit main event + chunked entity keys (Starknet event data limit = 300 felts).
            let total_keys_len = entity_keys_flat.len();
            let num_chunks = if total_keys_len == 0 {
                0_u32
            } else {
                let full = total_keys_len / MAX_KEYS_PER_CHUNK;
                if total_keys_len % MAX_KEYS_PER_CHUNK != 0 { full + 1 } else { full }
            };
            self.emit(ShardRequested { shard_id, entities, entity_key_chunks: num_chunks });

            let mut chunk_idx: u32 = 0;
            let mut key_offset: u32 = 0;
            while key_offset < total_keys_len {
                let remaining = total_keys_len - key_offset;
                let chunk_size = if remaining < MAX_KEYS_PER_CHUNK {
                    remaining
                } else {
                    MAX_KEYS_PER_CHUNK
                };
                let chunk = entity_keys_flat.slice(key_offset, chunk_size);
                self
                    .emit(
                        ShardEntityKeysChunk {
                            shard_id, chunk_index: chunk_idx, entity_keys_flat: chunk,
                        },
                    );
                key_offset += chunk_size;
                chunk_idx += 1;
            };

            // Notify the sharding proxy (event bus) so the operator discovers this shard.
            let proxy_addr = self.sharding_proxy.read();
            if proxy_addr != core::num::traits::Zero::zero() {
                let proxy = IShardingProxyDispatcher { contract_address: proxy_addr };
                proxy.notify_shard_requested(shard_id, entities, entity_keys_flat);
            }

            shard_id
        }

        fn settle(
            ref self: ComponentState<TContractState>,
            shard_id: felt252,
            changed_keys: Span<felt252>,
            changed_values: Span<felt252>,
            state_diff_hash: felt252,
            global_state_root: felt252,
            end_block_number: u64,
            slot_model_selectors: Span<felt252>,
            slot_entity_ids: Span<felt252>,
            slot_computation_keys: Span<felt252>,
            slot_member_selectors: Span<felt252>,
            slot_packed_offsets: Span<u32>,
            slot_initial_values: Span<felt252>,
        ) {
            assert(shard_id != 0, 'Shard: invalid shard id');

            // Verify shard is active and state diff hash matches changed keys.
            self.verify_shard_and_diff(shard_id, changed_keys, state_diff_hash);

            // StorageCommitment verification: only when registry is configured.
            // Production deployments MUST set the registry; tests may skip.
            let registry_addr = self.storage_commitment_registry.read();
            if registry_addr != core::num::traits::Zero::zero() {
                assert(end_block_number != 0, Errors::END_BLOCK_NOT_PROVEN);

                let mut commitment_data: Array<felt252> = ArrayTrait::new();
                let mut i: u32 = 0;
                while i < changed_keys.len() {
                    commitment_data.append(*changed_keys[i]);
                    i += 1;
                };
                let mut j: u32 = 0;
                while j < changed_values.len() {
                    commitment_data.append(*changed_values[j]);
                    j += 1;
                };
                let raw_storage_commitment = poseidon_hash_span(commitment_data.span());

                let verifier = IStorageCommitmentVerifierDispatcher {
                    contract_address: registry_addr,
                };
                assert(
                    verifier.verify(
                        raw_storage_commitment,
                        get_contract_address(),
                        global_state_root,
                        end_block_number,
                    ),
                    Errors::COMMITMENT_NOT_VERIFIED,
                );
            }

            // Apply changes with ownership verification and Torii notification.
            self
                .apply_settle(
                    shard_id,
                    changed_keys,
                    changed_values,
                    slot_model_selectors,
                    slot_entity_ids,
                    slot_computation_keys,
                    slot_member_selectors,
                    slot_packed_offsets,
                    slot_initial_values,
                );
        }

        fn set_storage_commitment_registry(
            ref self: ComponentState<TContractState>, registry: ContractAddress,
        ) {
            let current = self.storage_commitment_registry.read();
            assert(current == core::num::traits::Zero::zero(), Errors::REGISTRY_ALREADY_SET);
            self.storage_commitment_registry.write(registry);
        }

        fn set_sharding_proxy(
            ref self: ComponentState<TContractState>, proxy: ContractAddress,
        ) {
            let current = self.sharding_proxy.read();
            assert(current == core::num::traits::Zero::zero(), 'Shard: proxy already set');
            self.sharding_proxy.write(proxy);
        }

        fn cancel_shard(ref self: ComponentState<TContractState>, shard_id: felt252) {
            assert(self.shard_entity_count.read(shard_id) != 0, Errors::SHARD_NOT_FOUND);
            self.unlock_entities(shard_id);
            self.clear_shard_state(shard_id);
        }

        fn end_shard(ref self: ComponentState<TContractState>, shard_id: felt252) {
            self.emit(ShardFinished { shard_id });

            // Forward to proxy so the operator (watching proxy) sees ShardFinished.
            let proxy_addr = self.sharding_proxy.read();
            if proxy_addr != core::num::traits::Zero::zero() {
                let proxy = IShardingProxyDispatcher { contract_address: proxy_addr };
                proxy.end_shard(shard_id);
            }
        }

        fn register_shard_policy(
            ref self: ComponentState<TContractState>,
            model_selector: felt252,
            default_crdt: super::CRDVariant,
            field_overrides: Span<super::ShardField>,
        ) {
            assert(model_selector != 0, Errors::POLICY_NO_MODEL);
            let encoded_default = encode_crdt(default_crdt);

            // Clear previous field overrides if re-registering.
            let prev_count = self.model_policy_field_count.read(model_selector);
            let mut i: u32 = 0;
            while i < prev_count {
                let prev_field = self.model_policy_fields.read((model_selector, i));
                self.model_policy_field.write((model_selector, prev_field), 0);
                self.model_policy_fields.write((model_selector, i), 0);
                i += 1;
            };

            self.model_policy_default.write(model_selector, encoded_default);
            self.model_policy_field_count.write(model_selector, field_overrides.len());

            let mut j: u32 = 0;
            for field in field_overrides {
                let field = *field;
                let encoded_field_crdt = encode_crdt(field.crdt);
                self.model_policy_field.write((model_selector, field.selector), encoded_field_crdt);
                self.model_policy_fields.write((model_selector, j), field.selector);
                self.model_policy_max_elements.write((model_selector, field.selector), field.max_elements);
                j += 1;
            };
        }

        fn get_shard_policy(
            self: @ComponentState<TContractState>,
            model_selector: felt252,
        ) -> (felt252, Span<super::ShardField>) {
            let default_encoded = self.model_policy_default.read(model_selector);
            let field_count = self.model_policy_field_count.read(model_selector);
            let mut fields: Array<super::ShardField> = ArrayTrait::new();
            let mut i: u32 = 0;
            while i < field_count {
                let field_selector = self.model_policy_fields.read((model_selector, i));
                let field_crdt_encoded = self.model_policy_field.read((model_selector, field_selector));
                let max_elements = self.model_policy_max_elements.read((model_selector, field_selector));
                fields.append(super::ShardField {
                    selector: field_selector,
                    crdt: decode_crdt(field_crdt_encoded),
                    max_elements,
                });
                i += 1;
            };
            (default_encoded, fields.span())
        }

        #[cfg(feature: 'dev')]
        fn settle_dev(
            ref self: ComponentState<TContractState>,
            shard_id: felt252,
            changed_keys: Span<felt252>,
            changed_values: Span<felt252>,
            end_block_number: u64,
            slot_model_selectors: Span<felt252>,
            slot_entity_ids: Span<felt252>,
            slot_computation_keys: Span<felt252>,
            slot_member_selectors: Span<felt252>,
            slot_packed_offsets: Span<u32>,
            slot_initial_values: Span<felt252>,
        ) {
            assert(shard_id != 0, 'Shard: invalid shard id');

            // Dev mode: compute state_diff_hash locally from changed_keys.
            let state_diff_hash = poseidon_hash_span(changed_keys);
            self.verify_shard_and_diff(shard_id, changed_keys, state_diff_hash);

            self
                .apply_settle(
                    shard_id,
                    changed_keys,
                    changed_values,
                    slot_model_selectors,
                    slot_entity_ids,
                    slot_computation_keys,
                    slot_member_selectors,
                    slot_packed_offsets,
                    slot_initial_values,
                );
        }
    }

    #[generate_trait]
    pub impl InternalImpl<
        TContractState, +HasComponent<TContractState>,
    > of InternalTrait<TContractState> {
        fn is_entity_locked(self: @ComponentState<TContractState>, entity_id: felt252) -> bool {
            if self.shard_fork_mode.read() {
                return false;
            }
            self.entity_lock.read(entity_id) != 0
        }

        fn enable_shard_fork_mode(ref self: ComponentState<TContractState>) {
            assert(self.active_shard_count.read() > 0, 'Shard: no active shards');
            assert(!self.shard_fork_mode.read(), 'Shard: fork mode already set');
            self.shard_fork_mode.write(true);
        }

        fn is_shard_fork_mode(self: @ComponentState<TContractState>) -> bool {
            self.shard_fork_mode.read()
        }

        fn get_next_shard_id(self: @ComponentState<TContractState>) -> felt252 {
            self.next_shard_id.read()
        }

        fn get_active_shards(self: @ComponentState<TContractState>) -> Array<felt252> {
            let count = self.active_shard_count.read();
            let mut result = ArrayTrait::new();
            let mut i: u32 = 0;
            while i < count {
                result.append(self.active_shard_list.read(i));
                i += 1;
            };
            result
        }

        fn entity_shard_id(self: @ComponentState<TContractState>, entity_id: felt252) -> felt252 {
            self.entity_lock.read(entity_id)
        }

        fn shard_commitment(self: @ComponentState<TContractState>, shard_id: felt252) -> felt252 {
            self.shard_commitment.read(shard_id)
        }

        fn shard_attestation_fork_block_number(
            self: @ComponentState<TContractState>, shard_id: felt252,
        ) -> u64 {
            self.shard_fork_block_number.read(shard_id)
        }

        fn shard_entities_list(
            self: @ComponentState<TContractState>, shard_id: felt252,
        ) -> Array<felt252> {
            let count = self.shard_entity_count.read(shard_id);
            let mut entities: Array<felt252> = ArrayTrait::new();
            let mut i: u32 = 0;
            while i < count {
                entities.append(self.shard_entities.read((shard_id, i)));
                i += 1;
            };
            entities
        }

        /// Verify shard is active and state diff hash matches changed keys.
        fn verify_shard_and_diff(
            self: @ComponentState<TContractState>,
            shard_id: felt252,
            changed_keys: Span<felt252>,
            state_diff_hash: felt252,
        ) {
            // Commitment was set atomically in request_shard. Non-zero proves shard exists.
            let stored_commitment = self.shard_commitment.read(shard_id);
            assert(stored_commitment != 0, Errors::SHARD_NOT_FOUND);

            // Verify state_diff_hash = H(changed_keys).
            // This binds the changed keys to the TEE attestation (production)
            // or is locally recomputed (dev mode).
            let computed_diff = poseidon_hash_span(changed_keys);
            assert(computed_diff == state_diff_hash, Errors::STATE_DIFF_MISMATCH);
        }

        /// Shared apply logic for settle() and settle_dev():
        /// verify slot ownership, write with CRDT merge, emit ShardSettled, unlock, clear.
        ///
        /// `slot_initial_values` provides per-slot initial values for Add CRDT delta computation.
        /// For non-Add slots, the value is ignored (pass 0).
        fn apply_settle(
            ref self: ComponentState<TContractState>,
            shard_id: felt252,
            changed_keys: Span<felt252>,
            changed_values: Span<felt252>,
            slot_model_selectors: Span<felt252>,
            slot_entity_ids: Span<felt252>,
            slot_computation_keys: Span<felt252>,
            slot_member_selectors: Span<felt252>,
            slot_packed_offsets: Span<u32>,
            slot_initial_values: Span<felt252>,
        ) {
            let changes_len = changed_keys.len();
            assert(changes_len == changed_values.len(), Errors::CHANGED_KEYS_VALUES_LEN);
            assert(slot_model_selectors.len() == changes_len, Errors::METADATA_LEN_MISMATCH);
            assert(slot_entity_ids.len() == changes_len, Errors::METADATA_LEN_MISMATCH);
            assert(slot_computation_keys.len() == changes_len, Errors::METADATA_LEN_MISMATCH);
            assert(slot_member_selectors.len() == changes_len, Errors::METADATA_LEN_MISMATCH);
            assert(slot_packed_offsets.len() == changes_len, Errors::METADATA_LEN_MISMATCH);
            assert(slot_initial_values.len() == changes_len, Errors::METADATA_LEN_MISMATCH);

            let mut i: u32 = 0;
            while i < changes_len {
                let key = *changed_keys[i];
                let value = *changed_values[i];
                let model_sel = *slot_model_selectors[i];
                let entity_id = *slot_entity_ids[i];
                let comp_key = *slot_computation_keys[i];
                let member_sel = *slot_member_selectors[i];
                let offset: felt252 = (*slot_packed_offsets[i]).into();
                let initial_value = *slot_initial_values[i];

                // ── SLOT OWNERSHIP VERIFICATION ──
                // Generic verification via computation_key: works for ALL layout
                // types (Fixed, Struct, Enum, Tuple, FixedArray, any nesting depth).
                // comp_key == 0 → lock slot (skip Poseidon verify, entity_lock sufficient).
                if comp_key != 0 {
                    let expected_key = poseidon_hash_span(
                        [DOJO_STORAGE, model_sel, comp_key].span(),
                    ) + offset;
                    assert(expected_key == key, Errors::SLOT_OWNERSHIP_MISMATCH);
                }

                // Derive Add CRDT flag from on-chain CRDT policies.
                let field_crdt = self.model_policy_field.read((model_sel, member_sel));
                let default_crdt = self.model_policy_default.read(model_sel);
                let effective_crdt = if field_crdt != 0 { field_crdt } else { default_crdt };
                let is_add = (effective_crdt == 2); // 2 = Add encoded

                // Entity lock enforcement:
                // - Add CRDT slots: exempt (delta merge is safe without exclusive lock)
                // - Packed offset > 0: exempt (base slot at offset 0 already verified)
                // - Set CRDT slots at offset 0: MUST belong to a locked entity
                if !is_add && offset == 0 {
                    assert(
                        self.entity_lock.read(entity_id) == shard_id,
                        Errors::ENTITY_NOT_IN_SHARD,
                    );
                }

                // ── CRDT WRITE ──
                let storage_address: StorageAddress = key.try_into().unwrap();
                if is_add {
                    let current_value = storage_read_syscall(0, storage_address).unwrap_syscall();
                    let current_u256: u256 = current_value.into();
                    let shard_u256: u256 = value.into();
                    let initial_u256: u256 = initial_value.into();
                    assert(shard_u256 >= initial_u256, Errors::ADD_DELTA_UNDERFLOW);
                    let delta = shard_u256 - initial_u256;
                    let sum = current_u256 + delta;
                    let new_value: felt252 = sum.try_into().expect(Errors::ARITHMETIC_OVERFLOW);
                    storage_write_syscall(0, storage_address, new_value).unwrap_syscall();
                } else {
                    storage_write_syscall(0, storage_address, value).unwrap_syscall();
                }

                i += 1;
            };

            // Note: StoreUpdateRecord events are emitted by the world contract
            // wrapper (settle/settle_dev) after apply_settle returns, using
            // entity reads in layout order for correct Serde-format values.

            self.unlock_entities(shard_id);
            self.clear_shard_state(shard_id);
        }

        fn unlock_entities(ref self: ComponentState<TContractState>, shard_id: felt252) {
            let count = self.shard_entity_count.read(shard_id);
            let mut i: u32 = 0;
            while i < count {
                let entity_id = self.shard_entities.read((shard_id, i));
                self.entity_lock.write(entity_id, 0);
                i += 1;
            };
        }

        fn clear_shard_state(ref self: ComponentState<TContractState>, shard_id: felt252) {
            let entity_count = self.shard_entity_count.read(shard_id);
            let mut i: u32 = 0;
            while i < entity_count {
                self.shard_entities.write((shard_id, i), 0);
                i += 1;
            };
            self.shard_entity_count.write(shard_id, 0);

            self.shard_commitment.write(shard_id, 0);

            // Remove from active shard list (swap-remove for O(1)).
            let count = self.active_shard_count.read();
            let mut idx: u32 = 0;
            let mut found = false;
            while idx < count {
                if self.active_shard_list.read(idx) == shard_id {
                    found = true;
                    break;
                }
                idx += 1;
            };
            if found {
                let last = count - 1;
                if idx != last {
                    let last_val = self.active_shard_list.read(last);
                    self.active_shard_list.write(idx, last_val);
                }
                self.active_shard_list.write(last, 0);
                self.active_shard_count.write(last);

                // Auto-reset fork mode when no active shards remain.
                // Prevents permanent bypass if accidentally enabled on main chain.
                if last == 0 && self.shard_fork_mode.read() {
                    self.shard_fork_mode.write(false);
                }
            }
        }
    }

    /// Sort a span of felt252 values (insertion sort, O(n²)).
    /// Suitable for small arrays (entity counts typically < 50).
    fn sort_felt_span(values: Span<felt252>) -> Span<felt252> {
        let len = values.len();
        if len <= 1 {
            return values;
        }

        // Copy into mutable array.
        let mut arr: Array<felt252> = ArrayTrait::new();
        let mut k: u32 = 0;
        while k < len {
            arr.append(*values[k]);
            k += 1;
        };

        // Insertion sort using a secondary array (Cairo arrays are append-only).
        // We build a sorted index array, then reconstruct.
        let mut indices: Array<u32> = ArrayTrait::new();
        indices.append(0);

        let mut i: u32 = 1;
        while i < len {
            let val_i: u256 = (*values[i]).into();
            // Find insertion position.
            let mut pos: u32 = 0;
            let current_len = indices.len();
            let indices_span = indices.span();
            loop {
                if pos >= current_len {
                    break;
                }
                let idx_at_pos: u32 = *indices_span[pos];
                let val_at_pos: u256 = (*values[idx_at_pos]).into();
                if val_i < val_at_pos {
                    break;
                }
                pos += 1;
            };
            // Insert at position `pos` by rebuilding.
            let mut new_indices: Array<u32> = ArrayTrait::new();
            let mut j: u32 = 0;
            while j < pos {
                new_indices.append(*indices_span[j]);
                j += 1;
            };
            new_indices.append(i);
            let mut j2: u32 = pos;
            while j2 < current_len {
                new_indices.append(*indices_span[j2]);
                j2 += 1;
            };
            indices = new_indices;
            i += 1;
        };

        // Build sorted output from indices.
        let mut result: Array<felt252> = ArrayTrait::new();
        let indices_span = indices.span();
        let mut m: u32 = 0;
        while m < len {
            result.append(*values[*indices_span[m]]);
            m += 1;
        };

        result.span()
    }
}
