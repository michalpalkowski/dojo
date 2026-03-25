#[cfg(test)]
mod utils;

#[cfg(test)]
mod tests {
    mod benches {
        mod model {
            mod access;
        }
        mod storage {
            mod database;
            mod layout;
            mod packing;
            mod storage;
        }
        mod utils {
            mod layout;
        }
        mod bench_data;
        mod bench_utils;
    }

    mod contract;

    mod event {
        mod event;
    }

    mod expanded {
        pub(crate) mod bytearray_hash;
        pub(crate) mod selector_attack;
    }

    mod helpers {
        mod helpers;
        pub use helpers::{
            Abilities, Balance256, Case, Character, DOJO_NSH, EnumOne, Foo, IFooSetter,
            IFooSetterDispatcher, IFooSetterDispatcherTrait, Ibar, IbarDispatcher,
            IbarDispatcherTrait, MixedDynamic, MyEnum, MyNestedEnum, NestedFixed, NestedStats,
            NotCopiable, PackedPair, Score, SimpleEvent, Stats, Sword, Tile, TupleArrayOption,
            Weapon, WithOptionAndEnums, bar, deploy_world, deploy_world_and_bar, deploy_world_and_foo,
            deploy_world_with_all_kind_of_resources, deploy_world_with_balance256,
            deploy_world_with_mixed_dynamic, deploy_world_with_nested_fixed,
            deploy_world_with_not_copiable, deploy_world_with_packed_pair, deploy_world_with_score,
            deploy_world_with_tile, deploy_world_with_tuple_array_option,
            e_SimpleEvent, foo_setter, m_Foo, m_FooInvalidName, m_MixedDynamic, m_NestedFixed,
            test_contract,
            test_contract_with_dojo_init_args,
        };

        mod event;
        pub use event::deploy_world_for_event_upgrades;

        mod model;
        pub use model::deploy_world_for_model_upgrades;

        mod library;
        pub use library::*;
    }

    mod meta {
        mod introspect;
        mod layout;
    }

    mod model {
        mod model;
    }

    mod storage {
        mod database;
        mod dojo_store;
        mod layout;
        mod packing;
        mod storage;
    }

    mod utils {
        mod hash;
        mod key;
        mod layout;
        mod naming;
    }

    mod world {
        mod acl;
        mod contract;
        mod event;
        mod external_contract;
        mod metadata;
        mod model;
        mod namespace;
        mod storage;
        mod world;
    }

    mod sharding {
        mod slot;
        mod component;
        mod request;
        mod settlement_events;
    }
}
