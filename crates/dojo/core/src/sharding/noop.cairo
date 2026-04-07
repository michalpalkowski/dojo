/// Placeholder component used when the `sharding` feature is disabled.
/// Satisfies the `component!()` macro's type requirements with empty
/// Storage and Event — compiles to zero overhead (verified by measurement).
#[starknet::component]
pub mod sharding_noop {
    #[storage]
    pub struct Storage {}

    #[event]
    #[derive(Drop, starknet::Event)]
    pub enum Event {}
}
