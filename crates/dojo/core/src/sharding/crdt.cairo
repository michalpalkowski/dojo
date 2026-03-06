use starknet::ContractAddress;

pub type SlotKey = felt252;
pub type SlotValue = felt252;

#[derive(Drop, Serde, Hash, Copy, Debug, PartialEq, starknet::Store)]
pub enum CRDType {
    Add: (ContractAddress, SlotValue),
    SetLock: (ContractAddress, SlotValue),
    #[default]
    Set: (ContractAddress, SlotValue),
    Lock: (ContractAddress, SlotValue),
}

pub trait CRDTypeTrait {
    fn assert_is_base_set(self: CRDType);
    fn is_same_variant(self: CRDType, other: CRDType) -> bool;
    fn contract_address(self: CRDType) -> ContractAddress;
    fn slot(self: CRDType) -> SlotValue;
}

pub impl CRDTypeImpl of CRDTypeTrait {
    /// Asserts this type is the base `Set` state (init_count == 0).
    /// Set can transition to any type — this is the only valid starting point.
    fn assert_is_base_set(self: CRDType) {
        let is_valid = match self {
            CRDType::Set(_) => true,
            _ => false,
        };
        assert(is_valid, 'Component: Already initialized');
    }

    fn is_same_variant(self: CRDType, other: CRDType) -> bool {
        match (self, other) {
            (CRDType::Add(_), CRDType::Add(_)) => true,
            (CRDType::SetLock(_), CRDType::SetLock(_)) => true,
            (CRDType::Set(_), CRDType::Set(_)) => true,
            (CRDType::Lock(_), CRDType::Lock(_)) => true,
            _ => false,
        }
    }

    fn contract_address(self: CRDType) -> ContractAddress {
        match self {
            CRDType::Add((address, _)) | CRDType::SetLock((address, _)) |
            CRDType::Set((address, _)) | CRDType::Lock((address, _)) => address,
        }
    }

    fn slot(self: CRDType) -> felt252 {
        match self {
            CRDType::Add((_, slot)) | CRDType::SetLock((_, slot)) | CRDType::Set((_, slot)) |
            CRDType::Lock((_, slot)) => slot,
        }
    }
}

/// Increment a felt252 by 1 with overflow protection via u256.
pub fn safe_increment(value: felt252, error_msg: felt252) -> felt252 {
    let as_u256: u256 = value.into();
    (as_u256 + 1).try_into().expect(error_msg)
}
