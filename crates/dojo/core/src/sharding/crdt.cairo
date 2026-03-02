use starknet::ContractAddress;

pub type slot_key = felt252;
pub type slot_value = felt252;

#[derive(Drop, Serde, Hash, Copy, Debug, PartialEq, starknet::Store)]
pub enum CRDType {
    Add: (ContractAddress, slot_value),
    SetLock: (ContractAddress, slot_value),
    #[default]
    Set: (ContractAddress, slot_value),
    Lock: (ContractAddress, slot_value),
}

pub trait CRDTypeTrait {
    fn verify_crd_type(self: CRDType, crd_type: CRDType);
    fn is_same_variant(self: CRDType, other: CRDType) -> bool;
    fn contract_address(self: CRDType) -> ContractAddress;
    fn slot(self: CRDType) -> slot_value;
}

pub impl CRDTypeImpl of CRDTypeTrait {
    fn verify_crd_type(self: CRDType, crd_type: CRDType) {
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
