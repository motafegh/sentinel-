pragma solidity ^0.8.0;
contract SafeStorage {
    uint256 public value;
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    function setValue(uint256 _value) external {
        require(msg.sender == owner, "Not owner");
        value = _value;
    }

    function getValue() external view returns (uint256) {
        return value;
    }
}