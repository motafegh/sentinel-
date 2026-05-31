// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Safe: built-in overflow checks in ^0.8.0
contract IntegerUOSafe {
    mapping(address => uint256) public balances;
    function add(uint256 a, uint256 b) public pure returns (uint256) {
        return a + b;  // reverts on overflow in ^0.8.0
    }
    function transfer(address to, uint256 amount) external {
        require(balances[msg.sender] >= amount, "insufficient");
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }
}
