// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Vulnerable: explicit unchecked arithmetic — overflows silently (0.8.x equivalent of pre-0.8 behavior)
contract IntegerUOVulnerable {
    mapping(address => uint256) public balances;

    function add(uint256 a, uint256 b) public pure returns (uint256) {
        unchecked {
            return a + b;  // overflows silently — no revert on wrap-around
        }
    }

    function transfer(address to, uint256 amount) external {
        unchecked {
            balances[msg.sender] -= amount;  // underflows silently if insufficient
            balances[to] += amount;
        }
    }

    function withdraw(uint256 amount) external {
        unchecked {
            // Attacker can pass type(uint256).max to underflow balances[msg.sender]
            // and get a huge balance before the actual transfer
            balances[msg.sender] -= amount;
        }
    }
}
