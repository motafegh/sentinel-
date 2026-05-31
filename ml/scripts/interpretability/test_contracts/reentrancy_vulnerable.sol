// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Vulnerable: classic CEI violation — external call before state write
contract ReentrancyVulnerable {
    mapping(address => uint256) public balances;

    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "insufficient");
        (bool ok,) = msg.sender.call{value: amount}("");  // call BEFORE write — CEI violation
        require(ok, "transfer failed");
        balances[msg.sender] -= amount;  // state write AFTER external call
    }
}
