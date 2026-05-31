// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Safe: CEI pattern — state written before external call
contract ReentrancySafe {
    mapping(address => uint256) public balances;

    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "insufficient");
        balances[msg.sender] -= amount;  // write BEFORE call — CEI compliant
        (bool ok,) = msg.sender.call{value: amount}("");
        require(ok, "transfer failed");
    }
}
