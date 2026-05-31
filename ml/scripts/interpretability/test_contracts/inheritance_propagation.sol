// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Tests inheritance propagation: Contract B inherits reentrancy vulnerability from A
contract ReentrancyBase {
    mapping(address => uint256) internal _balances;
    function withdrawBase(uint256 amount) internal {
        require(_balances[msg.sender] >= amount);
        (bool ok,) = msg.sender.call{value: amount}("");  // call before write
        require(ok);
        _balances[msg.sender] -= amount;  // write after call — CEI violation
    }
}
contract InheritancePropagation is ReentrancyBase {
    function deposit() external payable { _balances[msg.sender] += msg.value; }
    function withdraw(uint256 amount) external { withdrawBase(amount); }
}
