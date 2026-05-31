// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Safe inheritance: base contract uses CEI correctly
contract SafeBase {
    mapping(address => uint256) internal _balances;
    function withdrawBase(uint256 amount) internal {
        require(_balances[msg.sender] >= amount);
        _balances[msg.sender] -= amount;  // write before call — CEI compliant
        (bool ok,) = msg.sender.call{value: amount}("");
        require(ok);
    }
}
contract InheritanceSafe is SafeBase {
    function deposit() external payable { _balances[msg.sender] += msg.value; }
    function withdraw(uint256 amount) external { withdrawBase(amount); }
}
