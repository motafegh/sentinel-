// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract WithdrawSafe01 {

    mapping(address => uint256) public balances;
    function deposit() external payable { balances[msg.sender] += msg.value; }
    function withdraw() external {
        uint256 amt = balances[msg.sender];
        require(amt > 0);
        balances[msg.sender] = 0;                      // update before call
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
