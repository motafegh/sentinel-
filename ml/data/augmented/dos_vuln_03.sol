// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln03 {

    mapping(address => uint256) public balances;
    address[] public holders;
    function deposit() external payable { balances[msg.sender] += msg.value; holders.push(msg.sender); }
    function refundAll() external {
        for (uint256 i = 0; i < holders.length; i++) {
            uint256 amt = balances[holders[i]];
            if (amt == 0) revert("No balance");
            (bool ok,) = holders[i].call{value: amt}("");
            require(ok);
            balances[holders[i]] = 0;
        }
    }
}
