// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln10 {

    mapping(address => uint256) public contributions;
    address[] public contributors;
    function contribute() external payable { contributions[msg.sender] += msg.value; contributors.push(msg.sender); }
    function refundAll() external {
        for (uint256 i = 0; i < contributors.length; i++) {
            uint256 amt = contributions[contributors[i]];
            if (amt > 0) {
                payable(contributors[i]).transfer(amt);
                contributions[contributors[i]] = 0;
            }
        }
    }
}
