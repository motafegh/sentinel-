// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln16 {

    address[] public investors;
    mapping(address => uint256) public invested;
    function invest() external payable { investors.push(msg.sender); invested[msg.sender] += msg.value; }
    function returnCapital() external {
        for (uint256 i = 0; i < investors.length; i++) {
            uint256 amt = invested[investors[i]];
            if (amt > 0) {
                payable(investors[i]).send(amt);
                invested[investors[i]] = 0;
            }
        }
    }
}
