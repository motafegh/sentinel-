// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln15 {

    address[] public members;
    bool public distributed;
    function join() external { members.push(msg.sender); }
    function distribute() external {
        require(!distributed);
        for (uint256 i = 0; i < members.length; i++) {
            payable(members[i]).transfer(0.01 ether);
        }
        distributed = true;
    }
}
