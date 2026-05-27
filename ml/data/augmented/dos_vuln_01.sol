// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln01 {

    address[] public participants;
    uint256 public pot;
    function join() external payable {
        require(msg.value >= 0.01 ether);
        participants.push(msg.sender);
        pot += msg.value;
    }
    function distribute() external {
        uint256 share = pot / participants.length;
        for (uint256 i = 0; i < participants.length; i++) {
            payable(participants[i]).transfer(share);
        }
        pot = 0;
        delete participants;
    }
}
