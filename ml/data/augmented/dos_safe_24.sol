// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe24 {

    mapping(address => bool) public approved;
    function approve(address a) external { approved[a] = true; }
    function claimApproval() external {
        require(approved[msg.sender]);
        approved[msg.sender] = false;
        (bool ok,) = msg.sender.call{value: 0.1 ether}("");
        require(ok);
    }
}
