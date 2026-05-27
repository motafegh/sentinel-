// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe15 {

    mapping(address => bool) public isMember;
    mapping(address => bool) public claimed;
    function join() external { isMember[msg.sender] = true; }
    function claim() external {
        require(isMember[msg.sender] && !claimed[msg.sender]);
        claimed[msg.sender] = true;
        (bool ok,) = msg.sender.call{value: 0.01 ether}("");
        require(ok);
    }
}
