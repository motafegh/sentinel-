// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe06 {

    mapping(address => uint256) public locked;
    function lock() external payable { locked[msg.sender] += msg.value; }
    function unlock() external {
        uint256 amt = locked[msg.sender];
        require(amt > 0);
        locked[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
