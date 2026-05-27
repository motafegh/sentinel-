// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe17 {

    mapping(address => bool) public whitelisted;
    mapping(address => uint256) public allocation;
    function addToWhitelist(address a) external { whitelisted[a] = true; }
    function setAllocation(address a, uint256 amt) external payable { allocation[a] = amt; }
    function claim() external {
        require(whitelisted[msg.sender]);
        uint256 amt = allocation[msg.sender];
        require(amt > 0);
        allocation[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
