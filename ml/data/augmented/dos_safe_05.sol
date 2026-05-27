// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe05 {

    mapping(address => uint256) public pendingPayout;
    function addPayout(address m, uint256 amt) external { pendingPayout[m] += amt; }
    function claim() external {
        uint256 amt = pendingPayout[msg.sender];
        require(amt > 0);
        pendingPayout[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
