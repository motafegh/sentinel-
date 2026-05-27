// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe09 {

    mapping(address => bool) public hasReward;
    function markReward(address a) external { hasReward[a] = true; }
    function claimReward() external {
        require(hasReward[msg.sender]);
        hasReward[msg.sender] = false;
        (bool ok,) = msg.sender.call{value: 0.001 ether}("");
        require(ok);
    }
}
