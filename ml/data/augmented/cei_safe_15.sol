// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ReferSafe15 {

    mapping(address => uint256) public referralRewards;
    function addReward(address ref, uint256 amt) external payable {
        referralRewards[ref] += amt;
    }
    function collectReward() external {
        uint256 reward = referralRewards[msg.sender];
        require(reward > 0);
        referralRewards[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: reward}("");
        require(ok);
    }
}
