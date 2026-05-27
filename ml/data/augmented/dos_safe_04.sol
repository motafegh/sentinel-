// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe04 {

    mapping(address => uint256) public stakes;
    mapping(address => uint256) public rewards;
    function stake() external payable { stakes[msg.sender] += msg.value; }
    function accumulateReward(uint256 rewardPerStaker) external {
        if (stakes[msg.sender] > 0) {
            rewards[msg.sender] += rewardPerStaker;
        }
    }
    function claimReward() external {
        uint256 amt = rewards[msg.sender];
        require(amt > 0);
        rewards[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
