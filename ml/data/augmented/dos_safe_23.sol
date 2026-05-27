// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe23 {

    mapping(address => uint256) public staked;
    mapping(address => uint256) public rewards;
    uint256 public totalStaked;
    function stake() external payable { staked[msg.sender] += msg.value; totalStaked += msg.value; }
    function accumulateRewards() external payable {
        if (totalStaked > 0 && staked[msg.sender] > 0) {
            rewards[msg.sender] += msg.value * staked[msg.sender] / totalStaked;
        }
    }
    function claimRewards() external {
        uint256 amt = rewards[msg.sender];
        require(amt > 0);
        rewards[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
