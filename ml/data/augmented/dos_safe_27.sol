// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe27 {

    mapping(address => uint256) public rewards;
    function registerMiner(address m) external pure {}
    function accrueReward(address m, uint256 amt) external { rewards[m] += amt; }
    function claimReward() external {
        uint256 amt = rewards[msg.sender];
        require(amt > 0);
        rewards[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
