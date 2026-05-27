// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln04 {

    mapping(address => uint256) public stakes;
    address[] public stakers;
    function stake() external payable { stakes[msg.sender] += msg.value; stakers.push(msg.sender); }
    function distributeReward(uint256 rewardPerStaker) external {
        for (uint256 i = 0; i < stakers.length; i++) {
            assert(stakes[stakers[i]] > 0);
            (bool ok,) = stakers[i].call{value: rewardPerStaker}("");
            require(ok);
        }
    }
}
