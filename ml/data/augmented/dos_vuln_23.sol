// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln23 {

    address[] public stakers;
    mapping(address => uint256) public staked;
    function stake() external payable { stakers.push(msg.sender); staked[msg.sender] += msg.value; }
    function distributeRewards() external payable {
        uint256 totalStaked;
        for (uint256 i = 0; i < stakers.length; i++) { totalStaked += staked[stakers[i]]; }
        for (uint256 i = 0; i < stakers.length; i++) {
            uint256 reward = msg.value * staked[stakers[i]] / totalStaked;
            (bool ok,) = stakers[i].call{value: reward}("");
            require(ok);
        }
    }
}
