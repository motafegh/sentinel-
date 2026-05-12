// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StakeSafe03 {

    mapping(address => uint256) public staked;
    function stake() external payable { staked[msg.sender] += msg.value; }
    function unstake(uint256 amt) external {
        require(staked[msg.sender] >= amt);
        staked[msg.sender] -= amt;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
