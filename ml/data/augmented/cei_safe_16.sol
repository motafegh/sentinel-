// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 { function transfer(address, uint256) external returns (bool); }

contract YieldSafe16 {
    IERC20 public rewardToken;
    mapping(address => uint256) public depositTime;
    mapping(address => uint256) public deposited;
    constructor(address t) { rewardToken = IERC20(t); }
    function deposit() external payable { depositTime[msg.sender] = block.timestamp; deposited[msg.sender] = msg.value; }
    function harvest() external {
        uint256 elapsed = block.timestamp - depositTime[msg.sender];
        uint256 reward = deposited[msg.sender] * elapsed / 365 days;
        depositTime[msg.sender] = block.timestamp;   // state before external
        rewardToken.transfer(msg.sender, reward);
    }
}
