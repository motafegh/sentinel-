// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 { function transfer(address to, uint256 amt) external returns (bool); }

contract RewardVuln02 {
    IERC20 public token;
    mapping(address => uint256) public pending;
    constructor(address t) { token = IERC20(t); }
    function claim() external {
        uint256 amt = pending[msg.sender];
        require(amt > 0);
        token.transfer(msg.sender, amt);   // external call before state clear
        pending[msg.sender] = 0;
    }
}
