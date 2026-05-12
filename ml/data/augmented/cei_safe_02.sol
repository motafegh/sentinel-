// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 { function transfer(address to, uint256 amt) external returns (bool); }

contract RewardSafe02 {
    IERC20 public token;
    mapping(address => uint256) public pending;
    constructor(address t) { token = IERC20(t); }
    function claim() external {
        uint256 amt = pending[msg.sender];
        require(amt > 0);
        pending[msg.sender] = 0;           // clear before external call
        token.transfer(msg.sender, amt);
    }
}
