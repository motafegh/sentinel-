// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe19 {

    interface IERC20 { function transfer(address to, uint256 amt) external returns (bool); }
    IERC20 public token;
    mapping(address => uint256) public pendingAirdrop;
    constructor(address t) { token = IERC20(t); }
    function registerAirdrop(address h, uint256 amt) external { pendingAirdrop[h] = amt; }
    function claimAirdrop() external {
        uint256 amt = pendingAirdrop[msg.sender];
        require(amt > 0);
        pendingAirdrop[msg.sender] = 0;
        token.transfer(msg.sender, amt);
    }
}
