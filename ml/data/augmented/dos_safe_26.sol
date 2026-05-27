// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe26 {

    mapping(address => uint256) public airdropAmount;
    function register(address r, uint256 amt) external { airdropAmount[r] = amt; }
    function claimAirdrop() external {
        uint256 amt = airdropAmount[msg.sender];
        require(amt > 0);
        airdropAmount[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
