// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln19 {

    interface IERC20 { function transfer(address to, uint256 amt) external returns (bool); }
    IERC20 public token;
    address[] public holders;
    constructor(address t) { token = IERC20(t); }
    function addHolder(address h) external { holders.push(h); }
    function airdrop(uint256 amt) external {
        for (uint256 i = 0; i < holders.length; i++) {
            token.transfer(holders[i], amt);
        }
    }
}
