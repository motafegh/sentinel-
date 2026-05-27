// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln25 {

    address[] public tokenHolders;
    mapping(address => uint256) public holdings;
    function addHolder(address h, uint256 tokens) external { tokenHolders.push(h); holdings[h] = tokens; }
    function payDividends() external payable {
        for (uint256 i = 0; i < tokenHolders.length; i++) {
            uint256 div = msg.value * holdings[tokenHolders[i]] / 10000;
            payable(tokenHolders[i]).transfer(div);
        }
    }
}
