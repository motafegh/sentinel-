// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe25 {

    mapping(address => uint256) public holdings;
    mapping(address => uint256) public dividendBalance;
    function addHolder(address h, uint256 tokens) external { holdings[h] = tokens; }
    function accrueDividend(address h, uint256 amt) external payable { dividendBalance[h] += amt; }
    function withdrawDividend() external {
        uint256 amt = dividendBalance[msg.sender];
        require(amt > 0);
        dividendBalance[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
