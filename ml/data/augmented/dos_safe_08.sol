// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe08 {

    mapping(address => uint256) public dividends;
    function addShareholder(address s) external pure {}
    function accrueDividend(address s, uint256 amt) external { dividends[s] += amt; }
    function claimDividend() external {
        uint256 amt = dividends[msg.sender];
        require(amt > 0);
        dividends[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
