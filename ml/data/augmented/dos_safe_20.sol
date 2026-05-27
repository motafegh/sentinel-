// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe20 {

    mapping(address => uint256) public feedCredits;
    function creditFeed(address f, uint256 amt) external payable { feedCredits[f] += amt; }
    function withdrawFeedCredit() external {
        uint256 amt = feedCredits[msg.sender];
        require(amt > 0);
        feedCredits[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
