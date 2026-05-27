// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe07 {

    mapping(address => uint256) public credits;
    function addCredit(address r, uint256 amt) external { credits[r] += amt; }
    function claim() external {
        uint256 amt = credits[msg.sender];
        require(amt > 0);
        credits[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
