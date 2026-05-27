// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe10 {

    mapping(address => uint256) public contributions;
    function contribute() external payable { contributions[msg.sender] += msg.value; }
    function withdrawContribution() external {
        uint256 amt = contributions[msg.sender];
        require(amt > 0);
        contributions[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
