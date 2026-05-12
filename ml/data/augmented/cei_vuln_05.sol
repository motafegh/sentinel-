// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract CrowdVuln05 {

    mapping(address => uint256) public contributions;
    bool public failed;
    function contribute() external payable { contributions[msg.sender] += msg.value; }
    function refund() external {
        require(failed);
        uint256 amt = contributions[msg.sender];
        require(amt > 0);
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
        contributions[msg.sender] = 0;
    }
}
