// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe16 {

    mapping(address => uint256) public invested;
    function invest() external payable { invested[msg.sender] += msg.value; }
    function withdrawCapital() external {
        uint256 amt = invested[msg.sender];
        require(amt > 0);
        invested[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
