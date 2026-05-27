// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe14 {

    mapping(address => uint256) public owed;
    function addOwed(address r, uint256 amt) external payable { owed[r] = amt; }
    function withdrawOwed() external {
        uint256 amt = owed[msg.sender];
        require(amt > 0);
        owed[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
