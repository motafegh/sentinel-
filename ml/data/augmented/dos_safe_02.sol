// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe02 {

    mapping(address => uint256) public owed;
    function addOwed(address p, uint256 amt) external { owed[p] = amt; }
    function withdraw() external {
        uint256 amt = owed[msg.sender];
        require(amt > 0);
        owed[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
