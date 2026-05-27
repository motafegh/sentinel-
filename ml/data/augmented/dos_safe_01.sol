// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe01 {

    mapping(address => uint256) public pendingWithdrawal;
    uint256 public pot;
    function join() external payable {
        require(msg.value >= 0.01 ether);
        pendingWithdrawal[msg.sender] += msg.value;
        pot += msg.value;
    }
    function withdraw() external {
        uint256 amount = pendingWithdrawal[msg.sender];
        require(amount > 0);
        pendingWithdrawal[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amount}("");
        require(ok);
    }
}
