// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IFlashReceiver { function onFlash(uint256 amt) external; }

contract FlashSafe11 {
    mapping(address => uint256) public reserves;
    function flashLoan(uint256 amt) external {
        uint256 before = address(this).balance;
        reserves[msg.sender] -= amt;               // state before callback
        IFlashReceiver(msg.sender).onFlash(amt);
        require(address(this).balance >= before);
    }
}
