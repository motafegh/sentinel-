// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IFlashReceiver { function onFlash(uint256 amt) external; }

contract FlashVuln11 {
    mapping(address => uint256) public reserves;
    function flashLoan(uint256 amt) external {
        uint256 before = address(this).balance;
        IFlashReceiver(msg.sender).onFlash(amt);   // callback before balance check
        reserves[msg.sender] -= amt;
        require(address(this).balance >= before);
    }
}
