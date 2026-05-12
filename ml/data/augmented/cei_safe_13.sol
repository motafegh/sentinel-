// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SavingsSafe13 {

    mapping(address => uint256) public savings;
    mapping(address => uint256) public lastWithdraw;
    uint256 constant COOLDOWN = 1 days;
    function save() external payable { savings[msg.sender] += msg.value; }
    function withdraw(uint256 amt) external {
        require(savings[msg.sender] >= amt);
        require(block.timestamp >= lastWithdraw[msg.sender] + COOLDOWN);
        savings[msg.sender] -= amt;
        lastWithdraw[msg.sender] = block.timestamp;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
