// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract MarginVuln25 {

    mapping(address => uint256) public margin;
    mapping(address => uint256) public position;
    function openPosition() external payable { margin[msg.sender] = msg.value; position[msg.sender] = msg.value * 5; }
    function closePosition() external {
        uint256 pnl = margin[msg.sender];
        require(pnl > 0);
        (bool ok,) = msg.sender.call{value: pnl}("");
        require(ok);
        margin[msg.sender] = 0;
        position[msg.sender] = 0;
    }
}
