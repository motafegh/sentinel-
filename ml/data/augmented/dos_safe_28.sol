// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe28 {

    mapping(address => uint256) public orderAmounts;
    function placeOrder() external payable { orderAmounts[msg.sender] += msg.value; }
    function refundOrder() external {
        uint256 amt = orderAmounts[msg.sender];
        require(amt > 0);
        orderAmounts[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
