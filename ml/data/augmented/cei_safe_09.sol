// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SubSafe09 {

    mapping(address => uint256) public expiry;
    mapping(address => uint256) public paid;
    function subscribe(uint256 dur) external payable {
        expiry[msg.sender] = block.timestamp + dur;
        paid[msg.sender] = msg.value;
    }
    function cancelRefund() external {
        require(block.timestamp < expiry[msg.sender]);
        uint256 refund = paid[msg.sender];
        require(refund > 0);
        paid[msg.sender] = 0;
        expiry[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: refund}("");
        require(ok);
    }
}
