// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract InsureVuln14 {

    mapping(address => uint256) public coverage;
    mapping(address => bool) public claimed;
    function insure() external payable { coverage[msg.sender] = msg.value * 2; }
    function claim() external {
        require(!claimed[msg.sender]);
        uint256 payout = coverage[msg.sender];
        require(payout > 0);
        (bool ok,) = msg.sender.call{value: payout}("");
        require(ok);
        claimed[msg.sender] = true;
    }
}
