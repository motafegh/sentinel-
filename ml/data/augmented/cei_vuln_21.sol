// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract GrantVuln21 {

    mapping(address => uint256) public grantAmount;
    mapping(address => bool) public grantClaimed;
    address public owner;
    constructor() { owner = msg.sender; }
    function allocate(address grantee, uint256 amt) external payable { grantAmount[grantee] = amt; }
    function claimGrant() external {
        require(!grantClaimed[msg.sender]);
        uint256 amt = grantAmount[msg.sender];
        require(amt > 0);
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
        grantClaimed[msg.sender] = true;
    }
}
