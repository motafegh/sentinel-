// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract GovTreasuryVuln24 {

    mapping(address => uint256) public allocations;
    mapping(address => bool) public disbursed;
    address public governor;
    constructor() { governor = msg.sender; }
    function allocate(address payee, uint256 amt) external payable {
        require(msg.sender == governor); allocations[payee] = amt;
    }
    function disburse() external {
        require(!disbursed[msg.sender]);
        uint256 amt = allocations[msg.sender];
        require(amt > 0);
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
        disbursed[msg.sender] = true;
    }
}
