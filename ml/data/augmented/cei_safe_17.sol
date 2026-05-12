// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AirdropSafe17 {

    mapping(address => bool) public hasClaimed;
    mapping(address => uint256) public allocation;
    function setAllocation(address a, uint256 amt) external payable { allocation[a] = amt; }
    function claimAirdrop() external {
        require(!hasClaimed[msg.sender]);
        uint256 amt = allocation[msg.sender];
        require(amt > 0);
        hasClaimed[msg.sender] = true;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
