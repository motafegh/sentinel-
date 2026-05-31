// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Vulnerable: transaction order dependence — first-come-first-served with state
contract TODVulnerable {
    address public claimant;
    uint256 public prize = 1 ether;
    function claim() external payable {
        require(msg.value >= prize);
        claimant = msg.sender;  // winner depends on tx ordering
        prize = msg.value;
    }
    function withdraw() external {
        require(msg.sender == claimant);
        payable(claimant).transfer(address(this).balance);
    }
}
