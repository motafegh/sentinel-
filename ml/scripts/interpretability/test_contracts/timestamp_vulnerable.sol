// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Vulnerable: uses block.timestamp for critical logic
contract TimestampVulnerable {
    uint256 public deadline;
    address public winner;
    function setDeadline(uint256 d) external { deadline = d; }
    function claimPrize() external {
        require(block.timestamp > deadline, "too early");  // timestamp manipulation
        winner = msg.sender;
    }
}
