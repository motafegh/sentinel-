// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Safe: uses block.number instead of block.timestamp
contract TimestampSafe {
    uint256 public deadlineBlock;
    address public winner;
    function setDeadline(uint256 d) external { deadlineBlock = d; }
    function claimPrize() external {
        require(block.number > deadlineBlock, "too early");  // block number is safer
        winner = msg.sender;
    }
}
