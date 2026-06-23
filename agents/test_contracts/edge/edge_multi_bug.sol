// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title TimedAuctionVault - an auction vault with TWO independent bugs in
///        different vulnerability classes.
/// @notice Bug 1 (Reentrancy): withdraw() updates state AFTER the external
///         call (CEI violation). A malicious bidder can re-enter withdraw()
///         before their balance is decremented.
/// @notice Bug 2 (Timestamp): auctionEnd is derived from block.timestamp,
///         which a miner can manipulate within ~15s bounds. For a short
///         auction this lets a miner-backed bidder extend or shorten the
///         window. Pattern-matching tools miss timestamp use in arithmetic
///         schedule derivation.
/// @dev Tests per-class verdict independence: the system must flag BOTH
///      Reentrancy AND Timestamp on this single contract, not just one.
contract TimedAuctionVault {
    IERC20 public token;
    address public highestBidder;
    uint256 public highestBid;
    uint256 public auctionStart;
    uint256 public auctionEnd;
    mapping(address => uint256) public bids;

    event NewBid(address indexed bidder, uint256 amount);
    event Withdraw(address indexed bidder, uint256 amount);

    constructor(address _token, uint256 duration) {
        token = IERC20(_token);
        auctionStart = block.timestamp;       // Bug 2: miner-manipulable start
        auctionEnd = block.timestamp + duration;  // Bug 2: miner-manipulable end
    }

    function bid(uint256 amount) external {
        require(block.timestamp < auctionEnd, "auction ended");
        require(token.transferFrom(msg.sender, address(this), amount), "tf fail");
        bids[msg.sender] += amount;
        if (bids[msg.sender] > highestBid) {
            highestBid = bids[msg.sender];
            highestBidder = msg.sender;
            emit NewBid(msg.sender, bids[msg.sender]);
        }
    }

    function withdraw() external {
        require(block.timestamp >= auctionEnd, "auction not ended");
        uint256 amount = bids[msg.sender];
        require(amount > 0, "nothing to withdraw");
        // Bug 1: external call BEFORE state update - reentrancy.
        require(token.transfer(msg.sender, amount), "transfer failed");
        bids[msg.sender] = 0;
        emit Withdraw(msg.sender, amount);
    }

    function claim() external {
        require(block.timestamp >= auctionEnd, "auction not ended");
        require(msg.sender == highestBidder, "not winner");
        uint256 pot = token.balanceOf(address(this));
        require(token.transfer(highestBidder, pot), "claim fail");
    }
}

interface IERC20 {
    function transfer(address to, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}
