// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AuctionSafe07 {

    address public winner;
    uint256 public highBid;
    mapping(address => uint256) public bids;
    function bid() external payable {
        require(msg.value > highBid);
        highBid = msg.value; winner = msg.sender;
    }
    function claimRefund() external {
        require(msg.sender != winner);
        uint256 amt = bids[msg.sender];
        require(amt > 0);
        bids[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
    }
}
