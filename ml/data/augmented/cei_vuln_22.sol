// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract LotteryVuln22 {

    address[] public tickets;
    uint256 public ticketPrice = 0.01 ether;
    bool public drawn;
    address public winner;
    function buyTicket() external payable { require(msg.value == ticketPrice); tickets.push(msg.sender); }
    function draw() external {
        require(!drawn);
        drawn = true;
        winner = tickets[block.timestamp % tickets.length];
    }
    function claimPrize() external {
        require(drawn && msg.sender == winner);
        uint256 prize = address(this).balance;
        (bool ok,) = msg.sender.call{value: prize}("");
        require(ok);
        winner = address(0);     // clear after call
    }
}
