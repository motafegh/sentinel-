// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe30 {

    struct Proposal { address payable recipient; uint256 amount; bool executed; uint256 voteCount; }
    Proposal[] public proposals;
    mapping(uint256 => mapping(address => bool)) public hasVoted;
    function createProposal(address payable r, uint256 a) external returns (uint256) {
        proposals.push(Proposal(r, a, false, 0));
        return proposals.length - 1;
    }
    function vote(uint256 id) external { require(!hasVoted[id][msg.sender]); hasVoted[id][msg.sender] = true; proposals[id].voteCount++; }
    function executeProposal(uint256 id) external {
        Proposal storage p = proposals[id];
        require(!p.executed);
        require(p.voteCount >= 3);
        p.executed = true;
        (bool ok,) = p.recipient.call{value: p.amount}("");
        require(ok);
    }
}
