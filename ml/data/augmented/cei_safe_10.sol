// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DaoSafe10 {

    struct Proposal { address payable recipient; uint256 amount; bool executed; }
    Proposal[] public proposals;
    mapping(uint256 => mapping(address => bool)) public voted;
    mapping(uint256 => uint256) public voteCount;
    function propose(address payable r, uint256 a) external payable returns (uint256) {
        proposals.push(Proposal(r, a, false)); return proposals.length - 1;
    }
    function vote(uint256 id) external { require(!voted[id][msg.sender]); voted[id][msg.sender] = true; voteCount[id]++; }
    function execute(uint256 id) external {
        Proposal storage p = proposals[id];
        require(!p.executed && voteCount[id] >= 2);
        p.executed = true;
        (bool ok,) = p.recipient.call{value: p.amount}("");
        require(ok);
    }
}
