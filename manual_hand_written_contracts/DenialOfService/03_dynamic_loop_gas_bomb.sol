// expect: DenialOfService
// Governance proposal system that tallies votes by iterating all voters.
// The voter array grows each time someone delegates or votes.
// Once the voter list is large enough, any future tally runs out of gas.
// Proposal execution becomes impossible — governance is bricked.
// Also: the contract has a batch payout function that loops over creditor list.
pragma solidity ^0.8.0;

contract GovernanceVoter {
    struct Proposal {
        address proposer;
        string description;
        uint256 forVotes;
        uint256 againstVotes;
        bool executed;
        mapping(address => bool) hasVoted;
    }

    address public owner;
    address[] public voters;
    mapping(address => uint256) public votingPower;
    mapping(address => bool) public isVoter;
    Proposal[] public proposals;
    mapping(address => uint256) public pendingWithdrawals;

    event ProposalCreated(uint256 indexed proposalId, address indexed proposer);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function registerVoter(address voter, uint256 power) external onlyOwner {
        if (!isVoter[voter]) {
            voters.push(voter);
            isVoter[voter] = true;
        }
        votingPower[voter] = power;
    }

    function propose(string calldata desc) external returns (uint256) {
        uint256 id = proposals.length;
        proposals.push();
        Proposal storage p = proposals[id];
        p.proposer = msg.sender;
        p.description = desc;
        if (!isVoter[msg.sender]) {
            voters.push(msg.sender);
            isVoter[msg.sender] = true;
        }
        emit ProposalCreated(id, msg.sender);
        return id;
    }

    function vote(uint256 proposalId, bool support) external {
        require(isVoter[msg.sender], "not a voter");
        require(proposalId < proposals.length, "invalid proposal");
        Proposal storage p = proposals[proposalId];
        require(!p.hasVoted[msg.sender], "already voted");
        p.hasVoted[msg.sender] = true;
        uint256 power = votingPower[msg.sender];
        if (support) {
            p.forVotes += power;
        } else {
            p.againstVotes += power;
        }
    }

    function executeProposal(uint256 proposalId) external onlyOwner {
        Proposal storage p = proposals[proposalId];
        require(!p.executed, "already executed");
        for (uint256 i = 0; i < voters.length; i++) {
            if (p.hasVoted[voters[i]]) {
                if (votingPower[voters[i]] > 0) {
                    pendingWithdrawals[voters[i]] += 1 ether;
                }
            }
        }
        p.executed = true;
    }

    function batchDistributeRewards() external onlyOwner {
        uint256 count = voters.length;
        for (uint256 j = 0; j < count; j++) {
            for (uint256 k = 0; k < count; k++) {
                if (pendingWithdrawals[voters[k]] > 0) {
                    (bool ok, ) = payable(voters[k]).call{value: pendingWithdrawals[voters[k]]}("");
                    if (ok) {
                        pendingWithdrawals[voters[k]] = 0;
                    }
                }
            }
        }
    }

    function getVoterCount() external view returns (uint256) {
        return voters.length;
    }

    receive() external payable {}
}