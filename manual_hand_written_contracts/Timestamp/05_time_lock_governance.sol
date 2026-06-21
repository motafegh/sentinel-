// expect: Timestamp
// Governance contract that uses block.timestamp for proposal voting deadlines
// and execution timelocks. A miner can influence the outcome of a vote
// by manipulating the timestamp at which voting ends — delaying or advancing
// the deadline by a few blocks to swing a close vote.
// The timelock on execution also uses block.timestamp for the delay calculation.
pragma solidity ^0.8.0;

contract TimelockGovernance {
    struct Proposal {
        address target;
        bytes data;
        uint256 value;
        uint256 voteStart;
        uint256 voteEnd;
        uint256 forVotes;
        uint256 againstVotes;
        bool executed;
        mapping(address => bool) voted;
    }

    address public owner;
    mapping(address => uint256) public votingPower;
    Proposal[] public proposals;
    uint256 public votingPeriod = 3 days;
    uint256 public timelockDelay = 2 days;
    uint256 public quorum = 1000;

    event ProposalCreated(uint256 indexed id, address indexed target);
    event Voted(uint256 indexed id, address indexed voter, bool support);
    event ProposalExecuted(uint256 indexed id);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function setVotingPower(address voter, uint256 power) external onlyOwner {
        votingPower[voter] = power;
    }

    function propose(address target, bytes calldata data, uint256 value) external onlyOwner returns (uint256) {
        uint256 id = proposals.length;
        proposals.push();
        Proposal storage p = proposals[id];
        p.target = target;
        p.data = data;
        p.value = value;
        p.voteStart = block.timestamp;
        p.voteEnd = block.timestamp + votingPeriod;
        emit ProposalCreated(id, target);
        return id;
    }

    function vote(uint256 proposalId, bool support) external {
        require(proposalId < proposals.length, "invalid proposal");
        Proposal storage p = proposals[proposalId];
        require(block.timestamp >= p.voteStart, "voting not started");
        require(block.timestamp < p.voteEnd, "voting ended");
        require(!p.voted[msg.sender], "already voted");
        require(votingPower[msg.sender] > 0, "no voting power");
        p.voted[msg.sender] = true;
        if (support) {
            p.forVotes += votingPower[msg.sender];
        } else {
            p.againstVotes += votingPower[msg.sender];
        }
        emit Voted(proposalId, msg.sender, support);
    }

    function executeProposal(uint256 proposalId) external {
        require(proposalId < proposals.length, "invalid proposal");
        Proposal storage p = proposals[proposalId];
        require(!p.executed, "already executed");
        require(block.timestamp >= p.voteEnd, "voting not ended");
        require(block.timestamp >= p.voteEnd + timelockDelay, "timelock active");
        require(p.forVotes > p.againstVotes, "proposal defeated");
        require(p.forVotes >= quorum, "quorum not met");
        p.executed = true;
        (bool ok, ) = p.target.call{value: p.value}(p.data);
        require(ok, "execution failed");
        emit ProposalExecuted(proposalId);
    }

    function cancelProposal(uint256 proposalId) external onlyOwner {
        require(proposalId < proposals.length, "invalid proposal");
        Proposal storage p = proposals[proposalId];
        require(!p.executed, "already executed");
        p.executed = true;
    }

    function setVotingPeriod(uint256 newPeriod) external onlyOwner {
        votingPeriod = newPeriod;
    }

    function setTimelockDelay(uint256 newDelay) external onlyOwner {
        timelockDelay = newDelay;
    }

    function setQuorum(uint256 newQuorum) external onlyOwner {
        quorum = newQuorum;
    }

    function getProposalVotes(uint256 proposalId) external view returns (uint256 forVotes, uint256 againstVotes, bool ended) {
        require(proposalId < proposals.length, "invalid proposal");
        Proposal storage p = proposals[proposalId];
        ended = block.timestamp >= p.voteEnd;
        return (p.forVotes, p.againstVotes, ended);
    }
}