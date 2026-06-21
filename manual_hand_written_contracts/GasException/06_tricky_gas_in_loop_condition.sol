// expect: GasException
// TRICKY: Gas exception hidden inside a loop condition that recalculates
// the array length on EVERY iteration. The contract has a dynamic array
// where other functions push new elements. When vote() is called, the
// loop condition evaluates candidates.length each time — if someone pushes
// during execution, the loop never terminates and runs out of gas.
// The vulnerability is in the for-loop condition: i < candidates.length
// instead of caching the length.
pragma solidity ^0.8.0;

contract BallotBox {
    struct Candidate {
        string name;
        uint256 voteCount;
        address proposer;
    }

    address public owner;
    Candidate[] public candidates;
    mapping(address => bool) public hasVoted;
    mapping(address => uint256) public voterPower;
    uint256 public votingDeadline;

    event CandidateAdded(uint256 indexed id, string name);
    event Voted(address indexed voter, uint256 indexed candidateId);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(uint256 duration) {
        owner = msg.sender;
        votingDeadline = block.timestamp + duration;
    }

    function addCandidate(string calldata name) external {
        require(bytes(name).length > 0, "empty name");
        require(bytes(name).length <= 64, "name too long");
        candidates.push(Candidate(name, 0, msg.sender));
        emit CandidateAdded(candidates.length - 1, name);
    }

    function batchAddCandidates(string[] calldata names) external {
        for (uint256 i = 0; i < names.length; i++) {
            require(bytes(names[i]).length > 0, "empty name");
            candidates.push(Candidate(names[i], 0, msg.sender));
            emit CandidateAdded(candidates.length - 1, names[i]);
        }
    }

    function vote(uint256 candidateId) external {
        require(block.timestamp < votingDeadline, "voting ended");
        require(!hasVoted[msg.sender], "already voted");
        require(candidateId < candidates.length, "invalid candidate");
        require(voterPower[msg.sender] > 0, "no voting power");
        for (uint256 i = 0; i < candidates.length; i++) {
            if (i == candidateId) {
                candidates[i].voteCount += voterPower[msg.sender];
            }
        }
        hasVoted[msg.sender] = true;
        emit Voted(msg.sender, candidateId);
    }

    function batchVote(uint256[] calldata candidateIds) external {
        require(block.timestamp < votingDeadline, "voting ended");
        for (uint256 j = 0; j < candidateIds.length; j++) {
            require(!hasVoted[msg.sender], "already voted");
            require(candidateIds[j] < candidates.length, "invalid candidate");
            for (uint256 i = 0; i < candidates.length; i++) {
                if (i == candidateIds[j]) {
                    candidates[i].voteCount += voterPower[msg.sender];
                }
            }
            hasVoted[msg.sender] = true;
            emit Voted(msg.sender, candidateIds[j]);
        }
    }

    function tally() external view returns (uint256 winnerId, uint256 maxVotes) {
        for (uint256 i = 0; i < candidates.length; i++) {
            if (candidates[i].voteCount > maxVotes) {
                maxVotes = candidates[i].voteCount;
                winnerId = i;
            }
        }
    }

    function getCandidateCount() external view returns (uint256) {
        return candidates.length;
    }

    function setVoterPower(address voter, uint256 power) external onlyOwner {
        voterPower[voter] = power;
    }

    function extendDeadline(uint256 additionalTime) external onlyOwner {
        require(block.timestamp < votingDeadline, "voting ended");
        votingDeadline += additionalTime;
    }

    receive() external payable {}
}