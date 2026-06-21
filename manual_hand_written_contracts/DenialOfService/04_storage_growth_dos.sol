// expect: DenialOfService
// On-chain message board that stores every entry in a dynamic array.
// Each message push costs more gas as the array grows (storage expansion).
// The contract has a function that loops over ALL entries to compute a summary.
// Once enough messages are stored, summary computation exceeds block gas limit.
// The contract cannot be deleted — only paused. Storage costs are permanent.
pragma solidity ^0.8.0;

contract MessageBoard {
    struct Message {
        address sender;
        uint256 timestamp;
        string content;
        uint256 blockNumber;
    }

    address public owner;
    Message[] public messages;
    mapping(address => uint256[]) public userMessageIds;
    mapping(address => uint256) public messageCount;
    bool public paused;
    uint256 public totalCharacters;

    event MessagePosted(uint256 indexed id, address indexed sender);
    event ContractPaused(address indexed pauser);

    modifier notPaused() {
        require(!paused, "contract paused");
        _;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function postMessage(string calldata content) external notPaused returns (uint256) {
        require(bytes(content).length > 0, "empty message");
        require(bytes(content).length <= 1024, "message too long");
        uint256 id = messages.length;
        messages.push(Message(msg.sender, block.timestamp, content, block.number));
        userMessageIds[msg.sender].push(id);
        messageCount[msg.sender]++;
        totalCharacters += bytes(content).length;
        emit MessagePosted(id, msg.sender);
        return id;
    }

    function batchPost(string[] calldata contents) external notPaused {
        for (uint256 i = 0; i < contents.length; i++) {
            require(bytes(contents[i]).length > 0, "empty message");
            uint256 id = messages.length;
            messages.push(Message(msg.sender, block.timestamp, contents[i], block.number));
            userMessageIds[msg.sender].push(id);
            messageCount[msg.sender]++;
            totalCharacters += bytes(contents[i]).length;
            emit MessagePosted(id, msg.sender);
        }
    }

    function computeSummary() external view returns (uint256 totalMsgs, uint256 avgLength, uint256 uniqueSenders) {
        totalMsgs = messages.length;
        if (totalMsgs == 0) return (0, 0, 0);
        avgLength = totalCharacters / totalMsgs;
        address[] memory seen = new address[](totalMsgs);
        uint256 seenCount = 0;
        for (uint256 i = 0; i < totalMsgs; i++) {
            address sender = messages[i].sender;
            bool found = false;
            for (uint256 j = 0; j < seenCount; j++) {
                if (seen[j] == sender) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                seen[seenCount] = sender;
                seenCount++;
            }
        }
        uniqueSenders = seenCount;
    }

    function getUserMessages(address user) external view returns (uint256[] memory) {
        return userMessageIds[user];
    }

    function togglePause() external onlyOwner {
        paused = !paused;
        emit ContractPaused(msg.sender);
    }

    function recoverFunds() external onlyOwner {
        (bool ok, ) = owner.call{value: address(this).balance}("");
        require(ok, "recovery failed");
    }
}