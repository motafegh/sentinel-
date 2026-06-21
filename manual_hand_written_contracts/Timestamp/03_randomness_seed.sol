// expect: Timestamp
// Lottery and gaming contract that uses block.timestamp as a source of randomness.
// Miners can influence block.timestamp by up to ~15 seconds, giving them
// an advantage in predicting or manipulating lottery outcomes.
// Combined with blockhash as a seed, the miner can grind through timestamps
// to find a favorable outcome before releasing the block.
pragma solidity ^0.8.0;

contract TimestampLottery {
    address public owner;
    uint256 public ticketPrice = 0.01 ether;
    address[] public tickets;
    mapping(address => uint256) public ticketCount;
    uint256 public roundNumber;
    bool public drawPaused;

    event TicketPurchased(address indexed buyer, uint256 count);
    event WinnerSelected(address indexed winner, uint256 prize);
    event RoundReset(uint256 roundNumber);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function buyTicket(uint256 count) external payable {
        require(count > 0, "zero count");
        require(msg.value == ticketPrice * count, "incorrect payment");
        require(!drawPaused, "draw paused");
        for (uint256 i = 0; i < count; i++) {
            tickets.push(msg.sender);
        }
        ticketCount[msg.sender] += count;
        emit TicketPurchased(msg.sender, count);
    }

    function drawWinner() external onlyOwner {
        require(tickets.length > 0, "no tickets");
        uint256 seed = uint256(keccak256(abi.encodePacked(block.timestamp, block.prevrandao, block.number)));
        uint256 winnerIndex = seed % tickets.length;
        address winner = tickets[winnerIndex];
        uint256 prize = address(this).balance;
        (bool ok, ) = winner.call{value: prize}("");
        require(ok, "prize transfer failed");
        roundNumber++;
        delete tickets;
        emit WinnerSelected(winner, prize);
        emit RoundReset(roundNumber);
    }

    function drawWinnerWithHash() external onlyOwner {
        require(tickets.length > 0, "no tickets");
        bytes32 blockHash = blockhash(block.number - 1);
        if (blockHash == bytes32(0)) {
            blockHash = keccak256(abi.encodePacked(block.timestamp));
        }
        uint256 seed = uint256(keccak256(abi.encodePacked(block.timestamp, blockHash, tickets.length)));
        uint256 winnerIndex = seed % tickets.length;
        address winner = tickets[winnerIndex];
        uint256 prize = address(this).balance;
        (bool ok, ) = winner.call{value: prize}("");
        require(ok, "prize transfer failed");
        roundNumber++;
        delete tickets;
        emit WinnerSelected(winner, prize);
        emit RoundReset(roundNumber);
    }

    function multiRoundDraw(uint256 rounds) external onlyOwner {
        for (uint256 r = 0; r < rounds; r++) {
            if (tickets.length == 0) break;
            uint256 seed = uint256(keccak256(abi.encodePacked(block.timestamp, r, block.prevrandao)));
            uint256 winnerIndex = seed % tickets.length;
            address winner = tickets[winnerIndex];
            uint256 share = address(this).balance / (rounds - r);
            (bool ok, ) = winner.call{value: share}("");
            require(ok, "multi round transfer failed");
            delete tickets[winnerIndex];
            tickets.pop();
            roundNumber++;
            emit WinnerSelected(winner, share);
        }
    }

    function togglePause() external onlyOwner {
        drawPaused = !drawPaused;
    }

    function getTicketCount() external view returns (uint256) {
        return tickets.length;
    }

    receive() external payable {}
}