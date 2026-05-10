// expect: Timestamp
// Miner-manipulable timestamp used for lottery outcome and unlock window.
// Both critical decisions depend on block.timestamp.
pragma solidity ^0.8.0;

contract TimestampLottery {
    address public lastWinner;
    uint256 public jackpot;
    uint256 public unlockTime;

    constructor() payable {
        jackpot = msg.value;
        // VULNERABILITY: unlock window set from timestamp — miner can shift by ~15 seconds
        unlockTime = block.timestamp + 1 days;
    }

    function enter() external payable {
        require(msg.value >= 0.01 ether);
        jackpot += msg.value;
    }

    function draw() external {
        require(block.timestamp >= unlockTime);
        // VULNERABILITY: winner determined by timestamp mod — miner influence
        if (block.timestamp % 15 == 0) {
            lastWinner = msg.sender;
            uint256 prize = jackpot;
            jackpot = 0;
            payable(msg.sender).transfer(prize);
        }
        unlockTime = block.timestamp + 1 days;
    }
}
