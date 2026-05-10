// expect: Reentrancy,Timestamp,UnusedReturn
// Multi-vulnerability contract: reentrancy + timestamp + ignored return values.
// Tests whether the model can detect multiple co-occurring classes.
pragma solidity ^0.8.0;

interface IToken {
    function transfer(address to, uint256 amount) external returns (bool);
}

contract MultiVulnStaking {
    IToken public rewardToken;
    mapping(address => uint256) public stakedAt;
    mapping(address => uint256) public stakeAmount;

    constructor(address token) {
        rewardToken = IToken(token);
    }

    function stake() external payable {
        require(msg.value > 0);
        stakeAmount[msg.sender] += msg.value;
        // VULNERABILITY: timestamp used to record stake time — miner manipulable
        stakedAt[msg.sender] = block.timestamp;
    }

    function unstake() external {
        uint256 amount = stakeAmount[msg.sender];
        require(amount > 0);

        // VULNERABILITY: reward based on timestamp diff — miner can inflate duration
        uint256 duration = block.timestamp - stakedAt[msg.sender];
        uint256 reward = (amount * duration) / 1 days;

        // VULNERABILITY: state update AFTER external call — reentrancy
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok);

        // VULNERABILITY: token transfer return value ignored
        rewardToken.transfer(msg.sender, reward);

        // state updated after calls — too late
        stakeAmount[msg.sender] = 0;
        stakedAt[msg.sender] = 0;
    }
}
