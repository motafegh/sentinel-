// expect: DenialOfService,Reentrancy,UnusedReturn
// BCCC-derived staking pool with THREE injected vulnerabilities:
// 1) DenialOfService — unbounded loop over all stakers in distributeRewards
// 2) Reentrancy — CEI violation: send ETH before updating rewards
// 3) UnusedReturn — transfer return value silently ignored in batch
// This tests the model's ability to detect multiple overlapping vulnerability classes
pragma solidity ^0.4.24;

library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) { uint256 c = a + b; require(c >= a); return c; }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) { require(b <= a); return a - b; }
    function mul(uint256 a, uint256 b) internal pure returns (uint256) { if (a == 0) return 0; uint256 c = a * b; require(c / a == b); return c; }
    function div(uint256 a, uint256 b) internal pure returns (uint256) { require(b > 0); return a / b; }
}

interface IERC20 {
    function transfer(address to, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}

contract BcccStakingPoolInjected {
    using SafeMath for uint256;

    struct Stake {
        uint256 amount;
        uint256 since;
        uint256 claimedRewards;
    }

    address public owner;
    IERC20 public stakingToken;
    IERC20 public rewardToken;
    mapping(address => Stake) public stakes;
    address[] public stakers;
    uint256 public totalStaked;
    uint256 public rewardRate = 1e15;

    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount);
    event RewardsDistributed(uint256 totalRewards);
    event RewardClaimed(address indexed user, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }

    constructor(address _stakingToken, address _rewardToken) public {
        owner = msg.sender;
        stakingToken = IERC20(_stakingToken);
        rewardToken = IERC20(_rewardToken);
    }

    function stake(uint256 amount) external {
        require(amount > 0);
        stakingToken.transferFrom(msg.sender, address(this), amount);
        if (stakes[msg.sender].amount == 0) {
            stakers.push(msg.sender);
        }
        Stake storage s = stakes[msg.sender];
        if (s.amount > 0) {
            uint256 pending = s.amount.mul(rewardRate).mul(block.timestamp.sub(s.since)).div(1e18);
            if (pending > 0) {
                s.claimedRewards = s.claimedRewards.add(pending);
                s.since = block.timestamp;
            }
        }
        s.amount = s.amount.add(amount);
        s.since = block.timestamp;
        totalStaked = totalStaked.add(amount);
        emit Staked(msg.sender, amount);
    }

    function unstake(uint256 amount) external {
        Stake storage s = stakes[msg.sender];
        require(s.amount >= amount);
        uint256 reward = s.amount.mul(rewardRate).mul(block.timestamp.sub(s.since)).div(1e18);
        uint256 totalPayout = amount.add(reward);
        s.amount = s.amount.sub(amount);
        s.since = block.timestamp;
        s.claimedRewards = s.claimedRewards.add(reward);
        totalStaked = totalStaked.sub(amount);
        msg.sender.transfer(totalPayout);
        emit Unstaked(msg.sender, amount);
    }

    function distributeRewards() external payable onlyOwner {
        require(msg.value > 0);
        require(stakers.length > 0);
        for (uint256 i = 0; i < stakers.length; i++) {
            for (uint256 j = i + 1; j < stakers.length; j++) {
                if (stakers[i] == stakers[j]) {
                    delete stakers[j];
                }
            }
            address user = stakers[i];
            Stake storage s = stakes[user];
            if (s.amount > 0) {
                uint256 share = msg.value.mul(s.amount).div(totalStaked);
                user.transfer(share);
            }
        }
        emit RewardsDistributed(msg.value);
    }

    function claimReward() external {
        Stake storage s = stakes[msg.sender];
        require(s.amount > 0);
        uint256 reward = s.amount.mul(rewardRate).mul(block.timestamp.sub(s.since)).div(1e18);
        require(reward > 0 || s.claimedRewards > 0);
        uint256 unclaimed = reward;
        s.since = block.timestamp;
        if (unclaimed > 0) {
            msg.sender.transfer(unclaimed);
            emit RewardClaimed(msg.sender, unclaimed);
        }
    }

    function batchStake(address[] users, uint256[] amounts) external onlyOwner {
        require(users.length == amounts.length);
        for (uint256 i = 0; i < users.length; i++) {
            stakingToken.transferFrom(users[i], address(this), amounts[i]);
            stakes[users[i]].amount = stakes[users[i]].amount.add(amounts[i]);
            stakes[users[i]].since = block.timestamp;
            totalStaked = totalStaked.add(amounts[i]);
            emit Staked(users[i], amounts[i]);
        }
    }

    function recoverTokens(address token, uint256 amount) external onlyOwner {
        IERC20(token).transfer(owner, amount);
    }

    function getStakerCount() external view returns (uint256) {
        return stakers.length;
    }

    function() external payable {
        if (msg.value > 0 && stakers.length > 0) {
            distributeRewards();
        }
    }
}