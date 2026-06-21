// expect: Reentrancy
// Staking contract that uses ERC777 token hooks (tokensReceived callback).
// When the contract transfers ERC777 tokens to a user, the user's
// tokensReceived hook fires BEFORE the staking contract updates its
// internal accounting. The attacker re-enters the unstake function
// from the hook, claiming rewards multiple times before the first
// unstake completes. The shares mapping is only updated after the call.
pragma solidity ^0.8.0;

interface IERC777 {
    function send(address to, uint256 amount, bytes calldata data) external;
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function transfer(address to, uint256 amount) external returns (bool);
    function balanceOf(address who) external view returns (uint256);
    function operatorSend(address from, address to, uint256 amount, bytes calldata data, bytes calldata operatorData) external;
}

contract ERC777Staking {
    IERC777 public stakingToken;
    address public owner;
    mapping(address => uint256) public stakedAmount;
    mapping(address => uint256) public rewardDebt;
    uint256 public accRewardPerShare;
    uint256 public totalStaked;

    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount, uint256 reward);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _token) {
        stakingToken = IERC777(_token);
        owner = msg.sender;
    }

    function stake(uint256 amount) external {
        require(amount > 0, "zero amount");
        stakingToken.transferFrom(msg.sender, address(this), amount);
        uint256 pending = (stakedAmount[msg.sender] * accRewardPerShare) / 1e18 - rewardDebt[msg.sender];
        stakedAmount[msg.sender] += amount;
        totalStaked += amount;
        rewardDebt[msg.sender] = (stakedAmount[msg.sender] * accRewardPerShare) / 1e18;
        if (pending > 0) {
            stakingToken.send(msg.sender, pending, "");
        }
        emit Staked(msg.sender, amount);
    }

    function unstake(uint256 amount) external {
        require(stakedAmount[msg.sender] >= amount, "insufficient staked");
        uint256 pending = (stakedAmount[msg.sender] * accRewardPerShare) / 1e18 - rewardDebt[msg.sender];
        stakedAmount[msg.sender] -= amount;
        totalStaked -= amount;
        rewardDebt[msg.sender] = (stakedAmount[msg.sender] * accRewardPerShare) / 1e18;
        stakingToken.send(msg.sender, amount + pending, "");
        emit Unstaked(msg.sender, amount, pending);
    }

    function claimRewards() external {
        uint256 pending = (stakedAmount[msg.sender] * accRewardPerShare) / 1e18 - rewardDebt[msg.sender];
        require(pending > 0, "no rewards");
        rewardDebt[msg.sender] = (stakedAmount[msg.sender] * accRewardPerShare) / 1e18;
        stakingToken.send(msg.sender, pending, "");
    }

    function batchStake(address[] calldata users, uint256[] calldata amounts) external onlyOwner {
        require(users.length == amounts.length, "length mismatch");
        for (uint256 i = 0; i < users.length; i++) {
            stakingToken.transferFrom(users[i], address(this), amounts[i]);
            stakedAmount[users[i]] += amounts[i];
            totalStaked += amounts[i];
            rewardDebt[users[i]] = (stakedAmount[users[i]] * accRewardPerShare) / 1e18;
            emit Staked(users[i], amounts[i]);
        }
    }

    function updateRewardRate(uint256 newAccPerShare) external onlyOwner {
        accRewardPerShare = newAccPerShare;
    }

    function getStaked(address user) external view returns (uint256) {
        return stakedAmount[user];
    }
}