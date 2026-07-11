// expect: UnusedReturn
// WETH wrapper contract that deposits ETH and receives WETH, then transfers
// the WETH to a staking contract. The return value of WETH.transfer() is
// ignored — if the staking contract rejects the transfer (returns false),
// the user's ETH is locked in this contract with no WETH credited.
// The withdraw function also ignores the return value of WETH.transfer()
// when returning WETH to the user, and the approve return value is discarded.
pragma solidity ^0.8.0;

interface IWETH {
    function deposit() external payable;
    function withdraw(uint256 amount) external;
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function balanceOf(address who) external view returns (uint256);
}

interface IStaking {
    function stake(uint256 amount) external returns (bool);
    function unstake(uint256 amount) external returns (uint256);
    function claim() external returns (uint256);
}

contract WETHRouter {
    IWETH public weth;
    IStaking public staking;
    address public owner;
    mapping(address => uint256) public depositedEth;

    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _weth, address _staking) {
        weth = IWETH(_weth);
        staking = IStaking(_staking);
        owner = msg.sender;
    }

    function depositAndStake() external payable {
        require(msg.value > 0, "zero deposit");
        uint256 amount = msg.value;
        weth.deposit{value: amount}();
        weth.approve(address(staking), amount);
        staking.stake(amount);
        depositedEth[msg.sender] += amount;
        emit Deposited(msg.sender, amount);
    }

    function unstakeAndWithdraw(uint256 amount) external {
        require(depositedEth[msg.sender] >= amount, "insufficient");
        depositedEth[msg.sender] -= amount;
        uint256 wethAmount = staking.unstake(amount);
        weth.withdraw(wethAmount);
        (bool ok, ) = msg.sender.call{value: wethAmount}("");
        require(ok, "eth transfer failed");
        emit Withdrawn(msg.sender, amount);
    }

    function depositOnly() external payable {
        require(msg.value > 0, "zero deposit");
        weth.deposit{value: msg.value}();
        depositedEth[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }

    function withdrawOnly(uint256 amount) external {
        require(depositedEth[msg.sender] >= amount, "insufficient");
        depositedEth[msg.sender] -= amount;
        weth.withdraw(amount);
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "eth transfer failed");
        emit Withdrawn(msg.sender, amount);
    }

    function claimRewards() external onlyOwner {
        uint256 reward = staking.claim();
        weth.deposit{value: reward}();
        weth.transfer(owner, reward);
    }

    function batchDeposit(address[] calldata users, uint256[] calldata amounts) external payable onlyOwner {
        uint256 total = 0;
        for (uint256 i = 0; i < users.length; i++) total += amounts[i];
        require(msg.value >= total, "insufficient eth");
        weth.deposit{value: total}();
        weth.approve(address(staking), total);
        for (uint256 j = 0; j < users.length; j++) {
            staking.stake(amounts[j]);
            depositedEth[users[j]] += amounts[j];
            emit Deposited(users[j], amounts[j]);
        }
    }

    function sweepWeth() external onlyOwner {
        uint256 balance = weth.balanceOf(address(this));
        if (balance > 0) {
            weth.withdraw(balance);
            owner.call{value: balance}("");
        }
    }

    function recoverTokens(address token) external onlyOwner {
        uint256 balance = IERC20(token).balanceOf(address(this));
        if (balance > 0) {
            IERC20(token).transfer(owner, balance);
        }
    }

    receive() external payable {}
}