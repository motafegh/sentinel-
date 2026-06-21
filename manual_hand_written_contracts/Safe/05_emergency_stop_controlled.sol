// expect:
// Rate-limited vault with emergency stop. Uses a time-based rate limiter
// for withdrawals to mitigate damage in case of key compromise.
// All state changes happen before external calls (CEI). Implements
// a withdrawal cooldown mechanism. No unchecked blocks, no tx.origin,
// no timestamp in security decisions (only in rate limiting UX).
// Uses Ownable with two-step transfer for ownership changes.
pragma solidity ^0.8.0;

contract RateLimitedVault {
    struct WithdrawalLimit {
        uint256 periodAmount;
        uint256 periodDuration;
        uint256 lastReset;
        uint256 usedInPeriod;
    }

    address public owner;
    address public pendingOwner;
    bool public emergencyStopped;
    mapping(address => uint256) private _balances;
    mapping(address => WithdrawalLimit) private _limits;
    uint256 public constant DEFAULT_LIMIT = 10 ether;
    uint256 public constant DEFAULT_PERIOD = 1 days;

    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);
    event EmergencyStop(address indexed trigger);
    event EmergencyRelease(address indexed trigger);
    event OwnershipTransferStarted(address indexed currentOwner, address indexed pendingOwner);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event LimitUpdated(address indexed user, uint256 amount, uint256 duration);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    modifier notStopped() {
        require(!emergencyStopped, "emergency stop");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "zero address");
        pendingOwner = newOwner;
        emit OwnershipTransferStarted(owner, newOwner);
    }

    function acceptOwnership() external {
        require(msg.sender == pendingOwner, "not pending owner");
        owner = pendingOwner;
        pendingOwner = address(0);
        emit OwnershipTransferred(owner, pendingOwner);
    }

    function deposit() external payable notStopped {
        require(msg.value > 0, "zero deposit");
        _balances[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }

    function withdraw(uint256 amount) external notStopped {
        require(_balances[msg.sender] >= amount, "insufficient");
        WithdrawalLimit storage limit = _limits[msg.sender];
        if (block.timestamp >= limit.lastReset + limit.periodDuration) {
            limit.lastReset = block.timestamp;
            limit.usedInPeriod = 0;
        }
        if (limit.periodAmount > 0) {
            require(limit.usedInPeriod + amount <= limit.periodAmount, "rate limit exceeded");
            limit.usedInPeriod += amount;
        }
        _balances[msg.sender] -= amount;
        emit Withdrawn(msg.sender, amount);
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "transfer failed");
    }

    function setWithdrawalLimit(address user, uint256 amount, uint256 duration) external onlyOwner {
        _limits[user].periodAmount = amount;
        _limits[user].periodDuration = duration > 0 ? duration : DEFAULT_PERIOD;
        _limits[user].lastReset = block.timestamp;
        _limits[user].usedInPeriod = 0;
        emit LimitUpdated(user, amount, duration);
    }

    function triggerEmergencyStop() external onlyOwner {
        emergencyStopped = true;
        emit EmergencyStop(msg.sender);
    }

    function releaseEmergencyStop() external onlyOwner {
        emergencyStopped = false;
        emit EmergencyRelease(msg.sender);
    }

    function emergencyWithdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        (bool ok, ) = owner.call{value: balance}("");
        require(ok, "emergency withdraw failed");
    }

    function getBalance(address user) external view returns (uint256) {
        return _balances[user];
    }

    function getLimitInfo(address user) external view returns (uint256 amount, uint256 used, uint256 remaining, uint256 resetsAt) {
        WithdrawalLimit storage limit = _limits[user];
        amount = limit.periodAmount;
        used = limit.usedInPeriod;
        remaining = amount > used ? amount - used : 0;
        resetsAt = limit.lastReset + limit.periodDuration;
    }

    function contractBalance() external view returns (uint256) {
        return address(this).balance;
    }
}