// expect:
// Contract using OpenZeppelin-style patterns: Ownable, ReentrancyGuard,
// Pausable. All external calls happen after state updates (CEI pattern).
// Uses pull-over-push for payments. No unsafe delegatecall or tx.origin.
// Uses SafeERC20-style checks (though implemented inline for simplicity).
// No unchecked blocks, no timestamp-based logic for auth decisions.
pragma solidity ^0.8.0;

contract OZManagedVault {
    address public owner;
    address public pendingOwner;
    bool private _reentrant;
    bool public paused;
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    event OwnershipTransferStarted(address indexed previousOwner, address indexed newOwner);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);
    event Paused(address indexed pauser);
    event Unpaused(address indexed unpauser);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    modifier nonReentrant() {
        require(!_reentrant, "reentrancy");
        _reentrant = true;
        _;
        _reentrant = false;
    }

    modifier whenNotPaused() {
        require(!paused, "paused");
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

    function deposit() external payable whenNotPaused {
        require(msg.value > 0, "zero deposit");
        _balances[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }

    function withdraw(uint256 amount) external nonReentrant whenNotPaused {
        require(_balances[msg.sender] >= amount, "insufficient");
        _balances[msg.sender] -= amount;
        emit Withdrawn(msg.sender, amount);
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "transfer failed");
    }

    function approve(address spender, uint256 amount) external whenNotPaused returns (bool) {
        _allowances[msg.sender][spender] = amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external nonReentrant whenNotPaused returns (bool) {
        require(_balances[from] >= amount, "insufficient");
        require(_allowances[from][msg.sender] >= amount, "no allowance");
        _allowances[from][msg.sender] -= amount;
        _balances[from] -= amount;
        _balances[to] += amount;
        return true;
    }

    function transfer(address to, uint256 amount) external whenNotPaused returns (bool) {
        require(_balances[msg.sender] >= amount, "insufficient");
        require(to != address(0), "zero address");
        _balances[msg.sender] -= amount;
        _balances[to] += amount;
        return true;
    }

    function pause() external onlyOwner {
        paused = true;
        emit Paused(msg.sender);
    }

    function unpause() external onlyOwner {
        paused = false;
        emit Unpaused(msg.sender);
    }

    function balanceOf(address user) external view returns (uint256) {
        return _balances[user];
    }

    function allowance(address from, address spender) external view returns (uint256) {
        return _allowances[from][spender];
    }

    function contractBalance() external view returns (uint256) {
        return address(this).balance;
    }
}