// expect:
// Clean contract following Checks-Effects-Interactions pattern strictly.
// All state changes happen BEFORE any external call. The withdraw function
// updates balance first, emits event, then makes the external call.
// The contract uses pull-over-push for all ETH transfers.
// No unchecked blocks, no tx.origin, no timestamp dependence in logic.
pragma solidity ^0.8.0;

contract ChecksEffectsVault {
    mapping(address => uint256) private _balances;
    mapping(address => uint256) private _nonces;
    address public immutable owner;
    bool private _reentrant;

    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

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

    constructor() {
        owner = msg.sender;
    }

    function deposit() external payable {
        require(msg.value > 0, "zero deposit");
        _balances[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }

    function withdraw(uint256 amount) external nonReentrant {
        require(_balances[msg.sender] >= amount, "insufficient balance");
        _balances[msg.sender] -= amount;
        emit Withdrawn(msg.sender, amount);
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "transfer failed");
    }

    function withdrawAll() external nonReentrant {
        uint256 amount = _balances[msg.sender];
        require(amount > 0, "zero balance");
        _balances[msg.sender] = 0;
        emit Withdrawn(msg.sender, amount);
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "transfer failed");
    }

    function transferBalance(address to, uint256 amount) external {
        require(_balances[msg.sender] >= amount, "insufficient");
        require(to != address(0), "zero address");
        require(to != address(this), "self transfer");
        _balances[msg.sender] -= amount;
        _balances[to] += amount;
    }

    function batchTransfer(address[] calldata recipients, uint256[] calldata amounts) external {
        require(recipients.length == amounts.length, "length mismatch");
        uint256 total = 0;
        for (uint256 i = 0; i < amounts.length; i++) {
            total += amounts[i];
        }
        require(_balances[msg.sender] >= total, "insufficient total");
        for (uint256 j = 0; j < recipients.length; j++) {
            _balances[msg.sender] -= amounts[j];
            _balances[recipients[j]] += amounts[j];
        }
    }

    function getBalance(address user) external view returns (uint256) {
        return _balances[user];
    }

    function getNonce(address user) external view returns (uint256) {
        return _nonces[user];
    }

    function contractBalance() external view returns (uint256) {
        return address(this).balance;
    }
}