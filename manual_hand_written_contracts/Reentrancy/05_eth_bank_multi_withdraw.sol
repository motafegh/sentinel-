// expect: Reentrancy
// ETH bank with multiple withdrawal methods that all share the same balance
// mapping and all violate CEI (Checks-Effects-Interactions). An attacker
// deploys a contract with a malicious fallback that calls one withdrawal
// method from another — the second call sees the balance not yet decremented
// and withdraws the same ETH again. All three withdrawal paths are affected.
pragma solidity ^0.8.0;

contract MultiWithdrawBank {
    mapping(address => uint256) public balances;
    mapping(address => bool) public vipMembers;
    address public owner;
    uint256 public maxWithdrawal = 10 ether;

    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function deposit() external payable {
        require(msg.value > 0, "zero deposit");
        balances[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }

    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "insufficient");
        require(amount <= maxWithdrawal, "exceeds max withdrawal");
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "withdraw failed");
        balances[msg.sender] -= amount;
        emit Withdrawn(msg.sender, amount);
    }

    function withdrawAll() external {
        uint256 amount = balances[msg.sender];
        require(amount > 0, "zero balance");
        require(amount <= maxWithdrawal, "exceeds max withdrawal");
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "withdrawAll failed");
        balances[msg.sender] = 0;
        emit Withdrawn(msg.sender, amount);
    }

    function vipWithdraw(uint256 amount) external {
        require(vipMembers[msg.sender], "not vip");
        require(balances[msg.sender] >= amount, "insufficient");
        require(amount <= maxWithdrawal * 2, "exceeds vip limit");
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "vip withdraw failed");
        balances[msg.sender] -= amount;
        emit Withdrawn(msg.sender, amount);
    }

    function batchWithdrawForUser(address user, uint256[] calldata amounts) external {
        require(msg.sender == user, "not authorized");
        for (uint256 i = 0; i < amounts.length; i++) {
            require(balances[user] >= amounts[i], "insufficient");
            require(amounts[i] <= maxWithdrawal, "exceeds limit");
            (bool ok, ) = user.call{value: amounts[i]}("");
            require(ok, "batch withdraw failed");
            balances[user] -= amounts[i];
            emit Withdrawn(user, amounts[i]);
        }
    }

    function setVip(address user, bool status) external onlyOwner {
        vipMembers[user] = status;
    }

    function setMaxWithdrawal(uint256 newMax) external onlyOwner {
        maxWithdrawal = newMax;
    }

    receive() external payable {
        balances[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }
}