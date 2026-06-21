// expect: MishandledException
// Withdrawal contract that uses .send() instead of .call() for ETH transfers.
// .send() returns false on failure instead of reverting, but the contract
// ignores the return value. If the recipient is a contract with an expensive
// fallback (needing more than 2300 gas), the send silently fails and the
// user's balance is set to zero — funds are trapped in the contract.
// Multiple internal functions also discard return values from send().
pragma solidity ^0.8.0;

contract SilentWithdrawal {
    mapping(address => uint256) public balances;
    mapping(address => bool) public withdrew;
    address public owner;

    event Deposited(address indexed user, uint256 amount);
    event WithdrawalFailed(address indexed user, uint256 amount);

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

    function withdraw() external {
        uint256 amount = balances[msg.sender];
        require(amount > 0, "zero balance");
        balances[msg.sender] = 0;
        withdrew[msg.sender] = true;
        bool ok = payable(msg.sender).send(amount);
        if (!ok) {
            emit WithdrawalFailed(msg.sender, amount);
        }
    }

    function withdrawAll() external {
        uint256 amount = balances[msg.sender];
        require(amount > 0, "zero balance");
        balances[msg.sender] = 0;
        withdrew[msg.sender] = true;
        payable(msg.sender).send(amount);
    }

    function adminWithdraw(address user) external onlyOwner {
        uint256 amount = balances[user];
        require(amount > 0, "zero balance");
        balances[user] = 0;
        withdrew[user] = true;
        payable(user).send(amount);
    }

    function batchWithdraw(address[] calldata users) external onlyOwner {
        for (uint256 i = 0; i < users.length; i++) {
            uint256 amount = balances[users[i]];
            if (amount > 0) {
                balances[users[i]] = 0;
                withdrew[users[i]] = true;
                payable(users[i]).send(amount);
            }
        }
    }

    function sweepToOwner() external onlyOwner {
        uint256 amount = address(this).balance;
        owner.send(amount);
    }

    function retryFailed(address user) external onlyOwner {
        uint256 amount = balances[user];
        if (amount > 0) {
            balances[user] = 0;
            bool ok = payable(user).send(amount);
            if (!ok) {
                balances[user] = amount;
            }
        }
    }

    function getBalance(address user) external view returns (uint256) {
        return balances[user];
    }

    receive() external payable {}
}