// expect: Reentrancy
// TRICKY: Reentrancy vulnerability hidden INSIDE a modifier.
// The modifier calls address.transfer() BEFORE executing the function body.
// This means any function using this modifier has a reentrancy vector,
// even if the function itself follows CEI pattern.
// The transfer in the modifier sends ETH before the state update in the function body.
pragma solidity ^0.8.0;

contract ModifierReentrancy {
    mapping(address => uint256) public balances;
    mapping(address => bool) public feePaid;
    address public owner;
    uint256 public fee = 0.01 ether;

    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);

    modifier collectFee() {
        require(!feePaid[msg.sender], "fee already paid");
        feePaid[msg.sender] = true;
        (bool ok, ) = owner.call{value: fee}("");
        require(ok, "fee transfer failed");
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

    function safeWithdraw(uint256 amount) external collectFee {
        require(balances[msg.sender] >= amount, "insufficient");
        balances[msg.sender] -= amount;
        emit Withdrawn(msg.sender, amount);
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "transfer failed");
    }

    function withdrawAll() external collectFee {
        uint256 amount = balances[msg.sender];
        require(amount > 0, "zero balance");
        balances[msg.sender] = 0;
        emit Withdrawn(msg.sender, amount);
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "transfer failed");
    }

    function multiWithdraw(uint256[] calldata amounts) external collectFee {
        uint256 total = 0;
        for (uint256 i = 0; i < amounts.length; i++) {
            total += amounts[i];
        }
        require(balances[msg.sender] >= total, "insufficient");
        balances[msg.sender] -= total;
        for (uint256 j = 0; j < amounts.length; j++) {
            (bool ok, ) = msg.sender.call{value: amounts[j]}("");
            require(ok, "multi transfer failed");
        }
        emit Withdrawn(msg.sender, total);
    }

    function transferBalance(address to, uint256 amount) external {
        require(balances[msg.sender] >= amount, "insufficient");
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }

    function setFee(uint256 newFee) external {
        require(msg.sender == owner, "not owner");
        fee = newFee;
    }

    receive() external payable {}
}