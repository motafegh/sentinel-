// expect: Reentrancy,TransactionOrderDependence
// Multi-vuln: Reentrancy + TransactionOrderDependence in a single contract.
// The batchWithdraw function re-enters via the first transfer before updating
// state for ALL recipients. The approve function has the classic allowance race.
// These are in separate functions but share the same balance mapping.
pragma solidity ^0.8.0;

contract MultiVulnBank {
    mapping(address => uint256) public balances;
    mapping(address => mapping(address => uint256)) public allowance;
    address public owner;

    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);

    constructor() {
        owner = msg.sender;
    }

    function deposit() external payable {
        balances[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }

    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "insufficient");
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "transfer failed");
        balances[msg.sender] -= amount;
        emit Withdrawn(msg.sender, amount);
    }

    function batchWithdraw(address[] calldata users, uint256[] calldata amounts) external {
        require(users.length == amounts.length, "length mismatch");
        for (uint256 i = 0; i < users.length; i++) {
            require(balances[users[i]] >= amounts[i], "insufficient");
            (bool ok, ) = users[i].call{value: amounts[i]}("");
            require(ok, "batch transfer failed");
            balances[users[i]] -= amounts[i];
            emit Withdrawn(users[i], amounts[i]);
        }
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        require(allowance[from][msg.sender] >= amount, "no allowance");
        require(balances[from] >= amount, "insufficient");
        allowance[from][msg.sender] -= amount;
        balances[from] -= amount;
        balances[to] += amount;
        return true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        require(balances[msg.sender] >= amount, "insufficient");
        balances[msg.sender] -= amount;
        balances[to] += amount;
        return true;
    }

    function batchTransfer(address[] calldata recipients, uint256[] calldata amounts) external {
        for (uint256 i = 0; i < recipients.length; i++) {
            balances[msg.sender] -= amounts[i];
            balances[recipients[i]] += amounts[i];
        }
    }

    receive() external payable {}
}