// expect: Reentrancy
// Cross-function reentrancy: the contract has a withdrawal function and
// a transfer function that share the same balance mapping. An attacker
// re-enters via the withdrawal callback and calls transfer() before
// the first withdrawal finishes — the balance is not yet updated, so
// the same ETH can be transferred to another account and withdrawn again.
// The state is only updated after the external call, enabling the overlap.
pragma solidity ^0.8.0;

contract CrossFunctionBank {
    mapping(address => uint256) public balances;
    mapping(address => uint256) public lockedUntil;
    address public owner;

    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);
    event Transferred(address indexed from, address indexed to, uint256 amount);

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
        require(block.timestamp >= lockedUntil[msg.sender], "locked");
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "withdraw failed");
        balances[msg.sender] -= amount;
        emit Withdrawn(msg.sender, amount);
    }

    function transfer(address to, uint256 amount) external {
        require(balances[msg.sender] >= amount, "insufficient");
        balances[msg.sender] -= amount;
        balances[to] += amount;
        emit Transferred(msg.sender, to, amount);
    }

    function batchTransfer(address[] calldata recipients, uint256[] calldata amounts) external {
        require(recipients.length == amounts.length, "length mismatch");
        for (uint256 i = 0; i < recipients.length; i++) {
            require(balances[msg.sender] >= amounts[i], "insufficient");
            balances[msg.sender] -= amounts[i];
            balances[recipients[i]] += amounts[i];
            emit Transferred(msg.sender, recipients[i], amounts[i]);
        }
    }

    function withdrawAll() external {
        uint256 amount = balances[msg.sender];
        require(amount > 0, "zero balance");
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "withdrawAll failed");
        balances[msg.sender] = 0;
        emit Withdrawn(msg.sender, amount);
    }

    function setLock(address user, uint256 duration) external onlyOwner {
        lockedUntil[user] = block.timestamp + duration;
    }

    function balanceOf(address user) external view returns (uint256) {
        return balances[user];
    }

    receive() external payable {}
}