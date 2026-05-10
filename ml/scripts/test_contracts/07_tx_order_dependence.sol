// expect: TransactionOrderDependence
// Classic front-running: approve-and-call race on ERC20 allowance.
// Also: price oracle set by owner tx that miners can front-run.
pragma solidity ^0.8.0;

contract Frontrunnable {
    mapping(address => mapping(address => uint256)) public allowance;
    mapping(address => uint256) public balance;
    uint256 public price;
    address public oracle;

    constructor(address _oracle) {
        oracle = _oracle;
    }

    // VULNERABILITY: allowance update is not atomic — front-runner can spend
    // old allowance then new allowance between the two txs
    function approve(address spender, uint256 amount) external {
        allowance[msg.sender][spender] = amount;
    }

    function transferFrom(address from, address to, uint256 amount) external {
        require(allowance[from][msg.sender] >= amount);
        allowance[from][msg.sender] -= amount;
        balance[from] -= amount;
        balance[to] += amount;
    }

    // VULNERABILITY: price visible in mempool — MEV bot can buy before update
    function setPrice(uint256 newPrice) external {
        require(msg.sender == oracle);
        price = newPrice;
    }

    function buy() external payable {
        require(msg.value >= price);
        balance[msg.sender] += msg.value / price;
    }
}
