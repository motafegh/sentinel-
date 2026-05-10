// expect:
// Clean contract — no expected vulnerabilities.
// Uses checks-effects-interactions, SafeMath via 0.8.0, pull payment, no timestamp logic.
pragma solidity ^0.8.0;

contract SafeVault {
    mapping(address => uint256) private _balances;
    address public immutable owner;

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
        _balances[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }

    function withdraw(uint256 amount) external {
        // checks
        require(_balances[msg.sender] >= amount, "insufficient");
        // effects
        _balances[msg.sender] -= amount;
        emit Withdrawn(msg.sender, amount);
        // interactions last
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "transfer failed");
    }

    function balanceOf(address user) external view returns (uint256) {
        return _balances[user];
    }
}
