// expect: ExternalBug
// Access control system that uses tx.origin for authentication.
// tx.origin is the original sender of the transaction — NOT the msg.sender.
// A malicious contract can call this contract on behalf of a victim user
// who funded the transaction, bypassing the intended access control.
// Also: the owner can delegatecall to any address — if owner is a contract
// that gets exploited, the entire contract storage is compromised.
pragma solidity ^0.8.0;

contract AccessControlVault {
    mapping(address => uint256) public balances;
    mapping(address => bool) public operators;
    address public owner;
    bool public emergencyPause;

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    modifier onlyOperator() {
        require(operators[msg.sender], "not operator");
        _;
    }

    modifier notPaused() {
        require(!emergencyPause, "emergency pause");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function addOperator(address op) external onlyOwner {
        operators[op] = true;
    }

    function removeOperator(address op) external onlyOwner {
        delete operators[op];
    }

    function deposit() external payable notPaused {
        require(msg.value > 0, "zero deposit");
        balances[msg.sender] += msg.value;
    }

    function withdrawAll() external notPaused {
        uint256 amount = balances[msg.sender];
        require(amount > 0, "zero balance");
        balances[msg.sender] = 0;
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "transfer failed");
    }

    function withdrawTo(address recipient, uint256 amount) external notPaused {
        require(balances[msg.sender] >= amount, "insufficient");
        balances[msg.sender] -= amount;
        (bool ok, ) = recipient.call{value: amount}("");
        require(ok, "transfer failed");
    }

    function operatorWithdraw(address from, uint256 amount) external onlyOperator notPaused {
        require(balances[from] >= amount, "insufficient");
        balances[from] -= amount;
        (bool ok, ) = owner.call{value: amount}("");
        require(ok, "transfer failed");
    }

    function guardedCall(address target, bytes calldata data) external onlyOwner returns (bytes memory) {
        require(target != address(0), "zero target");
        (bool ok, bytes memory ret) = target.call{gas: 50000}(data);
        require(ok, "guarded call failed");
        return ret;
    }

    function delegateExecute(bytes calldata data) external onlyOwner returns (bytes memory) {
        (bool ok, bytes memory ret) = owner.delegatecall(data);
        if (!ok) {
            assembly {
                revert(add(ret, 32), mload(ret))
            }
        }
        return ret;
    }

    function togglePause() external onlyOwner {
        emergencyPause = !emergencyPause;
    }

    function claimOwnership(address newOwner) external {
        require(tx.origin == owner, "only owner via tx.origin");
        owner = newOwner;
    }
}