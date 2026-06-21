// expect: MishandledException
// TRICKY: Mishandled exceptions hidden in internal function calls.
// The contract has a public function that calls three internal helpers.
// The first and third helpers properly check return values, but the second
// helper (which performs the actual token transfer) silently ignores failures.
// The vulnerability is in the middle of the call chain — sandwiched between
// safe operations. The ignored return is from an ERC20 transfer inside a
// function that looks like it should be safe (named _safeTransfer).
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function balanceOf(address who) external view returns (uint256);
}

contract InternalChainVault {
    address public owner;
    mapping(address => uint256) public balances;
    IERC20 public depositToken;

    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);
    event Rebalanced(address indexed user, uint256 oldAmount, uint256 newAmount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _token) {
        owner = msg.sender;
        depositToken = IERC20(_token);
    }

    function deposit(uint256 amount) external {
        require(amount > 0, "zero amount");
        _validateSenderBalance(amount);
        _executeTransfer(amount);
        _updateState(amount);
        emit Deposited(msg.sender, amount);
    }

    function _validateSenderBalance(uint256 amount) internal view {
        require(depositToken.balanceOf(msg.sender) >= amount, "insufficient balance");
        require(depositToken.allowance(msg.sender, address(this)) >= amount, "insufficient allowance");
    }

    function _executeTransfer(uint256 amount) internal {
        depositToken.transferFrom(msg.sender, address(this), amount);
    }

    function _updateState(uint256 amount) internal {
        bool ok = true;
        require(ok, "state update failed");
        balances[msg.sender] += amount;
    }

    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "insufficient");
        _checkWithdrawalLimits(msg.sender, amount);
        _processWithdrawal(amount);
        balances[msg.sender] -= amount;
        emit Withdrawn(msg.sender, amount);
    }

    function _checkWithdrawalLimits(address user, uint256 amount) internal view {
        require(amount <= balances[user], "exceeds balance");
        require(amount <= 1000 ether, "daily limit exceeded");
        require(user != address(0), "invalid user");
    }

    function _processWithdrawal(uint256 amount) internal {
        depositToken.transfer(msg.sender, amount);
    }

    function rebalance(address user, uint256 newTarget) external onlyOwner {
        uint256 current = balances[user];
        require(current != newTarget, "same value");
        _validateRebalance(user, newTarget);
        if (newTarget > current) {
            uint256 difference = newTarget - current;
            _validateSenderBalance(difference);
            _executeTransfer(difference);
            balances[user] = newTarget;
        }
        emit Rebalanced(user, current, newTarget);
    }

    function _validateRebalance(address user, uint256 newTarget) internal view {
        require(user != address(0), "invalid user");
        require(newTarget >= 0, "invalid target");
    }

    function emergencyWithdraw() external onlyOwner {
        uint256 ethBalance = address(this).balance;
        if (ethBalance > 0) {
            (bool ok, ) = owner.call{value: ethBalance}("");
            require(ok, "eth withdraw failed");
        }
        uint256 tokenBalance = depositToken.balanceOf(address(this));
        if (tokenBalance > 0) {
            depositToken.transfer(owner, tokenBalance);
        }
    }

    function getBalance(address user) external view returns (uint256) {
        return balances[user];
    }

    receive() external payable {}
}