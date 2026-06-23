// expect: UnusedReturn,MishandledException,Reentrancy
// DeFi strategy contract that chains multiple operations together.
// The return value of each intermediate step is discarded — only the final
// result is checked. If any intermediate token transfer or approval fails
// (returns false instead of reverting), the contract continues as if
// it succeeded. The deposit, withdraw, and reinvest functions all ignore
// the return values of the transfer and approve calls they make.
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function balanceOf(address who) external view returns (uint256);
}

interface IStrategy {
    function deposit(address token, uint256 amount) external returns (uint256 shares);
    function withdraw(address token, uint256 shares) external returns (uint256 amount);
    function reinvest() external returns (uint256 profit);
}

contract NestedCallStrategy {
    address public owner;
    IStrategy public strategy;
    mapping(address => mapping(address => uint256)) public depositedShares;

    event Deposited(address indexed user, address indexed token, uint256 amount);
    event Withdrawn(address indexed user, address indexed token, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _strategy) {
        owner = msg.sender;
        strategy = IStrategy(_strategy);
    }

    function deposit(address token, uint256 amount) external {
        IERC20(token).transferFrom(msg.sender, address(this), amount);
        IERC20(token).approve(address(strategy), amount);
        uint256 shares = strategy.deposit(token, amount);
        depositedShares[msg.sender][token] += shares;
        emit Deposited(msg.sender, token, amount);
    }

    function withdraw(address token, uint256 shares) external {
        require(depositedShares[msg.sender][token] >= shares, "insufficient shares");
        uint256 amount = strategy.withdraw(token, shares);
        depositedShares[msg.sender][token] -= shares;
        IERC20(token).transfer(msg.sender, amount);
        emit Withdrawn(msg.sender, token, amount);
    }

    function reinvest() external onlyOwner {
        uint256 profit = strategy.reinvest();
        owner.call{value: profit}("");
    }

    function multiAssetDeposit(address[] calldata tokens, uint256[] calldata amounts) external {
        require(tokens.length == amounts.length, "length mismatch");
        for (uint256 i = 0; i < tokens.length; i++) {
            IERC20(tokens[i]).transferFrom(msg.sender, address(this), amounts[i]);
            IERC20(tokens[i]).approve(address(strategy), amounts[i]);
            uint256 shares = strategy.deposit(tokens[i], amounts[i]);
            depositedShares[msg.sender][tokens[i]] += shares;
            emit Deposited(msg.sender, tokens[i], amounts[i]);
        }
    }

    function multiAssetWithdraw(address[] calldata tokens, uint256[] calldata shares) external {
        require(tokens.length == shares.length, "length mismatch");
        for (uint256 i = 0; i < tokens.length; i++) {
            require(depositedShares[msg.sender][tokens[i]] >= shares[i], "insufficient shares");
            uint256 amount = strategy.withdraw(tokens[i], shares[i]);
            depositedShares[msg.sender][tokens[i]] -= shares[i];
            IERC20(tokens[i]).transfer(msg.sender, amount);
            emit Withdrawn(msg.sender, tokens[i], amount);
        }
    }

    function harvestAndTransfer(address[] calldata tokens, address recipient) external onlyOwner {
        for (uint256 i = 0; i < tokens.length; i++) {
            uint256 balance = IERC20(tokens[i]).balanceOf(address(this));
            if (balance > 0) {
                IERC20(tokens[i]).transfer(recipient, balance);
            }
        }
    }

    function emergencyWithdraw(address token) external onlyOwner {
        uint256 shares = depositedShares[owner][token];
        if (shares > 0) {
            uint256 amount = strategy.withdraw(token, shares);
            depositedShares[owner][token] = 0;
            IERC20(token).transfer(owner, amount);
        }
    }

    function rescueETH() external onlyOwner {
        owner.call{value: address(this).balance}("");
    }

    receive() external payable {}
}