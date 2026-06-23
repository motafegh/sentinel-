// expect: ExternalBug,MishandledException
// Lending protocol that uses a spot-price oracle from a single DEX.
// The oracle returns the instantaneous pool price which can be manipulated
// via a flash loan: borrow large amount, swap on the DEX to move the price,
// trigger the oracle read on the manipulated price, profit, repay flash loan.
// This is a classic flash loan + oracle manipulation attack vector.
pragma solidity ^0.8.0;

interface IOracle {
    function getPrice(address token) external view returns (uint256);
}

interface IERC20 {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function transfer(address to, uint256 amount) external returns (bool);
    function balanceOf(address who) external view returns (uint256);
}

contract FlashLendingPool {
    IOracle public priceOracle;
    IERC20 public collateralToken;
    IERC20 public loanToken;
    address public owner;

    mapping(address => uint256) public depositedCollateral;
    mapping(address => uint256) public borrowedAmount;
    uint256 public totalLiquidity;
    uint256 public liquidationThreshold = 80;

    event LoanTaken(address indexed borrower, uint256 amount, uint256 collateral);
    event LoanRepaid(address indexed borrower, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _oracle, address _collateral, address _loanToken) {
        priceOracle = IOracle(_oracle);
        collateralToken = IERC20(_collateral);
        loanToken = IERC20(_loanToken);
        owner = msg.sender;
    }

    function depositCollateral(uint256 amount) external {
        collateralToken.transferFrom(msg.sender, address(this), amount);
        depositedCollateral[msg.sender] += amount;
    }

    function getLoan(uint256 loanAmount) external {
        uint256 collateral = depositedCollateral[msg.sender];
        require(collateral > 0, "no collateral");
        uint256 price = priceOracle.getPrice(address(collateralToken));
        uint256 collateralValue = (collateral * price) / 1e18;
        uint256 maxLoan = (collateralValue * liquidationThreshold) / 100;
        require(loanAmount <= maxLoan, "loan exceeds collateral value");
        require(loanAmount <= totalLiquidity, "insufficient liquidity");
        borrowedAmount[msg.sender] += loanAmount;
        totalLiquidity -= loanAmount;
        loanToken.transfer(msg.sender, loanAmount);
        emit LoanTaken(msg.sender, loanAmount, collateral);
    }

    function repayLoan(uint256 amount) external {
        require(borrowedAmount[msg.sender] >= amount, "repay exceeds borrowed");
        loanToken.transferFrom(msg.sender, address(this), amount);
        borrowedAmount[msg.sender] -= amount;
        totalLiquidity += amount;
        emit LoanRepaid(msg.sender, amount);
    }

    function liquidate(address borrower) external {
        uint256 borrowed = borrowedAmount[borrower];
        require(borrowed > 0, "no debt");
        uint256 collateral = depositedCollateral[borrower];
        uint256 price = priceOracle.getPrice(address(collateralToken));
        uint256 collateralValue = (collateral * price) / 1e18;
        uint256 debtValue = borrowed;
        require(collateralValue * 100 < debtValue * liquidationThreshold, "not liquidatable");
        loanToken.transferFrom(msg.sender, address(this), borrowed);
        borrowedAmount[borrower] = 0;
        depositedCollateral[borrower] = 0;
        collateralToken.transfer(msg.sender, collateral);
    }

    function addLiquidity(uint256 amount) external {
        loanToken.transferFrom(msg.sender, address(this), amount);
        totalLiquidity += amount;
    }

    function getAccountHealth(address user) external view returns (uint256 healthPercent) {
        uint256 borrowed = borrowedAmount[user];
        if (borrowed == 0) return 100;
        uint256 collateral = depositedCollateral[user];
        uint256 price = priceOracle.getPrice(address(collateralToken));
        uint256 collateralValue = (collateral * price) / 1e18;
        return (collateralValue * 100) / borrowed;
    }
}