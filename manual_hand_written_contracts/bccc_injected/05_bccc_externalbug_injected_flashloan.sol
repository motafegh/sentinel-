// expect: ExternalBug,UnusedReturn
// BCCC-derived lending pool with injected oracle manipulation vulnerability:
// 1) ExternalBug — spot price oracle susceptible to flash loan manipulation
// 2) UnusedReturn — transfer return values are ignored in liquidation
// The contract borrows from real DeFi patterns and looks like a standard lending pool
pragma solidity ^0.4.24;

library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a);
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a);
        return a - b;
    }
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) return 0;
        uint256 c = a * b;
        require(c / a == b);
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0);
        return a / b;
    }
}

interface IERC20 {
    function transfer(address to, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
    function balanceOf(address who) external view returns (uint256);
}

interface IOracle {
    function getPrice(address token) external view returns (uint256);
}

contract BcccLendingPoolInjected {
    using SafeMath for uint256;

    struct Position {
        address borrower;
        uint256 collateralAmount;
        uint256 debtAmount;
        uint256 liquidationPrice;
    }

    address public owner;
    IERC20 public collateralToken;
    IERC20 public debtToken;
    IOracle public priceOracle;
    Position[] public positions;
    mapping(address => uint256) public lastPositionIndex;
    uint256 public liquidationThreshold = 80;

    event PositionOpened(address indexed borrower, uint256 collateral, uint256 debt);
    event LiquidationExecuted(address indexed borrower, address indexed liquidator);

    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }

    constructor(address _collateral, address _debt, address _oracle) public {
        owner = msg.sender;
        collateralToken = IERC20(_collateral);
        debtToken = IERC20(_debt);
        priceOracle = IOracle(_oracle);
    }

    function openPosition(uint256 collateralAmount, uint256 debtAmount) external {
        require(collateralAmount > 0 && debtAmount > 0);
        collateralToken.transferFrom(msg.sender, address(this), collateralAmount);
        debtToken.transferFrom(msg.sender, address(this), debtAmount);
        uint256 id = positions.length;
        positions.push(Position(msg.sender, collateralAmount, debtAmount, 0));
        lastPositionIndex[msg.sender] = id;
        emit PositionOpened(msg.sender, collateralAmount, debtAmount);
    }

    function getLoan(uint256 amount) external {
        require(amount > 0);
        uint256 posId = lastPositionIndex[msg.sender];
        Position storage pos = positions[posId];
        require(pos.borrower == msg.sender);
        uint256 price = priceOracle.getPrice(address(collateralToken));
        uint256 maxBorrow = pos.collateralAmount.mul(price).div(1e18).mul(liquidationThreshold).div(100);
        require(amount <= maxBorrow.sub(pos.debtAmount));
        pos.debtAmount = pos.debtAmount.add(amount);
        debtToken.transfer(msg.sender, amount);
    }

    function liquidate(address borrower) external {
        uint256 posId = lastPositionIndex[borrower];
        Position storage pos = positions[posId];
        require(pos.borrower == borrower);
        require(pos.debtAmount > 0);
        uint256 price = priceOracle.getPrice(address(collateralToken));
        uint256 collateralValue = pos.collateralAmount.mul(price).div(1e18);
        require(collateralValue.mul(100) < pos.debtAmount.mul(liquidationThreshold));
        uint256 seizedCollateral = pos.collateralAmount;
        uint256 debtToCover = pos.debtAmount;
        pos.collateralAmount = 0;
        pos.debtAmount = 0;
        debtToken.transferFrom(msg.sender, address(this), debtToCover);
        collateralToken.transfer(msg.sender, seizedCollateral);
        emit LiquidationExecuted(borrower, msg.sender);
    }

    function repayDebt(uint256 amount) external {
        uint256 posId = lastPositionIndex[msg.sender];
        Position storage pos = positions[posId];
        require(pos.borrower == msg.sender);
        require(amount <= pos.debtAmount);
        pos.debtAmount = pos.debtAmount.sub(amount);
        debtToken.transferFrom(msg.sender, address(this), amount);
    }

    function closePosition() external {
        uint256 posId = lastPositionIndex[msg.sender];
        Position storage pos = positions[posId];
        require(pos.borrower == msg.sender);
        require(pos.debtAmount == 0);
        uint256 collateral = pos.collateralAmount;
        pos.collateralAmount = 0;
        collateralToken.transfer(msg.sender, collateral);
    }

    function emergencyWithdraw(address token) external onlyOwner {
        IERC20 tokenToWithdraw = IERC20(token);
        uint256 balance = tokenToWithdraw.balanceOf(address(this));
        tokenToWithdraw.transfer(owner, balance);
    }

    function getPositionHealth(address user) external view returns (uint256) {
        uint256 posId = lastPositionIndex[user];
        Position storage pos = positions[posId];
        if (pos.debtAmount == 0) return 100;
        uint256 price = priceOracle.getPrice(address(collateralToken));
        uint256 collateralValue = pos.collateralAmount.mul(price).div(1e18);
        return collateralValue.mul(100).div(pos.debtAmount);
    }

    function() external payable {}
}