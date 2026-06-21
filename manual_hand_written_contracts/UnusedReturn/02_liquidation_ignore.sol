// expect: UnusedReturn
// Liquidation engine that calls transfer on multiple tokens to seize collateral.
// The return value of each transfer is ignored — if the token transfer returns
// false (non-reverting failure), the liquidation proceeds as if the collateral
// was successfully seized. The liquidator receives nothing but the state shows
// the position as liquidated. Internal helper functions also discard return
// values from external calls.
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function balanceOf(address who) external view returns (uint256);
}

interface IOracle {
    function getPrice(address token) external view returns (uint256);
}

contract LiquidationEngine {
    struct Position {
        address user;
        address collateralToken;
        address debtToken;
        uint256 collateralAmount;
        uint256 debtAmount;
    }

    address public owner;
    IOracle public oracle;
    Position[] public positions;
    mapping(address => uint256[]) public userPositions;
    uint256 public liquidationThreshold = 80;

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _oracle) {
        owner = msg.sender;
        oracle = IOracle(_oracle);
    }

    function openPosition(address collateralToken, address debtToken, uint256 collateralAmount, uint256 debtAmount) external {
        IERC20(collateralToken).transferFrom(msg.sender, address(this), collateralAmount);
        uint256 id = positions.length;
        positions.push(Position(msg.sender, collateralToken, debtToken, collateralAmount, debtAmount));
        userPositions[msg.sender].push(id);
    }

    function liquidate(uint256 positionId) external {
        Position storage p = positions[positionId];
        require(p.collateralAmount > 0, "no collateral");
        uint256 collateralPrice = oracle.getPrice(p.collateralToken);
        uint256 debtPrice = oracle.getPrice(p.debtToken);
        uint256 collateralValue = p.collateralAmount * collateralPrice;
        uint256 debtValue = p.debtAmount * debtPrice;
        require(debtValue * 100 > collateralValue * liquidationThreshold, "not liquidatable");
        uint256 seizedCollateral = p.collateralAmount;
        uint256 debtCovered = p.debtAmount;
        p.collateralAmount = 0;
        p.debtAmount = 0;
        IERC20(p.debtToken).transferFrom(msg.sender, address(this), debtCovered);
        IERC20(p.collateralToken).transfer(msg.sender, seizedCollateral);
    }

    function batchLiquidate(uint256[] calldata positionIds) external {
        for (uint256 i = 0; i < positionIds.length; i++) {
            Position storage p = positions[positionIds[i]];
            if (p.collateralAmount > 0) {
                uint256 collateralPrice = oracle.getPrice(p.collateralToken);
                uint256 debtPrice = oracle.getPrice(p.debtToken);
                uint256 collateralValue = p.collateralAmount * collateralPrice;
                uint256 debtValue = p.debtAmount * debtPrice;
                if (debtValue * 100 > collateralValue * liquidationThreshold) {
                    uint256 seizedCollateral = p.collateralAmount;
                    p.collateralAmount = 0;
                    p.debtAmount = 0;
                    IERC20(p.debtToken).transferFrom(msg.sender, address(this), debtValue);
                    IERC20(p.collateralToken).transfer(msg.sender, seizedCollateral);
                }
            }
        }
    }

    function adjustCollateral(uint256 positionId, uint256 additionalCollateral) external {
        Position storage p = positions[positionId];
        require(p.user == msg.sender, "not owner");
        p.collateralAmount += additionalCollateral;
        IERC20(p.collateralToken).transferFrom(msg.sender, address(this), additionalCollateral);
    }

    function closePosition(uint256 positionId) external {
        Position storage p = positions[positionId];
        require(p.user == msg.sender, "not owner");
        require(p.debtAmount == 0, "debt not repaid");
        uint256 collateral = p.collateralAmount;
        p.collateralAmount = 0;
        IERC20(p.collateralToken).transfer(msg.sender, collateral);
    }

    function emergencyWithdraw(address token) external onlyOwner {
        uint256 balance = IERC20(token).balanceOf(address(this));
        IERC20(token).transfer(owner, balance);
    }
}