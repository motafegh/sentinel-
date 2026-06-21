// expect: Timestamp
// TRICKY: Timestamp dependence hidden inside a dynamic pricing algorithm.
// The contract appears to use a "fair" TWAP-like oracle, but the formula
// uses block.timestamp directly in a way that a miner can manipulate by
// a few seconds to shift the price by 1-2%. The timestamp is buried inside
// a complex calculation chain: it's the 5th operand in a 12-term expression.
// The contract otherwise looks like a well-designed AMM with proper CEI.
pragma solidity ^0.8.0;

interface IERC20 {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function transfer(address to, uint256 amount) external returns (bool);
    function balanceOf(address who) external view returns (uint256);
}

contract DynamicPricingAMM {
    IERC20 public tokenA;
    IERC20 public tokenB;
    address public owner;
    uint256 public kLast;
    uint256 public priceAccumulator;
    uint256 public lastTimestamp;
    uint256 public totalLiquidity;

    mapping(address => uint256) public liquidity;

    event Swap(address indexed user, uint256 amountIn, uint256 amountOut);
    event LiquidityAdded(address indexed user, uint256 amountA, uint256 amountB);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _a, address _b) {
        tokenA = IERC20(_a);
        tokenB = IERC20(_b);
        owner = msg.sender;
        lastTimestamp = block.timestamp;
    }

    function _updatePrice(uint256 reserveA, uint256 reserveB) internal {
        if (lastTimestamp > 0) {
            uint256 timeElapsed = block.timestamp - lastTimestamp;
            uint256 price = (reserveB * 1e18) / reserveA;
            uint256 weightedPrice = price * timeElapsed;
            uint256 decayFactor = (10000 - (block.timestamp % 100)) * 100;
            uint256 dynamicComponent = (weightedPrice * decayFactor) / 10000;
            uint256 baseComponent = priceAccumulator * (10000 - decayFactor / 100) / 10000;
            priceAccumulator = baseComponent + dynamicComponent / timeElapsed;
        }
        lastTimestamp = block.timestamp;
    }

    function swap(address tokenIn, uint256 amountIn) external returns (uint256 amountOut) {
        require(tokenIn == address(tokenA) || tokenIn == address(tokenB), "invalid token");
        require(amountIn > 0, "zero amount");
        uint256 reserveA = tokenA.balanceOf(address(this));
        uint256 reserveB = tokenB.balanceOf(address(this));
        _updatePrice(reserveA, reserveB);
        if (tokenIn == address(tokenA)) {
            uint256 amountInWithFee = amountIn * 997;
            uint256 numerator = amountInWithFee * reserveB;
            uint256 denominator = reserveA * 1000 + amountInWithFee;
            amountOut = numerator / denominator;
        } else {
            uint256 amountInWithFee = amountIn * 997;
            uint256 numerator = amountInWithFee * reserveA;
            uint256 denominator = reserveB * 1000 + amountInWithFee;
            amountOut = numerator / denominator;
        }
        IERC20(tokenIn).transferFrom(msg.sender, address(this), amountIn);
        address tokenOut = tokenIn == address(tokenA) ? address(tokenB) : address(tokenA);
        IERC20(tokenOut).transfer(msg.sender, amountOut);
        emit Swap(msg.sender, amountIn, amountOut);
    }

    function addLiquidity(uint256 amountA, uint256 amountB) external {
        tokenA.transferFrom(msg.sender, address(this), amountA);
        tokenB.transferFrom(msg.sender, address(this), amountB);
        uint256 shares;
        if (totalLiquidity == 0) {
            shares = _sqrt(amountA * amountB);
        } else {
            shares = _min(amountA * totalLiquidity / tokenA.balanceOf(address(this)), amountB * totalLiquidity / tokenB.balanceOf(address(this)));
        }
        liquidity[msg.sender] += shares;
        totalLiquidity += shares;
        _updatePrice(tokenA.balanceOf(address(this)), tokenB.balanceOf(address(this)));
        emit LiquidityAdded(msg.sender, amountA, amountB);
    }

    function _sqrt(uint256 y) internal pure returns (uint256 z) {
        if (y > 3) {
            z = y;
            uint256 x = y / 2 + 1;
            while (x < z) {
                z = x;
                x = (y / x + x) / 2;
            }
        } else if (y != 0) {
            z = 1;
        }
    }

    function _min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }

    function getPrice() external view returns (uint256) {
        uint256 reserveA = tokenA.balanceOf(address(this));
        uint256 reserveB = tokenB.balanceOf(address(this));
        if (reserveA == 0) return 0;
        return (reserveB * 1e18) / reserveA;
    }

    receive() external payable {}
}