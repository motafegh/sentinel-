// expect: Reentrancy
// Read-only reentrancy: a view function checks the contract's state and
// returns a value based on it. During a reentrancy attack where state
// hasn't been updated yet, a third contract calling this view function
// can observe inconsistent state and make decisions based on it.
// The liquidity pool's getReserve() is called by external protocols that
// trust the snapshot — but during reentrancy, it returns stale data.
pragma solidity ^0.8.0;

interface IERC20 {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function transfer(address to, uint256 amount) external returns (bool);
    function balanceOf(address who) external view returns (uint256);
}

contract ReadOnlyReentrancyPool {
    IERC20 public tokenA;
    IERC20 public tokenB;
    address public owner;
    mapping(address => uint256) public lpBalances;
    uint256 public totalLiquidity;

    event LiquidityAdded(address indexed provider, uint256 amountA, uint256 amountB);
    event LiquidityRemoved(address indexed provider, uint256 amountA, uint256 amountB);
    event Swap(address indexed swapper, address tokenIn, address tokenOut, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _a, address _b) {
        tokenA = IERC20(_a);
        tokenB = IERC20(_b);
        owner = msg.sender;
    }

    function addLiquidity(uint256 amountA, uint256 amountB) external {
        tokenA.transferFrom(msg.sender, address(this), amountA);
        tokenB.transferFrom(msg.sender, address(this), amountB);
        uint256 shares;
        if (totalLiquidity == 0) {
            shares = amountA;
        } else {
            shares = (amountA * totalLiquidity) / tokenA.balanceOf(address(this));
        }
        lpBalances[msg.sender] += shares;
        totalLiquidity += shares;
        emit LiquidityAdded(msg.sender, amountA, amountB);
    }

    function removeLiquidity(uint256 shares) external {
        require(lpBalances[msg.sender] >= shares, "insufficient shares");
        uint256 balanceA = tokenA.balanceOf(address(this));
        uint256 balanceB = tokenB.balanceOf(address(this));
        uint256 amountA = (shares * balanceA) / totalLiquidity;
        uint256 amountB = (shares * balanceB) / totalLiquidity;

        lpBalances[msg.sender] -= shares;
        totalLiquidity -= shares;
        tokenA.transfer(msg.sender, amountA);
        tokenB.transfer(msg.sender, amountB);

        emit LiquidityRemoved(msg.sender, amountA, amountB);
    }

    function swap(address tokenIn, address tokenOut, uint256 amountIn) external returns (uint256 amountOut) {
        require(tokenIn == address(tokenA) || tokenIn == address(tokenB), "invalid tokenIn");
        require(tokenOut == address(tokenA) || tokenOut == address(tokenB), "invalid tokenOut");
        require(tokenIn != tokenOut, "same token");

        uint256 reserveIn = IERC20(tokenIn).balanceOf(address(this));
        uint256 reserveOut = IERC20(tokenOut).balanceOf(address(this));

        uint256 amountInWithFee = amountIn * 997;
        uint256 numerator = amountInWithFee * reserveOut;
        uint256 denominator = (reserveIn * 1000) + amountInWithFee;
        amountOut = numerator / denominator;

        tokenA.transferFrom(msg.sender, address(this), amountIn == tokenA ? amountIn : 0);
        tokenB.transferFrom(msg.sender, address(this), amountIn == tokenB ? amountIn : 0);

        if (tokenOut == address(tokenA)) {
            tokenA.transfer(msg.sender, amountOut);
        } else {
            tokenB.transfer(msg.sender, amountOut);
        }

        emit Swap(msg.sender, tokenIn, tokenOut, amountIn);
    }

    function getReserves() external view returns (uint256 reserveA, uint256 reserveB) {
        reserveA = tokenA.balanceOf(address(this));
        reserveB = tokenB.balanceOf(address(this));
    }

    function getShares(address user) external view returns (uint256) {
        return lpBalances[user];
    }

    function transferShares(address to, uint256 shares) external {
        require(lpBalances[msg.sender] >= shares, "insufficient shares");
        lpBalances[msg.sender] -= shares;
        lpBalances[to] += shares;
    }
}