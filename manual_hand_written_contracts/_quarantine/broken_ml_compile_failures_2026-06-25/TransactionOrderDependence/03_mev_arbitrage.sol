// expect: TransactionOrderDependence
// DEX aggregator that reads the best price from an external oracle and executes
// a swap. The oracle price is visible in the mempool as part of the tx calldata.
// A MEV bot sees the profitable trade, front-runs it by buying the asset first
// (driving up the price), then sells into the original transaction's price impact.
// The sandwich attack exploits the transaction ordering in the mempool.
pragma solidity ^0.8.0;

interface IERC20 {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function transfer(address to, uint256 amount) external returns (bool);
    function balanceOf(address who) external view returns (uint256);
}

interface IDEX {
    function swapExactTokensForTokens(uint256 amountIn, uint256 amountOutMin, address[] calldata path, address to, uint256 deadline) external returns (uint256[] memory amounts);
    function getAmountsOut(uint256 amountIn, address[] calldata path) external view returns (uint256[] memory amounts);
}

contract MEVSniper {
    address public owner;
    IERC20 public tokenA;
    IERC20 public tokenB;
    IDEX public dex;
    uint256 public slippageTolerance = 50;

    event SwapExecuted(address indexed user, uint256 amountIn, uint256 amountOut);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _tokenA, address _tokenB, address _dex) {
        owner = msg.sender;
        tokenA = IERC20(_tokenA);
        tokenB = IERC20(_tokenB);
        dex = IDEX(_dex);
    }

    function executeSwap(uint256 amountIn, uint256 minAmountOut) external {
        uint256 amountOut = dex.getAmountsOut(amountIn, getPath())[1];
        require(amountOut >= minAmountOut, "slippage too high");
        tokenA.transferFrom(msg.sender, address(this), amountIn);
        tokenA.approve(address(dex), amountIn);
        uint256[] memory results = dex.swapExactTokensForTokens(amountIn, minAmountOut, getPath(), msg.sender, block.timestamp + 60);
        emit SwapExecuted(msg.sender, amountIn, results[1]);
    }

    function executeSwapWithPermit(uint256 amountIn, uint256 minAmountOut, uint8 v, bytes32 r, bytes32 s, uint256 deadline) external {
        require(deadline >= block.timestamp, "permit expired");
        tokenA.transferFrom(msg.sender, address(this), amountIn);
        tokenA.approve(address(dex), amountIn);
        uint256[] memory results = dex.swapExactTokensForTokens(amountIn, minAmountOut, getPath(), msg.sender, deadline);
        emit SwapExecuted(msg.sender, amountIn, results[1]);
    }

    function sandwich(uint256 amountIn, uint256 frontRunAmount, address victim, uint256 victimMinOut) external onlyOwner {
        tokenA.transferFrom(msg.sender, address(this), frontRunAmount);
        tokenA.approve(address(dex), frontRunAmount);
        dex.swapExactTokensForTokens(frontRunAmount, 0, getPath(), address(this), block.timestamp + 60);
        dex.swapExactTokensForTokens(amountIn, victimMinOut, getPath(), victim, block.timestamp + 60);
        uint256 balanceB = tokenB.balanceOf(address(this));
        tokenB.approve(address(dex), balanceB);
        address[] memory reversePath = new address[](2);
        reversePath[0] = address(tokenB);
        reversePath[1] = address(tokenA);
        uint256[] memory back = dex.swapExactTokensForTokens(balanceB, 0, reversePath, msg.sender, block.timestamp + 60);
    }

    function getPath() internal view returns (address[] memory) {
        address[] memory path = new address[](2);
        path[0] = address(tokenA);
        path[1] = address(tokenB);
        return path;
    }

    function batchSwap(address[] calldata users, uint256[] calldata amounts, uint256 minOut) external onlyOwner {
        require(users.length == amounts.length, "length mismatch");
        for (uint256 i = 0; i < users.length; i++) {
            tokenA.transferFrom(users[i], address(this), amounts[i]);
            tokenA.approve(address(dex), amounts[i]);
            dex.swapExactTokensForTokens(amounts[i], minOut, getPath(), users[i], block.timestamp + 60);
        }
    }

    function withdrawToken(address token, uint256 amount) external onlyOwner {
        IERC20(token).transfer(owner, amount);
    }

    receive() external payable {}
}