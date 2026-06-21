// expect: UnusedReturn
// DEX router contract that approves tokens and then executes swaps.
// The approve function's return value is discarded throughout — if the token
// returns false (non-reverting), the swap proceeds without the approval
// being set, causing the subsequent transferFrom to fail silently.
// Multiple internal call return values are also discarded throughout
// the settlement and withdrawal functions.
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
}

interface IDEX {
    function swapExactTokensForTokens(uint256 amountIn, uint256 amountOutMin, address[] calldata path, address to, uint256 deadline) external returns (uint256[] memory amounts);
    function addLiquidity(address tokenA, address tokenB, uint256 amountADesired, uint256 amountBDesired, uint256 amountAMin, uint256 amountBMin, address to, uint256 deadline) external returns (uint256 amountA, uint256 amountB, uint256 liquidity);
}

contract ApproveIgnorerRouter {
    address public owner;
    IDEX public dex;

    event Swapped(address indexed user, uint256 amountIn, uint256 amountOut);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _dex) {
        owner = msg.sender;
        dex = IDEX(_dex);
    }

    function swapExactTokensForTokens(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path
    ) external {
        IERC20(tokenIn).transferFrom(msg.sender, address(this), amountIn);
        IERC20(tokenIn).approve(address(dex), amountIn);
        uint256[] memory amounts = dex.swapExactTokensForTokens(amountIn, amountOutMin, path, msg.sender, block.timestamp + 60);
        emit Swapped(msg.sender, amountIn, amounts[amounts.length - 1]);
    }

    function addLiquidity(
        address tokenA,
        address tokenB,
        uint256 amountA,
        uint256 amountB
    ) external {
        IERC20(tokenA).transferFrom(msg.sender, address(this), amountA);
        IERC20(tokenB).transferFrom(msg.sender, address(this), amountB);
        IERC20(tokenA).approve(address(dex), amountA);
        IERC20(tokenB).approve(address(dex), amountB);
        dex.addLiquidity(tokenA, tokenB, amountA, amountB, 0, 0, msg.sender, block.timestamp + 60);
    }

    function batchSwap(
        address[] calldata tokensIn,
        uint256[] calldata amountsIn,
        address[][] calldata paths
    ) external {
        require(tokensIn.length == amountsIn.length, "length mismatch");
        require(tokensIn.length == paths.length, "path mismatch");
        for (uint256 i = 0; i < tokensIn.length; i++) {
            IERC20(tokensIn[i]).transferFrom(msg.sender, address(this), amountsIn[i]);
            IERC20(tokensIn[i]).approve(address(dex), amountsIn[i]);
            dex.swapExactTokensForTokens(amountsIn[i], 0, paths[i], msg.sender, block.timestamp + 60);
            emit Swapped(msg.sender, amountsIn[i], 0);
        }
    }

    function batchApprove(address[] calldata tokens, address spender, uint256[] calldata amounts) external onlyOwner {
        require(tokens.length == amounts.length, "length mismatch");
        for (uint256 i = 0; i < tokens.length; i++) {
            IERC20(tokens[i]).approve(spender, amounts[i]);
        }
    }

    function batchTransferTokens(address[] calldata tokens, address[] calldata recipients, uint256[] calldata amounts) external onlyOwner {
        require(tokens.length == recipients.length, "length mismatch");
        require(tokens.length == amounts.length, "amount mismatch");
        for (uint256 i = 0; i < tokens.length; i++) {
            IERC20(tokens[i]).transfer(recipients[i], amounts[i]);
        }
    }

    function rescueTokens(address[] calldata tokens) external onlyOwner {
        for (uint256 i = 0; i < tokens.length; i++) {
            uint256 balance = IERC20(tokens[i]).balanceOf(address(this));
            if (balance > 0) {
                IERC20(tokens[i]).transfer(owner, balance);
            }
        }
    }

    receive() external payable {}
}