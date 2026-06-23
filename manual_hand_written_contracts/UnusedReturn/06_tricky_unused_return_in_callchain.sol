// expect: UnusedReturn,CallToUnknown,MishandledException
// TRICKY: Unused return values hidden in a complex multi-step DeFi operation.
// The contract chains 5 external calls together. The first 4 calls have
// their return values properly checked. The 5th call — a token transfer
// deep inside the final step — silently ignores its return value.
// The CallToUnknown is from a user-supplied swap router address that gets
// called without validation. Both vulnerabilities are at the end of a
// long function body where reviewers stop paying attention.
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function balanceOf(address who) external view returns (uint256);
}

interface IDEXRouter {
    function swapExactTokensForTokens(uint256 amountIn, uint256 amountOutMin, address[] calldata path, address to, uint256 deadline) external returns (uint256[] memory amounts);
    function getAmountsOut(uint256 amountIn, address[] calldata path) external view returns (uint256[] memory amounts);
}

contract DeFiOperator {
    address public owner;
    IERC20 public tokenA;
    IERC20 public tokenB;
    IERC20 public tokenC;
    IDEXRouter public router;

    event SwapExecuted(address indexed user, uint256 amountIn, uint256 amountOut);
    event RebalanceCompleted(address indexed user, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _router, address _a, address _b, address _c) {
        owner = msg.sender;
        router = IDEXRouter(_router);
        tokenA = IERC20(_a);
        tokenB = IERC20(_b);
        tokenC = IERC20(_c);
    }

    function executeMultiStepSwap(uint256 amountIn, uint256 minOutA, uint256 minOutB) external returns (uint256) {
        require(amountIn > 0, "zero amount");
        require(tokenA.transferFrom(msg.sender, address(this), amountIn), "step1: transferFrom failed");
        require(tokenA.approve(address(router), amountIn), "step2: approve failed");
        address[] memory pathAB = new address[](2);
        pathAB[0] = address(tokenA);
        pathAB[1] = address(tokenB);
        uint256[] memory amountsAB = router.swapExactTokensForTokens(amountIn, minOutA, pathAB, address(this), block.timestamp + 60);
        require(amountsAB.length > 0, "step3: swap failed");
        uint256 amountB = amountsAB[amountsAB.length - 1];
        require(amountB >= minOutA, "step3: slippage");
        require(tokenB.approve(address(router), amountB), "step4: approve B failed");
        address[] memory pathBC = new address[](2);
        pathBC[0] = address(tokenB);
        pathBC[1] = address(tokenC);
        uint256[] memory amountsBC = router.swapExactTokensForTokens(amountB, minOutB, pathBC, address(this), block.timestamp + 60);
        uint256 amountC = amountsBC[amountsBC.length - 1];
        tokenC.transfer(msg.sender, amountC);
        emit SwapExecuted(msg.sender, amountIn, amountC);
        return amountC;
    }

    function approveAndSwap(address tokenIn, uint256 amount, address customRouter, address[] calldata path, uint256 minOut) external returns (uint256) {
        require(tokenIn != address(0), "zero token");
        IERC20(tokenIn).transferFrom(msg.sender, address(this), amount);
        IERC20(tokenIn).approve(customRouter, amount);
        uint256[] memory amounts = IDEXRouter(customRouter).swapExactTokensForTokens(amount, minOut, path, msg.sender, block.timestamp + 60);
        return amounts[amounts.length - 1];
    }

    function batchRebalance(address[] calldata users, uint256[] calldata amountsA, uint256[] calldata minOuts) external onlyOwner {
        require(users.length == amountsA.length && amountsA.length == minOuts.length, "length mismatch");
        for (uint256 i = 0; i < users.length; i++) {
            require(tokenA.transferFrom(users[i], address(this), amountsA[i]), "batch step1 failed");
            require(tokenA.approve(address(router), amountsA[i]), "batch step2 failed");
            address[] memory path = new address[](2);
            path[0] = address(tokenA);
            path[1] = address(tokenB);
            uint256[] memory amounts = router.swapExactTokensForTokens(amountsA[i], minOuts[i], path, users[i], block.timestamp + 60);
            tokenB.transfer(users[i], amounts[amounts.length - 1]);
            emit RebalanceCompleted(users[i], amounts[amounts.length - 1]);
        }
    }

    function withdrawFees() external onlyOwner {
        uint256 ethBalance = address(this).balance;
        if (ethBalance > 0) {
            (bool ok, ) = owner.call{value: ethBalance}("");
            require(ok, "eth withdraw failed");
        }
        uint256 balanceA = tokenA.balanceOf(address(this));
        uint256 balanceB = tokenB.balanceOf(address(this));
        uint256 balanceC = tokenC.balanceOf(address(this));
        if (balanceA > 0) { tokenA.transfer(owner, balanceA); }
        if (balanceB > 0) { tokenB.transfer(owner, balanceB); }
        if (balanceC > 0) { tokenC.transfer(owner, balanceC); }
    }

    receive() external payable {}
}