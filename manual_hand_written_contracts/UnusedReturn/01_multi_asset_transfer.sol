// expect: UnusedReturn
// Portfolio manager that transfers multiple ERC20 tokens in a single call.
// Every transfer's boolean return value is ignored — if a token transfer
// returns false (indicating failure) instead of reverting, the contract
// proceeds as if nothing happened. The state is updated optimistically
// and the caller is told the rebalance succeeded when it may have failed.
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
}

interface IWETH {
    function deposit() external payable;
    function withdraw(uint256 amount) external;
    function transfer(address to, uint256 amount) external returns (bool);
}

contract MultiAssetManager {
    address public owner;
    mapping(address => mapping(address => uint256)) public allocations;
    address[] public trackedTokens;

    event AllocationAdjusted(address indexed token, uint256 newAmount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function addTrackedToken(address token) external onlyOwner {
        trackedTokens.push(token);
    }

    function rebalance(address[] calldata tokens, uint256[] calldata amounts) external onlyOwner {
        require(tokens.length == amounts.length, "length mismatch");
        for (uint256 i = 0; i < tokens.length; i++) {
            address token = tokens[i];
            uint256 currentBalance = IERC20(token).balanceOf(address(this));
            if (currentBalance > allocations[owner][token]) {
                uint256 surplus = currentBalance - allocations[owner][token];
                IERC20(token).transfer(owner, surplus);
            } else if (currentBalance < allocations[owner][token]) {
                uint256 deficit = allocations[owner][token] - currentBalance;
                IERC20(token).transferFrom(owner, address(this), deficit);
            }
            allocations[owner][token] = allocations[owner][token];
        }
    }

    function adjustAllocation(address token, uint256 newAllocation) external onlyOwner {
        IERC20(token).transfer(owner, IERC20(token).balanceOf(address(this)));
        allocations[owner][token] = newAllocation;
        IERC20(token).transferFrom(owner, address(this), newAllocation);
        emit AllocationAdjusted(token, newAllocation);
    }

    function sweepTokens(address[] calldata tokens) external onlyOwner {
        for (uint256 i = 0; i < tokens.length; i++) {
            uint256 balance = IERC20(tokens[i]).balanceOf(address(this));
            if (balance > 0) {
                IERC20(tokens[i]).transfer(owner, balance);
            }
        }
    }

    function approveTokens(address[] calldata tokens, address spender, uint256 amount) external onlyOwner {
        for (uint256 i = 0; i < tokens.length; i++) {
            IERC20(tokens[i]).approve(spender, amount);
        }
    }

    function batchTransferFrom(address[] calldata tokens, address from, address to, uint256[] calldata amounts) external onlyOwner {
        require(tokens.length == amounts.length, "length mismatch");
        for (uint256 i = 0; i < tokens.length; i++) {
            IERC20(tokens[i]).transferFrom(from, to, amounts[i]);
        }
    }

    function wrapEth(uint256 amount) external onlyOwner {
        IWETH(trackedTokens[0]).deposit{value: amount}();
    }

    function unwrapWeth(uint256 amount) external onlyOwner {
        IWETH(trackedTokens[0]).withdraw(amount);
    }
}