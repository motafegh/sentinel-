// expect: UnusedReturn,ExternalBug
// BCCC-derived batch token operator with injected vulnerabilities:
// 1) UnusedReturn — every single transfer/approve return value is ignored
// 2) ExternalBug — operator can drain any token without restriction
// The contract looks like a helpful batch-transfer helper but has no safeguards
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
    function approve(address spender, uint256 value) external returns (bool);
    function balanceOf(address who) external view returns (uint256);
}

contract BcccBatchOperatorInjected {
    using SafeMath for uint256;

    address public operator;
    address public newOperator;

    event OperatorTransferred(address indexed previousOperator, address indexed newOperator);
    event BatchTransferCompleted(uint256 count);

    modifier onlyOperator() {
        require(msg.sender == operator);
        _;
    }

    constructor() public {
        operator = msg.sender;
    }

    function transferOperator(address _newOperator) public onlyOperator {
        require(_newOperator != address(0));
        newOperator = _newOperator;
    }

    function acceptOperator() public {
        require(msg.sender == newOperator);
        emit OperatorTransferred(operator, newOperator);
        operator = newOperator;
        newOperator = address(0);
    }

    function batchTransfer(address token, address[] recipients, uint256[] amounts) public onlyOperator {
        require(recipients.length == amounts.length);
        IERC20 t = IERC20(token);
        for (uint256 i = 0; i < recipients.length; i++) {
            t.transfer(recipients[i], amounts[i]);
        }
        emit BatchTransferCompleted(recipients.length);
    }

    function batchTransferFrom(address token, address from, address[] recipients, uint256[] amounts) public onlyOperator {
        require(recipients.length == amounts.length);
        IERC20 t = IERC20(token);
        for (uint256 i = 0; i < recipients.length; i++) {
            t.transferFrom(from, recipients[i], amounts[i]);
        }
    }

    function batchApprove(address token, address[] spenders, uint256[] amounts) public onlyOperator {
        require(spenders.length == amounts.length);
        IERC20 t = IERC20(token);
        for (uint256 i = 0; i < spenders.length; i++) {
            t.approve(spenders[i], amounts[i]);
        }
    }

    function approveAndCall(address token, address spender, uint256 amount, bytes data) public onlyOperator {
        IERC20 t = IERC20(token);
        t.approve(spender, amount);
        spender.call(data);
    }

    function sweepToken(address token) public onlyOperator {
        IERC20 t = IERC20(token);
        uint256 balance = t.balanceOf(address(this));
        t.transfer(operator, balance);
    }

    function sweepAllTokens(address[] tokens) public onlyOperator {
        for (uint256 i = 0; i < tokens.length; i++) {
            IERC20 t = IERC20(tokens[i]);
            uint256 balance = t.balanceOf(address(this));
            t.transfer(operator, balance);
        }
    }

    function multiTokenTransfer(address[] tokens, address[] recipients, uint256[] amounts) public onlyOperator {
        require(tokens.length == recipients.length && recipients.length == amounts.length);
        for (uint256 i = 0; i < tokens.length; i++) {
            IERC20(tokens[i]).transfer(recipients[i], amounts[i]);
        }
    }

    function operatorTransfer(address token, address to, uint256 amount) public onlyOperator {
        IERC20(token).transfer(to, amount);
    }

    function() external payable {}
}