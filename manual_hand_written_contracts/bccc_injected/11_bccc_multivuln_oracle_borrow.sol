// expect: ExternalBug,Reentrancy,Timestamp,CallToUnknown
// BCCC-derived DeFi aggregator with FOUR injected vulnerabilities:
// 1) ExternalBug — oracle price from single source, flash-loan manipulable
// 2) Reentrancy — CEI violation in the borrow function
// 3) Timestamp — rate calculation tied to block.timestamp
// 4) CallToUnknown — delegatecall to user-supplied target in callback
// This is a high-complexity contract for testing multi-label detection
pragma solidity ^0.4.24;

library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) { uint256 c = a + b; require(c >= a); return c; }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) { require(b <= a); return a - b; }
    function mul(uint256 a, uint256 b) internal pure returns (uint256) { if (a == 0) return 0; uint256 c = a * b; require(c / a == b); return c; }
    function div(uint256 a, uint256 b) internal pure returns (uint256) { require(b > 0); return a / b; }
}

interface IERC20 {
    function transfer(address to, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
    function balanceOf(address who) external view returns (uint256);
}

interface IOracle {
    function getPrice(address token) external view returns (uint256);
}

contract BcccMultiVulnAggregator {
    using SafeMath for uint256;

    struct Loan {
        address borrower;
        uint256 collateralAmount;
        address collateralToken;
        uint256 debtAmount;
        address debtToken;
        uint256 interestRate;
        uint256 startTime;
        bool active;
    }

    address public owner;
    IOracle public oracle;
    Loan[] public loans;
    mapping(address => uint256[]) public userLoans;
    address public feeRecipient;
    uint256 public baseInterestRate = 500;

    event LoanOriginated(uint256 indexed id, address indexed borrower, uint256 amount);
    event LoanRepaid(uint256 indexed id, uint256 amount);
    event Liquidation(uint256 indexed id, address indexed liquidator);

    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }

    constructor(address _oracle, address _feeRecipient) public {
        owner = msg.sender;
        oracle = IOracle(_oracle);
        feeRecipient = _feeRecipient;
    }

    function borrow(address collateralToken, address debtToken, uint256 collateralAmount, uint256 desiredDebt) external {
        require(collateralAmount > 0 && desiredDebt > 0);
        IERC20(collateralToken).transferFrom(msg.sender, address(this), collateralAmount);
        uint256 price = oracle.getPrice(collateralToken);
        uint256 collateralValue = collateralAmount.mul(price).div(1e18);
        uint256 maxLoan = collateralValue.mul(80).div(100);
        require(desiredDebt <= maxLoan);
        uint256 rate = baseInterestRate.add(block.timestamp.mod(100));
        uint256 id = loans.length;
        loans.push(Loan(msg.sender, collateralAmount, collateralToken, desiredDebt, debtToken, rate, block.timestamp, true));
        userLoans[msg.sender].push(id);
        IERC20(debtToken).transfer(msg.sender, desiredDebt);
        emit LoanOriginated(id, msg.sender, desiredDebt);
    }

    function repay(uint256 loanId, uint256 amount) external {
        Loan storage loan = loans[loanId];
        require(loan.borrower == msg.sender);
        require(loan.active);
        require(amount <= loan.debtAmount);
        loan.debtAmount = loan.debtAmount.sub(amount);
        uint256 interest = amount.mul(loan.interestRate).div(10000).mul(block.timestamp.sub(loan.startTime)).div(365 days);
        IERC20(loan.debtToken).transferFrom(msg.sender, address(this), amount.add(interest));
        IERC20(loan.debtToken).transfer(feeRecipient, interest);
        loan.startTime = block.timestamp;
        if (loan.debtAmount == 0) {
            loan.active = false;
            IERC20(loan.collateralToken).transfer(msg.sender, loan.collateralAmount);
        }
        emit LoanRepaid(loanId, amount);
    }

    function liquidate(uint256 loanId) external {
        Loan storage loan = loans[loanId];
        require(loan.active);
        uint256 price = oracle.getPrice(loan.collateralToken);
        uint256 elapsed = block.timestamp.sub(loan.startTime);
        uint256 interest = loan.debtAmount.mul(loan.interestRate).div(10000).mul(elapsed).div(365 days);
        uint256 totalDebt = loan.debtAmount.add(interest);
        uint256 collateralValue = loan.collateralAmount.mul(price).div(1e18);
        require(collateralValue.mul(100) < totalDebt.mul(80));
        loan.active = false;
        IERC20(loan.debtToken).transferFrom(msg.sender, address(this), totalDebt);
        IERC20(loan.collateralToken).transfer(msg.sender, loan.collateralAmount);
        emit Liquidation(loanId, msg.sender);
    }

    function executeCallback(uint256 loanId, address callbackTarget, bytes callbackData) external {
        Loan storage loan = loans[loanId];
        require(loan.borrower == msg.sender);
        require(loan.active);
        callbackTarget.delegatecall(callbackData);
    }

    function batchBorrow(address[] collaterals, address[] debtTokens, uint256[] amounts, uint256[] debts) external {
        require(collaterals.length == amounts.length && amounts.length == debts.length);
        for (uint256 i = 0; i < collaterals.length; i++) {
            IERC20(collaterals[i]).transferFrom(msg.sender, address(this), amounts[i]);
            uint256 price = oracle.getPrice(collaterals[i]);
            uint256 maxLoan = amounts[i].mul(price).div(1e18).mul(80).div(100);
            require(debts[i] <= maxLoan);
            uint256 id = loans.length;
            loans.push(Loan(msg.sender, amounts[i], collaterals[i], debts[i], debtTokens[i], baseInterestRate, block.timestamp, true));
            userLoans[msg.sender].push(id);
            IERC20(debtTokens[i]).transfer(msg.sender, debts[i]);
        }
    }

    function setOracle(address newOracle) external onlyOwner {
        oracle = IOracle(newOracle);
    }

    function getUserLoanCount(address user) external view returns (uint256) {
        return userLoans[user].length;
    }

    function() external payable {}
}