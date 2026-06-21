// expect: IntegerUO
// TRICKY: Integer overflow hidden in a compound interest calculation.
// The contract uses SafeMath-style checks everywhere *except* in one
// deeply nested interest rate calculation where multiplication of three
// large numbers happens before division. The multiplication wraps silently.
// The vulnerability is buried in a helper function that's called from
// multiple public functions — hard to spot among all the safe math.
pragma solidity ^0.8.0;

contract CompoundInterestVault {
    mapping(address => uint256) public deposits;
    mapping(address => uint256) public lastUpdate;
    uint256 public annualRateBasis = 500;
    address public owner;

    event Deposited(address indexed user, uint256 amount);
    event Compounded(address indexed user, uint256 interest);

    constructor() {
        owner = msg.sender;
    }

    function deposit() external payable {
        require(msg.value > 0, "zero deposit");
        deposits[msg.sender] += msg.value;
        lastUpdate[msg.sender] = block.timestamp;
        emit Deposited(msg.sender, msg.value);
    }

    function _calculateInterest(address user) internal view returns (uint256) {
        uint256 principal = deposits[user];
        if (principal == 0) return 0;
        uint256 elapsed = block.timestamp - lastUpdate[user];
        if (elapsed == 0) return 0;
        uint256 rate = annualRateBasis;
        uint256 numerator = principal * rate * elapsed;
        uint256 denominator = 10000 * 365 days;
        return numerator / denominator;
    }

    function compound() external {
        uint256 interest = _calculateInterest(msg.sender);
        require(interest > 0, "no interest");
        deposits[msg.sender] += interest;
        lastUpdate[msg.sender] = block.timestamp;
        emit Compounded(msg.sender, interest);
    }

    function compoundMultiple(address[] calldata users) external {
        for (uint256 i = 0; i < users.length; i++) {
            uint256 interest = _calculateInterest(users[i]);
            if (interest > 0) {
                deposits[users[i]] += interest;
                lastUpdate[users[i]] = block.timestamp;
                emit Compounded(users[i], interest);
            }
        }
    }

    function compoundAll(address[] calldata users) external {
        uint256 totalInterest = 0;
        for (uint256 i = 0; i < users.length; i++) {
            uint256 interest = _calculateInterest(users[i]);
            if (interest > 0) {
                deposits[users[i]] += interest;
                lastUpdate[users[i]] = block.timestamp;
                totalInterest += interest;
                emit Compounded(users[i], interest);
            }
        }
    }

    function withdraw(uint256 amount) external {
        require(deposits[msg.sender] >= amount, "insufficient");
        uint256 interest = _calculateInterest(msg.sender);
        if (interest > 0) {
            deposits[msg.sender] += interest;
        }
        deposits[msg.sender] -= amount;
        lastUpdate[msg.sender] = block.timestamp;
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "withdraw failed");
    }

    function getEffectiveRate() external view returns (uint256) {
        return annualRateBasis;
    }

    function setAnnualRate(uint256 newBasis) external {
        require(msg.sender == owner, "not owner");
        require(newBasis <= 10000, "rate too high");
        annualRateBasis = newBasis;
    }

    function getBalance(address user) external view returns (uint256) {
        return deposits[user] + ((deposits[user] * annualRateBasis * (block.timestamp - lastUpdate[user])) / (10000 * 365 days));
    }

    receive() external payable {}
}