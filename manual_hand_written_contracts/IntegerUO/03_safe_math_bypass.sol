// expect: IntegerUO
// Lending contract that uses a custom "safe math" library that wraps on overflow.
// The library attempts to check arithmetic but has a bug: it only reverts on
// some patterns while letting others through. The batch operations use unchecked
// arithmetic that silently wraps. The compound interest calculation overflows
// when interest accumulates beyond uint256 capacity.
pragma solidity ^0.8.0;

library BuggyMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        return c;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "sub underflow");
        uint256 c = a - b;
        return c;
    }

    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) return 0;
        uint256 c = a * b;
        return c;
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "div by zero");
        return a / b;
    }

    function pow(uint256 base, uint256 exp) internal pure returns (uint256) {
        uint256 result = 1;
        for (uint256 i = 0; i < exp; i++) {
            result = mul(result, base);
        }
        return result;
    }
}

contract BuggyLending {
    using BuggyMath for uint256;

    mapping(address => uint256) public deposits;
    mapping(address => uint256) public borrows;
    mapping(address => uint256) public interestAccrued;
    uint256 public interestRate = 105;
    uint256 public baseRate = 100;
    address public owner;

    event Deposited(address indexed user, uint256 amount);
    event Borrowed(address indexed user, uint256 amount);

    constructor() {
        owner = msg.sender;
    }

    function deposit() external payable {
        require(msg.value > 0, "zero deposit");
        deposits[msg.sender] = deposits[msg.sender].add(msg.value);
        emit Deposited(msg.sender, msg.value);
    }

    function borrow(uint256 amount) external {
        require(deposits[msg.sender] >= amount, "insufficient deposit");
        deposits[msg.sender] = deposits[msg.sender].sub(amount);
        borrows[msg.sender] = borrows[msg.sender].add(amount);
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "transfer failed");
        emit Borrowed(msg.sender, amount);
    }

    function repay() external payable {
        require(borrows[msg.sender] >= msg.value, "repay exceeds borrow");
        borrows[msg.sender] = borrows[msg.sender].sub(msg.value);
        deposits[msg.sender] = deposits[msg.sender].add(msg.value);
    }

    function accrueInterest(address user) external {
        uint256 borrowed = borrows[user];
        if (borrowed > 0) {
            uint256 interest = borrowed.mul(interestRate).div(baseRate).sub(borrowed);
            interestAccrued[user] = interestAccrued[user].add(interest);
            borrows[user] = borrowed.add(interest);
        }
    }

    function compoundInterest(address user, uint256 periods) external {
        uint256 borrowed = borrows[user];
        if (borrowed > 0) {
            uint256 rate = interestRate.div(baseRate);
            uint256 compounded = borrowed.mul(rate.pow(periods));
            borrows[user] = compounded;
        }
    }

    function batchAccrue(address[] calldata users) external {
        for (uint256 i = 0; i < users.length; i++) {
            uint256 borrowed = borrows[users[i]];
            if (borrowed > 0) {
                uint256 interest = borrowed.mul(interestRate).div(baseRate).sub(borrowed);
                interestAccrued[users[i]] = interestAccrued[users[i]].add(interest);
                borrows[users[i]] = borrowed.add(interest);
            }
        }
    }

    function getPosition(address user) external view returns (uint256 deposit, uint256 borrow, uint256 interest) {
        return (deposits[user], borrows[user], interestAccrued[user]);
    }
}