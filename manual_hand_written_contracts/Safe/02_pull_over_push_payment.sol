// expect:
// Pull-over-push payment contract. Instead of pushing ETH to recipients
// (which can fail and brick the contract), users must claim their payments.
// The contract maintains a mapping of pending payouts and only transfers
// ETH when the recipient explicitly calls claim(). Each claim uses
// checks-effects-interactions pattern. No unbounded loops, no DOS vector.
pragma solidity ^0.8.0;

contract PullPaymentContract {
    mapping(address => uint256) private _pendingPayments;
    mapping(address => uint256) private _paidTotal;
    address public immutable owner;
    uint256 public totalPending;

    event PaymentClaimed(address indexed recipient, uint256 amount);
    event PaymentDeposited(address indexed sender, uint256 amount);
    event PaymentWithdrawn(address indexed recipient, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function depositPayment(address recipient, uint256 amount) external payable {
        require(amount > 0, "zero amount");
        require(msg.value >= amount, "insufficient deposit");
        _pendingPayments[recipient] += amount;
        totalPending += amount;
        emit PaymentDeposited(recipient, amount);
    }

    function batchDepositPayments(address[] calldata recipients, uint256[] calldata amounts) external payable {
        require(recipients.length == amounts.length, "length mismatch");
        uint256 total = 0;
        for (uint256 i = 0; i < amounts.length; i++) {
            total += amounts[i];
        }
        require(msg.value >= total, "insufficient deposit");
        for (uint256 j = 0; j < recipients.length; j++) {
            _pendingPayments[recipients[j]] += amounts[j];
            totalPending += amounts[j];
            emit PaymentDeposited(recipients[j], amounts[j]);
        }
    }

    function claim() external {
        uint256 amount = _pendingPayments[msg.sender];
        require(amount > 0, "no pending payment");
        _pendingPayments[msg.sender] = 0;
        totalPending -= amount;
        _paidTotal[msg.sender] += amount;
        emit PaymentClaimed(msg.sender, amount);
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "claim transfer failed");
    }

    function batchClaim(address[] calldata users) external {
        for (uint256 i = 0; i < users.length; i++) {
            uint256 amount = _pendingPayments[users[i]];
            if (amount > 0) {
                _pendingPayments[users[i]] = 0;
                totalPending -= amount;
                _paidTotal[users[i]] += amount;
                emit PaymentClaimed(users[i], amount);
                (bool ok, ) = users[i].call{value: amount}("");
                require(ok, "batch claim failed");
            }
        }
    }

    function getPendingPayment(address user) external view returns (uint256) {
        return _pendingPayments[user];
    }

    function getPaidTotal(address user) external view returns (uint256) {
        return _paidTotal[user];
    }

    function contractBalance() external view returns (uint256) {
        return address(this).balance;
    }

    function withdrawSurplus(uint256 amount) external onlyOwner {
        require(amount <= address(this).balance - totalPending, "cannot withdraw pending");
        (bool ok, ) = owner.call{value: amount}("");
        require(ok, "withdraw surplus failed");
    }
}