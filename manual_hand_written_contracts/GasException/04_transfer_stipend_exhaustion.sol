// expect: GasException
// Payment distributor that uses .transfer() and .send() for ETH payouts.
// These methods forward only 2300 gas to the recipient — insufficient for
// any non-trivial fallback logic. If the recipient is a multisig wallet
// or a contract that logs events, the transfer reverts silently.
// The contract has a batch payout that loops over many recipients,
// each using .transfer() — a single failing recipient can brick the batch.
// Also uses .send() in a loop without checking the return value.
pragma solidity ^0.8.0;

contract PaymentDistributor {
    struct Payee {
        address payable addr;
        uint256 amount;
        bool paid;
    }

    address public owner;
    Payee[] public payees;
    mapping(address => uint256) public pendingPayments;
    uint256 public totalPending;

    event PayeeAdded(address indexed payee, uint256 amount);
    event PaymentSent(address indexed payee, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function addPayee(address payable payee, uint256 amount) external onlyOwner {
        pendingPayments[payee] += amount;
        totalPending += amount;
        payees.push(Payee(payee, amount, false));
        emit PayeeAdded(payee, amount);
    }

    function batchAddPayees(address payable[] calldata recipients, uint256[] calldata amounts) external onlyOwner {
        require(recipients.length == amounts.length, "length mismatch");
        for (uint256 i = 0; i < recipients.length; i++) {
            pendingPayments[recipients[i]] += amounts[i];
            totalPending += amounts[i];
            payees.push(Payee(recipients[i], amounts[i], false));
            emit PayeeAdded(recipients[i], amounts[i]);
        }
    }

    function distributeAll() external onlyOwner {
        for (uint256 i = 0; i < payees.length; i++) {
            if (!payees[i].paid) {
                payees[i].paid = true;
                payees[i].addr.transfer(payees[i].amount);
            }
        }
    }

    function distributeSend() external onlyOwner {
        for (uint256 i = 0; i < payees.length; i++) {
            if (!payees[i].paid) {
                payees[i].paid = true;
                bool ok = payees[i].addr.send(payees[i].amount);
                if (!ok) {
                    payees[i].paid = false;
                }
            }
        }
    }

    function paySingleWithTransfer(address payable recipient) external payable onlyOwner {
        require(msg.value > 0, "zero amount");
        recipient.transfer(msg.value);
    }

    function paySingleWithSend(address payable recipient) external payable onlyOwner returns (bool) {
        require(msg.value > 0, "zero amount");
        return recipient.send(msg.value);
    }

    function payBatchWithTransfer(address payable[] calldata recipients, uint256[] calldata amounts) external onlyOwner {
        require(recipients.length == amounts.length, "length mismatch");
        for (uint256 i = 0; i < recipients.length; i++) {
            recipients[i].transfer(amounts[i]);
        }
    }

    function claimPending() external {
        uint256 amount = pendingPayments[msg.sender];
        require(amount > 0, "no pending payment");
        pendingPayments[msg.sender] = 0;
        totalPending -= amount;
        payable(msg.sender).transfer(amount);
    }

    function getPayeeCount() external view returns (uint256) {
        return payees.length;
    }

    receive() external payable {}
}