// expect: TransactionOrderDependence
// TRICKY: Transaction order dependence buried in a FIFO payout queue.
// The contract processes payments in order of submission. A user can monitor
// the mempool for a large payout transaction and front-run it with their own
// smaller payout to get priority in the queue. The ordering vulnerability
// is masked by the "fair" FIFO algorithm — it looks deterministic but is
// actually dependent on mempool ordering. The vulnerability is in the
// processQueue function which processes items sequentially from an array.
pragma solidity ^0.8.0;

contract FIFOPayoutQueue {
    struct QueueItem {
        address recipient;
        uint256 amount;
        uint256 submittedAt;
        uint256 priority;
    }

    address public owner;
    QueueItem[] public queue;
    mapping(address => uint256) public pendingPayouts;
    uint256 public currentIndex;
    uint256 public nonce;

    event Enqueued(address indexed recipient, uint256 amount, uint256 priority);
    event Processed(address indexed recipient, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function enqueue(address recipient, uint256 amount) external {
        require(recipient != address(0), "zero address");
        require(amount > 0, "zero amount");
        uint256 priority = nonce++;
        queue.push(QueueItem(recipient, amount, block.timestamp, priority));
        pendingPayouts[recipient] += amount;
        emit Enqueued(recipient, amount, priority);
    }

    function enqueueAll(address[] calldata recipients, uint256[] calldata amounts) external {
        require(recipients.length == amounts.length, "length mismatch");
        for (uint256 i = 0; i < recipients.length; i++) {
            uint256 priority = nonce++;
            queue.push(QueueItem(recipients[i], amounts[i], block.timestamp, priority));
            pendingPayouts[recipients[i]] += amounts[i];
            emit Enqueued(recipients[i], amounts[i], priority);
        }
    }

    function processQueue(uint256 batchSize) external onlyOwner {
        require(batchSize > 0, "zero batch");
        uint256 processed = 0;
        while (currentIndex < queue.length && processed < batchSize) {
            QueueItem storage item = queue[currentIndex];
            if (address(this).balance >= item.amount) {
                pendingPayouts[item.recipient] -= item.amount;
                (bool ok, ) = item.recipient.call{value: item.amount}("");
                if (ok) {
                    emit Processed(item.recipient, item.amount);
                } else {
                    pendingPayouts[item.recipient] += item.amount;
                }
            }
            currentIndex++;
            processed++;
        }
    }

    function getQueueLength() external view returns (uint256) {
        return queue.length;
    }

    function getRemaining() external view returns (uint256) {
        return queue.length - currentIndex;
    }

    function addFunds() external payable {}

    function emergencyWithdraw() external onlyOwner {
        (bool ok, ) = owner.call{value: address(this).balance}("");
        require(ok, "withdraw failed");
    }

    receive() external payable {}
}