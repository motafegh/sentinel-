// expect: Timestamp,CallToUnknown,MishandledException
// Multi-vuln: Timestamp dependence in rate limiter + CallToUnknown in execute.
// The rate limiter resets based on block.timestamp allowing miner manipulation,
// and the execute function calls arbitrary targets from a dynamically updated whitelist.
// The whitelist can be updated by the owner (via a timelock using block.timestamp).
pragma solidity ^0.8.0;

contract TimelockExecutor {
    struct PendingCall {
        address target;
        bytes data;
        uint256 value;
        uint256 readyAt;
        bool executed;
    }

    address public owner;
    PendingCall[] public pendingCalls;
    uint256 public timelockDelay = 2 days;

    event CallScheduled(uint256 indexed id, address indexed target, uint256 readyAt);
    event CallExecuted(uint256 indexed id, address indexed target);
    event TimelockUpdated(uint256 oldDelay, uint256 newDelay);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function scheduleCall(address target, bytes calldata data, uint256 value) external onlyOwner returns (uint256) {
        uint256 id = pendingCalls.length;
        pendingCalls.push(PendingCall(target, data, value, block.timestamp + timelockDelay, false));
        emit CallScheduled(id, target, block.timestamp + timelockDelay);
        return id;
    }

    function executeCall(uint256 id) external onlyOwner {
        require(id < pendingCalls.length, "invalid id");
        PendingCall storage pc = pendingCalls[id];
        require(!pc.executed, "already executed");
        require(block.timestamp >= pc.readyAt, "timelock active");
        pc.executed = true;
        (bool ok, ) = pc.target.call{value: pc.value}(pc.data);
        require(ok, "call failed");
        emit CallExecuted(id, pc.target);
    }

    function executeBatch(uint256[] calldata ids) external onlyOwner {
        for (uint256 i = 0; i < ids.length; i++) {
            PendingCall storage pc = pendingCalls[ids[i]];
            if (!pc.executed && block.timestamp >= pc.readyAt) {
                pc.executed = true;
                pc.target.call{value: pc.value}(pc.data);
                emit CallExecuted(ids[i], pc.target);
            }
        }
    }

    function cancelCall(uint256 id) external onlyOwner {
        require(id < pendingCalls.length, "invalid id");
        pendingCalls[id].executed = true;
    }

    function updateTimelock(uint256 newDelay) external onlyOwner {
        emit TimelockUpdated(timelockDelay, newDelay);
        timelockDelay = newDelay;
    }

    function getPendingCount() external view returns (uint256) {
        return pendingCalls.length;
    }

    receive() external payable {}
}