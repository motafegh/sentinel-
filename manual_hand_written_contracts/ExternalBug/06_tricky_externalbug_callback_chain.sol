// expect: ExternalBug,MishandledException
// TRICKY: ExternalBug buried in a complex callback chain across three contracts.
// This contract registers external callbacks that fire during token transfer.
// The onTokenTransfer callback does a delegatecall to a user-supplied address.
// The return value of the delegatecall is not checked. The callback registration
// looks like a harmless "notification" feature but enables arbitrary code execution.
// The MishandledException is in the ignored return value.
pragma solidity ^0.8.0;

contract CallbackRegistry {
    struct Callback {
        address target;
        bytes4 selector;
        bool active;
        uint256 gasLimit;
    }

    address public owner;
    mapping(bytes32 => Callback) public callbacks;
    mapping(address => bytes32[]) public userCallbacks;
    mapping(address => uint256) public totalFees;

    event CallbackRegistered(bytes32 indexed id, address indexed target, bytes4 selector);
    event CallbackExecuted(bytes32 indexed id, bool success);
    event CallbackRemoved(bytes32 indexed id);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function registerCallback(address target, bytes4 selector, uint256 gasLimit) external returns (bytes32) {
        require(target != address(0), "zero target");
        bytes32 id = keccak256(abi.encodePacked(msg.sender, target, selector, block.timestamp));
        callbacks[id] = Callback(target, selector, true, gasLimit);
        userCallbacks[msg.sender].push(id);
        emit CallbackRegistered(id, target, selector);
        return id;
    }

    function executeCallback(bytes32 id) external {
        Callback storage cb = callbacks[id];
        require(cb.active, "not active");
        bytes memory data = abi.encodePacked(cb.selector, abi.encode(msg.sender, block.timestamp));
        uint256 gasToForward = cb.gasLimit > 0 ? cb.gasLimit : gasleft();
        cb.target.call{gas: gasToForward}(data);
    }

    function executeAllCallbacks() external {
        bytes32[] storage ids = userCallbacks[msg.sender];
        for (uint256 i = 0; i < ids.length; i++) {
            Callback storage cb = callbacks[ids[i]];
            if (cb.active) {
                bytes memory data = abi.encodePacked(cb.selector, abi.encode(msg.sender, totalFees[msg.sender]));
                cb.target.call(data);
            }
        }
    }

    function executeDelegateCallback(bytes32 id, bytes memory extraData) external {
        Callback storage cb = callbacks[id];
        require(cb.active, "not active");
        bytes memory fullData = abi.encodePacked(cb.selector, extraData);
        cb.target.delegatecall(fullData);
        emit CallbackExecuted(id, true);
    }

    function batchExecuteCallbacks(bytes32[] calldata ids) external {
        for (uint256 i = 0; i < ids.length; i++) {
            Callback storage cb = callbacks[ids[i]];
            if (cb.active) {
                bytes memory data = abi.encodePacked(cb.selector, abi.encode(msg.sender, i));
                cb.target.call(data);
                emit CallbackExecuted(ids[i], true);
            }
        }
    }

    function removeCallback(bytes32 id) external {
        Callback storage cb = callbacks[id];
        require(cb.active, "already inactive");
        cb.active = false;
        emit CallbackRemoved(id);
    }

    function addFee(uint256 amount) external {
        totalFees[msg.sender] += amount;
    }

    function withdrawFees() external onlyOwner {
        (bool ok, ) = owner.call{value: address(this).balance}("");
        require(ok, "withdraw failed");
    }

    function getCallbackCount(address user) external view returns (uint256) {
        return userCallbacks[user].length;
    }

    receive() external payable {}
}