// expect: MishandledException,CallToUnknown
// Contract that makes delegatecall to upgrade its logic without checking
// the return value. If the delegatecall fails (e.g., target has no code,
// or execution reverts), the function continues as if nothing happened.
// The contract state may be left in an inconsistent or uninitialized state.
// Multiple internal call return values are also discarded silently.
pragma solidity ^0.8.0;

contract MutedDelegatecall {
    address public implementation;
    address public owner;
    mapping(address => uint256) public values;
    bool public initialized;

    event ImplementationSet(address indexed impl);
    event ValueUpdated(address indexed updater, uint256 value);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function setImplementation(address impl) external onlyOwner {
        implementation = impl;
        emit ImplementationSet(impl);
    }

    function initialize(bytes calldata data) external onlyOwner {
        if (implementation != address(0)) {
            implementation.delegatecall(data);
        }
        initialized = true;
    }

    function upgradeAndCall(address newImpl, bytes calldata data) external onlyOwner {
        implementation = newImpl;
        emit ImplementationSet(newImpl);
        newImpl.delegatecall(data);
    }

    function mutedDelegateCall(bytes calldata data) external onlyOwner {
        require(implementation != address(0), "no impl");
        implementation.delegatecall(data);
    }

    function tryDelegateCallBatch(bytes[] calldata calldatas) external onlyOwner {
        for (uint256 i = 0; i < calldatas.length; i++) {
            implementation.delegatecall(calldatas[i]);
        }
    }

    function setValue(uint256 val) external {
        values[msg.sender] = val;
        emit ValueUpdated(msg.sender, val);
    }

    function batchSetValues(address[] calldata users, uint256[] calldata vals) external onlyOwner {
        for (uint256 i = 0; i < users.length; i++) {
            bytes memory data = abi.encodeWithSignature("setValue(uint256)", vals[i]);
            users[i].call(data);
        }
    }

    function forwardCall(address target, bytes calldata data) external onlyOwner {
        target.call{gas: 100000}(data);
    }

    function tryMulticall(address[] calldata targets, bytes[] calldata calldatas) external onlyOwner {
        require(targets.length == calldatas.length, "length mismatch");
        for (uint256 i = 0; i < targets.length; i++) {
            targets[i].call(calldatas[i]);
        }
    }

    function recoverFunds() external onlyOwner {
        owner.call{value: address(this).balance}("");
    }

    receive() external payable {}
}