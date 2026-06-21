// expect: CallToUnknown
// Generic forwarder contract that accepts an arbitrary target and data.
// Relays calls to user-specified addresses without any verification.
// No whitelist, no signature check, no governance — anyone can trigger any call.
pragma solidity ^0.8.0;

contract GenericForwarder {
    address public owner;
    uint256 public nonce;
    mapping(bytes32 => bool) public executedHashes;

    event Forwarded(address indexed target, bytes data, bool success);
    event Deposited(address indexed from, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function forward(address target, bytes calldata data) external payable returns (bytes memory) {
        require(target != address(0), "zero target");
        (bool ok, bytes memory ret) = target.call{value: msg.value}(data);
        emit Forwarded(target, data, ok);
        if (!ok) {
            assembly {
                revert(add(ret, 32), mload(ret))
            }
        }
        return ret;
    }

    function forwardWithGasLimit(address target, bytes calldata data, uint256 gasLimit) external payable returns (bytes memory) {
        (bool ok, bytes memory ret) = target.call{gas: gasLimit, value: msg.value}(data);
        emit Forwarded(target, data, ok);
        if (!ok) {
            assembly {
                revert(add(ret, 32), mload(ret))
            }
        }
        return ret;
    }

    function batchForward(address[] calldata targets, bytes[] calldata calldatas) external payable returns (bytes[] memory) {
        require(targets.length == calldatas.length, "length mismatch");
        bytes[] memory results = new bytes[](targets.length);
        uint256 totalValue = msg.value;
        for (uint256 i = 0; i < targets.length; i++) {
            uint256 valueForCall = totalValue / (targets.length - i);
            (bool ok, bytes memory ret) = targets[i].call{value: valueForCall}(calldatas[i]);
            require(ok, "batch forward call failed");
            results[i] = ret;
            totalValue -= valueForCall;
        }
        return results;
    }

    function delegateForward(address target, bytes calldata data) external payable onlyOwner returns (bytes memory) {
        (bool ok, bytes memory ret) = target.delegatecall(data);
        if (!ok) {
            assembly {
                revert(add(ret, 32), mload(ret))
            }
        }
        return ret;
    }

    function staticForward(address target, bytes calldata data) external view returns (bytes memory) {
        (bool ok, bytes memory ret) = target.staticcall(data);
        require(ok, "static forward failed");
        return ret;
    }

    receive() external payable {
        emit Deposited(msg.sender, msg.value);
    }
}