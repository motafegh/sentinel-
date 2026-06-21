// expect: CallToUnknown
// Factory contract that creates child contracts via CREATE2 and stores them.
// Later interactions with the created contracts use low-level calls.
// The bytecode at creation time is known, but the factory owner can set
// arbitrary implementation addresses for the routing layer.
// The delegatecall to a user-supplied library address is the critical vector.
pragma solidity ^0.8.0;

contract OpaqueFactory {
    address public template;
    address public owner;
    address public libraryAddress;
    mapping(address => bool) public isChild;
    address[] public children;

    event ChildCreated(address indexed child, address indexed creator);
    event LibraryUpdated(address indexed oldLib, address indexed newLib);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _template) {
        template = _template;
        owner = msg.sender;
    }

    function setLibrary(address lib) external onlyOwner {
        address old = libraryAddress;
        libraryAddress = lib;
        emit LibraryUpdated(old, lib);
    }

    function createChild(bytes32 salt) external returns (address child) {
        bytes memory initCode = abi.encodePacked(
            type(MinimalChild).creationCode,
            abi.encode(msg.sender, template)
        );
        assembly {
            child := create2(0, add(initCode, 32), mload(initCode), salt)
        }
        require(child != address(0), "create2 failed");
        isChild[child] = true;
        children.push(child);
        emit ChildCreated(child, msg.sender);
    }

    function executeOnChild(address child, bytes calldata data) external onlyOwner returns (bytes memory) {
        require(isChild[child], "not a child");
        (bool ok, bytes memory ret) = child.call(data);
        require(ok, "child call failed");
        return ret;
    }

    function delegateToLibrary(bytes calldata data) external onlyOwner returns (bytes memory) {
        require(libraryAddress != address(0), "library not set");
        (bool ok, bytes memory ret) = libraryAddress.delegatecall(data);
        if (!ok) {
            assembly {
                revert(add(ret, 32), mload(ret))
            }
        }
        return ret;
    }

    function batchExecuteOnChildren(bytes[] calldata calldatas) external onlyOwner returns (bytes[] memory) {
        bytes[] memory results = new bytes[](calldatas.length);
        for (uint256 i = 0; i < calldatas.length; i++) {
            (bool ok, bytes memory ret) = address(this).call(calldatas[i]);
            require(ok, "self call failed");
            results[i] = ret;
        }
        return results;
    }

    function delegateBatch(address[] calldata targets, bytes[] calldata calldatas) external onlyOwner returns (bytes[] memory) {
        require(targets.length == calldatas.length, "length mismatch");
        bytes[] memory results = new bytes[](targets.length);
        for (uint256 i = 0; i < targets.length; i++) {
            (bool ok, bytes memory ret) = targets[i].delegatecall(calldatas[i]);
            require(ok, "delegate batch call failed");
            results[i] = ret;
        }
        return results;
    }

    function getChildCount() external view returns (uint256) {
        return children.length;
    }

    function destroyChild(address child) external onlyOwner {
        require(isChild[child], "not a child");
        isChild[child] = false;
    }
}

contract MinimalChild {
    address public parent;
    address public implementation;

    constructor(address _parent, address _impl) {
        parent = _parent;
        implementation = _impl;
    }

    fallback(bytes calldata data) external payable returns (bytes memory) {
        (bool ok, bytes memory ret) = implementation.delegatecall(data);
        if (!ok) {
            assembly {
                revert(add(ret, 32), mload(ret))
            }
        }
        return ret;
    }

    receive() external payable {}
}