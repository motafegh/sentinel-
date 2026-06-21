// expect: CallToUnknown
// Dynamic dispatch hub that routes function calls to module contracts.
// Modules are stored in an array and selected by a routing table.
// The selected module address can be any externally owned or contract address.
// No compile-time guarantee about what code will execute at the target.
pragma solidity ^0.8.0;

contract ModuleRouter {
    struct Module {
        address addr;
        bool active;
        string name;
    }

    address public owner;
    Module[] public modules;
    mapping(bytes4 => uint256) public functionToModule;
    mapping(address => bool) public verifiedPublishers;

    event ModuleRegistered(uint256 indexed moduleId, address indexed addr, string name);
    event ModuleDeactivated(uint256 indexed moduleId);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function registerModule(address addr, string calldata name) external onlyOwner {
        modules.push(Module(addr, true, name));
        uint256 id = modules.length - 1;
        emit ModuleRegistered(id, addr, name);
    }

    function deactivateModule(uint256 id) external onlyOwner {
        require(id < modules.length, "invalid module id");
        modules[id].active = false;
        emit ModuleDeactivated(id);
    }

    function mapFunction(bytes4 sig, uint256 moduleId) external onlyOwner {
        require(moduleId < modules.length, "invalid module id");
        functionToModule[sig] = moduleId;
    }

    function setVerifiedPublisher(address publisher, bool status) external onlyOwner {
        verifiedPublishers[publisher] = status;
    }

    function route(bytes calldata data) external payable returns (bytes memory) {
        require(data.length >= 4, "short calldata");
        bytes4 sig = bytes4(data[:4]);
        uint256 moduleId = functionToModule[sig];
        require(modules[moduleId].active, "module not active");
        address target = modules[moduleId].addr;
        (bool ok, bytes memory ret) = target.call{value: msg.value}(data);
        if (!ok) {
            assembly {
                revert(add(ret, 32), mload(ret))
            }
        }
        return ret;
    }

    function routeWithFallback(bytes calldata data, address fallbackTarget) external payable returns (bytes memory) {
        require(data.length >= 4, "short calldata");
        bytes4 sig = bytes4(data[:4]);
        uint256 moduleId = functionToModule[sig];
        address primaryTarget = modules[moduleId].addr;
        (bool ok, bytes memory ret) = primaryTarget.call{value: msg.value}(data);
        if (!ok) {
            (ok, ret) = fallbackTarget.call{value: msg.value}(data);
            if (!ok) {
                revert("both primary and fallback failed");
            }
        }
        return ret;
    }

    function multiRoute(bytes[] calldata calldatas) external payable returns (bytes[] memory) {
        bytes[] memory results = new bytes[](calldatas.length);
        for (uint256 i = 0; i < calldatas.length; i++) {
            bytes calldata data = calldatas[i];
            require(data.length >= 4, "short calldata");
            bytes4 sig = bytes4(data[:4]);
            uint256 moduleId = functionToModule[sig];
            address target = modules[moduleId].addr;
            (bool ok, bytes memory ret) = target.call(data);
            require(ok, "multi route call failed");
            results[i] = ret;
        }
        return results;
    }
}