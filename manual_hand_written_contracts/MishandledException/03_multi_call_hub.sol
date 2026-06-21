// expect: MishandledException
// Multicall hub that dispatches calls to multiple targets and aggregates results.
// The return values of individual calls are parsed, but some are ignored
// entirely — the function only checks success for the first call in a batch.
// Subsequent calls can fail silently and the caller never knows.
// The internal helper function for low-level calls also discards return data.
pragma solidity ^0.8.0;

contract MultiCallHub {
    address public owner;
    mapping(bytes4 => address) public routingTable;
    uint256 public callNonce;

    event CallDispatched(address indexed target, bytes4 indexed signature, bool success);
    event RouteUpdated(bytes4 indexed sig, address indexed target);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function setRoute(bytes4 sig, address target) external onlyOwner {
        routingTable[sig] = target;
        emit RouteUpdated(sig, target);
    }

    function dispatchSingle(bytes calldata data) external payable returns (bytes memory) {
        require(data.length >= 4, "short data");
        bytes4 sig = bytes4(data[:4]);
        address target = routingTable[sig];
        require(target != address(0), "no route");
        callNonce++;
        (bool ok, bytes memory ret) = target.call{value: msg.value}(data);
        emit CallDispatched(target, sig, ok);
        return ret;
    }

    function dispatchBatch(bytes[] calldata calldatas) external payable returns (bytes[] memory results) {
        results = new bytes[](calldatas.length);
        uint256 totalValue = msg.value;
        for (uint256 i = 0; i < calldatas.length; i++) {
            bytes calldata data = calldatas[i];
            bytes4 sig = bytes4(data[:4]);
            address target = routingTable[sig];
            if (target == address(0)) continue;
            callNonce++;
            uint256 valueForCall = totalValue / (calldatas.length - i);
            (bool ok, bytes memory ret) = target.call{value: valueForCall}(data);
            emit CallDispatched(target, sig, ok);
            results[i] = ret;
            totalValue -= valueForCall;
        }
    }

    function dispatchWithFallback(bytes calldata primaryData, bytes calldata fallbackData) external payable returns (bytes memory) {
        bytes4 sig = bytes4(primaryData[:4]);
        address primaryTarget = routingTable[sig];
        callNonce++;
        (bool ok, bytes memory ret) = primaryTarget.call{value: msg.value}(primaryData);
        emit CallDispatched(primaryTarget, sig, ok);
        if (!ok) {
            bytes4 fbSig = bytes4(fallbackData[:4]);
            address fbTarget = routingTable[fbSig];
            if (fbTarget != address(0)) {
                fbTarget.call(fallbackData);
            }
        }
        return ret;
    }

    function dispatchToArbitrary(address target, bytes calldata data) external payable onlyOwner returns (bytes memory) {
        callNonce++;
        (bool ok, bytes memory ret) = target.call{value: msg.value}(data);
        emit CallDispatched(target, bytes4(data[:4]), ok);
        return ret;
    }

    function forwardAndForget(address[] calldata targets, bytes[] calldata calldatas) external onlyOwner {
        require(targets.length == calldatas.length, "length mismatch");
        for (uint256 i = 0; i < targets.length; i++) {
            callNonce++;
            (bool ok, ) = targets[i].call(calldatas[i]);
            emit CallDispatched(targets[i], bytes4(calldatas[i][:4]), ok);
        }
    }

    function ownerCall(address target, bytes calldata data) external onlyOwner {
        target.call(data);
    }

    receive() external payable {}
}