// expect: CallToUnknown
// Proxy contract that delegates to an upgradeable implementation address.
// The implementation address is stored in state and can be swapped by the owner.
// Low-level delegatecall to an arbitrary address with arbitrary calldata.
// Since the address is user-settable, this can call any code in the proxy's storage context.
pragma solidity ^0.8.0;

contract UpgradeableProxy {
    address public implementation;
    address public owner;
    mapping(bytes4 => bool) public blacklistedSignatures;

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _impl) {
        implementation = _impl;
        owner = msg.sender;
    }

    function upgradeTo(address newImpl) external onlyOwner {
        require(newImpl != address(0), "zero address");
        implementation = newImpl;
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "zero address");
        owner = newOwner;
    }

    function addToBlacklist(bytes4 sig) external onlyOwner {
        blacklistedSignatures[sig] = true;
    }

    function removeFromBlacklist(bytes4 sig) external onlyOwner {
        delete blacklistedSignatures[sig];
    }

    fallback(bytes calldata data) external payable returns (bytes memory) {
        bytes4 sig = bytes4(data[:4]);
        if (blacklistedSignatures[sig]) {
            revert("signature blacklisted");
        }
        (bool ok, bytes memory ret) = implementation.delegatecall(data);
        if (!ok) {
            assembly {
                revert(add(ret, 32), mload(ret))
            }
        }
        return ret;
    }

    receive() external payable {}

    function executeBatch(bytes[] calldata calls) external onlyOwner returns (bytes[] memory) {
        bytes[] memory results = new bytes[](calls.length);
        for (uint256 i = 0; i < calls.length; i++) {
            (bool ok, bytes memory ret) = implementation.delegatecall(calls[i]);
            require(ok, "batch call failed");
            results[i] = ret;
        }
        return results;
    }

    function executeOnTarget(address target, bytes calldata data) external onlyOwner returns (bytes memory) {
        (bool ok, bytes memory ret) = target.call{value: address(this).balance}(data);
        require(ok, "target call failed");
        return ret;
    }
}