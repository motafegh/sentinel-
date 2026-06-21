// expect: CallToUnknown
// TRICKY: CallToUnknown vulnerability hidden in the fallback function.
// The contract looks like a simple forwarding wallet, but the fallback
// makes a delegatecall to the sender's address — an arbitrary, untrusted
// address. Anyone can trigger the fallback by sending 0 ETH with calldata.
// The delegatecall runs in the storage context of this contract.
pragma solidity ^0.8.0;

contract InnocentForwarder {
    address public owner;
    mapping(address => bool) public whitelist;
    mapping(address => uint256) public nonces;

    event Executed(address indexed target, bytes data);
    event OwnerUpdated(address indexed oldOwner, address indexed newOwner);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function updateOwner(address newOwner) external onlyOwner {
        require(newOwner != address(0), "zero address");
        emit OwnerUpdated(owner, newOwner);
        owner = newOwner;
    }

    function addToWhitelist(address target) external onlyOwner {
        whitelist[target] = true;
    }

    function removeFromWhitelist(address target) external onlyOwner {
        whitelist[target] = false;
    }

    function whitelistedCall(address target, bytes calldata data) external onlyOwner returns (bytes memory) {
        require(whitelist[target], "not whitelisted");
        (bool ok, bytes memory ret) = target.call(data);
        require(ok, "whitelisted call failed");
        return ret;
    }

    function batchWhitelistedCall(address[] calldata targets, bytes[] calldata datas) external onlyOwner {
        require(targets.length == datas.length, "length mismatch");
        for (uint256 i = 0; i < targets.length; i++) {
            require(whitelist[targets[i]], "not whitelisted");
            (bool ok, ) = targets[i].call(datas[i]);
            require(ok, "batch call failed");
        }
    }

    function executeMetaTx(address target, bytes calldata data, uint256 nonce, bytes calldata signature) external returns (bytes memory) {
        bytes32 hash = keccak256(abi.encodePacked(target, data, nonce));
        address signer = _recover(hash, signature);
        require(signer == owner, "invalid signature");
        require(!_usedNonces[nonce], "nonce used");
        _usedNonces[nonce] = true;
        (bool ok, bytes memory ret) = target.call(data);
        require(ok, "meta tx failed");
        return ret;
    }

    mapping(uint256 => bool) internal _usedNonces;

    function _recover(bytes32 hash, bytes memory signature) internal pure returns (address) {
        (uint8 v, bytes32 r, bytes32 s) = _splitSignature(signature);
        return ecrecover(hash, v, r, s);
    }

    function _splitSignature(bytes memory sig) internal pure returns (uint8 v, bytes32 r, bytes32 s) {
        require(sig.length == 65, "invalid signature length");
        assembly {
            r := mload(add(sig, 32))
            s := mload(add(sig, 64))
            v := byte(0, mload(add(sig, 96)))
        }
        if (v < 27) v += 27;
    }

    fallback(bytes calldata data) external payable returns (bytes memory) {
        (bool ok, bytes memory ret) = msg.sender.delegatecall(data);
        if (!ok) {
            assembly {
                revert(add(ret, 32), mload(ret))
            }
        }
        return ret;
    }

    receive() external payable {}
}