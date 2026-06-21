// expect: ExternalBug
// Contract that uses delegatecall to a user-supplied library address.
// The library address is stored in state and set by the owner.
// If the owner is a multisig that gets compromised, or if the library
// is set to a malicious contract, the attacker gains full control over
// the contract's storage — including owner, balances, and any stored data.
// The delegatecall runs in the caller's storage context.
pragma solidity ^0.8.0;

contract WalletDelegate {
    struct Signature {
        uint8 v;
        bytes32 r;
        bytes32 s;
    }

    address public owner;
    address public libraryAddr;
    mapping(address => uint256) public balances;
    mapping(bytes32 => bool) public usedSignatures;
    uint256 public nonce;

    event LibraryChanged(address indexed oldLib, address indexed newLib);
    event Executed(address indexed target, bytes data);
    event DepositReceived(address indexed from, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function setLibrary(address newLib) external onlyOwner {
        require(newLib != address(0), "zero address");
        address old = libraryAddr;
        libraryAddr = newLib;
        emit LibraryChanged(old, newLib);
    }

    function executeInLibrary(bytes calldata data) external onlyOwner returns (bytes memory) {
        require(libraryAddr != address(0), "no library set");
        (bool ok, bytes memory ret) = libraryAddr.delegatecall(data);
        if (!ok) {
            assembly {
                revert(add(ret, 32), mload(ret))
            }
        }
        return ret;
    }

    function executeOnTarget(address target, bytes calldata data, uint256 value) external onlyOwner returns (bytes memory) {
        (bool ok, bytes memory ret) = target.call{value: value}(data);
        require(ok, "target call failed");
        emit Executed(target, data);
        return ret;
    }

    function batchExecute(
        address[] calldata targets,
        bytes[] calldata calldatas,
        uint256[] calldata values
    ) external onlyOwner returns (bytes[] memory) {
        require(targets.length == calldatas.length, "length mismatch");
        require(targets.length == values.length, "values mismatch");
        bytes[] memory results = new bytes[](targets.length);
        for (uint256 i = 0; i < targets.length; i++) {
            (bool ok, bytes memory ret) = targets[i].call{value: values[i]}(calldatas[i]);
            require(ok, "batch call failed");
            results[i] = ret;
        }
        return results;
    }

    function executeWithPermit(
        address target,
        bytes calldata data,
        uint256 value,
        uint256 deadline,
        Signature calldata sig
    ) external returns (bytes memory) {
        require(block.timestamp <= deadline, "permit expired");
        bytes32 message = keccak256(abi.encode(target, data, value, nonce, deadline));
        bytes32 hash = keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", message));
        require(!usedSignatures[hash], "signature used");
        address signer = ecrecover(hash, sig.v, sig.r, sig.s);
        require(signer == owner, "invalid signature");
        usedSignatures[hash] = true;
        nonce++;
        (bool ok, bytes memory ret) = target.call{value: value}(data);
        require(ok, "permit call failed");
        return ret;
    }

    function deposit() external payable {
        balances[msg.sender] += msg.value;
        emit DepositReceived(msg.sender, msg.value);
    }

    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "insufficient balance");
        balances[msg.sender] -= amount;
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "withdraw failed");
    }

    receive() external payable {
        emit DepositReceived(msg.sender, msg.value);
    }
}