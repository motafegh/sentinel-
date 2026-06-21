// expect: GasException
// Registry contract that stores large byte arrays in calldata during function calls.
// Each call with large calldata costs gas proportional to calldata size
// (68 gas per non-zero byte for tx data, 4 gas per zero byte).
// The contract requires submitting large proofs or data blobs on-chain.
// An attacker can spam the contract with oversized calldata to congest the chain.
// The dynamic array expansion in storage read also causes gas spikes.
pragma solidity ^0.8.0;

contract LargeDataRegistry {
    struct Proof {
        bytes32 root;
        bytes data;
        address submitter;
        uint256 timestamp;
    }

    address public owner;
    Proof[] public proofs;
    mapping(bytes32 => bool) public submittedRoots;
    uint256 public maxProofSize = 100000;
    uint256 public totalDataStored;

    event ProofSubmitted(bytes32 indexed root, address indexed submitter, uint256 dataSize);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function submitProof(bytes32 root, bytes calldata data) external {
        require(data.length <= maxProofSize, "proof too large");
        require(!submittedRoots[root], "root already submitted");
        submittedRoots[root] = true;
        proofs.push(Proof(root, data, msg.sender, block.timestamp));
        totalDataStored += data.length;
        emit ProofSubmitted(root, msg.sender, data.length);
    }

    function batchSubmitProofs(bytes32[] calldata roots, bytes[] calldata dataBlobs) external {
        require(roots.length == dataBlobs.length, "length mismatch");
        for (uint256 i = 0; i < roots.length; i++) {
            require(dataBlobs[i].length <= maxProofSize, "proof too large");
            require(!submittedRoots[roots[i]], "root already submitted");
            submittedRoots[roots[i]] = true;
            proofs.push(Proof(roots[i], dataBlobs[i], msg.sender, block.timestamp));
            totalDataStored += dataBlobs[i].length;
            emit ProofSubmitted(roots[i], msg.sender, dataBlobs[i].length);
        }
    }

    function verifyProof(bytes32 root) external view returns (bool found, uint256 index) {
        for (uint256 i = 0; i < proofs.length; i++) {
            if (proofs[i].root == root) {
                return (true, i);
            }
        }
        return (false, 0);
    }

    function getProofData(uint256 index) external view returns (bytes memory) {
        require(index < proofs.length, "invalid index");
        return proofs[index].data;
    }

    function concatenateProofs(uint256 from, uint256 to) external view returns (bytes memory) {
        require(to > from, "invalid range");
        require(to <= proofs.length, "out of bounds");
        uint256 totalLen = 0;
        for (uint256 i = from; i < to; i++) {
            totalLen += proofs[i].data.length;
        }
        bytes memory result = new bytes(totalLen);
        uint256 offset = 0;
        for (uint256 j = from; j < to; j++) {
            bytes storage d = proofs[j].data;
            for (uint256 k = 0; k < d.length; k++) {
                result[offset + k] = d[k];
            }
            offset += d.length;
        }
        return result;
    }

    function setMaxProofSize(uint256 newMax) external onlyOwner {
        maxProofSize = newMax;
    }

    function pruneOldProofs(uint256 threshold) external onlyOwner {
        uint256 pruneBefore = block.timestamp - threshold;
        uint256 newLength = 0;
        for (uint256 i = 0; i < proofs.length; i++) {
            if (proofs[i].timestamp >= pruneBefore) {
                proofs[newLength] = proofs[i];
                newLength++;
            }
        }
        while (proofs.length > newLength) {
            proofs.pop();
        }
    }
}