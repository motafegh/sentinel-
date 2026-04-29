// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/PausableUpgradeable.sol";
import "./IZKMLVerifier.sol";
import "./SentinelToken.sol";

// RECALL - Three upgradeable base contracts required for UUPS pattern.
// Initializable: enables initialize() instead of constructor.
// UUPSUpgradeable: upgrade logic lives in implementation, not proxy.
// OwnableUpgradeable: upgradeable-compatible version of Ownable.
// PausableUpgradeable: emergency stop - owner can halt audit submissions.
contract AuditRegistry is Initializable, UUPSUpgradeable, OwnableUpgradeable, PausableUpgradeable {

    // RECALL - Verifier referenced via interface not concrete type (ADR-002).
    // Swappable when EZKL regenerates ZKMLVerifier for a new model version.
    IZKMLVerifier public zkmlVerifier;
    SentinelToken public sentinelToken;

    // RECALL - Full audit record stored per submission.
    // proofHash = keccak256(proof) not raw bytes - 32 bytes vs 29KB.
    // scoreFieldElement is uint256 - EZKL encodes model output as a scaled
    // BN254 field element (model_output * 2^13).
    // Off-chain: divide by 8192 to get human-readable probability.
    struct AuditResult {
        uint256 scoreFieldElement;
        bytes32 proofHash;
        uint256 timestamp;
        address agent;
        bool verified;
    }

    mapping(address => AuditResult[]) private _audits;

    event AuditSubmitted(
        address indexed contractAddress,
        bytes32 proofHash,
        address indexed agent,
        uint256 scoreFieldElement
    );

    event ImplementationUpgraded(address indexed newImplementation);

    // RECALL - _disableInitializers() prevents direct initialisation of the
    // implementation contract (standard UUPS security pattern).
    constructor() {
        _disableInitializers();
    }

    function initialize(
        address verifierAddress,
        address tokenAddress
    ) public initializer {
        __Ownable_init(msg.sender);
        __Pausable_init();
        zkmlVerifier = IZKMLVerifier(verifierAddress);
        sentinelToken = SentinelToken(tokenAddress);
    }

    // --- Emergency controls -----------------------------------------------

    function pause() external onlyOwner { _pause(); }
    function unpause() external onlyOwner { _unpause(); }

    // --- Core logic -------------------------------------------------------

    // RECALL - Three guards enforce SENTINEL's full trust model.
    // Guard 1: economic  - agent must be staked >= MIN_STAKE
    // Guard 2: cryptographic - ZK proof must be valid on-chain
    // Guard 3: consistency   - scoreFieldElement must match publicSignals[64]
    //
    // ENCODING NOTE - little-endian field element in proof.json:
    //   Python: int.from_bytes(bytes.fromhex(instances[64]), byteorder='little')
    //   NOT:    int(instances[64], 16)  <- big-endian, produces wrong value
    //   Scale:  scoreFieldElement = round(model_output * 8192)
    function submitAudit(
        address contractAddress,
        uint256 scoreFieldElement,
        bytes calldata proof,
        uint256[] calldata publicSignals
    ) external whenNotPaused {

        require(
            sentinelToken.stakedBalance(msg.sender) >= sentinelToken.MIN_STAKE(),
            "AuditRegistry: insufficient stake"
        );

        require(
            zkmlVerifier.verifyProof(proof, publicSignals),
            "AuditRegistry: invalid ZK proof"
        );

        require(
            publicSignals[64] == scoreFieldElement,
            "AuditRegistry: score mismatch with proof"
        );

        _audits[contractAddress].push(AuditResult({
            scoreFieldElement: scoreFieldElement,
            proofHash:         keccak256(proof),
            timestamp:         block.timestamp,
            agent:             msg.sender,
            verified:          true
        }));

        emit AuditSubmitted(contractAddress, keccak256(proof), msg.sender, scoreFieldElement);
    }

    // --- Queries ----------------------------------------------------------

    function hasAudit(address contractAddress) external view returns (bool) {
        return _audits[contractAddress].length > 0;
    }

    function getLatestAudit(
        address contractAddress
    ) external view returns (AuditResult memory) {
        AuditResult[] storage audits = _audits[contractAddress];
        require(audits.length > 0, "AuditRegistry: no audits found");
        return audits[audits.length - 1];
    }

    function getAuditHistory(
        address contractAddress
    ) external view returns (AuditResult[] memory) {
        return _audits[contractAddress];
    }

    function getAuditCount(
        address contractAddress
    ) external view returns (uint256) {
        return _audits[contractAddress].length;
    }

    // --- Upgrade ----------------------------------------------------------

    function _authorizeUpgrade(
        address newImplementation
    ) internal override onlyOwner {
        emit ImplementationUpgraded(newImplementation);
    }
}
