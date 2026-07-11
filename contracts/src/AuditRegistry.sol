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

    // --- V2: multi-class audit result (10-class) ---
    // CLASS_NAMES index mapping (must match graph_schema.py):
    //   0: CallToUnknown          5: MishandledException
    //   1: DenialOfService         6: Reentrancy
    //   2: ExternalBug             7: Timestamp
    //   3: GasException            8: TransactionOrderDependence
    //   4: IntegerUO               9: UnusedReturn

    uint256 public constant NUM_CLASSES = 10;
    uint256 public constant INPUT_OFFSET = 128;  // CrossAttentionFusion output dim

    struct AuditResultV2 {
        uint256[10] classScores;  // each = round(prob * 8192), little-endian field element
        bytes32 proofHash;
        bytes32 modelHash;        // SHA-256 of teacher checkpoint (from provenance manifest)
        uint256 timestamp;
        address agent;
        bool verified;
    }

    mapping(address => AuditResultV2[]) private _auditsV2;

    event AuditSubmitted(
        address indexed contractAddress,
        bytes32 proofHash,
        address indexed agent,
        uint256 scoreFieldElement
    );

    event AuditSubmittedV2(
        address indexed contractAddress,
        bytes32 proofHash,
        address indexed agent,
        uint256[10] classScores,
        bytes32 modelHash
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

    // --- V2: multi-class submission --------------------------------------

    // Guard 3 for V2 verifies ALL 10 class scores against publicSignals.
    // publicSignals layout: [fusion_0..fusion_127, class_score_0..class_score_9]
    function submitAuditV2(
        address contractAddress,
        uint256[10] calldata classScores,
        bytes calldata proof,
        uint256[] calldata publicSignals,
        bytes32 modelHash
    ) external whenNotPaused {
        require(
            sentinelToken.stakedBalance(msg.sender) >= sentinelToken.MIN_STAKE(),
            "AuditRegistry: insufficient stake"
        );

        require(
            publicSignals.length >= INPUT_OFFSET + NUM_CLASSES,
            "AuditRegistry: insufficient publicSignals"
        );

        require(
            zkmlVerifier.verifyProof(proof, publicSignals),
            "AuditRegistry: invalid ZK proof"
        );

        for (uint256 i = 0; i < NUM_CLASSES; i++) {
            require(
                publicSignals[INPUT_OFFSET + i] == classScores[i],
                "AuditRegistry: class score mismatch"
            );
        }

        _auditsV2[contractAddress].push(AuditResultV2({
            classScores: classScores,
            proofHash: keccak256(proof),
            modelHash: modelHash,
            timestamp: block.timestamp,
            agent: msg.sender,
            verified: true
        }));

        emit AuditSubmittedV2(contractAddress, keccak256(proof), msg.sender, classScores, modelHash);
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

    // --- V2 queries -------------------------------------------------------

    function hasAuditV2(address contractAddress) external view returns (bool) {
        return _auditsV2[contractAddress].length > 0;
    }

    function getLatestAuditV2(
        address contractAddress
    ) external view returns (AuditResultV2 memory) {
        AuditResultV2[] storage audits = _auditsV2[contractAddress];
        require(audits.length > 0, "AuditRegistry: no V2 audits found");
        return audits[audits.length - 1];
    }

    function getAuditHistoryV2(
        address contractAddress
    ) external view returns (AuditResultV2[] memory) {
        return _auditsV2[contractAddress];
    }

    function getAuditCountV2(
        address contractAddress
    ) external view returns (uint256) {
        return _auditsV2[contractAddress].length;
    }

    // --- Upgrade ----------------------------------------------------------

    function _authorizeUpgrade(
        address newImplementation
    ) internal override onlyOwner {
        emit ImplementationUpgraded(newImplementation);
    }
}
