// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/PausableUpgradeable.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "./IZKMLVerifier.sol";
import "./SentinelToken.sol";

/// @notice Upgradeable SENTINEL audit registry.
///
/// Trust boundaries are versioned explicitly:
/// - V1: historical scalar proof path (65-signal protocol).
/// - V2: historical 128-input/10-output proxy proof path. The proof does not
///   bind audit context/model identity; retained for pre-V3 compatibility only.
/// - V3: the same proxy-proof class is accepted only together with an EIP-712
///   policy attestation binding proof + public signals + target bytecode +
///   agent + round + model/data/schema identities to this chain and registry.
///
/// V3 does NOT claim that EZKL proves the full teacher or Solidity source-code
/// execution. It combines (1) proxy-computation proof and (2) a separately
/// authenticated policy/provenance statement. This distinction is intentional.
contract AuditRegistry is Initializable, UUPSUpgradeable, OwnableUpgradeable, PausableUpgradeable {
    using ECDSA for bytes32;

    // ---------------------------------------------------------------------
    // Existing storage — NEVER reorder (UUPS compatibility)
    // ---------------------------------------------------------------------

    IZKMLVerifier public zkmlVerifier;
    SentinelToken public sentinelToken;

    struct AuditResult {
        uint256 scoreFieldElement;
        bytes32 proofHash;
        uint256 timestamp;
        address agent;
        bool verified;
    }

    mapping(address => AuditResult[]) private _audits;

    uint256 public constant NUM_CLASSES = 10;
    uint256 public constant INPUT_OFFSET = 128;
    uint256 public constant V2_TOTAL_SIGNALS = INPUT_OFFSET + NUM_CLASSES;

    struct AuditResultV2 {
        uint256[10] classScores;
        bytes32 proofHash;
        bytes32 modelHash;
        uint256 timestamp;
        address agent;
        bool verified;
    }

    mapping(address => AuditResultV2[]) private _auditsV2;

    // ---------------------------------------------------------------------
    // V3 append-only storage
    // ---------------------------------------------------------------------

    /// @dev Dedicated verifier for the context-attested V3 path. Keeping this
    /// separate avoids silently changing the meaning of historical V1/V2 rows.
    IZKMLVerifier public zkmlVerifierV3;

    /// @dev Dedicated service identity allowed to attest audit context.
    address public auditPolicySignerV3;

    /// @dev Set permanently by initializeV3(). Historical queries remain live;
    /// only new V1/V2 submissions are disabled after V3 activation.
    bool public legacySubmissionsDisabled;

    struct AuditContextV3 {
        address contractAddress;
        uint256 roundId;
        bytes32 teacherModelHash;
        bytes32 proxyBundleHash;
        bytes32 dataVersionHash;
        bytes32 classSchemaHash;
        uint256 deadline;
    }

    struct AuditResultV3 {
        uint256[10] classScoreFelts;
        bytes32 proofHash;
        bytes32 requestDigest;
        bytes32 publicSignalsHash;
        bytes32 contractCodeHash;
        bytes32 teacherModelHash;
        bytes32 proxyBundleHash;
        bytes32 dataVersionHash;
        bytes32 classSchemaHash;
        uint256 roundId;
        uint256 timestamp;
        address agent;
        address policySigner;
        address verifier;
        bool verified;
    }

    mapping(address => AuditResultV3[]) private _auditsV3;
    mapping(bytes32 => bool) private _usedV3RequestDigests;

    // EIP-712 domain is calculated dynamically so chain-id changes/forks do not
    // accidentally reuse signatures across domains.
    bytes32 private constant _EIP712_DOMAIN_TYPEHASH = keccak256(
        "EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"
    );
    bytes32 private constant _EIP712_NAME_HASH = keccak256("SENTINEL Audit Registry");
    bytes32 private constant _EIP712_VERSION_HASH = keccak256("3");
    bytes32 public constant AUDIT_REQUEST_V3_TYPEHASH = keccak256(
        "SentinelAuditV3(address agent,address contractAddress,bytes32 contractCodeHash,uint256 roundId,bytes32 teacherModelHash,bytes32 proxyBundleHash,bytes32 dataVersionHash,bytes32 classSchemaHash,bytes32 proofHash,bytes32 publicSignalsHash,bytes32 classScoreFeltsHash,uint256 deadline)"
    );

    // ---------------------------------------------------------------------
    // Events
    // ---------------------------------------------------------------------

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

    event AuditSubmittedV3(
        address indexed contractAddress,
        bytes32 indexed requestDigest,
        bytes32 indexed proofHash,
        address agent,
        uint256[10] classScoreFelts,
        bytes32 teacherModelHash,
        bytes32 proxyBundleHash,
        bytes32 dataVersionHash,
        bytes32 classSchemaHash,
        uint256 roundId
    );

    event V3VerifierUpdated(address indexed previousVerifier, address indexed newVerifier);
    event V3PolicySignerUpdated(address indexed previousSigner, address indexed newSigner);
    event LegacySubmissionsDisabled();
    event ImplementationUpgraded(address indexed newImplementation);

    constructor() {
        _disableInitializers();
    }

    function initialize(
        address verifierAddress,
        address tokenAddress
    ) public initializer {
        require(verifierAddress != address(0), "AuditRegistry: zero verifier");
        require(tokenAddress != address(0), "AuditRegistry: zero token");
        __Ownable_init(msg.sender);
        __Pausable_init();
        zkmlVerifier = IZKMLVerifier(verifierAddress);
        sentinelToken = SentinelToken(tokenAddress);
    }

    /// @notice Activate the context-bound V3 protocol after upgrading an
    /// existing proxy. This is intentionally a one-way containment boundary:
    /// once V3 is activated, new V1/V2 submissions stay disabled.
    function initializeV3(
        address verifierAddress,
        address policySignerAddress
    ) external reinitializer(2) onlyOwner {
        require(verifierAddress != address(0), "AuditRegistry: zero V3 verifier");
        require(policySignerAddress != address(0), "AuditRegistry: zero V3 signer");
        zkmlVerifierV3 = IZKMLVerifier(verifierAddress);
        auditPolicySignerV3 = policySignerAddress;
        legacySubmissionsDisabled = true;
        emit V3VerifierUpdated(address(0), verifierAddress);
        emit V3PolicySignerUpdated(address(0), policySignerAddress);
        emit LegacySubmissionsDisabled();
    }

    // ---------------------------------------------------------------------
    // Emergency / V3 trust-root controls
    // ---------------------------------------------------------------------

    function pause() external onlyOwner { _pause(); }
    function unpause() external onlyOwner { _unpause(); }

    function setZkmlVerifierV3(address verifierAddress) external onlyOwner {
        require(verifierAddress != address(0), "AuditRegistry: zero V3 verifier");
        address previous = address(zkmlVerifierV3);
        zkmlVerifierV3 = IZKMLVerifier(verifierAddress);
        emit V3VerifierUpdated(previous, verifierAddress);
    }

    function setAuditPolicySignerV3(address signerAddress) external onlyOwner {
        require(signerAddress != address(0), "AuditRegistry: zero V3 signer");
        address previous = auditPolicySignerV3;
        auditPolicySignerV3 = signerAddress;
        emit V3PolicySignerUpdated(previous, signerAddress);
    }

    // ---------------------------------------------------------------------
    // Historical V1/V2 writes
    // ---------------------------------------------------------------------

    function submitAudit(
        address contractAddress,
        uint256 scoreFieldElement,
        bytes calldata proof,
        uint256[] calldata publicSignals
    ) external whenNotPaused {
        require(!legacySubmissionsDisabled, "AuditRegistry: legacy submissions disabled");
        require(
            sentinelToken.stakedBalance(msg.sender) >= sentinelToken.MIN_STAKE(),
            "AuditRegistry: insufficient stake"
        );
        require(publicSignals.length > 64, "AuditRegistry: insufficient public signals");
        require(
            zkmlVerifier.verifyProof(proof, publicSignals),
            "AuditRegistry: invalid ZK proof"
        );
        require(
            publicSignals[64] == scoreFieldElement,
            "AuditRegistry: score mismatch with proof"
        );

        bytes32 proofHash = keccak256(proof);
        _audits[contractAddress].push(AuditResult({
            scoreFieldElement: scoreFieldElement,
            proofHash: proofHash,
            timestamp: block.timestamp,
            agent: msg.sender,
            verified: true
        }));
        emit AuditSubmitted(contractAddress, proofHash, msg.sender, scoreFieldElement);
    }

    function submitAuditV2(
        address contractAddress,
        uint256[10] calldata classScores,
        bytes calldata proof,
        uint256[] calldata publicSignals,
        bytes32 modelHash
    ) external whenNotPaused {
        require(!legacySubmissionsDisabled, "AuditRegistry: legacy submissions disabled");
        require(
            sentinelToken.stakedBalance(msg.sender) >= sentinelToken.MIN_STAKE(),
            "AuditRegistry: insufficient stake"
        );
        require(
            publicSignals.length == V2_TOTAL_SIGNALS,
            "AuditRegistry: invalid V2 public signal count"
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

        bytes32 proofHash = keccak256(proof);
        _auditsV2[contractAddress].push(AuditResultV2({
            classScores: classScores,
            proofHash: proofHash,
            modelHash: modelHash,
            timestamp: block.timestamp,
            agent: msg.sender,
            verified: true
        }));
        emit AuditSubmittedV2(contractAddress, proofHash, msg.sender, classScores, modelHash);
    }

    // ---------------------------------------------------------------------
    // V3 context-bound submission
    // ---------------------------------------------------------------------

    function domainSeparatorV3() public view returns (bytes32) {
        return keccak256(abi.encode(
            _EIP712_DOMAIN_TYPEHASH,
            _EIP712_NAME_HASH,
            _EIP712_VERSION_HASH,
            block.chainid,
            address(this)
        ));
    }

    function computeAuditDigestV3(
        address agent,
        AuditContextV3 calldata context,
        bytes32 proofHash,
        bytes32 publicSignalsHash,
        bytes32 classScoreFeltsHash
    ) public view returns (bytes32) {
        bytes32 contractCodeHash = context.contractAddress.codehash;
        bytes32 structHash = keccak256(abi.encode(
            AUDIT_REQUEST_V3_TYPEHASH,
            agent,
            context.contractAddress,
            contractCodeHash,
            context.roundId,
            context.teacherModelHash,
            context.proxyBundleHash,
            context.dataVersionHash,
            context.classSchemaHash,
            proofHash,
            publicSignalsHash,
            classScoreFeltsHash,
            context.deadline
        ));
        return keccak256(abi.encodePacked("\x19\x01", domainSeparatorV3(), structHash));
    }

    function submitAuditV3(
        AuditContextV3 calldata context,
        uint256[10] calldata classScoreFelts,
        bytes calldata proof,
        uint256[] calldata publicSignals,
        bytes calldata policySignature
    ) external whenNotPaused {
        require(legacySubmissionsDisabled, "AuditRegistry: V3 not initialized");
        require(address(zkmlVerifierV3) != address(0), "AuditRegistry: V3 verifier unset");
        require(auditPolicySignerV3 != address(0), "AuditRegistry: V3 signer unset");
        require(context.contractAddress.code.length > 0, "AuditRegistry: target has no code");
        require(block.timestamp <= context.deadline, "AuditRegistry: policy signature expired");
        require(
            sentinelToken.stakedBalance(msg.sender) >= sentinelToken.MIN_STAKE(),
            "AuditRegistry: insufficient stake"
        );
        require(
            publicSignals.length == V2_TOTAL_SIGNALS,
            "AuditRegistry: invalid V3 public signal count"
        );
        for (uint256 i = 0; i < NUM_CLASSES; i++) {
            require(
                publicSignals[INPUT_OFFSET + i] == classScoreFelts[i],
                "AuditRegistry: class score mismatch"
            );
        }

        bytes32 proofHash = keccak256(proof);
        bytes32 publicSignalsHash = keccak256(abi.encode(publicSignals));
        bytes32 classScoreFeltsHash = keccak256(abi.encode(classScoreFelts));
        bytes32 requestDigest = computeAuditDigestV3(
            msg.sender,
            context,
            proofHash,
            publicSignalsHash,
            classScoreFeltsHash
        );

        require(
            !_usedV3RequestDigests[requestDigest],
            "AuditRegistry: V3 request already used"
        );
        require(
            requestDigest.recover(policySignature) == auditPolicySignerV3,
            "AuditRegistry: invalid V3 policy signature"
        );

        // Mark before the external verifier call to prevent same-digest
        // re-entrancy. Any verifier failure/revert rolls this state change back.
        _usedV3RequestDigests[requestDigest] = true;
        require(
            zkmlVerifierV3.verifyProof(proof, publicSignals),
            "AuditRegistry: invalid V3 ZK proof"
        );

        bytes32 contractCodeHash = context.contractAddress.codehash;
        _auditsV3[context.contractAddress].push(AuditResultV3({
            classScoreFelts: classScoreFelts,
            proofHash: proofHash,
            requestDigest: requestDigest,
            publicSignalsHash: publicSignalsHash,
            contractCodeHash: contractCodeHash,
            teacherModelHash: context.teacherModelHash,
            proxyBundleHash: context.proxyBundleHash,
            dataVersionHash: context.dataVersionHash,
            classSchemaHash: context.classSchemaHash,
            roundId: context.roundId,
            timestamp: block.timestamp,
            agent: msg.sender,
            policySigner: auditPolicySignerV3,
            verifier: address(zkmlVerifierV3),
            verified: true
        }));

        emit AuditSubmittedV3(
            context.contractAddress,
            requestDigest,
            proofHash,
            msg.sender,
            classScoreFelts,
            context.teacherModelHash,
            context.proxyBundleHash,
            context.dataVersionHash,
            context.classSchemaHash,
            context.roundId
        );
    }

    function isV3RequestUsed(bytes32 requestDigest) external view returns (bool) {
        return _usedV3RequestDigests[requestDigest];
    }

    // ---------------------------------------------------------------------
    // Queries — all historical versions remain readable
    // ---------------------------------------------------------------------

    function hasAudit(address contractAddress) external view returns (bool) {
        return _audits[contractAddress].length > 0;
    }

    function getLatestAudit(address contractAddress) external view returns (AuditResult memory) {
        AuditResult[] storage audits = _audits[contractAddress];
        require(audits.length > 0, "AuditRegistry: no audits found");
        return audits[audits.length - 1];
    }

    function getAuditHistory(address contractAddress) external view returns (AuditResult[] memory) {
        return _audits[contractAddress];
    }

    function getAuditCount(address contractAddress) external view returns (uint256) {
        return _audits[contractAddress].length;
    }

    function hasAuditV2(address contractAddress) external view returns (bool) {
        return _auditsV2[contractAddress].length > 0;
    }

    function getLatestAuditV2(address contractAddress) external view returns (AuditResultV2 memory) {
        AuditResultV2[] storage audits = _auditsV2[contractAddress];
        require(audits.length > 0, "AuditRegistry: no V2 audits found");
        return audits[audits.length - 1];
    }

    function getAuditHistoryV2(address contractAddress) external view returns (AuditResultV2[] memory) {
        return _auditsV2[contractAddress];
    }

    function getAuditCountV2(address contractAddress) external view returns (uint256) {
        return _auditsV2[contractAddress].length;
    }

    function hasAuditV3(address contractAddress) external view returns (bool) {
        return _auditsV3[contractAddress].length > 0;
    }

    function getLatestAuditV3(address contractAddress) external view returns (AuditResultV3 memory) {
        AuditResultV3[] storage audits = _auditsV3[contractAddress];
        require(audits.length > 0, "AuditRegistry: no V3 audits found");
        return audits[audits.length - 1];
    }

    function getAuditHistoryV3(address contractAddress) external view returns (AuditResultV3[] memory) {
        return _auditsV3[contractAddress];
    }

    function getAuditCountV3(address contractAddress) external view returns (uint256) {
        return _auditsV3[contractAddress].length;
    }

    function _authorizeUpgrade(address newImplementation) internal override onlyOwner {
        emit ImplementationUpgraded(newImplementation);
    }
}
