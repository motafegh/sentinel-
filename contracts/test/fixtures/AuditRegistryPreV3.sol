// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/PausableUpgradeable.sol";
import "../../src/IZKMLVerifier.sol";
import "../../src/SentinelToken.sol";

/// @notice Test-only frozen copy of the canonical pre-V3 registry storage seam.
/// @dev The state declaration order intentionally matches `main` before the
/// system-alignment V3 extension. Do not "clean up" or reorder this fixture: it
/// exists to detect UUPS storage/history regressions during upgrade tests.
contract AuditRegistryPreV3 is Initializable, UUPSUpgradeable, OwnableUpgradeable, PausableUpgradeable {
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

    struct AuditResultV2 {
        uint256[10] classScores;
        bytes32 proofHash;
        bytes32 modelHash;
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

    constructor() {
        _disableInitializers();
    }

    function initialize(address verifierAddress, address tokenAddress) public initializer {
        __Ownable_init(msg.sender);
        __Pausable_init();
        zkmlVerifier = IZKMLVerifier(verifierAddress);
        sentinelToken = SentinelToken(tokenAddress);
    }

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

        _audits[contractAddress].push(
            AuditResult({
                scoreFieldElement: scoreFieldElement,
                proofHash: keccak256(proof),
                timestamp: block.timestamp,
                agent: msg.sender,
                verified: true
            })
        );
        emit AuditSubmitted(contractAddress, keccak256(proof), msg.sender, scoreFieldElement);
    }

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

        _auditsV2[contractAddress].push(
            AuditResultV2({
                classScores: classScores,
                proofHash: keccak256(proof),
                modelHash: modelHash,
                timestamp: block.timestamp,
                agent: msg.sender,
                verified: true
            })
        );
        emit AuditSubmittedV2(contractAddress, keccak256(proof), msg.sender, classScores, modelHash);
    }

    function getLatestAudit(address contractAddress) external view returns (AuditResult memory) {
        AuditResult[] storage audits = _audits[contractAddress];
        require(audits.length > 0, "AuditRegistry: no audits found");
        return audits[audits.length - 1];
    }

    function getAuditCount(address contractAddress) external view returns (uint256) {
        return _audits[contractAddress].length;
    }

    function getLatestAuditV2(address contractAddress) external view returns (AuditResultV2 memory) {
        AuditResultV2[] storage audits = _auditsV2[contractAddress];
        require(audits.length > 0, "AuditRegistry: no V2 audits found");
        return audits[audits.length - 1];
    }

    function getAuditCountV2(address contractAddress) external view returns (uint256) {
        return _auditsV2[contractAddress].length;
    }

    function _authorizeUpgrade(address newImplementation) internal override onlyOwner {
        emit ImplementationUpgraded(newImplementation);
    }
}
