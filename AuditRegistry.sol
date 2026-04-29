// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/PausableUpgradeable.sol";
import "./IZKMLVerifier.sol";
import "./SentinelToken.sol";

// RECALL — Three upgradeable base contracts required for UUPS pattern.
// Initializable: enables initialize() instead of constructor.
// UUPSUpgradeable: upgrade logic lives in implementation, not proxy.
// OwnableUpgradeable: upgradeable-compatible version of Ownable.
// PausableUpgradeable: emergency stop — owner can halt audit submissions.
contract AuditRegistry is Initializable, UUPSUpgradeable, OwnableUpgradeable, PausableUpgradeable {

    // RECALL — Verifier referenced via interface not concrete type (ADR-002).
    // Swappable when EZKL regenerates ZKMLVerifier for a new model version.
    IZKMLVerifier public zkmlVerifier;
    SentinelToken public sentinelToken;

    // RECALL — Full audit record stored per submission.
    // proofHash = keccak256(proof) not raw bytes — 32 bytes vs 29KB.
    // Storing full proof on-chain would cost thousands in gas.
    // verified is always true if stored — the three guards enforce it.
    // scoreFieldElement is uint256 — EZKL encodes model output as a scaled
    // BN254 field element (model_output * 2^13). Storing the raw field element
    // preserves on-chain verifiability. Off-chain: divide by 8192 to get
    // the human-readable risk probability (e.g. 4497 / 8192 = 0.5490).
    struct AuditResult {
        uint256 scoreFieldElement;
        bytes32 proofHash;
        uint256 timestamp;
        address agent;
        bool verified;
    }

    // RECALL — Array per contract address gives full audit history.
    // Latest audit = audits[contractAddress][length - 1].
    mapping(address => AuditResult[]) private _audits;

    // RECALL — scoreFieldElement is the raw EZKL field element (model_output * 2^13).
    // Off-chain: divide by 8192 to get human-readable probability.
    // Example: score = 4497 → 4497 / 8192 = 0.5490 vulnerability probability.
    event AuditSubmitted(
        address indexed contractAddress,
        bytes32 proofHash,
        address indexed agent,
        uint256 scoreFieldElement
    );

    // RECALL — Emitted when the implementation is upgraded via UUPS.
    // Lets off-chain listeners detect upgrades without monitoring proxy storage.
    event ImplementationUpgraded(address indexed newImplementation);

    // RECALL — _disableInitializers() prevents direct initialisation of the
    // implementation contract. Without it, anyone could call initialize()
    // on the implementation directly and become its owner — a critical security
    // flaw for production UUPS deployments.
    // PRODUCTION: this constructor is correct and must NOT be removed.
    // TESTING: unit tests deploy implementation directly and call initialize(),
    // which works because Foundry tests bypass the initializer guard.
    constructor() {
        _disableInitializers();
    }

    // RECALL — Replaces constructor for upgradeable contracts.
    // initializer modifier ensures this runs exactly once.
    // Sets verifier + token references used by submitAudit().
    function initialize(
        address verifierAddress,
        address tokenAddress
    ) public initializer {
        __Ownable_init(msg.sender);
        __Pausable_init();
        zkmlVerifier = IZKMLVerifier(verifierAddress);
        sentinelToken = SentinelToken(tokenAddress);
    }

    // ─── Emergency controls ───────────────────────────────────────────────────

    // RECALL — pause() stops all audit submissions immediately.
    // Use if a bug in the ZK verifier is discovered or keys are compromised.
    // Does NOT affect reads — getLatestAudit() and getAuditHistory() still work.
    // Production: replace onlyOwner with multi-sig or DAO governance.
    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
    }

    // ─── Core logic ──────────────────────────────────────────────────────────

    // RECALL — Three guards enforce SENTINEL's full trust model.
    // All three must pass or the entire tx reverts — nothing stored.
    // Guard 1: economic (staking)
    // Guard 2: cryptographic (ZK proof validity)
    // Guard 3: consistency (scoreFieldElement in calldata matches field element in proof)
    //
    // ENCODING NOTE — scoreFieldElement vs human score:
    //   EZKL encodes the model's output probability as a BN254 field element:
    //     field_element = round(model_output * 2^13)
    //   Example: model outputs 0.5490 → field element = round(0.5490 * 8192) = 4497
    //   The caller MUST pass this raw uint256 value — not 0.5490, not 54, not 4497/8192.
    //   Off-chain: scoreFieldElement / 8192 gives the human-readable probability.
    //
    //   IMPORTANT — little-endian encoding in proof.json:
    //   proof.json stores field elements as 32-byte little-endian hex strings.
    //   To extract the correct uint256:
    //     Python: int.from_bytes(bytes.fromhex(instances[64]), byteorder='little')
    //   NOT:  int(instances[64], 16)  ← treats as big-endian, produces wrong value
    function submitAudit(
        address contractAddress,
        uint256 scoreFieldElement,
        bytes calldata proof,
        uint256[] calldata publicSignals
    ) external whenNotPaused {

        // Guard 1 — agent must have skin in the game
        require(
            sentinelToken.stakedBalance(msg.sender) >= sentinelToken.MIN_STAKE(),
            "AuditRegistry: insufficient stake"
        );

        // Guard 2 — ZK proof must be cryptographically valid
        // RECALL — Delegates to ZKMLVerifier.verifyProof() via interface.
        // If proof was generated by a different model or tampered with,
        // this returns false and the tx reverts. Costs ~250K gas.
        require(
            zkmlVerifier.verifyProof(proof, publicSignals),
            "AuditRegistry: invalid ZK proof"
        );

        // Guard 3 — scoreFieldElement in calldata must match proof output
        // RECALL — EZKL encodes model output as a field element.
        // publicSignals[64] = model_output * 2^13 (scale=13 from settings.json)
        // Caller must pass this exact uint256 value.
        // The full uint256 is compared — no truncation to uint8 or any other type.
        require(
            publicSignals[64] == scoreFieldElement,
            "AuditRegistry: score mismatch with proof"
        );

        // All guards passed — store the result
        _audits[contractAddress].push(AuditResult({
            scoreFieldElement: scoreFieldElement,
            proofHash:         keccak256(proof),
            timestamp:         block.timestamp,
            agent:             msg.sender,
            verified:          true
        }));

        emit AuditSubmitted(
            contractAddress,
            keccak256(proof),
            msg.sender,
            scoreFieldElement
        );
    }

    // ─── Queries ──────────────────────────────────────────────────────────────

    // RECALL — Check before calling getLatestAudit() to avoid a revert.
    // Use pattern: if (registry.hasAudit(addr)) { registry.getLatestAudit(addr); }
    function hasAudit(address contractAddress) external view returns (bool) {
        return _audits[contractAddress].length > 0;
    }

    // RECALL — Returns most recent audit for a contract.
    // Reverts if no audits exist — use hasAudit() first to guard against this.
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

    // ─── Upgrade ──────────────────────────────────────────────────────────────

    // RECALL — Required by UUPS. Controls who can upgrade the contract.
    // onlyOwner modifier does the access control work.
    // Production: replace onlyOwner with DAO governance or multi-sig.
    // ImplementationUpgraded event lets off-chain systems detect upgrades
    // without relying solely on ERC1967Proxy's built-in Upgraded event.
    function _authorizeUpgrade(
        address newImplementation
    ) internal override onlyOwner {
        emit ImplementationUpgraded(newImplementation);
    }
}
