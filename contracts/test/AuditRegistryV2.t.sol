// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import "../src/AuditRegistry.sol";
import "../src/SentinelToken.sol";
import "./mocks/MockZKMLVerifier.sol";

/// @notice Tests the currently tracked 128-input/10-output legacy V2 seam.
/// @dev These tests intentionally document that V2 does NOT bind model/context
/// identity. That limitation is a testable property to eliminate in V3, not a
/// security claim hidden in prose.
contract AuditRegistryV2Test is Test {
    AuditRegistry internal registry;
    SentinelToken internal token;
    MockZKMLVerifier internal verifier;

    address internal agent;
    address internal other;
    address internal target;

    uint256 internal constant MIN_STAKE = 1000 * 10 ** 18;
    uint256 internal constant INPUT_OFFSET = 128;
    uint256 internal constant NUM_CLASSES = 10;
    uint256 internal constant TOTAL_SIGNALS = 138;
    bytes internal constant PROOF = hex"deadbeef";

    function setUp() public {
        agent = makeAddr("v2-agent");
        other = makeAddr("v2-other");
        target = makeAddr("v2-target");

        token = new SentinelToken();
        token.transfer(agent, 10_000 * 10 ** 18);
        verifier = new MockZKMLVerifier();

        AuditRegistry impl = new AuditRegistry();
        bytes memory initData = abi.encodeCall(
            AuditRegistry.initialize,
            (address(verifier), address(token))
        );
        ERC1967Proxy proxy = new ERC1967Proxy(address(impl), initData);
        registry = AuditRegistry(address(proxy));

        vm.startPrank(agent);
        token.approve(address(token), MIN_STAKE);
        token.stake(MIN_STAKE);
        vm.stopPrank();
    }

    function _scores() internal pure returns (uint256[10] memory scores) {
        for (uint256 i = 0; i < NUM_CLASSES; i++) {
            scores[i] = 1000 + i * 37;
        }
    }

    function _signals(uint256[10] memory scores)
        internal
        pure
        returns (uint256[] memory signals)
    {
        signals = new uint256[](TOTAL_SIGNALS);
        for (uint256 i = 0; i < NUM_CLASSES; i++) {
            signals[INPUT_OFFSET + i] = scores[i];
        }
    }

    function test_v2_happy_path_records_all_scores_and_identities() public {
        uint256[10] memory scores = _scores();
        uint256[] memory signals = _signals(scores);
        bytes32 modelHash = sha256("teacher-checkpoint");

        vm.prank(agent);
        registry.submitAuditV2(target, scores, PROOF, signals, modelHash);

        assertTrue(registry.hasAuditV2(target));
        assertEq(registry.getAuditCountV2(target), 1);
        AuditRegistry.AuditResultV2 memory result = registry.getLatestAuditV2(target);
        assertEq(result.proofHash, keccak256(PROOF));
        assertEq(result.modelHash, modelHash);
        assertEq(result.agent, agent);
        assertTrue(result.verified);
        for (uint256 i = 0; i < NUM_CLASSES; i++) {
            assertEq(result.classScores[i], scores[i]);
        }
    }

    function test_v2_requires_minimum_public_signal_layout() public {
        uint256[10] memory scores = _scores();
        uint256[] memory shortSignals = new uint256[](TOTAL_SIGNALS - 1);

        vm.prank(agent);
        vm.expectRevert("AuditRegistry: insufficient public signals");
        registry.submitAuditV2(target, scores, PROOF, shortSignals, bytes32(uint256(1)));
    }

    function test_v2_rejects_any_class_score_mismatch() public {
        uint256[10] memory scores = _scores();
        uint256[] memory signals = _signals(scores);
        signals[INPUT_OFFSET + 7] += 1;

        vm.prank(agent);
        vm.expectRevert("AuditRegistry: class score mismatch");
        registry.submitAuditV2(target, scores, PROOF, signals, bytes32(uint256(1)));
    }

    function test_v2_rejects_invalid_proof() public {
        verifier.setReturnValue(false);
        uint256[10] memory scores = _scores();

        vm.prank(agent);
        vm.expectRevert("AuditRegistry: invalid ZK proof");
        registry.submitAuditV2(
            target,
            scores,
            PROOF,
            _signals(scores),
            bytes32(uint256(1))
        );
    }

    function test_v2_requires_stake() public {
        uint256[10] memory scores = _scores();
        vm.prank(other);
        vm.expectRevert("AuditRegistry: insufficient stake");
        registry.submitAuditV2(
            target,
            scores,
            PROOF,
            _signals(scores),
            bytes32(uint256(1))
        );
    }

    function test_v2_model_hash_is_caller_metadata_not_proof_bound() public {
        uint256[10] memory scores = _scores();
        uint256[] memory signals = _signals(scores);
        bytes32 claimedModelA = keccak256("model-A");
        bytes32 claimedModelB = keccak256("model-B");

        vm.startPrank(agent);
        registry.submitAuditV2(target, scores, PROOF, signals, claimedModelA);
        registry.submitAuditV2(target, scores, PROOF, signals, claimedModelB);
        vm.stopPrank();

        AuditRegistry.AuditResultV2[] memory history = registry.getAuditHistoryV2(target);
        assertEq(history.length, 2);
        assertEq(history[0].modelHash, claimedModelA);
        assertEq(history[1].modelHash, claimedModelB);
        // Same proof/signals verify under the mock for both model identities.
        // V3 must make this impossible by binding context into public signals.
        assertEq(history[0].proofHash, history[1].proofHash);
    }

    function test_v2_proof_is_not_bound_to_target_contract() public {
        uint256[10] memory scores = _scores();
        uint256[] memory signals = _signals(scores);
        bytes32 modelHash = keccak256("same-model");
        address targetB = makeAddr("v2-target-b");

        vm.startPrank(agent);
        registry.submitAuditV2(target, scores, PROOF, signals, modelHash);
        registry.submitAuditV2(targetB, scores, PROOF, signals, modelHash);
        vm.stopPrank();

        assertEq(registry.getLatestAuditV2(target).proofHash, keccak256(PROOF));
        assertEq(registry.getLatestAuditV2(targetB).proofHash, keccak256(PROOF));
        // This is the contained V2 replay/context gap that V3 must close.
    }
}
