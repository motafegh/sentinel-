// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import "../src/AuditRegistry.sol";
import "../src/SentinelToken.sol";
import "./mocks/MockZKMLVerifier.sol";

contract V3AuditTarget {
    uint256 public value = 1;
}

contract V3AuditTargetB {
    uint256 public value = 2;
}

contract AuditRegistryV3Test is Test {
    AuditRegistry internal registry;
    SentinelToken internal token;
    MockZKMLVerifier internal legacyVerifier;
    MockZKMLVerifier internal v3Verifier;
    V3AuditTarget internal target;
    V3AuditTargetB internal targetB;

    uint256 internal constant POLICY_SIGNER_KEY = 0xA11CE;
    address internal policySigner;
    address internal agent;
    address internal otherAgent;

    uint256 internal constant MIN_STAKE = 1000 * 10 ** 18;
    uint256 internal constant INPUT_OFFSET = 128;
    uint256 internal constant NUM_CLASSES = 10;
    uint256 internal constant TOTAL_SIGNALS = 138;
    bytes internal constant PROOF = hex"deadbeef";

    function setUp() public {
        policySigner = vm.addr(POLICY_SIGNER_KEY);
        agent = makeAddr("v3-agent");
        otherAgent = makeAddr("v3-other-agent");
        target = new V3AuditTarget();
        targetB = new V3AuditTargetB();

        token = new SentinelToken();
        token.transfer(agent, 10_000 * 10 ** 18);
        token.transfer(otherAgent, 10_000 * 10 ** 18);
        legacyVerifier = new MockZKMLVerifier();
        v3Verifier = new MockZKMLVerifier();

        AuditRegistry impl = new AuditRegistry();
        bytes memory initData = abi.encodeCall(
            AuditRegistry.initialize,
            (address(legacyVerifier), address(token))
        );
        ERC1967Proxy proxy = new ERC1967Proxy(address(impl), initData);
        registry = AuditRegistry(address(proxy));
        registry.initializeV3(address(v3Verifier), policySigner);

        _stake(agent);
        _stake(otherAgent);
    }

    function _stake(address who) internal {
        vm.prank(who);
        token.stake(MIN_STAKE);
    }

    function _scores() internal pure returns (uint256[10] memory scores) {
        for (uint256 i = 0; i < NUM_CLASSES; i++) {
            scores[i] = 1200 + i * 41;
        }
    }

    function _signals(uint256[10] memory scores)
        internal
        pure
        returns (uint256[] memory signals)
    {
        signals = new uint256[](TOTAL_SIGNALS);
        for (uint256 i = 0; i < INPUT_OFFSET; i++) {
            signals[i] = i + 17;
        }
        for (uint256 i = 0; i < NUM_CLASSES; i++) {
            signals[INPUT_OFFSET + i] = scores[i];
        }
    }

    function _context(address targetAddress)
        internal
        view
        returns (AuditRegistry.AuditContextV3 memory context)
    {
        context = AuditRegistry.AuditContextV3({
            contractAddress: targetAddress,
            roundId: 77,
            teacherModelHash: sha256("run12-teacher"),
            proxyBundleHash: sha256("zkml-v2-bundle"),
            dataVersionHash: sha256("sentinel-r4-data"),
            classSchemaHash: sha256("sentinel-ten-class-v1"),
            deadline: block.timestamp + 1 days
        });
    }

    function _digest(
        AuditRegistry targetRegistry,
        address submittingAgent,
        AuditRegistry.AuditContextV3 memory context,
        uint256[10] memory scores,
        bytes memory proof,
        uint256[] memory signals
    ) internal view returns (bytes32) {
        return targetRegistry.computeAuditDigestV3(
            submittingAgent,
            context,
            keccak256(proof),
            keccak256(abi.encode(signals)),
            keccak256(abi.encode(scores))
        );
    }

    function _sign(bytes32 digest) internal returns (bytes memory) {
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(POLICY_SIGNER_KEY, digest);
        return abi.encodePacked(r, s, v);
    }

    function _validSubmission()
        internal
        returns (
            AuditRegistry.AuditContextV3 memory context,
            uint256[10] memory scores,
            uint256[] memory signals,
            bytes memory signature,
            bytes32 digest
        )
    {
        context = _context(address(target));
        scores = _scores();
        signals = _signals(scores);
        digest = _digest(registry, agent, context, scores, PROOF, signals);
        signature = _sign(digest);
    }

    function test_v3_happy_path_binds_context_and_proof() public {
        (
            AuditRegistry.AuditContextV3 memory context,
            uint256[10] memory scores,
            uint256[] memory signals,
            bytes memory signature,
            bytes32 digest
        ) = _validSubmission();

        vm.prank(agent);
        registry.submitAuditV3(context, scores, PROOF, signals, signature);

        assertTrue(registry.hasAuditV3(address(target)));
        assertTrue(registry.isV3RequestUsed(digest));
        AuditRegistry.AuditResultV3 memory result = registry.getLatestAuditV3(address(target));
        assertEq(result.requestDigest, digest);
        assertEq(result.proofHash, keccak256(PROOF));
        assertEq(result.publicSignalsHash, keccak256(abi.encode(signals)));
        assertEq(result.contractCodeHash, address(target).codehash);
        assertEq(result.teacherModelHash, context.teacherModelHash);
        assertEq(result.proxyBundleHash, context.proxyBundleHash);
        assertEq(result.dataVersionHash, context.dataVersionHash);
        assertEq(result.classSchemaHash, context.classSchemaHash);
        assertEq(result.roundId, context.roundId);
        assertEq(result.agent, agent);
        assertEq(result.policySigner, policySigner);
        assertEq(result.verifier, address(v3Verifier));
        assertTrue(result.verified);
        for (uint256 i = 0; i < NUM_CLASSES; i++) {
            assertEq(result.classScoreFelts[i], scores[i]);
        }
    }

    function test_v3_rejects_exact_replay() public {
        (
            AuditRegistry.AuditContextV3 memory context,
            uint256[10] memory scores,
            uint256[] memory signals,
            bytes memory signature,
            bytes32 digest
        ) = _validSubmission();

        vm.prank(agent);
        registry.submitAuditV3(context, scores, PROOF, signals, signature);
        assertTrue(registry.isV3RequestUsed(digest));

        vm.prank(agent);
        vm.expectRevert("AuditRegistry: V3 request already used");
        registry.submitAuditV3(context, scores, PROOF, signals, signature);
    }

    function test_v3_rejects_target_substitution() public {
        (
            AuditRegistry.AuditContextV3 memory context,
            uint256[10] memory scores,
            uint256[] memory signals,
            bytes memory signature,
        ) = _validSubmission();
        context.contractAddress = address(targetB);

        vm.prank(agent);
        vm.expectRevert("AuditRegistry: invalid V3 policy signature");
        registry.submitAuditV3(context, scores, PROOF, signals, signature);
    }

    function test_v3_rejects_teacher_model_substitution() public {
        (
            AuditRegistry.AuditContextV3 memory context,
            uint256[10] memory scores,
            uint256[] memory signals,
            bytes memory signature,
        ) = _validSubmission();
        context.teacherModelHash = sha256("different-teacher");

        vm.prank(agent);
        vm.expectRevert("AuditRegistry: invalid V3 policy signature");
        registry.submitAuditV3(context, scores, PROOF, signals, signature);
    }

    function test_v3_rejects_fusion_signal_substitution() public {
        (
            AuditRegistry.AuditContextV3 memory context,
            uint256[10] memory scores,
            uint256[] memory signals,
            bytes memory signature,
        ) = _validSubmission();
        signals[0] += 1; // outputs remain unchanged; signature must still fail.

        vm.prank(agent);
        vm.expectRevert("AuditRegistry: invalid V3 policy signature");
        registry.submitAuditV3(context, scores, PROOF, signals, signature);
    }

    function test_v3_rejects_proof_substitution() public {
        (
            AuditRegistry.AuditContextV3 memory context,
            uint256[10] memory scores,
            uint256[] memory signals,
            bytes memory signature,
        ) = _validSubmission();
        bytes memory differentProof = hex"cafebabe";

        vm.prank(agent);
        vm.expectRevert("AuditRegistry: invalid V3 policy signature");
        registry.submitAuditV3(context, scores, differentProof, signals, signature);
    }

    function test_v3_rejects_agent_substitution() public {
        (
            AuditRegistry.AuditContextV3 memory context,
            uint256[10] memory scores,
            uint256[] memory signals,
            bytes memory signature,
        ) = _validSubmission();

        vm.prank(otherAgent);
        vm.expectRevert("AuditRegistry: invalid V3 policy signature");
        registry.submitAuditV3(context, scores, PROOF, signals, signature);
    }

    function test_v3_rejects_expired_policy_signature() public {
        AuditRegistry.AuditContextV3 memory context = _context(address(target));
        context.deadline = block.timestamp + 10;
        uint256[10] memory scores = _scores();
        uint256[] memory signals = _signals(scores);
        bytes memory signature = _sign(_digest(registry, agent, context, scores, PROOF, signals));

        vm.warp(context.deadline + 1);
        vm.prank(agent);
        vm.expectRevert("AuditRegistry: policy signature expired");
        registry.submitAuditV3(context, scores, PROOF, signals, signature);
    }

    function test_v3_invalid_proof_rolls_back_replay_marker() public {
        (
            AuditRegistry.AuditContextV3 memory context,
            uint256[10] memory scores,
            uint256[] memory signals,
            bytes memory signature,
            bytes32 digest
        ) = _validSubmission();
        v3Verifier.setReturnValue(false);

        vm.prank(agent);
        vm.expectRevert("AuditRegistry: invalid V3 ZK proof");
        registry.submitAuditV3(context, scores, PROOF, signals, signature);
        assertFalse(registry.isV3RequestUsed(digest));
    }

    function test_v3_rejects_cross_registry_replay() public {
        AuditRegistry impl2 = new AuditRegistry();
        bytes memory initData = abi.encodeCall(
            AuditRegistry.initialize,
            (address(legacyVerifier), address(token))
        );
        ERC1967Proxy proxy2 = new ERC1967Proxy(address(impl2), initData);
        AuditRegistry registry2 = AuditRegistry(address(proxy2));
        registry2.initializeV3(address(v3Verifier), policySigner);

        AuditRegistry.AuditContextV3 memory context = _context(address(target));
        uint256[10] memory scores = _scores();
        uint256[] memory signals = _signals(scores);
        bytes memory signature = _sign(_digest(registry, agent, context, scores, PROOF, signals));

        assertTrue(registry.domainSeparatorV3() != registry2.domainSeparatorV3());
        vm.prank(agent);
        vm.expectRevert("AuditRegistry: invalid V3 policy signature");
        registry2.submitAuditV3(context, scores, PROOF, signals, signature);
    }

    function test_v3_activation_disables_legacy_writes() public {
        uint256[10] memory scores = _scores();
        uint256[] memory signals = _signals(scores);

        vm.prank(agent);
        vm.expectRevert("AuditRegistry: legacy submissions disabled");
        registry.submitAuditV2(address(target), scores, PROOF, signals, bytes32(uint256(1)));

        uint256[] memory oldSignals = new uint256[](65);
        vm.prank(agent);
        vm.expectRevert("AuditRegistry: legacy submissions disabled");
        registry.submitAudit(address(target), 0, PROOF, oldSignals);
    }

    function test_v3_rejects_eoa_target() public {
        AuditRegistry.AuditContextV3 memory context = _context(makeAddr("not-contract"));
        uint256[10] memory scores = _scores();
        uint256[] memory signals = _signals(scores);
        bytes memory signature = _sign(_digest(registry, agent, context, scores, PROOF, signals));

        vm.prank(agent);
        vm.expectRevert("AuditRegistry: target has no code");
        registry.submitAuditV3(context, scores, PROOF, signals, signature);
    }
}
