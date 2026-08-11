// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import "../src/AuditRegistry.sol";
import "../src/SentinelToken.sol";
import "./fixtures/AuditRegistryPreV3.sol";
import "./mocks/MockZKMLVerifier.sol";

contract AuditRegistryV3UpgradeTest is Test {
    AuditRegistryPreV3 internal oldRegistry;
    SentinelToken internal token;
    MockZKMLVerifier internal legacyVerifier;
    MockZKMLVerifier internal v3Verifier;

    address internal agent;
    address internal target;
    address internal policySigner;

    uint256 internal constant MIN_STAKE = 1000 * 10 ** 18;
    bytes internal constant V1_PROOF = hex"01020304";
    bytes internal constant V2_PROOF = hex"deadbeef";

    function setUp() public {
        agent = makeAddr("upgrade-agent");
        target = makeAddr("upgrade-target");
        policySigner = makeAddr("upgrade-policy-signer");

        token = new SentinelToken();
        token.transfer(agent, 10_000 * 10 ** 18);
        legacyVerifier = new MockZKMLVerifier();
        v3Verifier = new MockZKMLVerifier();

        AuditRegistryPreV3 implementation = new AuditRegistryPreV3();
        bytes memory initData = abi.encodeCall(
            AuditRegistryPreV3.initialize,
            (address(legacyVerifier), address(token))
        );
        ERC1967Proxy proxy = new ERC1967Proxy(address(implementation), initData);
        oldRegistry = AuditRegistryPreV3(address(proxy));

        vm.prank(agent);
        token.stake(MIN_STAKE);
    }

    function _v1Signals(uint256 score) internal pure returns (uint256[] memory signals) {
        signals = new uint256[](65);
        signals[64] = score;
    }

    function _v2Scores() internal pure returns (uint256[10] memory scores) {
        for (uint256 i = 0; i < 10; i++) {
            scores[i] = 900 + i * 13;
        }
    }

    function _v2Signals(uint256[10] memory scores)
        internal
        pure
        returns (uint256[] memory signals)
    {
        signals = new uint256[](138);
        for (uint256 i = 0; i < 10; i++) {
            signals[128 + i] = scores[i];
        }
    }

    function test_pre_v3_history_and_core_storage_survive_upgrade() public {
        uint256 v1Score = 4096;
        uint256[10] memory v2Scores = _v2Scores();
        bytes32 v2ModelHash = sha256("pre-v3-model");

        vm.startPrank(agent);
        oldRegistry.submitAudit(target, v1Score, V1_PROOF, _v1Signals(v1Score));
        oldRegistry.submitAuditV2(
            target,
            v2Scores,
            V2_PROOF,
            _v2Signals(v2Scores),
            v2ModelHash
        );
        vm.stopPrank();

        AuditRegistryPreV3.AuditResult memory beforeV1 = oldRegistry.getLatestAudit(target);
        AuditRegistryPreV3.AuditResultV2 memory beforeV2 = oldRegistry.getLatestAuditV2(target);
        address beforeOwner = oldRegistry.owner();
        address beforeLegacyVerifier = address(oldRegistry.zkmlVerifier());
        address beforeToken = address(oldRegistry.sentinelToken());

        AuditRegistry newImplementation = new AuditRegistry();
        bytes memory v3InitData = abi.encodeCall(
            AuditRegistry.initializeV3,
            (address(v3Verifier), policySigner)
        );
        oldRegistry.upgradeToAndCall(address(newImplementation), v3InitData);

        AuditRegistry upgraded = AuditRegistry(address(oldRegistry));

        // Existing inherited/base and registry slots remain intact.
        assertEq(upgraded.owner(), beforeOwner);
        assertEq(address(upgraded.zkmlVerifier()), beforeLegacyVerifier);
        assertEq(address(upgraded.sentinelToken()), beforeToken);
        assertEq(upgraded.getAuditCount(target), 1);
        assertEq(upgraded.getAuditCountV2(target), 1);

        AuditRegistry.AuditResult memory afterV1 = upgraded.getLatestAudit(target);
        AuditRegistry.AuditResultV2 memory afterV2 = upgraded.getLatestAuditV2(target);
        assertEq(afterV1.scoreFieldElement, beforeV1.scoreFieldElement);
        assertEq(afterV1.proofHash, beforeV1.proofHash);
        assertEq(afterV1.timestamp, beforeV1.timestamp);
        assertEq(afterV1.agent, beforeV1.agent);
        assertEq(afterV1.verified, beforeV1.verified);

        assertEq(afterV2.proofHash, beforeV2.proofHash);
        assertEq(afterV2.modelHash, beforeV2.modelHash);
        assertEq(afterV2.timestamp, beforeV2.timestamp);
        assertEq(afterV2.agent, beforeV2.agent);
        assertEq(afterV2.verified, beforeV2.verified);
        for (uint256 i = 0; i < 10; i++) {
            assertEq(afterV2.classScores[i], beforeV2.classScores[i]);
        }

        // Appended V3 state is initialized without corrupting old state.
        assertEq(address(upgraded.zkmlVerifierV3()), address(v3Verifier));
        assertEq(upgraded.auditPolicySignerV3(), policySigner);
        assertTrue(upgraded.legacySubmissionsDisabled());
        assertEq(upgraded.getAuditCountV3(target), 0);
        assertEq(token.stakedBalance(agent), MIN_STAKE);
    }

    function test_initialize_v3_cannot_be_replayed_after_upgrade() public {
        AuditRegistry newImplementation = new AuditRegistry();
        oldRegistry.upgradeToAndCall(
            address(newImplementation),
            abi.encodeCall(AuditRegistry.initializeV3, (address(v3Verifier), policySigner))
        );
        AuditRegistry upgraded = AuditRegistry(address(oldRegistry));

        vm.expectRevert();
        upgraded.initializeV3(address(v3Verifier), policySigner);
    }
}
