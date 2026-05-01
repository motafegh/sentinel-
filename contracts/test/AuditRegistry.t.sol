// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import "../src/AuditRegistry.sol";
import "../src/SentinelToken.sol";
import "./mocks/MockZKMLVerifier.sol";

contract AuditRegistryTest is Test {
    AuditRegistry registry;
    SentinelToken token;
    MockZKMLVerifier verifier;

    address owner;
    address agent;
    address other;
    address targetContract;

    uint256 constant MIN_STAKE = 1000 * 10 ** 18;

    // Arbitrary proof bytes and publicSignals used in happy-path tests
    bytes   constant PROOF        = hex"deadbeef";
    uint256 constant SCORE        = 4096; // ~0.5 in 2^13 scaling
    // publicSignals needs 65 elements; index 64 == SCORE
    uint256[] public  signals;

    function setUp() public {
        owner          = address(this);
        agent          = makeAddr("agent");
        other          = makeAddr("other");
        targetContract = makeAddr("targetContract");

        // Deploy token and fund agent
        token = new SentinelToken();
        token.transfer(agent, 10_000 * 10 ** 18);

        // Deploy mock verifier (defaults to returning true)
        verifier = new MockZKMLVerifier();

        // Deploy AuditRegistry via UUPS proxy
        AuditRegistry impl = new AuditRegistry();
        bytes memory initData = abi.encodeCall(
            AuditRegistry.initialize,
            (address(verifier), address(token))
        );
        ERC1967Proxy proxy = new ERC1967Proxy(address(impl), initData);
        registry = AuditRegistry(address(proxy));

        // Build signals array: 65 slots, index 64 == SCORE
        for (uint256 i = 0; i < 65; i++) {
            signals.push(i == 64 ? SCORE : 0);
        }

        // Agent stakes the minimum required
        vm.startPrank(agent);
        token.approve(address(token), MIN_STAKE);
        token.stake(MIN_STAKE);
        vm.stopPrank();
    }

    // --- happy path -------------------------------------------------------

    function test_submit_audit_happy_path() public {
        vm.prank(agent);
        registry.submitAudit(targetContract, SCORE, PROOF, signals);

        assertTrue(registry.hasAudit(targetContract), "audit should be recorded");
        assertEq(registry.getAuditCount(targetContract), 1);

        AuditRegistry.AuditResult memory result = registry.getLatestAudit(targetContract);
        assertEq(result.scoreFieldElement, SCORE);
        assertEq(result.proofHash, keccak256(PROOF));
        assertEq(result.agent, agent);
        assertTrue(result.verified);
    }

    function test_submit_audit_emits_event() public {
        vm.expectEmit(true, false, true, true);
        emit AuditRegistry.AuditSubmitted(targetContract, keccak256(PROOF), agent, SCORE);
        vm.prank(agent);
        registry.submitAudit(targetContract, SCORE, PROOF, signals);
    }

    function test_get_audit_history_multiple_submissions() public {
        vm.startPrank(agent);
        registry.submitAudit(targetContract, SCORE, PROOF, signals);
        registry.submitAudit(targetContract, SCORE + 1, PROOF, _buildSignals(SCORE + 1));
        vm.stopPrank();

        AuditRegistry.AuditResult[] memory history = registry.getAuditHistory(targetContract);
        assertEq(history.length, 2);
        assertEq(history[0].scoreFieldElement, SCORE);
        assertEq(history[1].scoreFieldElement, SCORE + 1);
    }

    // --- guard 1: insufficient stake --------------------------------------

    function test_guard1_insufficient_stake() public {
        // `other` has never staked
        vm.prank(other);
        vm.expectRevert("AuditRegistry: insufficient stake");
        registry.submitAudit(targetContract, SCORE, PROOF, signals);
    }

    function test_guard1_partial_stake_reverts() public {
        address lowAgent = makeAddr("lowAgent");
        token.transfer(lowAgent, MIN_STAKE - 1);

        vm.startPrank(lowAgent);
        token.approve(address(token), MIN_STAKE - 1);
        token.stake(MIN_STAKE - 1);
        vm.expectRevert("AuditRegistry: insufficient stake");
        registry.submitAudit(targetContract, SCORE, PROOF, signals);
        vm.stopPrank();
    }

    // --- guard 2: invalid proof -------------------------------------------

    function test_guard2_invalid_proof() public {
        verifier.setReturnValue(false);

        vm.prank(agent);
        vm.expectRevert("AuditRegistry: invalid ZK proof");
        registry.submitAudit(targetContract, SCORE, PROOF, signals);
    }

    // --- guard 3: score mismatch ------------------------------------------

    function test_guard3_score_mismatch() public {
        uint256 wrongScore = SCORE + 999;

        vm.prank(agent);
        vm.expectRevert("AuditRegistry: score mismatch with proof");
        // Pass signals where index 64 == SCORE, but tell registry score == wrongScore
        registry.submitAudit(targetContract, wrongScore, PROOF, signals);
    }

    // --- queries ----------------------------------------------------------

    function test_has_audit_returns_false_for_unknown() public view {
        assertFalse(registry.hasAudit(makeAddr("unknown")));
    }

    function test_get_latest_audit_reverts_when_none() public {
        vm.expectRevert("AuditRegistry: no audits found");
        registry.getLatestAudit(makeAddr("noAudits"));
    }

    function test_audit_count_increments() public {
        assertEq(registry.getAuditCount(targetContract), 0);

        vm.prank(agent);
        registry.submitAudit(targetContract, SCORE, PROOF, signals);
        assertEq(registry.getAuditCount(targetContract), 1);

        vm.prank(agent);
        registry.submitAudit(targetContract, SCORE, PROOF, signals);
        assertEq(registry.getAuditCount(targetContract), 2);
    }

    // --- pause / unpause --------------------------------------------------

    function test_pause_halts_submission() public {
        registry.pause();

        vm.prank(agent);
        vm.expectRevert();
        registry.submitAudit(targetContract, SCORE, PROOF, signals);
    }

    function test_unpause_restores_submission() public {
        registry.pause();
        registry.unpause();

        vm.prank(agent);
        registry.submitAudit(targetContract, SCORE, PROOF, signals);
        assertTrue(registry.hasAudit(targetContract));
    }

    function test_pause_by_non_owner_reverts() public {
        vm.prank(other);
        vm.expectRevert();
        registry.pause();
    }

    // --- UUPS upgrade -----------------------------------------------------

    function test_upgrade_via_uups() public {
        // Deploy a new implementation (same contract — just tests the upgrade path)
        AuditRegistry newImpl = new AuditRegistry();

        registry.upgradeToAndCall(address(newImpl), "");

        // State survives upgrade
        vm.prank(agent);
        registry.submitAudit(targetContract, SCORE, PROOF, signals);
        assertEq(registry.getAuditCount(targetContract), 1);
    }

    function test_upgrade_by_non_owner_reverts() public {
        AuditRegistry newImpl = new AuditRegistry();

        vm.prank(other);
        vm.expectRevert();
        registry.upgradeToAndCall(address(newImpl), "");
    }

    // --- helpers ----------------------------------------------------------

    function _buildSignals(uint256 score) internal pure returns (uint256[] memory s) {
        s = new uint256[](65);
        s[64] = score;
    }
}
