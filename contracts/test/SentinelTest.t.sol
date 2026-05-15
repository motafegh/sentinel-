// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import "../src/SentinelToken.sol";
import "../src/IZKMLVerifier.sol";
import "../src/AuditRegistry.sol";

// RECALL — Mock verifier lets us test AuditRegistry without a real ZK proof.
// In unit tests we control what verifyProof() returns so we can test each guard
// independently. Real ZK verification is tested in integration/on-chain tests.
//
// IMPORTANT — method name must be verifyProof(), not verify().
// IZKMLVerifier.verifyProof() matches the EZKL Halo2Verifier ABI exactly.
// Using verify() would compile (Solidity doesn't enforce interface method names
// on concrete types) but AuditRegistry.submitAudit() calls verifyProof() and
// would receive a fallback revert instead of the mock's bool return.
contract MockZKMLVerifier is IZKMLVerifier {
    bool private _shouldPass;

    constructor(bool shouldPass) {
        _shouldPass = shouldPass;
    }

    // RECALL — nonpayable not view, matching IZKMLVerifier exactly.
    // EZKL's generated verifier mutates memory internally even though it
    // does not change storage — that's why it's nonpayable, not view.
    function verifyProof(
        bytes calldata,
        uint256[] calldata
    ) external returns (bool) {
        return _shouldPass;
    }
}

contract SentinelTest is Test {

    SentinelToken token;
    AuditRegistry registry;
    MockZKMLVerifier verifierPass;
    MockZKMLVerifier verifierFail;

    address owner = address(this);
    address agent = address(0xBEEF);
    address target = address(0xDEAD); // contract being audited

    // RECALL — setUp() runs before every single test function.
    // Fresh deployments each time — tests never share state.
    //
    // RECALL — UUPS tests must deploy via ERC1967Proxy, not directly.
    // AuditRegistry's constructor calls _disableInitializers() which blocks
    // any direct call to initialize() on the implementation.
    // The proxy's constructor calls initialize() atomically on deployment —
    // this is the same flow as production.
    function setUp() public {
        // Deploy token and mint to owner
        token = new SentinelToken();

        // Deploy two mock verifiers — one that passes, one that fails
        verifierPass = new MockZKMLVerifier(true);
        verifierFail = new MockZKMLVerifier(false);

        // Deploy registry via proxy (production-equivalent pattern)
        AuditRegistry impl = new AuditRegistry();
        bytes memory initData = abi.encodeWithSelector(
            AuditRegistry.initialize.selector,
            address(verifierPass),
            address(token)
        );
        ERC1967Proxy proxy = new ERC1967Proxy(address(impl), initData);
        registry = AuditRegistry(address(proxy));

        // Fund agent with enough tokens to stake
        token.transfer(agent, 2000 * 10 ** 18);
    }

    // ─── SentinelToken Tests ───────────────────────────────────────────

    // RECALL — Happy path: agent stakes tokens, balance updates correctly
    function test_StakeUpdatesBalance() public {
        vm.prank(agent); // next call comes from agent
        token.stake(1000 * 10 ** 18);

        assertEq(token.stakedBalance(agent), 1000 * 10 ** 18);
    }

    function test_UnstakeReturnsTokens() public {
        vm.prank(agent);
        token.stake(1000 * 10 ** 18);

        uint256 balanceBefore = token.balanceOf(agent);

        vm.prank(agent);
        token.unstake(1000 * 10 ** 18);

        assertEq(token.stakedBalance(agent), 0);
        assertEq(token.balanceOf(agent), balanceBefore + 1000 * 10 ** 18);
    }

    function test_SlashBurnsTokens() public {
        vm.prank(agent);
        token.stake(1000 * 10 ** 18);

        uint256 supplyBefore = token.totalSupply();

        // owner slashes agent
        token.slash(agent, 500 * 10 ** 18);

        assertEq(token.stakedBalance(agent), 500 * 10 ** 18);
        assertEq(token.totalSupply(), supplyBefore - 500 * 10 ** 18);
    }

    function test_UnstakeRevertsIfInsufficientStake() public {
        vm.prank(agent);
        // RECALL — vm.expectRevert tells Foundry the next call must revert
        // with exactly this message. Test fails if it doesn't revert.
        vm.expectRevert("SentinelToken: insufficient staked balance");
        token.unstake(1000 * 10 ** 18); // never staked anything
    }

    // RECALL — Fuzz test: Foundry generates hundreds of random amounts.
    // vm.assume() filters out inputs that would make the test invalid
    // rather than testing the wrong thing.
    function testFuzz_StakeAndUnstake(uint256 amount) public {
        vm.assume(amount > 0 && amount <= 2000 * 10 ** 18);

        vm.prank(agent);
        token.stake(amount);
        assertEq(token.stakedBalance(agent), amount);

        vm.prank(agent);
        token.unstake(amount);
        assertEq(token.stakedBalance(agent), 0);
    }

    function testFuzz_SlashRevertsIfAmountExceedsStake(uint256 stake, uint256 slash) public {
        vm.assume(stake > 0 && stake <= 2000 * 10 ** 18);
        vm.assume(slash > stake);

        vm.prank(agent);
        token.stake(stake);

        vm.expectRevert("SentinelToken: insufficient staked balance to slash");
        token.slash(agent, slash);
    }

    // ─── AuditRegistry Tests ───────────────────────────────────────────

    // RECALL — Build valid publicSignals array for Guard 3.
    // Index 64 must match the scoreFieldElement we pass to submitAudit().
    // All other indices are zero for testing purposes.
    // In production, indices 0-63 contain the 64 contract feature values
    // as BN254 field elements, and index 64 holds the model output score.
    function _buildSignals(uint256 scoreFieldElement) internal pure returns (uint256[] memory) {
        uint256[] memory signals = new uint256[](65);
        signals[64] = scoreFieldElement;
        return signals;
    }

    function test_SubmitAuditSucceeds() public {
        // Agent stakes first
        vm.prank(agent);
        token.stake(1000 * 10 ** 18);

        uint256 score = 4497; // 0.5490 * 8192 — realistic field element value
        bytes memory proof = abi.encodePacked("mock_proof");
        uint256[] memory signals = _buildSignals(score);

        vm.prank(agent);
        registry.submitAudit(target, score, proof, signals);

        // Verify it was stored
        AuditRegistry.AuditResult memory result = registry.getLatestAudit(target);
        assertEq(result.scoreFieldElement, score);
        assertEq(result.agent, agent);
        assertTrue(result.verified);
    }

    function test_SubmitAuditRevertsIfNotStaked() public {
        // Agent has tokens but never staked
        uint256[] memory signals = _buildSignals(4497);

        vm.prank(agent);
        vm.expectRevert("AuditRegistry: insufficient stake");
        registry.submitAudit(target, 4497, abi.encodePacked("proof"), signals);
    }

    function test_SubmitAuditRevertsIfProofInvalid() public {
        // Deploy registry with FAILING verifier — same proxy pattern as setUp()
        AuditRegistry badImpl = new AuditRegistry();
        bytes memory initData = abi.encodeWithSelector(
            AuditRegistry.initialize.selector,
            address(verifierFail),
            address(token)
        );
        ERC1967Proxy badProxy = new ERC1967Proxy(address(badImpl), initData);
        AuditRegistry badRegistry = AuditRegistry(address(badProxy));

        vm.prank(agent);
        token.stake(1000 * 10 ** 18);

        uint256[] memory signals = _buildSignals(4497);

        vm.prank(agent);
        vm.expectRevert("AuditRegistry: invalid ZK proof");
        badRegistry.submitAudit(target, 4497, abi.encodePacked("proof"), signals);
    }

    function test_SubmitAuditRevertsIfScoreMismatch() public {
        vm.prank(agent);
        token.stake(1000 * 10 ** 18);

        // signals say score=4497 but we claim score=9999
        uint256[] memory signals = _buildSignals(4497);

        vm.prank(agent);
        vm.expectRevert("AuditRegistry: score mismatch with proof");
        registry.submitAudit(target, 9999, abi.encodePacked("proof"), signals);
    }

    function testFuzz_ScoreMismatchAlwaysReverts(uint256 claimedScore, uint256 proofScore) public {
        // RECALL — Any time claimed score != proof score, must revert.
        // Foundry will find edge cases we'd never think to test manually.
        vm.assume(claimedScore != proofScore);

        vm.prank(agent);
        token.stake(1000 * 10 ** 18);

        uint256[] memory signals = _buildSignals(proofScore);

        vm.prank(agent);
        vm.expectRevert("AuditRegistry: score mismatch with proof");
        registry.submitAudit(target, claimedScore, abi.encodePacked("proof"), signals);
    }

    // ─── hasAudit / audit history ────────────────────────────────────────────

    function test_HasAuditReturnsFalseBeforeSubmission() public {
        assertFalse(registry.hasAudit(target));
    }

    function test_HasAuditReturnsTrueAfterSubmission() public {
        vm.prank(agent);
        token.stake(1000 * 10 ** 18);

        uint256[] memory signals = _buildSignals(4497);
        vm.prank(agent);
        registry.submitAudit(target, 4497, abi.encodePacked("proof"), signals);

        assertTrue(registry.hasAudit(target));
    }

    function test_GetLatestAuditRevertsIfNoAudits() public {
        vm.expectRevert("AuditRegistry: no audits found");
        registry.getLatestAudit(target);
    }

    // RECALL — Multiple audits on the same contract should accumulate.
    // getLatestAudit() always returns the most recent one.
    // getAuditCount() returns how many total audits exist.
    function test_MultipleAuditsReturnLatest() public {
        vm.prank(agent);
        token.stake(1000 * 10 ** 18);

        // First audit — score 4497 (0.5490)
        uint256[] memory signals1 = _buildSignals(4497);
        vm.prank(agent);
        registry.submitAudit(target, 4497, abi.encodePacked("proof_1"), signals1);

        // Second audit — score 6144 (0.75)
        uint256[] memory signals2 = _buildSignals(6144);
        vm.prank(agent);
        registry.submitAudit(target, 6144, abi.encodePacked("proof_2"), signals2);

        // Latest audit should be the second one
        AuditRegistry.AuditResult memory latest = registry.getLatestAudit(target);
        assertEq(latest.scoreFieldElement, 6144);

        // Total count should be 2
        assertEq(registry.getAuditCount(target), 2);

        // Full history should have both
        AuditRegistry.AuditResult[] memory history = registry.getAuditHistory(target);
        assertEq(history.length, 2);
        assertEq(history[0].scoreFieldElement, 4497);
        assertEq(history[1].scoreFieldElement, 6144);
    }

    function testFuzz_MultipleAuditsAccumulate(uint8 numAudits) public {
        vm.assume(numAudits > 0 && numAudits <= 20);

        vm.prank(agent);
        token.stake(1000 * 10 ** 18);

        uint256 lastScore;
        for (uint256 i = 0; i < numAudits; i++) {
            lastScore = (i + 1) * 100; // distinct score per round
            uint256[] memory signals = _buildSignals(lastScore);
            vm.prank(agent);
            registry.submitAudit(
                target,
                lastScore,
                abi.encodePacked("proof_", i),
                signals
            );
        }

        assertEq(registry.getAuditCount(target), numAudits);
        assertEq(registry.getLatestAudit(target).scoreFieldElement, lastScore);
    }

    // ─── Pause / Unpause ─────────────────────────────────────────────────────

    function test_PauseBlocksSubmissions() public {
        registry.pause();

        vm.prank(agent);
        token.stake(1000 * 10 ** 18);

        uint256[] memory signals = _buildSignals(4497);

        vm.prank(agent);
        vm.expectRevert();  // EnforcedPause from PausableUpgradeable
        registry.submitAudit(target, 4497, abi.encodePacked("proof"), signals);
    }

    function test_UnpauseResumesSubmissions() public {
        registry.pause();
        registry.unpause();

        vm.prank(agent);
        token.stake(1000 * 10 ** 18);

        uint256[] memory signals = _buildSignals(4497);
        vm.prank(agent);
        registry.submitAudit(target, 4497, abi.encodePacked("proof"), signals);

        assertTrue(registry.hasAudit(target));
    }

    function test_OnlyOwnerCanPause() public {
        vm.prank(agent);
        vm.expectRevert();
        registry.pause();
    }

    // ─── AuditSubmitted event ────────────────────────────────────────────────

    // RECALL — vm.expectEmit verifies an event is emitted with the exact
    // values we specify. Arguments: (checkTopic1, checkTopic2, checkData).
    // topics are the indexed params; data is the non-indexed params.
    function test_AuditSubmittedEventEmitted() public {
        vm.prank(agent);
        token.stake(1000 * 10 ** 18);

        uint256 score = 4497;
        bytes memory proof = abi.encodePacked("mock_proof");
        bytes32 expectedHash = keccak256(proof);
        uint256[] memory signals = _buildSignals(score);

        vm.expectEmit(true, true, false, true);
        emit AuditRegistry.AuditSubmitted(target, expectedHash, agent, score);

        vm.prank(agent);
        registry.submitAudit(target, score, proof, signals);
    }
}
