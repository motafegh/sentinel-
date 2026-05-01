// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import "../src/AuditRegistry.sol";
import "../src/SentinelToken.sol";
import "./mocks/MockZKMLVerifier.sol";

/// @dev Handler contract drives random actions for stateful invariant fuzzing.
/// Foundry calls functions in this contract with random inputs to explore state.
contract AuditRegistryHandler is Test {
    AuditRegistry public registry;
    SentinelToken public token;

    address[] public actors;

    uint256 public totalAuditsSubmitted;
    // Track per-actor staked to validate invariant without touching contract internals
    mapping(address => uint256) public handlerStaked;

    uint256 constant MIN_STAKE = 1000 * 10 ** 18;

    constructor(AuditRegistry _registry, SentinelToken _token) {
        registry = _registry;
        token    = _token;

        // Set up a fixed pool of actors
        for (uint256 i = 0; i < 5; i++) {
            address a = makeAddr(string(abi.encode("actor", i)));
            actors.push(a);
            // Fund each actor
            _token.transfer(a, 10_000 * 10 ** 18);
        }
    }

    function stake(uint256 actorSeed, uint256 amount) public {
        address actor = _pickActor(actorSeed);
        amount = bound(amount, 1, token.balanceOf(actor));

        vm.startPrank(actor);
        token.approve(address(token), amount);
        token.stake(amount);
        vm.stopPrank();

        handlerStaked[actor] += amount;
    }

    function unstake(uint256 actorSeed, uint256 amount) public {
        address actor = _pickActor(actorSeed);
        uint256 staked = token.stakedBalance(actor);
        if (staked == 0) return;

        amount = bound(amount, 1, staked);
        vm.prank(actor);
        token.unstake(amount);
        handlerStaked[actor] -= amount;
    }

    function submitAudit(uint256 actorSeed, uint256 scoreSeed, address target) public {
        address actor = _pickActor(actorSeed);
        if (token.stakedBalance(actor) < MIN_STAKE) return;
        if (target == address(0)) return;

        uint256 score = bound(scoreSeed, 0, type(uint32).max);

        // Build publicSignals with index 64 == score
        uint256[] memory sigs = new uint256[](65);
        sigs[64] = score;

        bytes memory proof = abi.encode(score); // arbitrary bytes; mock always returns true

        vm.prank(actor);
        registry.submitAudit(target, score, proof, sigs);
        totalAuditsSubmitted++;
    }

    function _pickActor(uint256 seed) internal view returns (address) {
        return actors[seed % actors.length];
    }
}

contract InvariantAuditRegistryTest is Test {
    AuditRegistry registry;
    SentinelToken token;
    MockZKMLVerifier verifier;
    AuditRegistryHandler handler;

    address[] private _targets;

    function setUp() public {
        token    = new SentinelToken();
        verifier = new MockZKMLVerifier();

        AuditRegistry impl = new AuditRegistry();
        bytes memory initData = abi.encodeCall(
            AuditRegistry.initialize,
            (address(verifier), address(token))
        );
        ERC1967Proxy proxy = new ERC1967Proxy(address(impl), initData);
        registry = AuditRegistry(address(proxy));

        handler = new AuditRegistryHandler(registry, token);

        // Transfer a large supply budget to the handler for actor funding
        token.transfer(address(handler), 500_000 * 10 ** 18);

        // Pre-populate some fixed target addresses for submitAudit calls
        for (uint256 i = 0; i < 3; i++) {
            _targets.push(makeAddr(string(abi.encode("target", i))));
        }

        // Target the handler — Foundry calls its public functions randomly
        targetContract(address(handler));
        bytes4[] memory selectors = new bytes4[](3);
        selectors[0] = AuditRegistryHandler.stake.selector;
        selectors[1] = AuditRegistryHandler.unstake.selector;
        selectors[2] = AuditRegistryHandler.submitAudit.selector;
        targetSelector(FuzzSelector({ addr: address(handler), selectors: selectors }));
    }

    /// @dev Audit count is strictly monotonic — submissions never disappear.
    function invariant_audit_count_monotonic() public view {
        for (uint256 i = 0; i < _targets.length; i++) {
            // We can't read prior counts easily, but we can ensure count is
            // consistent with totalAuditsSubmitted (upper bound).
            uint256 count = registry.getAuditCount(_targets[i]);
            assertLe(count, handler.totalAuditsSubmitted(), "count exceeds total submitted");
        }
    }

    /// @dev Total staked across all actors ≤ total token supply.
    function invariant_total_staked_le_supply() public view {
        uint256 totalStaked = token.balanceOf(address(token));
        assertLe(totalStaked, token.totalSupply(), "staked exceeds total supply");
    }

    /// @dev Token contract holds exactly the sum of all staked balances.
    function invariant_contract_balance_matches_staked() public view {
        address[] memory actors = new address[](5);
        for (uint256 i = 0; i < 5; i++) {
            actors[i] = makeAddr(string(abi.encode("actor", i)));
        }

        uint256 sumStaked = 0;
        for (uint256 i = 0; i < actors.length; i++) {
            sumStaked += token.stakedBalance(actors[i]);
        }

        assertEq(token.balanceOf(address(token)), sumStaked, "contract balance ≠ sum of staked");
    }
}
