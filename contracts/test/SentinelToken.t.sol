// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "../src/SentinelToken.sol";

contract SentinelTokenTest is Test {
    SentinelToken token;
    address owner;
    address agent;
    address other;

    uint256 constant MIN_STAKE = 1000 * 10 ** 18;

    function setUp() public {
        owner = address(this);
        agent = makeAddr("agent");
        other = makeAddr("other");

        token = new SentinelToken();
        // Fund agent with enough tokens to stake
        token.transfer(agent, 10_000 * 10 ** 18);
    }

    // --- stake / unstake --------------------------------------------------

    function test_stake_and_unstake() public {
        vm.startPrank(agent);
        token.approve(address(token), MIN_STAKE);
        token.stake(MIN_STAKE);

        assertEq(token.stakedBalance(agent), MIN_STAKE, "staked balance should equal staked amount");
        assertEq(token.balanceOf(address(token)), MIN_STAKE, "contract should hold staked tokens");

        token.unstake(MIN_STAKE);
        assertEq(token.stakedBalance(agent), 0, "staked balance should be zero after unstake");
        assertEq(token.balanceOf(agent), 10_000 * 10 ** 18, "agent balance should be fully restored");
        vm.stopPrank();
    }

    function test_stake_emits_event() public {
        vm.startPrank(agent);
        token.approve(address(token), MIN_STAKE);
        vm.expectEmit(true, false, false, true);
        emit SentinelToken.Staked(agent, MIN_STAKE);
        token.stake(MIN_STAKE);
        vm.stopPrank();
    }

    function test_unstake_emits_event() public {
        vm.startPrank(agent);
        token.approve(address(token), MIN_STAKE);
        token.stake(MIN_STAKE);
        vm.expectEmit(true, false, false, true);
        emit SentinelToken.Unstaked(agent, MIN_STAKE);
        token.unstake(MIN_STAKE);
        vm.stopPrank();
    }

    function test_stake_zero_reverts() public {
        vm.prank(agent);
        vm.expectRevert("SentinelToken: amount must be > 0");
        token.stake(0);
    }

    function test_over_unstake_reverts() public {
        vm.startPrank(agent);
        token.approve(address(token), MIN_STAKE);
        token.stake(MIN_STAKE);
        vm.expectRevert("SentinelToken: insufficient staked balance");
        token.unstake(MIN_STAKE + 1);
        vm.stopPrank();
    }

    function test_unstake_without_stake_reverts() public {
        vm.prank(agent);
        vm.expectRevert("SentinelToken: insufficient staked balance");
        token.unstake(1);
    }

    // --- slash ------------------------------------------------------------

    function test_slash_by_owner() public {
        // Give agent a stake
        vm.startPrank(agent);
        token.approve(address(token), MIN_STAKE);
        token.stake(MIN_STAKE);
        vm.stopPrank();

        uint256 supplyBefore = token.totalSupply();

        // Owner slashes half the stake
        uint256 slashAmount = MIN_STAKE / 2;
        token.slash(agent, slashAmount);

        assertEq(token.stakedBalance(agent), MIN_STAKE - slashAmount, "slashed balance incorrect");
        assertEq(token.totalSupply(), supplyBefore - slashAmount, "slashed tokens must be burned");
    }

    function test_slash_emits_event() public {
        vm.startPrank(agent);
        token.approve(address(token), MIN_STAKE);
        token.stake(MIN_STAKE);
        vm.stopPrank();

        vm.expectEmit(true, false, false, true);
        emit SentinelToken.Slashed(agent, MIN_STAKE);
        token.slash(agent, MIN_STAKE);
    }

    function test_slash_by_non_owner_reverts() public {
        vm.startPrank(agent);
        token.approve(address(token), MIN_STAKE);
        token.stake(MIN_STAKE);
        vm.stopPrank();

        vm.prank(other);
        vm.expectRevert();
        token.slash(agent, MIN_STAKE);
    }

    function test_slash_more_than_staked_reverts() public {
        vm.startPrank(agent);
        token.approve(address(token), MIN_STAKE);
        token.stake(MIN_STAKE);
        vm.stopPrank();

        vm.expectRevert("SentinelToken: insufficient staked balance to slash");
        token.slash(agent, MIN_STAKE + 1);
    }

    // --- transfers while staked -------------------------------------------

    function test_transfer_while_staked() public {
        uint256 agentBalance = token.balanceOf(agent);
        uint256 transferable = agentBalance - MIN_STAKE;

        vm.startPrank(agent);
        token.approve(address(token), MIN_STAKE);
        token.stake(MIN_STAKE);

        // Agent can still transfer un-staked tokens
        token.transfer(other, transferable);
        assertEq(token.balanceOf(other), transferable, "other should receive transferred tokens");
        assertEq(token.stakedBalance(agent), MIN_STAKE, "stake unchanged after transfer");
        vm.stopPrank();
    }

    // --- constants / initial state ----------------------------------------

    function test_initial_supply_minted_to_owner() public view {
        assertEq(token.totalSupply(), 1_000_000 * 10 ** 18);
        // Owner transferred 10_000 to agent in setUp; remainder is owner's
        assertEq(token.balanceOf(owner), 990_000 * 10 ** 18);
    }

    function test_min_stake_constant() public view {
        assertEq(token.MIN_STAKE(), MIN_STAKE);
    }

    function test_staked_balance_of_unstaked_address_is_zero() public view {
        assertEq(token.stakedBalance(other), 0);
    }
}
