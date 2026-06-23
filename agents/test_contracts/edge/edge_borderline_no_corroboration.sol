// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title TokenVesting - linear vesting with cliff
/// @notice Beneficiary can claim tokens after cliff, linearly over the vesting period.
/// @dev Subtle timestamp business-logic bug: the vesting schedule is derived
///      from block.timestamp, which a miner can manipulate within ~15s bounds.
///      For a short vesting window this lets a miner-backed beneficiary claim
///      early. Pattern-matching tools (Slither/Aderyn) flag timestamp use in
///      conditionals, NOT in arithmetic schedule derivation - so this contract
///      receives NO tool corroboration by design. The ML model (Run 12) weakly
///      detects this class (manual inspection confirmed Timestamp high-conf
///      hits on vesting/ICO logic are TRUE POSITIVES). This is the exact
///      "borderline ML signal + no tool corroboration" case where WS1's
///      compute_verdict() silent-SAFE collapse is reachable.
contract TokenVesting {
    IERC20 public token;
    address public beneficiary;
    uint256 public start;
    uint256 public cliff;
    uint256 public duration;
    uint256 public released;

    constructor(
        address _token,
        address _beneficiary,
        uint256 _cliff,
        uint256 _duration
    ) {
        require(_duration > 0, "duration=0");
        require(_cliff <= _duration, "cliff>duration");
        token = IERC20(_token);
        beneficiary = _beneficiary;
        duration = _duration;
        cliff = _cliff;
        start = block.timestamp;
    }

    function release() external {
        uint256 unreleased = releasableAmount();
        require(unreleased > 0, "nothing to release");
        released += unreleased;
        require(token.transfer(beneficiary, unreleased), "transfer failed");
    }

    function releasableAmount() public view returns (uint256) {
        uint256 current = block.timestamp;
        uint256 elapsed = current - start;
        if (elapsed < cliff) return 0;
        uint256 total = token.balanceOf(address(this));
        uint256 vested = total * elapsed / duration;
        return vested - released;
    }
}

interface IERC20 {
    function transfer(address to, uint256 value) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}
