// expect: Reentrancy
// Tricky: reentrancy hidden inside an internal helper. Not an obvious pattern.
pragma solidity ^0.8.0;

contract TreasuryRouter {
    mapping(address => uint256) private _shares;
    uint256 public totalShares;

    function stake() external payable {
        _shares[msg.sender] += msg.value;
        totalShares += msg.value;
    }

    function _sendReward(address to, uint256 amount) internal {
        // VULNERABILITY: external call inside internal helper before shares updated
        (bool ok, ) = to.call{value: amount}("");
        require(ok);
    }

    function claimAndRestake(uint256 keepAmount) external {
        uint256 owned = _shares[msg.sender];
        require(owned > keepAmount);

        uint256 payout = owned - keepAmount;
        // state not updated before _sendReward — reentrancy possible
        _sendReward(msg.sender, payout);

        // these run after the call
        _shares[msg.sender] = keepAmount;
        totalShares -= payout;
    }
}
