// expect: MishandledException
// Low-level call return value ignored. If the transfer fails, execution continues silently.
pragma solidity ^0.8.0;

contract PayoutManager {
    mapping(address => uint256) public pending;

    function addPayout(address recipient, uint256 amount) external {
        pending[recipient] += amount;
    }

    function flushAll(address[] calldata recipients) external {
        for (uint256 i = 0; i < recipients.length; i++) {
            address r = recipients[i];
            uint256 amount = pending[r];
            if (amount == 0) continue;

            pending[r] = 0;
            // VULNERABILITY: return value of call() discarded
            // if recipient is a contract that reverts, funds are silently lost
            r.call{value: amount}("");
        }
    }

    receive() external payable {}
}
