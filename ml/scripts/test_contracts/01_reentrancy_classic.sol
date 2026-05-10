// expect: Reentrancy
// Classic reentrancy: state update AFTER external call.
// The model should catch this. Ether bank with withdrawal before balance zeroed.
pragma solidity ^0.8.0;

contract EtherBank {
    mapping(address => uint256) public balances;

    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw() external {
        uint256 amount = balances[msg.sender];
        require(amount > 0, "nothing to withdraw");

        // VULNERABILITY: call before state update
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "transfer failed");

        // state updated AFTER the external call — attacker can re-enter
        balances[msg.sender] = 0;
    }

    receive() external payable {}
}
