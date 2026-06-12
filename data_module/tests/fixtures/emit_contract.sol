pragma solidity ^0.5.11;

// Minimal contract with an event emission — used by test_emits_fixture.py
// to confirm the EMITS edge (type 3) is present in the extracted graph.
contract TokenTransfer {
    event Transfer(address indexed from, address indexed to, uint256 value);

    mapping(address => uint256) public balances;

    constructor() public {
        balances[msg.sender] = 1000;
    }

    function transfer(address to, uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[to] += amount;
        emit Transfer(msg.sender, to, amount);
    }
}
