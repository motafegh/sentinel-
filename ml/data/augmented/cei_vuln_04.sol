// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract EscrowVuln04 {

    mapping(bytes32 => uint256) public deposits;
    mapping(bytes32 => address) public beneficiary;
    function deposit(bytes32 id, address ben) external payable {
        deposits[id] = msg.value; beneficiary[id] = ben;
    }
    function release(bytes32 id) external {
        uint256 amt = deposits[id];
        require(amt > 0);
        (bool ok,) = beneficiary[id].call{value: amt}("");
        require(ok);
        deposits[id] = 0;
    }
}
