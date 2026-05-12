// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract BridgeSafe23 {

    mapping(bytes32 => bool) public processed;
    mapping(bytes32 => address) public recipient;
    mapping(bytes32 => uint256) public amount;
    function lock(bytes32 id, address to) external payable { recipient[id] = to; amount[id] = msg.value; }
    function unlock(bytes32 id) external {
        require(!processed[id]);
        processed[id] = true;
        (bool ok,) = recipient[id].call{value: amount[id]}("");
        require(ok);
    }
}
