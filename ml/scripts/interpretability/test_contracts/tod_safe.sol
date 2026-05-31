// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Safe: commit-reveal prevents front-running
contract TODSafe {
    mapping(address => bytes32) public commitments;
    mapping(address => uint256) public deposits;
    function commit(bytes32 hash) external payable {
        commitments[msg.sender] = hash;
        deposits[msg.sender] = msg.value;
    }
    function reveal(uint256 value, bytes32 nonce) external {
        require(keccak256(abi.encodePacked(value, nonce)) == commitments[msg.sender]);
        uint256 amount = deposits[msg.sender];
        deposits[msg.sender] = 0;
        payable(msg.sender).transfer(amount);
    }
}
