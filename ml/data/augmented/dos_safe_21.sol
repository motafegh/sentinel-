// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe21 {

    mapping(uint256 => bytes32) public hashes;
    uint256 public count;
    function addEntry(bytes32 h) external { hashes[count++] = h; }
    function rehashEntry(uint256 i) external {
        require(i < count);
        hashes[i] = keccak256(abi.encodePacked(hashes[i]));
    }
}
