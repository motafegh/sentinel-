// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln21 {

    mapping(uint256 => bytes32) public hashes;
    uint256 public count;
    function addEntry(bytes32 h) external { hashes[count++] = h; }
    function rehashAll() external {
        for (uint256 i = 0; i < count; i++) {
            hashes[i] = keccak256(abi.encodePacked(hashes[i]));
        }
    }
}
