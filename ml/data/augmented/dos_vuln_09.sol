// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln09 {

    mapping(uint256 => bytes32) public dataStore;
    uint256 public numEntries;
    function storeEntry(bytes32 data) external { dataStore[numEntries++] = data; }
    function processAll() external {
        for (uint256 i = 0; i < numEntries; i++) {
            bytes32 data = dataStore[i];
            if (data != bytes32(0)) {
                payable(msg.sender).transfer(0.001 ether);
            }
        }
    }
}
