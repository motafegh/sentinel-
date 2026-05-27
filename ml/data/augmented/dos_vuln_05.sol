// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln05 {

    mapping(uint256 => address[]) public groupMembers;
    uint256 public numGroups;
    function addMember(uint256 gid, address m) external { groupMembers[gid].push(m); }
    function payoutAll(uint256 perMember) external payable {
        for (uint256 g = 0; g < numGroups; g++) {
            address[] storage members = groupMembers[g];
            for (uint256 i = 0; i < members.length; i++) {
                (bool ok,) = members[i].call{value: perMember}("");
                require(ok);
            }
        }
    }
}
