// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln12 {

    address[] public childContracts;
    function addChild(address c) external { childContracts.push(c); }
    function destroyAll() external {
        for (uint256 i = 0; i < childContracts.length; i++) {
            IChild(childContracts[i]).destroy();
        }
    }
    interface IChild { function destroy() external; }
}
