// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln17 {

    address[] public whitelist;
    function addToWhitelist(address a) external { whitelist.push(a); }
    function batchTransfer() external payable {
        uint256 perAddr = msg.value / whitelist.length;
        for (uint256 i = 0; i < whitelist.length; i++) {
            (bool ok,) = whitelist[i].call{value: perAddr}("");
            require(ok);
        }
    }
}
