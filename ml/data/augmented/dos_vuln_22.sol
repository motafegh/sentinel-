// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln22 {

    address[] public processors;
    function addProcessor(address p) external { processors.push(p); }
    function processChain() external {
        for (uint256 i = 0; i < processors.length; i++) {
            (bool ok,) = processors[i].call(abi.encodeWithSignature("onProcess()"));
            require(ok);
        }
    }
}
