// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln11 {

    address[] public modules;
    function addModule(address m) external { modules.push(m); }
    function executeAll() external {
        for (uint256 i = 0; i < modules.length; i++) {
            (bool ok,) = modules[i].delegatecall(abi.encodeWithSignature("execute()"));
            require(ok);
        }
    }
}
