// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe11 {

    mapping(address => bool) public isModule;
    function addModule(address m) external { isModule[m] = true; }
    function executeModule(address m) external {
        require(isModule[m]);
        (bool ok,) = m.delegatecall(abi.encodeWithSignature("execute()"));
        require(ok);
    }
}
