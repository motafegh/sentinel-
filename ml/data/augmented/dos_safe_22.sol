// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe22 {

    mapping(address => bool) public isProcessor;
    function addProcessor(address p) external { isProcessor[p] = true; }
    function callProcessor(address p) external {
        require(isProcessor[p]);
        (bool ok,) = p.call(abi.encodeWithSignature("onProcess()"));
        require(ok);
    }
}
