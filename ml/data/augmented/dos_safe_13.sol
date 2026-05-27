// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe13 {

    mapping(address => bool) public isDelegate;
    function addDelegate(address d) external { isDelegate[d] = true; }
    function notifyDelegate(address d) external {
        require(isDelegate[d]);
        (bool ok,) = d.call{value: 0, gas: 100000}("");
        require(ok);
    }
}
