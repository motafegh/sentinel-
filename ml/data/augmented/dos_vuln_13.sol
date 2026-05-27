// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln13 {

    address[] public delegates;
    function addDelegate(address d) external { delegates.push(d); }
    function notifyAll() external {
        for (uint256 i = 0; i < delegates.length; i++) {
            (bool ok,) = delegates[i].call{value: 0, gas: 100000}("");
            require(ok);
        }
    }
}
