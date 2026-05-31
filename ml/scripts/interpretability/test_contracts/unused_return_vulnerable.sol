// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Vulnerable: return value of external call ignored
contract UnusedReturnVulnerable {
    address public target;
    constructor(address t) { target = t; }
    function callTarget(bytes calldata data) external {
        target.call(data);  // return value not checked
    }
}
