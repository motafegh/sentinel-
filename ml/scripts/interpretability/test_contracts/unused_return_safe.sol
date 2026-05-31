// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
// Safe: return value checked
contract UnusedReturnSafe {
    address public target;
    constructor(address t) { target = t; }
    function callTarget(bytes calldata data) external {
        (bool success,) = target.call(data);
        require(success, "call failed");  // return value checked
    }
}
