// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe29 {

    mapping(address => bool) public isValidator;
    function addValidator(address v) external { isValidator[v] = true; }
    function notifyValidator(address v, bytes calldata data) external {
        require(isValidator[v]);
        (bool ok,) = v.call(data);
        require(ok);
    }
}
