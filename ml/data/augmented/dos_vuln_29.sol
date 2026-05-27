// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln29 {

    address[] public validators;
    function addValidator(address v) external { validators.push(v); }
    function notifyValidators(bytes calldata data) external {
        for (uint256 i = 0; i < validators.length; i++) {
            (bool ok,) = validators[i].call(data);
            require(ok);
        }
    }
}
