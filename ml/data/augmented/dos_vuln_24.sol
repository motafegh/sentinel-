// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln24 {

    interface IRegistry { function isRegistered(address) external view returns (bool); }
    IRegistry public registry;
    address[] public applicants;
    constructor(address r) { registry = IRegistry(r); }
    function applyFor(address a) external { applicants.push(a); }
    function processApplications() external {
        for (uint256 i = 0; i < applicants.length; i++) {
            require(registry.isRegistered(applicants[i]));
            payable(applicants[i]).transfer(0.1 ether);
        }
    }
}
