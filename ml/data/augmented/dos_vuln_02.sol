// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln02 {

    address[] public payees;
    mapping(address => uint256) public owed;
    function addPayee(address p) external { payees.push(p); }
    function payAll() external {
        for (uint256 i = 0; i < payees.length; i++) {
            uint256 amt = owed[payees[i]];
            if (amt > 0) {
                payable(payees[i]).send(amt);
                owed[payees[i]] = 0;
            }
        }
    }
}
