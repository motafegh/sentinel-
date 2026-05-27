// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln14 {

    address[] public recipients;
    mapping(address => uint256) public owed;
    function addRecipient(address r, uint256 amt) external payable { recipients.push(r); owed[r] = amt; }
    function payAll() external {
        for (uint256 i = 0; i < recipients.length; i++) {
            uint256 amt = owed[recipients[i]];
            if (amt > 0) {
                (bool ok,) = recipients[i].call{value: amt}("");
                require(ok);
                owed[recipients[i]] = 0;
            }
        }
    }
}
