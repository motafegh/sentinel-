// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln18 {

    address payable[] public recipients;
    function addRecipient(address payable r) external { recipients.push(r); }
    function splitEven() external payable {
        uint256 share = msg.value / recipients.length;
        for (uint256 i = 0; i < recipients.length; i++) {
            recipients[i].transfer(share);
        }
    }
}
