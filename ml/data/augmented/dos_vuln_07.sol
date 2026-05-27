// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln07 {

    address[] public recipients;
    uint256 public amountPerRecipient;
    function addRecipient(address r) external { recipients.push(r); }
    function processAll() external {
        for (uint256 i = 0; i < recipients.length; i++) {
            (bool ok,) = recipients[i].call{value: amountPerRecipient}("");
            if (ok) {
                recipients.push(recipients[i]);
            }
            require(ok);
        }
    }
}
