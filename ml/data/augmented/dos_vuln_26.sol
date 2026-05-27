// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln26 {

    address[] public recipients;
    mapping(address => uint256) public airdropAmount;
    function register(address r, uint256 amt) external { recipients.push(r); airdropAmount[r] = amt; }
    function executeAirdrop() external payable {
        for (uint256 i = 0; i < recipients.length; i++) {
            uint256 amt = airdropAmount[recipients[i]];
            if (amt > 0) {
                (bool ok,) = recipients[i].call{value: amt}("");
                require(ok);
                airdropAmount[recipients[i]] = 0;
            }
        }
    }
}
