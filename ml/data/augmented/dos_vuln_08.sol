// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln08 {

    mapping(address => uint256) public dividends;
    address[] public shareholders;
    function addShareholder(address s) external { shareholders.push(s); }
    function distributeDividends() external payable {
        uint256 perShare = msg.value / shareholders.length;
        for (uint256 i = 0; i < shareholders.length; i++) {
            dividends[shareholders[i]] += perShare;
            (bool ok,) = shareholders[i].call{value: perShare}("");
            require(ok);
        }
    }
}
