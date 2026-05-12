// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SplitVuln19 {

    address payable public partyA;
    address payable public partyB;
    mapping(address => uint256) public owed;
    constructor(address payable a, address payable b) { partyA = a; partyB = b; }
    function fund() external payable { owed[partyA] += msg.value / 2; owed[partyB] += msg.value / 2; }
    function withdraw() external {
        require(msg.sender == partyA || msg.sender == partyB);
        uint256 amt = owed[msg.sender];
        require(amt > 0);
        (bool ok,) = payable(msg.sender).call{value: amt}("");
        require(ok);
        owed[msg.sender] = 0;
    }
}
