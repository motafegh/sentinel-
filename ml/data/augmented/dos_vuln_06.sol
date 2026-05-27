// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln06 {

    mapping(address => uint256) public locked;
    function lock() external payable { locked[msg.sender] += msg.value; }
    function unlockAll(address[] calldata users) external {
        uint256 i = 0;
        while (i < users.length) {
            uint256 amt = locked[users[i]];
            if (amt > 0) {
                (bool ok,) = users[i].call{value: amt}("");
                require(ok);
                locked[users[i]] = 0;
            }
            i++;
        }
    }
}
