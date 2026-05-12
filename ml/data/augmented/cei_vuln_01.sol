// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract WithdrawVuln01 {

    mapping(address => uint256) public balances;
    function deposit() external payable { balances[msg.sender] += msg.value; }
    function withdraw() external {
        uint256 amt = balances[msg.sender];
        require(amt > 0);
        (bool ok,) = msg.sender.call{value: amt}("");  // call before update
        require(ok);
        balances[msg.sender] = 0;
    }
}
