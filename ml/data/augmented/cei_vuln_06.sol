// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract VaultVuln06 {

    mapping(address => uint256) public shares;
    uint256 public totalShares;
    function deposit() external payable { shares[msg.sender] += msg.value; totalShares += msg.value; }
    function redeem(uint256 s) external {
        require(shares[msg.sender] >= s);
        uint256 amt = s * address(this).balance / totalShares;
        (bool ok,) = msg.sender.call{value: amt}("");
        require(ok);
        shares[msg.sender] -= s;
        totalShares -= s;
    }
}
