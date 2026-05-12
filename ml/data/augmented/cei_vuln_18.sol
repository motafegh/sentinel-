// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract BondVuln18 {

    struct Bond { uint256 principal; uint256 maturity; bool redeemed; }
    mapping(address => Bond) public bonds;
    function issueBond(uint256 mat) external payable {
        bonds[msg.sender] = Bond(msg.value, block.timestamp + mat, false);
    }
    function redeem() external {
        Bond storage b = bonds[msg.sender];
        require(!b.redeemed && block.timestamp >= b.maturity);
        uint256 payout = b.principal + b.principal / 10;
        (bool ok,) = msg.sender.call{value: payout}("");
        require(ok);
        b.redeemed = true;
    }
}
