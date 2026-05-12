// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract LendSafe08 {

    mapping(address => uint256) public collateral;
    mapping(address => uint256) public borrowed;
    function depositCollateral() external payable { collateral[msg.sender] += msg.value; }
    function repayAndWithdraw() external payable {
        require(msg.value >= borrowed[msg.sender]);
        uint256 col = collateral[msg.sender];
        collateral[msg.sender] = 0;
        borrowed[msg.sender] = 0;
        (bool ok,) = msg.sender.call{value: col}("");
        require(ok);
    }
}
