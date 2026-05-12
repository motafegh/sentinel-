// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract LendVuln08 {

    mapping(address => uint256) public collateral;
    mapping(address => uint256) public borrowed;
    function depositCollateral() external payable { collateral[msg.sender] += msg.value; }
    function repayAndWithdraw() external payable {
        require(msg.value >= borrowed[msg.sender]);
        (bool ok,) = msg.sender.call{value: collateral[msg.sender]}("");
        require(ok);
        collateral[msg.sender] = 0;
        borrowed[msg.sender] = 0;
    }
}
