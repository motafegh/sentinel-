// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract OptionSafe20 {

    mapping(address => uint256) public optionSize;
    mapping(address => uint256) public strikePrice;
    mapping(address => bool) public exercised;
    function writeOption(address buyer, uint256 size, uint256 strike) external payable {
        optionSize[buyer] = size; strikePrice[buyer] = strike;
    }
    function exercise() external payable {
        require(!exercised[msg.sender]);
        require(msg.value >= strikePrice[msg.sender]);
        uint256 payout = optionSize[msg.sender];
        exercised[msg.sender] = true;
        (bool ok,) = msg.sender.call{value: payout}("");
        require(ok);
    }
}
