// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract NftSafe12 {

    mapping(address => uint256) public minted;
    uint256 public price = 0.01 ether;
    function mint() external payable {
        require(msg.value >= price);
        uint256 excess = msg.value - price;
        minted[msg.sender]++;                               // state before refund
        if (excess > 0) {
            (bool ok,) = msg.sender.call{value: excess}("");
            require(ok);
        }
    }
}
