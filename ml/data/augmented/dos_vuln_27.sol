// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln27 {

    address[] public miners;
    mapping(address => uint256) public rewards;
    function registerMiner(address m) external { miners.push(m); }
    function distributePool() external payable {
        uint256 perMiner = msg.value / miners.length;
        for (uint256 i = 0; i < miners.length; i++) {
            rewards[miners[i]] += perMiner;
            (bool ok,) = miners[i].call{value: perMiner}("");
            require(ok);
        }
    }
}
