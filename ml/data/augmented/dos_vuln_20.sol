// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosVuln20 {

    interface IOracle { function getFeedCount() external view returns (uint256); function getFeed(uint256 i) external view returns (address); }
    IOracle public oracle;
    constructor(address o) { oracle = IOracle(o); }
    function payoutFeeds() external payable {
        uint256 count = oracle.getFeedCount();
        uint256 perFeed = msg.value / count;
        for (uint256 i = 0; i < count; i++) {
            address feed = oracle.getFeed(i);
            payable(feed).transfer(perFeed);
        }
    }
}
