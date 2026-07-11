// expect: GasException
// On-chain verification contract that calls multiple external oracles in a loop.
// Each staticcall forwards remaining gas — if one oracle has an expensive
// view function, the whole transaction reverts due to out-of-gas.
// The contract also has a function that writes large structs to storage
// in an unbounded loop, hitting the block gas limit from SSTORE costs.
// The recursive call pattern chains gas costs exponentially.
pragma solidity ^0.8.0;

interface IDataFeed {
    function getLatestData() external view returns (uint256);
    function getHistoricalData(uint256 from, uint256 to) external view returns (uint256[] memory);
}

contract DataAggregator {
    address public owner;
    IDataFeed[] public dataFeeds;
    uint256[] public aggregatedValues;
    mapping(address => uint256) public lastUpdate;

    event FeedAdded(address indexed feed);
    event DataAggregated(uint256 indexed round, uint256 value);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function addFeed(address feed) external onlyOwner {
        dataFeeds.push(IDataFeed(feed));
        emit FeedAdded(feed);
    }

    function aggregateAll() external returns (uint256 sum) {
        require(dataFeeds.length > 0, "no feeds");
        for (uint256 i = 0; i < dataFeeds.length; i++) {
            uint256 val = dataFeeds[i].getLatestData();
            sum += val;
            lastUpdate[address(dataFeeds[i])] = block.timestamp;
        }
        uint256 round = aggregatedValues.length;
        aggregatedValues.push(sum);
        emit DataAggregated(round, sum);
    }

    function aggregateRange(uint256 from, uint256 to) external returns (uint256[] memory results) {
        require(to > from, "invalid range");
        require(dataFeeds.length > 0, "no feeds");
        results = new uint256[](to - from);
        uint256 index = 0;
        for (uint256 j = from; j < to; j++) {
            uint256 sum = 0;
            for (uint256 i = 0; i < dataFeeds.length; i++) {
                uint256[] memory historical = dataFeeds[i].getHistoricalData(j, j + 1);
                if (historical.length > 0) {
                    sum += historical[0];
                }
            }
            results[index] = sum;
            index++;
        }
        for (uint256 k = 0; k < results.length; k++) {
            aggregatedValues.push(results[k]);
        }
    }

    function recursiveAggregate(uint256 depth) external view returns (uint256) {
        require(depth <= 10, "max depth 10");
        if (depth == 0) {
            uint256 total = 0;
            for (uint256 i = 0; i < dataFeeds.length; i++) {
                total += dataFeeds[i].getLatestData();
            }
            return total;
        }
        uint256 sub = recursiveAggregate(depth - 1);
        uint256 current = 0;
        for (uint256 j = 0; j < dataFeeds.length; j++) {
            current += dataFeeds[j].getLatestData();
        }
        return sub + current;
    }

    function heavyStorageWrite(uint256 rounds) external onlyOwner {
        require(rounds <= 100, "max 100 rounds");
        for (uint256 i = 0; i < rounds; i++) {
            aggregatedValues.push(block.timestamp);
            for (uint256 j = 0; j < 50; j++) {
                aggregatedValues.push(aggregatedValues.length);
            }
        }
    }

    function getFeedCount() external view returns (uint256) {
        return dataFeeds.length;
    }
}