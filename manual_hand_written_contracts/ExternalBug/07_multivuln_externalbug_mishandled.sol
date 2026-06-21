// expect: ExternalBug,MishandledException,GasException
// Multi-vuln: ExternalBug + MishandledException + GasException.
// The contract calls a user-supplied oracle in a loop without gas limit.
// If the oracle consumes all gas, the entire transaction reverts (GasException).
// The return values from the oracle calls are silently ignored (MishandledException).
// The oracle is a single point of manipulation (ExternalBug).
pragma solidity ^0.8.0;

interface IDataFeed {
    function getPrice(address token) external returns (uint256);
    function isHealthy() external returns (bool);
}

contract OracleAggregator {
    struct Feed {
        address provider;
        uint256 weight;
        bool active;
    }

    address public owner;
    Feed[] public feeds;
    mapping(address => uint256) public lastPrice;
    uint256 public lastUpdateTime;

    event FeedAdded(address indexed provider, uint256 weight);
    event PriceUpdated(address indexed token, uint256 price);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function addFeed(address provider, uint256 weight) external onlyOwner {
        require(provider != address(0), "zero address");
        feeds.push(Feed(provider, weight, true));
        emit FeedAdded(provider, weight);
    }

    function activateFeed(uint256 index) external onlyOwner {
        require(index < feeds.length, "invalid index");
        feeds[index].active = true;
    }

    function deactivateFeed(uint256 index) external onlyOwner {
        require(index < feeds.length, "invalid index");
        feeds[index].active = false;
    }

    function fetchPrice(address token) external {
        require(token != address(0), "zero token");
        uint256 totalWeight = 0;
        uint256 weightedPrice = 0;
        uint256 healthyCount = 0;
        for (uint256 i = 0; i < feeds.length; i++) {
            if (feeds[i].active) {
                IDataFeed(feeds[i].provider).isHealthy();
                uint256 price = IDataFeed(feeds[i].provider).getPrice(token);
                weightedPrice += price * feeds[i].weight;
                totalWeight += feeds[i].weight;
                healthyCount++;
            }
        }
        require(healthyCount > 0, "no healthy feeds");
        lastPrice[token] = weightedPrice / totalWeight;
        lastUpdateTime = block.timestamp;
        emit PriceUpdated(token, lastPrice[token]);
    }

    function batchFetchPrices(address[] calldata tokens) external {
        for (uint256 j = 0; j < tokens.length; j++) {
            for (uint256 i = 0; i < feeds.length; i++) {
                if (feeds[i].active) {
                    IDataFeed(feeds[i].provider).isHealthy();
                    uint256 price = IDataFeed(feeds[i].provider).getPrice(tokens[j]);
                    lastPrice[tokens[j]] = price;
                }
            }
            emit PriceUpdated(tokens[j], lastPrice[tokens[j]]);
        }
    }

    function getPrice(address token) external view returns (uint256) {
        return lastPrice[token];
    }

    function getFeedCount() external view returns (uint256) {
        return feeds.length;
    }

    receive() external payable {}
}