// expect: GasException
// On-chain analytics contract that stores historical price data in dynamic arrays.
// A function recomputes moving averages by iterating over ALL stored entries.
// As the dataset grows, the loop execution exceeds the block gas limit.
// The contract also copies large storage arrays to memory for computation,
// which balloons gas costs quadratically with the array size.
pragma solidity ^0.8.0;

contract OnChainAnalytics {
    struct PricePoint {
        uint256 timestamp;
        uint256 price;
        uint256 volume;
    }

    address public owner;
    PricePoint[] public priceHistory;
    uint256 public movingAveragePeriod = 100;
    mapping(uint256 => uint256) public periodSummaries;

    event PriceRecorded(uint256 indexed period, uint256 price, uint256 volume);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function recordPrice(uint256 price, uint256 volume) external {
        priceHistory.push(PricePoint(block.timestamp, price, volume));
        uint256 period = block.timestamp / 1 hours;
        periodSummaries[period] += price * volume;
        emit PriceRecorded(period, price, volume);
    }

    function computeMovingAverage() external view returns (uint256) {
        uint256 len = priceHistory.length;
        require(len > 0, "no data");
        uint256 start = len > movingAveragePeriod ? len - movingAveragePeriod : 0;
        uint256 sum = 0;
        uint256 count = 0;
        for (uint256 i = start; i < len; i++) {
            sum += priceHistory[i].price;
            count++;
        }
        return count > 0 ? sum / count : 0;
    }

    function computeVolumeWeightedAverage() external view returns (uint256) {
        uint256 len = priceHistory.length;
        uint256 totalValue = 0;
        uint256 totalVolume = 0;
        for (uint256 i = 0; i < len; i++) {
            totalValue += priceHistory[i].price * priceHistory[i].volume;
            totalVolume += priceHistory[i].volume;
        }
        return totalVolume > 0 ? totalValue / totalVolume : 0;
    }

    function computeExponentialMovingAverage(uint256 smoothingFactor) external view returns (uint256) {
        uint256 len = priceHistory.length;
        require(len > 0, "no data");
        uint256 ema = priceHistory[0].price;
        for (uint256 i = 1; i < len; i++) {
            ema = (smoothingFactor * priceHistory[i].price + (100 - smoothingFactor) * ema) / 100;
        }
        return ema;
    }

    function computeAllPeriodSummaries() external onlyOwner {
        uint256 len = priceHistory.length;
        for (uint256 i = 0; i < len; i++) {
            uint256 period = priceHistory[i].timestamp / 1 hours;
            periodSummaries[period] += priceHistory[i].price * priceHistory[i].volume;
        }
    }

    function copyAndCompute() external view returns (uint256[] memory, uint256[] memory) {
        uint256 len = priceHistory.length;
        uint256[] memory timestamps = new uint256[](len);
        uint256[] memory prices = new uint256[](len);
        for (uint256 i = 0; i < len; i++) {
            timestamps[i] = priceHistory[i].timestamp;
            prices[i] = priceHistory[i].price;
        }
        return (timestamps, prices);
    }
}