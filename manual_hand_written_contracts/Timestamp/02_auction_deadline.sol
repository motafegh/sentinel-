// expect: Timestamp
// Dutch auction contract where the price declines linearly with block.timestamp.
// A miner can choose to include the transaction in a block with a slightly
// earlier or later timestamp to influence the clearing price.
// The auction end time also uses block.timestamp — a miner can extend
// or shorten the auction by a few seconds, potentially benefiting
// themselves or a preferred bidder.
pragma solidity ^0.8.0;

contract TimestampAuction {
    address public seller;
    address public winner;
    uint256 public startTime;
    uint256 public endTime;
    uint256 public startPrice;
    uint256 public reservePrice;
    uint256 public priceDropPerSecond;
    bool public settled;
    uint256 public winningBid;

    event BidPlaced(address indexed bidder, uint256 price);
    event AuctionSettled(address indexed winner, uint256 price);

    modifier notSettled() {
        require(!settled, "already settled");
        _;
    }

    constructor(uint256 _duration, uint256 _startPrice, uint256 _reservePrice, uint256 _dropPerSecond) {
        seller = msg.sender;
        startTime = block.timestamp;
        endTime = block.timestamp + _duration;
        startPrice = _startPrice;
        reservePrice = _reservePrice;
        priceDropPerSecond = _dropPerSecond;
    }

    function getCurrentPrice() public view returns (uint256) {
        uint256 elapsed = block.timestamp - startTime;
        uint256 priceDrop = elapsed * priceDropPerSecond;
        if (priceDrop >= startPrice) return reservePrice;
        uint256 price = startPrice - priceDrop;
        return price < reservePrice ? reservePrice : price;
    }

    function bid() external payable notSettled {
        require(block.timestamp <= endTime, "auction ended");
        require(block.timestamp >= startTime, "auction not started");
        uint256 currentPrice = getCurrentPrice();
        require(msg.value >= currentPrice, "bid too low");
        if (winner != address(0)) {
            uint256 refund = winningBid;
            (bool ok, ) = winner.call{value: refund}("");
            require(ok, "refund failed");
        }
        winner = msg.sender;
        winningBid = msg.value;
        emit BidPlaced(msg.sender, msg.value);
    }

    function settle() external notSettled {
        require(block.timestamp >= endTime || msg.sender == seller, "auction still active");
        settled = true;
        uint256 price = winningBid;
        uint256 sellerProceeds = price;
        (bool ok, ) = seller.call{value: sellerProceeds}("");
        require(ok, "settlement failed");
        emit AuctionSettled(winner, price);
    }

    function getTimeRemaining() external view returns (uint256) {
        if (block.timestamp >= endTime) return 0;
        return endTime - block.timestamp;
    }

    function extendAuction(uint256 additionalTime) external {
        require(msg.sender == seller, "not seller");
        require(!settled, "already settled");
        endTime = block.timestamp + additionalTime;
    }

    receive() external payable {}
}