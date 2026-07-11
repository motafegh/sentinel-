// expect: TransactionOrderDependence
// Dutch auction for NFT sale where price decreases over time.
// Bidders monitor the mempool for the first bid transaction and try to
// front-run it with higher gas to buy at a marginally higher price.
// The race condition is inherent: the first tx to confirm gets the NFT,
// and all other pending bids revert. MEV bots compete to order their
// bid just before the intended bidder's transaction.
// The batch claim for airdrops also suffers from ordering races.
pragma solidity ^0.8.0;

interface IERC721 {
    function transferFrom(address from, address to, uint256 tokenId) external;
    function mint(address to, uint256 tokenId) external;
}

contract DutchAuctionRace {
    struct Auction {
        uint256 tokenId;
        address seller;
        uint256 startPrice;
        uint256 reservePrice;
        uint256 startTime;
        uint256 duration;
        bool active;
    }

    IERC721 public nft;
    address public owner;
    mapping(uint256 => Auction) public auctions;
    uint256 public priceDropPerSecond;

    event AuctionCreated(uint256 indexed tokenId, uint256 startPrice, uint256 reservePrice);
    event AuctionWon(uint256 indexed tokenId, address indexed winner, uint256 price);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _nft, uint256 _dropPerSec) {
        nft = IERC721(_nft);
        owner = msg.sender;
        priceDropPerSecond = _dropPerSec;
    }

    function createAuction(uint256 tokenId, uint256 startPrice, uint256 reservePrice, uint256 duration) external {
        require(nft.ownerOf(tokenId) == msg.sender, "not owner");
        require(startPrice > reservePrice, "start must exceed reserve");
        nft.transferFrom(msg.sender, address(this), tokenId);
        auctions[tokenId] = Auction(tokenId, msg.sender, startPrice, reservePrice, block.timestamp, duration, true);
        emit AuctionCreated(tokenId, startPrice, reservePrice);
    }

    function getCurrentPrice(uint256 tokenId) public view returns (uint256) {
        Auction storage a = auctions[tokenId];
        require(a.active, "not active");
        uint256 elapsed = block.timestamp - a.startTime;
        if (elapsed >= a.duration) return a.reservePrice;
        uint256 priceDrop = elapsed * priceDropPerSecond;
        if (priceDrop >= a.startPrice) return a.reservePrice;
        uint256 price = a.startPrice - priceDrop;
        return price < a.reservePrice ? a.reservePrice : price;
    }

    function bid(uint256 tokenId) external payable {
        Auction storage a = auctions[tokenId];
        require(a.active, "auction not active");
        uint256 price = getCurrentPrice(tokenId);
        require(msg.value >= price, "bid too low");
        a.active = false;
        uint256 excess = msg.value - price;
        nft.transferFrom(address(this), msg.sender, tokenId);
        (bool ok, ) = a.seller.call{value: price}("");
        require(ok, "seller payment failed");
        if (excess > 0) {
            (bool okRefund, ) = msg.sender.call{value: excess}("");
            require(okRefund, "refund failed");
        }
        emit AuctionWon(tokenId, msg.sender, price);
    }

    function batchBid(uint256[] calldata tokenIds) external payable {
        uint256 totalCost = 0;
        for (uint256 i = 0; i < tokenIds.length; i++) {
            Auction storage a = auctions[tokenIds[i]];
            require(a.active, "not active");
            totalCost += getCurrentPrice(tokenIds[i]);
        }
        require(msg.value >= totalCost, "insufficient");
        uint256 remaining = msg.value;
        for (uint256 j = 0; j < tokenIds.length; j++) {
            Auction storage a = auctions[tokenIds[j]];
            uint256 price = getCurrentPrice(tokenIds[j]);
            a.active = false;
            nft.transferFrom(address(this), msg.sender, tokenIds[j]);
            (bool ok, ) = a.seller.call{value: price}("");
            require(ok, "seller payment failed");
            remaining -= price;
            emit AuctionWon(tokenIds[j], msg.sender, price);
        }
        if (remaining > 0) {
            (bool okRefund, ) = msg.sender.call{value: remaining}("");
            require(okRefund, "refund failed");
        }
    }

    function cancelAuction(uint256 tokenId) external {
        Auction storage a = auctions[tokenId];
        require(a.seller == msg.sender, "not seller");
        require(a.active, "not active");
        a.active = false;
        nft.transferFrom(address(this), msg.sender, tokenId);
    }

    function setPriceDrop(uint256 newDrop) external onlyOwner {
        priceDropPerSecond = newDrop;
    }

    function withdrawFees() external onlyOwner {
        (bool ok, ) = owner.call{value: address(this).balance}("");
        require(ok, "withdraw failed");
    }

    receive() external payable {}
}