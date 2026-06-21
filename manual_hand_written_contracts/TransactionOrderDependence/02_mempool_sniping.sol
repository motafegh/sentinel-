// expect: TransactionOrderDependence
// NFT listing contract where the seller puts a token up for sale at a fixed price.
// The buy function is visible in the mempool — a front-running bot can see the
// buy transaction and submit their own with higher gas to buy the token first.
// The seller also has a cancel function that can be front-run to prevent cancellation.
// The first transaction to confirm gets the asset, regardless of intent.
pragma solidity ^0.8.0;

interface IERC721 {
    function transferFrom(address from, address to, uint256 tokenId) external;
    function ownerOf(uint256 tokenId) external view returns (address);
}

contract NFTSnipeMarket {
    struct Listing {
        address seller;
        uint256 price;
        uint256 createdAt;
    }

    IERC721 public nft;
    address public owner;
    mapping(uint256 => Listing) public listings;
    uint256 public listingFee = 0.01 ether;

    event TokenListed(uint256 indexed tokenId, uint256 price);
    event TokenBought(uint256 indexed tokenId, address indexed buyer, uint256 price);
    event ListingCanceled(uint256 indexed tokenId);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _nft) {
        nft = IERC721(_nft);
        owner = msg.sender;
    }

    function list(uint256 tokenId, uint256 price) external payable {
        require(msg.value >= listingFee, "listing fee required");
        require(nft.ownerOf(tokenId) == msg.sender, "not owner");
        require(price > 0, "zero price");
        nft.transferFrom(msg.sender, address(this), tokenId);
        listings[tokenId] = Listing(msg.sender, price, block.timestamp);
        emit TokenListed(tokenId, price);
    }

    function buy(uint256 tokenId) external payable {
        Listing storage l = listings[tokenId];
        require(l.price > 0, "not listed");
        require(msg.value >= l.price, "insufficient payment");

        uint256 excess = msg.value - l.price;
        delete listings[tokenId];
        nft.transferFrom(address(this), msg.sender, tokenId);

        (bool okSeller, ) = l.seller.call{value: l.price}("");
        require(okSeller, "seller payment failed");

        if (excess > 0) {
            (bool okRefund, ) = msg.sender.call{value: excess}("");
            require(okRefund, "refund failed");
        }

        emit TokenBought(tokenId, msg.sender, msg.value);
    }

    function batchBuy(uint256[] calldata tokenIds) external payable {
        uint256 totalCost = 0;
        for (uint256 i = 0; i < tokenIds.length; i++) {
            Listing storage l = listings[tokenIds[i]];
            require(l.price > 0, "not listed");
            totalCost += l.price;
        }
        require(msg.value >= totalCost, "insufficient payment");
        for (uint256 j = 0; j < tokenIds.length; j++) {
            Listing storage l = listings[tokenIds[j]];
            delete listings[tokenIds[j]];
            nft.transferFrom(address(this), msg.sender, tokenIds[j]);
            (bool ok, ) = l.seller.call{value: l.price}("");
            require(ok, "batch seller payment failed");
            emit TokenBought(tokenIds[j], msg.sender, l.price);
        }
    }

    function cancel(uint256 tokenId) external {
        Listing storage l = listings[tokenId];
        require(l.seller == msg.sender, "not seller");
        delete listings[tokenId];
        nft.transferFrom(address(this), msg.sender, tokenId);
        emit ListingCanceled(tokenId);
    }

    function buyWithPermit(uint256 tokenId, uint256 maxPrice, bytes calldata permitData) external payable {
        Listing storage l = listings[tokenId];
        require(l.price > 0, "not listed");
        require(msg.value >= l.price, "insufficient payment");
        require(l.price <= maxPrice, "price exceeded max");

        delete listings[tokenId];
        nft.transferFrom(address(this), msg.sender, tokenId);
        (bool ok, ) = l.seller.call{value: l.price}("");
        require(ok, "seller payment failed");
        emit TokenBought(tokenId, msg.sender, l.price);
    }

    function setListingFee(uint256 newFee) external onlyOwner {
        listingFee = newFee;
    }

    function withdrawFees() external onlyOwner {
        (bool ok, ) = owner.call{value: address(this).balance}("");
        require(ok, "fee withdraw failed");
    }
}