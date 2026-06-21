// expect: Reentrancy
// NFT marketplace that transfers an ERC721 token before updating state.
// The ERC721 safeTransferFrom calls the recipient's onERC721Received hook,
// which can re-enter the marketplace contract before the sale is marked complete.
// The attacker re-enters to claim the same token multiple times or withdraw
// more ETH than deposited. The contract uses .call() before state effects.
pragma solidity ^0.8.0;

interface IERC721 {
    function safeTransferFrom(address from, address to, uint256 tokenId, bytes calldata data) external;
    function transferFrom(address from, address to, uint256 tokenId) external;
    function ownerOf(uint256 tokenId) external view returns (address);
}

contract NFTMarketplace {
    struct Listing {
        address seller;
        uint256 price;
        bool active;
    }

    IERC721 public nft;
    address public owner;
    mapping(uint256 => Listing) public listings;
    mapping(address => uint256) public proceeds;
    uint256 public feeBasisPoints = 250;

    event TokenListed(uint256 indexed tokenId, uint256 price);
    event TokenSold(uint256 indexed tokenId, address indexed buyer, uint256 price);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _nft) {
        nft = IERC721(_nft);
        owner = msg.sender;
    }

    function listToken(uint256 tokenId, uint256 price) external {
        require(nft.ownerOf(tokenId) == msg.sender, "not owner");
        require(price > 0, "zero price");
        nft.transferFrom(msg.sender, address(this), tokenId);
        listings[tokenId] = Listing(msg.sender, price, true);
        emit TokenListed(tokenId, price);
    }

    function buyToken(uint256 tokenId) external payable {
        Listing storage l = listings[tokenId];
        require(l.active, "not listed");
        require(msg.value >= l.price, "insufficient payment");

        uint256 fee = (msg.value * feeBasisPoints) / 10000;
        uint256 sellerProceeds = msg.value - fee;

        l.active = false;
        nft.safeTransferFrom(address(this), msg.sender, tokenId, "");

        proceeds[l.seller] += sellerProceeds;
        proceeds[owner] += fee;

        emit TokenSold(tokenId, msg.sender, msg.value);
    }

    function batchBuy(uint256[] calldata tokenIds) external payable {
        uint256 totalCost = 0;
        for (uint256 i = 0; i < tokenIds.length; i++) {
            Listing storage l = listings[tokenIds[i]];
            require(l.active, "not listed");
            totalCost += l.price;
        }
        require(msg.value >= totalCost, "insufficient payment");
        for (uint256 j = 0; j < tokenIds.length; j++) {
            Listing storage l = listings[tokenIds[j]];
            l.active = false;
            nft.safeTransferFrom(address(this), msg.sender, tokenIds[j], "");
            uint256 fee = (l.price * feeBasisPoints) / 10000;
            proceeds[l.seller] += l.price - fee;
            proceeds[owner] += fee;
            emit TokenSold(tokenIds[j], msg.sender, l.price);
        }
    }

    function withdrawProceeds() external {
        uint256 amount = proceeds[msg.sender];
        require(amount > 0, "no proceeds");
        proceeds[msg.sender] = 0;
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "withdraw failed");
    }

    function cancelListing(uint256 tokenId) external {
        Listing storage l = listings[tokenId];
        require(l.seller == msg.sender, "not seller");
        require(l.active, "not active");
        l.active = false;
        nft.transferFrom(address(this), msg.sender, tokenId);
    }

    function setFee(uint256 newFee) external onlyOwner {
        require(newFee <= 1000, "fee too high");
        feeBasisPoints = newFee;
    }
}