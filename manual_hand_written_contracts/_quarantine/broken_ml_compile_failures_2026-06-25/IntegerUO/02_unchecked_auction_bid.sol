// expect: IntegerUO
// Blind auction contract where bids are processed in an unchecked block.
// The refund calculation for losing bidders uses unchecked subtraction.
// If a bidder's balance in the contract is manipulated (e.g., via reentrancy
// or insufficient tracking), the refund calculation can underflow.
// The totalBid tracking also wraps on overflow from accumulated bids.
pragma solidity ^0.8.0;

contract UncheckedAuction {
    address public beneficiary;
    uint256 public auctionEnd;
    uint256 public highestBid;
    address public highestBidder;
    mapping(address => uint256) public pendingReturns;
    mapping(address => bytes32) public sealedBids;
    mapping(address => uint256) public bidDeposits;
    uint256 public totalBids;
    bool public ended;

    event AuctionEnded(address winner, uint256 highestBid);

    modifier notEnded() {
        require(!ended, "auction ended");
        _;
    }

    constructor(uint256 _biddingTime, address _beneficiary) {
        beneficiary = _beneficiary;
        auctionEnd = block.timestamp + _biddingTime;
    }

    function bid(bytes32 _blindedBid) external payable notEnded {
        require(block.timestamp < auctionEnd, "auction ended");
        require(msg.value > 0, "zero bid");
        sealedBids[msg.sender] = _blindedBid;
        bidDeposits[msg.sender] += msg.value;
        unchecked {
            totalBids += msg.value;
        }
    }

    function reveal(string[] calldata _values, bool[] calldata _fake) external notEnded {
        require(block.timestamp >= auctionEnd, "not ended yet");
        uint256 length = _values.length;
        for (uint256 i = 0; i < length; i++) {
            bytes32 sealed = sealedBids[msg.sender];
            bytes32 check = keccak256(abi.encodePacked(_values[i], _fake[i]));
            require(sealed == check, "bid mismatch");
            uint256 value = stringToUint(_values[i]);
            if (!_fake[i] && value > highestBid) {
                unchecked {
                    pendingReturns[highestBidder] += highestBid;
                }
                highestBid = value;
                highestBidder = msg.sender;
            } else {
                unchecked {
                    pendingReturns[msg.sender] += bidDeposits[msg.sender];
                }
            }
            delete sealedBids[msg.sender];
        }
    }

    function stringToUint(string memory s) internal pure returns (uint256) {
        bytes memory b = bytes(s);
        uint256 result = 0;
        for (uint256 i = 0; i < b.length; i++) {
            if (b[i] >= 0x30 && b[i] <= 0x39) {
                unchecked {
                    result = result * 10 + (uint256(uint8(b[i])) - 48);
                }
            }
        }
        return result;
    }

    function withdraw() external returns (bool) {
        uint256 amount = pendingReturns[msg.sender];
        if (amount > 0) {
            pendingReturns[msg.sender] = 0;
            if (!payable(msg.sender).send(amount)) {
                pendingReturns[msg.sender] = amount;
                return false;
            }
        }
        return true;
    }

    function auctionEnd() external {
        require(block.timestamp >= auctionEnd, "auction not ended");
        require(!ended, "already ended");
        ended = true;
        emit AuctionEnded(highestBidder, highestBid);
        payable(beneficiary).transfer(highestBid);
    }

    function batchWithdraw(address[] calldata bidders) external returns (bool[] memory results) {
        results = new bool[](bidders.length);
        for (uint256 i = 0; i < bidders.length; i++) {
            uint256 amount = pendingReturns[bidders[i]];
            if (amount > 0) {
                pendingReturns[bidders[i]] = 0;
                results[i] = payable(bidders[i]).send(amount);
                if (!results[i]) {
                    pendingReturns[bidders[i]] = amount;
                }
            }
        }
    }
}