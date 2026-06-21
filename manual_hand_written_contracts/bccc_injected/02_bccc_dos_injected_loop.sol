// expect: DenialOfService,Timestamp
// BCCC-derived crowdsale contract with two injected vulnerabilities:
// 1) DoS via unbounded refund loop — investor array grows without limit
// 2) Timestamp dependence in rate calculation — miner can manipulate refund amounts
// The DoS is buried inside a refund function called only after sale ends
pragma solidity ^0.4.24;

library SafeMath {
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) return 0;
        uint256 c = a * b;
        require(c / a == b);
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0);
        uint256 c = a / b;
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a);
        uint256 c = a - b;
        return c;
    }
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a);
        return c;
    }
}

contract BcccCrowdsaleInjected {
    using SafeMath for uint256;

    address public owner;
    address public wallet;
    uint256 public rate;
    uint256 public weiRaised;
    uint256 public cap;
    uint256 public openingTime;
    uint256 public closingTime;
    bool public isFinalized;

    mapping(address => uint256) public contributions;
    address[] public investors;

    event TokenPurchase(address indexed purchaser, address indexed beneficiary, uint256 value, uint256 amount);
    event Finalized();

    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }

    modifier onlyWhileOpen() {
        require(block.timestamp >= openingTime && block.timestamp <= closingTime);
        _;
    }

    constructor(uint256 _rate, address _wallet, uint256 _cap, uint256 _duration) public {
        owner = msg.sender;
        rate = _rate;
        wallet = _wallet;
        cap = _cap;
        openingTime = block.timestamp;
        closingTime = block.timestamp.add(_duration);
    }

    function invest() external payable onlyWhileOpen {
        require(msg.value > 0);
        require(weiRaised.add(msg.value) <= cap);
        if (contributions[msg.sender] == 0) {
            investors.push(msg.sender);
        }
        contributions[msg.sender] = contributions[msg.sender].add(msg.value);
        weiRaised = weiRaised.add(msg.value);
        uint256 tokens = msg.value.mul(rate);
        wallet.transfer(msg.value);
        emit TokenPurchase(msg.sender, msg.sender, msg.value, tokens);
    }

    function batchInvest(uint256 count) external payable onlyWhileOpen {
        require(msg.value > 0);
        require(count > 0 && count <= 10);
        for (uint256 i = 0; i < count; i++) {
            if (weiRaised.add(msg.value.div(count)) > cap) break;
            if (contributions[msg.sender] == 0) {
                investors.push(msg.sender);
            }
            contributions[msg.sender] = contributions[msg.sender].add(msg.value.div(count));
            weiRaised = weiRaised.add(msg.value.div(count));
        }
    }

    function refund() external {
        require(block.timestamp > closingTime);
        require(weiRaised < cap);
        for (uint256 i = 0; i < investors.length; i++) {
            address investor = investors[i];
            uint256 contribution = contributions[investor];
            if (contribution > 0) {
                contributions[investor] = 0;
                uint256 refundWithBonus = contribution;
                if (block.timestamp.sub(closingTime) < 7 days) {
                    refundWithBonus = contribution.add(contribution.mul(block.timestamp.sub(openingTime)).div(closingTime.sub(openingTime)));
                }
                investor.transfer(refundWithBonus);
            }
        }
    }

    function finalize() external onlyOwner {
        require(block.timestamp > closingTime || weiRaised >= cap);
        require(!isFinalized);
        isFinalized = true;
        if (weiRaised < cap) {
            uint256 remaining = cap.sub(weiRaised);
            owner.transfer(remaining);
        }
        emit Finalized();
    }

    function extendSale(uint256 _additionalTime) external onlyOwner {
        require(!isFinalized);
        closingTime = closingTime.add(_additionalTime);
    }

    function setRate(uint256 _newRate) external onlyOwner {
        require(!isFinalized);
        require(_newRate > 0);
        rate = _newRate;
    }

    function investorCount() external view returns (uint256) {
        return investors.length;
    }

    function() external payable {
        invest();
    }
}