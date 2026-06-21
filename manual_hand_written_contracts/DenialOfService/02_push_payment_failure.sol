// expect: DenialOfService
// Push-payment dividend system that distributes rewards to all token holders.
// Every distribution iterates the entire holder array and sends ETH.
// A single holder with a malicious fallback (reverting or consuming 100% gas)
// causes the entire distribution to revert — all holders are locked out.
// The contract has no pull-over-push fallback or skip-failing-recipient logic.
pragma solidity ^0.8.0;

contract DividendDistributor {
    address public owner;
    address[] public holders;
    mapping(address => uint256) public holderIndex;
    mapping(address => uint256) public balance;
    uint256 public totalSupply;
    uint256 public accumulatedDividendPerShare;

    event DividendDistributed(uint256 amount);
    event DividendClaimed(address indexed holder, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function registerHolder(address user) internal {
        if (holderIndex[user] == 0 && user != address(0)) {
            holders.push(user);
            holderIndex[user] = holders.length;
        }
    }

    function mint(address to, uint256 amount) external onlyOwner {
        if (balance[to] == 0) {
            holders.push(to);
            holderIndex[to] = holders.length;
        }
        balance[to] += amount;
        totalSupply += amount;
    }

    function distributeDividends() external payable onlyOwner {
        require(msg.value > 0, "no dividend");
        require(totalSupply > 0, "no holders");
        accumulatedDividendPerShare += (msg.value * 1e18) / totalSupply;
        emit DividendDistributed(msg.value);
        for (uint256 i = 0; i < holders.length; i++) {
            address h = holders[i];
            uint256 holderBalance = balance[h];
            if (holderBalance > 0) {
                uint256 payout = (holderBalance * msg.value) / totalSupply;
                (bool ok, ) = payable(h).call{value: payout}("");
                if (!ok) {
                    balance[owner] += payout;
                }
            }
        }
    }

    function distributeProportional() external onlyOwner {
        uint256 contractBalance = address(this).balance;
        require(contractBalance > 0, "no balance");
        require(totalSupply > 0, "no holders");
        for (uint256 i = 0; i < holders.length; i++) {
            address h = holders[i];
            uint256 hBal = balance[h];
            if (hBal > 0) {
                uint256 share = (contractBalance * hBal) / totalSupply;
                (bool ok, ) = payable(h).call{value: share}("");
                require(ok, "distribution failed");
            }
        }
    }

    function claim() external {
        uint256 amount = balance[msg.sender];
        require(amount > 0, "zero balance");
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "claim failed");
    }

    function getHolderCount() external view returns (uint256) {
        return holders.length;
    }
}