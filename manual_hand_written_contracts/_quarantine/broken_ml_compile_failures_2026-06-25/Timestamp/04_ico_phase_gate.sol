// expect: Timestamp
// ICO contract with phase-based pricing that uses block.timestamp for
// phase transitions. The price increases each phase, but a miner who
// controls block.timestamp can keep the ICO in an earlier cheaper phase
// for longer, or rush it into a later phase to disadvantage buyers.
// Vesting for team tokens also relies on block.timestamp for cliff unlocks.
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

contract ICOPhaseManager {
    struct Phase {
        uint256 startTime;
        uint256 endTime;
        uint256 pricePerToken;
        uint256 cap;
        uint256 sold;
    }

    address public owner;
    IERC20 public token;
    Phase[] public phases;
    mapping(address => uint256) public purchased;
    uint256 public totalSold;
    bool public finalized;

    event TokensPurchased(address indexed buyer, uint256 amount, uint256 phase);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _token) {
        owner = msg.sender;
        token = IERC20(_token);
    }

    function addPhase(uint256 startTime, uint256 endTime, uint256 price, uint256 cap) external onlyOwner {
        require(startTime < endTime, "invalid times");
        require(price > 0, "zero price");
        phases.push(Phase(startTime, endTime, price, cap, 0));
    }

    function buyTokens(uint256 phaseId, uint256 tokenAmount) external payable {
        require(phaseId < phases.length, "invalid phase");
        Phase storage p = phases[phaseId];
        require(block.timestamp >= p.startTime, "phase not started");
        require(block.timestamp <= p.endTime, "phase ended");
        require(p.sold + tokenAmount <= p.cap, "cap exceeded");
        uint256 cost = tokenAmount * p.pricePerToken;
        require(msg.value >= cost, "insufficient payment");
        p.sold += tokenAmount;
        totalSold += tokenAmount;
        purchased[msg.sender] += tokenAmount;
        token.transfer(msg.sender, tokenAmount);
        emit TokensPurchased(msg.sender, tokenAmount, phaseId);
        uint256 excess = msg.value - cost;
        if (excess > 0) {
            (bool ok, ) = msg.sender.call{value: excess}("");
            require(ok, "refund failed");
        }
    }

    function getCurrentPhase() external view returns (int256 phaseId, uint256 price) {
        for (uint256 i = 0; i < phases.length; i++) {
            if (block.timestamp >= phases[i].startTime && block.timestamp <= phases[i].endTime) {
                return (int256(i), phases[i].pricePerToken);
            }
        }
        if (block.timestamp < phases[0].startTime) {
            return (-1, 0);
        }
        return (-2, 0);
    }

    function finalizeICO() external onlyOwner {
        require(!finalized, "already finalized");
        require(block.timestamp > phases[phases.length - 1].endTime, "ico not ended");
        finalized = true;
        uint256 remaining = token.balanceOf(address(this));
        if (remaining > 0) {
            token.transfer(owner, remaining);
        }
        uint256 ethBalance = address(this).balance;
        if (ethBalance > 0) {
            (bool ok, ) = owner.call{value: ethBalance}("");
            require(ok, "eth transfer failed");
        }
    }

    function extendPhase(uint256 phaseId, uint256 newEndTime) external onlyOwner {
        require(phaseId < phases.length, "invalid phase");
        require(newEndTime > phases[phaseId].endTime, "must extend");
        phases[phaseId].endTime = newEndTime;
    }

    function getPhaseInfo(uint256 phaseId) external view returns (uint256 start, uint256 end, uint256 price, uint256 cap, uint256 sold) {
        Phase storage p = phases[phaseId];
        return (p.startTime, p.endTime, p.pricePerToken, p.cap, p.sold);
    }

    receive() external payable {}
}