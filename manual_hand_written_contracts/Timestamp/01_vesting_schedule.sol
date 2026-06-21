// expect: Timestamp
// Token vesting contract that uses block.timestamp for cliff and release calculations.
// A miner can manipulate the timestamp by up to ~15 seconds to influence whether
// a cliff is reached in a given block. The vesting schedule computes released
// amounts based on elapsed time — timestamp manipulation shifts the proportion.
// The contract also uses timestamp for a lockup penalty calculation.
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

contract TimestampVesting {
    struct Grant {
        address beneficiary;
        uint256 totalAmount;
        uint256 cliffDuration;
        uint256 vestingDuration;
        uint256 startTime;
        uint256 released;
        bool revocable;
    }

    address public owner;
    IERC20 public token;
    Grant[] public grants;
    mapping(address => uint256[]) public beneficiaryGrants;

    event GrantCreated(uint256 indexed grantId, address indexed beneficiary, uint256 amount);
    event TokensReleased(uint256 indexed grantId, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _token) {
        owner = msg.sender;
        token = IERC20(_token);
    }

    function createGrant(address beneficiary, uint256 totalAmount, uint256 cliffDuration, uint256 vestingDuration, bool revocable) external onlyOwner returns (uint256) {
        uint256 id = grants.length;
        grants.push(Grant(beneficiary, totalAmount, cliffDuration, vestingDuration, block.timestamp, 0, revocable));
        beneficiaryGrants[beneficiary].push(id);
        token.transferFrom(msg.sender, address(this), totalAmount);
        emit GrantCreated(id, beneficiary, totalAmount);
        return id;
    }

    function release(uint256 grantId) external {
        Grant storage g = grants[grantId];
        require(msg.sender == g.beneficiary || msg.sender == owner, "not authorized");
        uint256 elapsed = block.timestamp - g.startTime;
        uint256 releasable;
        if (elapsed < g.cliffDuration) {
            releasable = 0;
        } else if (elapsed >= g.vestingDuration) {
            releasable = g.totalAmount - g.released;
        } else {
            releasable = (g.totalAmount * elapsed) / g.vestingDuration - g.released;
        }
        require(releasable > 0, "nothing to release");
        g.released += releasable;
        token.transfer(g.beneficiary, releasable);
        emit TokensReleased(grantId, releasable);
    }

    function batchRelease(address beneficiary) external {
        uint256[] storage ids = beneficiaryGrants[beneficiary];
        for (uint256 i = 0; i < ids.length; i++) {
            Grant storage g = grants[ids[i]];
            uint256 elapsed = block.timestamp - g.startTime;
            uint256 releasable;
            if (elapsed < g.cliffDuration) {
                releasable = 0;
            } else if (elapsed >= g.vestingDuration) {
                releasable = g.totalAmount - g.released;
            } else {
                releasable = (g.totalAmount * elapsed) / g.vestingDuration - g.released;
            }
            if (releasable > 0) {
                g.released += releasable;
                token.transfer(beneficiary, releasable);
                emit TokensReleased(ids[i], releasable);
            }
        }
    }

    function revoke(uint256 grantId) external onlyOwner {
        Grant storage g = grants[grantId];
        require(g.revocable, "not revocable");
        uint256 remaining = g.totalAmount - g.released;
        g.totalAmount = g.released;
        token.transfer(owner, remaining);
    }

    function getVestedAmount(uint256 grantId) external view returns (uint256) {
        Grant storage g = grants[grantId];
        uint256 elapsed = block.timestamp - g.startTime;
        if (elapsed < g.cliffDuration) return 0;
        if (elapsed >= g.vestingDuration) return g.totalAmount;
        return (g.totalAmount * elapsed) / g.vestingDuration;
    }
}