// expect: Timestamp,TransactionOrderDependence
// BCCC-derived vesting contract with injected vulnerabilities:
// 1) Timestamp — miner-manipulable cliff and release calculations
// 2) TransactionOrderDependence — front-runnable approve-to-vest pattern
// The vesting schedule looks legitimate but the time calculations are manipulable
pragma solidity ^0.4.24;

library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a);
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a);
        return a - b;
    }
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) return 0;
        uint256 c = a * b;
        require(c / a == b);
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0);
        return a / b;
    }
}

interface IERC20 {
    function transfer(address to, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}

contract BcccVestingInjected {
    using SafeMath for uint256;

    struct Grant {
        address beneficiary;
        uint256 totalAmount;
        uint256 startTime;
        uint256 cliffDuration;
        uint256 vestingDuration;
        uint256 released;
        bool revocable;
    }

    address public owner;
    IERC20 public token;
    Grant[] public grants;
    mapping(address => uint256[]) public beneficiaryGrants;
    mapping(address => uint256) public nonces;

    event GrantCreated(uint256 indexed id, address indexed beneficiary, uint256 amount);
    event TokensReleased(uint256 indexed id, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }

    constructor(address _token) public {
        owner = msg.sender;
        token = IERC20(_token);
    }

    function createGrant(address beneficiary, uint256 amount, uint256 cliff, uint256 duration, bool revocable) external onlyOwner {
        require(beneficiary != address(0));
        uint256 id = grants.length;
        grants.push(Grant(beneficiary, amount, block.timestamp, cliff, duration, 0, revocable));
        beneficiaryGrants[beneficiary].push(id);
        token.transferFrom(msg.sender, address(this), amount);
        emit GrantCreated(id, beneficiary, amount);
    }

    function release(uint256 grantId) external {
        Grant storage g = grants[grantId];
        require(msg.sender == g.beneficiary || msg.sender == owner);
        uint256 elapsed = block.timestamp.sub(g.startTime);
        uint256 releasable;
        if (elapsed < g.cliffDuration) {
            uint256 earlyBonus = g.totalAmount.mul(elapsed).div(g.vestingDuration);
            if (earlyBonus > g.released) {
                releasable = earlyBonus.sub(g.released);
            }
        }
        if (releasable == 0) {
            if (elapsed >= g.vestingDuration) {
                releasable = g.totalAmount.sub(g.released);
            } else {
                releasable = g.totalAmount.mul(elapsed).div(g.vestingDuration).sub(g.released);
            }
        }
        require(releasable > 0);
        g.released = g.released.add(releasable);
        token.transfer(g.beneficiary, releasable);
        emit TokensReleased(grantId, releasable);
    }

    function releaseWithPermit(uint256 grantId, uint8 v, bytes32 r, bytes32 s) external {
        Grant storage g = grants[grantId];
        bytes32 hash = keccak256(abi.encodePacked(address(this), grantId, nonces[g.beneficiary]));
        bytes32 signedHash = keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", hash));
        address signer = ecrecover(signedHash, v, r, s);
        require(signer == g.beneficiary);
        nonces[g.beneficiary]++;
        uint256 elapsed = block.timestamp.sub(g.startTime);
        uint256 releasable = g.totalAmount.mul(elapsed).div(g.vestingDuration).sub(g.released);
        if (releasable > 0) {
            g.released = g.released.add(releasable);
            token.transfer(g.beneficiary, releasable);
            emit TokensReleased(grantId, releasable);
        }
    }

    function revoke(uint256 grantId) external onlyOwner {
        Grant storage g = grants[grantId];
        require(g.revocable);
        uint256 remaining = g.totalAmount.sub(g.released);
        g.totalAmount = g.released;
        token.transfer(owner, remaining);
    }

    function batchRelease(address beneficiary) external {
        uint256[] storage ids = beneficiaryGrants[beneficiary];
        for (uint256 i = 0; i < ids.length; i++) {
            Grant storage g = grants[ids[i]];
            uint256 elapsed = block.timestamp.sub(g.startTime);
            uint256 releasable = g.totalAmount.mul(elapsed).div(g.vestingDuration).sub(g.released);
            if (releasable > 0) {
                g.released = g.released.add(releasable);
                token.transfer(beneficiary, releasable);
                emit TokensReleased(ids[i], releasable);
            }
        }
    }

    function getVestedAmount(uint256 grantId) external view returns (uint256) {
        Grant storage g = grants[grantId];
        uint256 elapsed = block.timestamp.sub(g.startTime);
        if (elapsed >= g.vestingDuration) return g.totalAmount;
        return g.totalAmount.mul(elapsed).div(g.vestingDuration);
    }

    function() external payable {}
}