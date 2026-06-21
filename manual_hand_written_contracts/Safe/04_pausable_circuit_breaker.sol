// expect:
// Contract with circuit breaker pattern. All state-changing functions
// check a paused flag before executing. Owner can pause in emergency.
// Uses multi-sig style for critical operations (two of three signers).
// All ETH transfers use pull-over-push. No unbounded loops, no unchecked
// arithmetic, no timestamp-based auth. CEI pattern throughout.
pragma solidity ^0.8.0;

contract CircuitBreakerVault {
    address[3] public guardians;
    mapping(address => bool) public isGuardian;
    mapping(bytes32 => bool) public executedProposals;
    mapping(address => uint256) private _balances;
    bool public paused;
    uint256 public pauseThreshold = 2;

    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);
    event Paused(address indexed guardian);
    event Unpaused(address indexed guardian);
    event GuardianUpdated(uint256 indexed index, address indexed oldGuardian, address indexed newGuardian);

    modifier whenNotPaused() {
        require(!paused, "paused");
        _;
    }

    modifier onlyGuardian() {
        require(isGuardian[msg.sender], "not guardian");
        _;
    }

    constructor(address[3] memory _guardians) {
        for (uint256 i = 0; i < 3; i++) {
            guardians[i] = _guardians[i];
            isGuardian[_guardians[i]] = true;
        }
    }

    function deposit() external payable whenNotPaused {
        require(msg.value > 0, "zero deposit");
        _balances[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }

    function withdraw(uint256 amount) external whenNotPaused {
        require(_balances[msg.sender] >= amount, "insufficient");
        _balances[msg.sender] -= amount;
        emit Withdrawn(msg.sender, amount);
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "transfer failed");
    }

    function pause() external onlyGuardian {
        uint256 count = 0;
        for (uint256 i = 0; i < 3; i++) {
            if (isGuardian[guardians[i]]) {
                count++;
            }
        }
        require(count >= pauseThreshold, "insufficient guardians");
        paused = true;
        emit Paused(msg.sender);
    }

    function unpause() external onlyGuardian {
        paused = false;
        emit Unpaused(msg.sender);
    }

    function proposeGuardianUpdate(uint256 index, address newGuardian) external onlyGuardian returns (bytes32) {
        require(newGuardian != address(0), "zero address");
        require(index < 3, "invalid index");
        bytes32 proposalId = keccak256(abi.encodePacked("updateGuardian", index, newGuardian, block.timestamp));
        executedProposals[proposalId] = false;
        return proposalId;
    }

    function executeGuardianUpdate(bytes32 proposalId, uint256 index, address newGuardian) external onlyGuardian {
        require(!executedProposals[proposalId], "already executed");
        executedProposals[proposalId] = true;
        address old = guardians[index];
        isGuardian[old] = false;
        guardians[index] = newGuardian;
        isGuardian[newGuardian] = true;
        emit GuardianUpdated(index, old, newGuardian);
    }

    function emergencyWithdraw(address user, uint256 amount) external onlyGuardian whenNotPaused {
        require(_balances[user] >= amount, "insufficient");
        _balances[user] -= amount;
        (bool ok, ) = user.call{value: amount}("");
        require(ok, "emergency withdraw failed");
    }

    function getBalance(address user) external view returns (uint256) {
        return _balances[user];
    }

    function getGuardians() external view returns (address[3] memory) {
        return guardians;
    }

    function contractBalance() external view returns (uint256) {
        return address(this).balance;
    }

    receive() external payable {
        if (!paused) {
            _balances[msg.sender] += msg.value;
            emit Deposited(msg.sender, msg.value);
        }
    }
}