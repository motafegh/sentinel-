// expect: DenialOfService
// TRICKY: DoS vulnerability that activates from the constructor.
// The constructor pushes a large number of entries to an array, making
// any function that iterates over the array cost more gas than the block limit.
// The contract deploys fine but the first call to cleanup() fails permanently.
// The vulnerability is in the constructor — invisible to source analysis that
// skips constructor code.
pragma solidity ^0.8.0;

contract ConstructorDoSBomb {
    struct Registration {
        address user;
        uint256 timestamp;
        bytes32 data;
        uint256 value;
    }

    address public owner;
    Registration[] public registrations;
    mapping(address => bool) public blacklisted;
    uint256 public constant MAX_REGISTRATIONS = 10000;
    bool public cleaned;

    event Registered(address indexed user);
    event Cleaned(uint256 count);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(uint256 bombSize) {
        owner = msg.sender;
        for (uint256 i = 0; i < bombSize; i++) {
            registrations.push(Registration(address(uint160(i)), block.timestamp, keccak256(abi.encodePacked(i)), i));
        }
    }

    function register() external {
        require(registrations.length < MAX_REGISTRATIONS, "max registrations");
        require(!blacklisted[msg.sender], "blacklisted");
        registrations.push(Registration(msg.sender, block.timestamp, keccak256(abi.encodePacked(msg.sender, block.timestamp)), msg.value));
        emit Registered(msg.sender);
    }

    function cleanup() external onlyOwner {
        require(!cleaned, "already cleaned");
        uint256 count = 0;
        for (uint256 i = 0; i < registrations.length; i++) {
            address user = registrations[i].user;
            if (blacklisted[user] || registrations[i].value == 0) {
                delete registrations[i];
                count++;
            }
        }
        for (uint256 j = 0; j < registrations.length; j++) {
            for (uint256 k = j + 1; k < registrations.length; k++) {
                if (registrations[j].user == address(0) && registrations[k].user != address(0)) {
                    registrations[j] = registrations[k];
                    delete registrations[k];
                }
            }
        }
        cleaned = true;
        emit Cleaned(count);
    }

    function computeStats() external view returns (uint256 totalUsers, uint256 totalValue, uint256 uniqueUsers) {
        totalUsers = registrations.length;
        for (uint256 i = 0; i < registrations.length; i++) {
            totalValue += registrations[i].value;
            bool unique = true;
            for (uint256 j = 0; j < i; j++) {
                if (registrations[j].user == registrations[i].user) {
                    unique = false;
                    break;
                }
            }
            if (unique) uniqueUsers++;
        }
    }

    function getRegistrationCount() external view returns (uint256) {
        return registrations.length;
    }

    function blacklistUser(address user) external onlyOwner {
        blacklisted[user] = true;
    }

    function removeBlacklist(address user) external onlyOwner {
        blacklisted[user] = false;
    }

    function withdrawFunds() external onlyOwner {
        (bool ok, ) = owner.call{value: address(this).balance}("");
        require(ok, "withdraw failed");
    }

    receive() external payable {}
}