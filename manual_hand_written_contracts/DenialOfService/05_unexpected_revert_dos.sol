// expect: DenialOfService
// Airdrop contract that mints tokens to multiple recipients in a single call.
// If any recipient is a contract that reverts on ERC20 transfer,
// the entire airdrop transaction reverts — all recipients fail.
// Attacker can register a blacklisted contract address as a recipient
// to permanently brick the airdrop function.
// Same pattern appears in the batch stake/unstake functions.
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

contract AirDropDistributor {
    address public owner;
    IERC20 public token;
    mapping(address => bool) public blacklisted;
    mapping(address => uint256) public lastAirdropIndex;

    event AirDropExecuted(uint256 recipientCount);
    event BlacklistUpdated(address indexed user, bool status);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _token) {
        owner = msg.sender;
        token = IERC20(_token);
    }

    function setBlacklist(address user, bool status) external onlyOwner {
        blacklisted[user] = status;
        emit BlacklistUpdated(user, status);
    }

    function batchTransfer(address[] calldata recipients, uint256[] calldata amounts) external onlyOwner {
        require(recipients.length == amounts.length, "length mismatch");
        for (uint256 i = 0; i < recipients.length; i++) {
            require(!blacklisted[recipients[i]], "recipient blacklisted");
            bool ok = token.transfer(recipients[i], amounts[i]);
            require(ok, "transfer failed");
            lastAirdropIndex[recipients[i]] = i;
        }
        emit AirDropExecuted(recipients.length);
    }

    function batchTransferFrom(address from, address[] calldata recipients, uint256 amount) external onlyOwner {
        for (uint256 i = 0; i < recipients.length; i++) {
            require(!blacklisted[recipients[i]], "recipient blacklisted");
            bool ok = token.transferFrom(from, recipients[i], amount);
            require(ok, "transferFrom failed");
        }
    }

    function batchMintWithCommission(address[] calldata recipients, uint256[] calldata amounts, address commissionRecipient, uint256 commissionAmount) external onlyOwner {
        require(recipients.length == amounts.length, "length mismatch");
        for (uint256 i = 0; i < recipients.length; i++) {
            bool ok = token.transfer(recipients[i], amounts[i]);
            require(ok, "mint transfer failed");
        }
        if (commissionAmount > 0) {
            bool ok = token.transfer(commissionRecipient, commissionAmount);
            require(ok, "commission transfer failed");
        }
    }

    function multiStageAirdrop(address[] calldata stage1, address[] calldata stage2, uint256 amount) external onlyOwner {
        for (uint256 i = 0; i < stage1.length; i++) {
            bool ok = token.transfer(stage1[i], amount);
            require(ok, "stage1 transfer failed");
        }
        for (uint256 j = 0; j < stage2.length; j++) {
            bool ok = token.transfer(stage2[j], amount);
            require(ok, "stage2 transfer failed");
        }
    }

    function ownerWithdraw(uint256 amount) external onlyOwner {
        token.transfer(owner, amount);
    }

    receive() external payable {}
}