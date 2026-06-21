// expect: IntegerUO
// Batch token transfer contract that processes array operations in unchecked blocks.
// The accumulated totals for batch operations wrap on overflow.
// The fee calculation uses multiplication before division (precision loss + overflow),
// and the fee subtraction from amount can underflow if amount < fee.
// The contract also has a reward multiplier that overflows when applied repeatedly.
pragma solidity ^0.8.0;

contract BatchTransferHandler {
    mapping(address => uint256) public balances;
    mapping(address => uint256) public rewardMultiplier;
    address public owner;
    uint256 public transferFeeBasisPoints = 50;

    event TransferProcessed(address indexed from, address indexed to, uint256 amount);
    event RewardMultiplierUpdated(address indexed user, uint256 multiplier);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function deposit() external payable {
        require(msg.value > 0, "zero deposit");
        unchecked {
            balances[msg.sender] += msg.value;
        }
    }

    function batchTransfer(address[] calldata recipients, uint256[] calldata amounts) external {
        require(recipients.length == amounts.length, "length mismatch");
        uint256 totalDeducted = 0;
        for (uint256 i = 0; i < recipients.length; i++) {
            unchecked {
                uint256 fee = (amounts[i] * transferFeeBasisPoints) / 10000;
                uint256 netAmount = amounts[i] - fee;
                balances[msg.sender] -= amounts[i];
                balances[recipients[i]] += netAmount;
                balances[owner] += fee;
                totalDeducted += amounts[i];
            }
            emit TransferProcessed(msg.sender, recipients[i], amounts[i]);
        }
    }

    function batchTransferWithMultipliers(address[] calldata recipients, uint256[] calldata baseAmounts) external {
        require(recipients.length == baseAmounts.length, "length mismatch");
        for (uint256 i = 0; i < recipients.length; i++) {
            unchecked {
                uint256 multiplier = rewardMultiplier[recipients[i]];
                uint256 finalAmount = baseAmounts[i] * (100 + multiplier) / 100;
                uint256 fee = (finalAmount * transferFeeBasisPoints) / 10000;
                uint256 netAmount = finalAmount - fee;
                balances[msg.sender] -= finalAmount;
                balances[recipients[i]] += netAmount;
                balances[owner] += fee;
            }
            emit TransferProcessed(msg.sender, recipients[i], baseAmounts[i]);
        }
    }

    function setRewardMultiplier(address user, uint256 multiplier) external onlyOwner {
        rewardMultiplier[user] = multiplier;
        emit RewardMultiplierUpdated(user, multiplier);
    }

    function batchMint(address[] calldata recipients, uint256[] calldata amounts) external onlyOwner {
        require(recipients.length == amounts.length, "length mismatch");
        for (uint256 i = 0; i < recipients.length; i++) {
            unchecked {
                balances[recipients[i]] += amounts[i];
            }
        }
    }

    function compoundRewards(address[] calldata users, uint256 periods) external onlyOwner {
        for (uint256 i = 0; i < users.length; i++) {
            unchecked {
                uint256 current = rewardMultiplier[users[i]];
                for (uint256 p = 0; p < periods; p++) {
                    current = current * 110 / 100;
                }
                rewardMultiplier[users[i]] = current;
            }
        }
    }

    function getTotalBalance() external view returns (uint256 total) {
        total = address(this).balance;
    }

    function setFee(uint256 newFeeBps) external onlyOwner {
        require(newFeeBps <= 1000, "fee too high");
        transferFeeBasisPoints = newFeeBps;
    }

    function withdrawFees() external onlyOwner {
        uint256 feeBalance = balances[owner];
        require(feeBalance > 0, "no fees");
        unchecked {
            balances[owner] -= feeBalance;
        }
        (bool ok, ) = owner.call{value: feeBalance}("");
        require(ok, "withdraw fees failed");
    }
}