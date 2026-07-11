// expect: MishandledException
// Batch payout contract that sends ETH to multiple recipients in a loop.
// The return value of each call() is completely ignored — no check, no revert.
// If a recipient is a contract that reverts on receive, the ETH is silently lost
// and subsequent recipients never get paid because the loop continues past
// the failure. The contract also discards the return values of token transfers.
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

contract SilentBatchPayout {
    address public owner;
    IERC20 public payoutToken;

    event PayoutSent(address indexed recipient, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _token) {
        owner = msg.sender;
        payoutToken = IERC20(_token);
    }

    function batchPayoutEth(address[] calldata recipients, uint256[] calldata amounts) external payable onlyOwner {
        require(recipients.length == amounts.length, "length mismatch");
        for (uint256 i = 0; i < recipients.length; i++) {
            recipients[i].call{value: amounts[i]}("");
            emit PayoutSent(recipients[i], amounts[i]);
        }
    }

    function batchPayoutToken(address[] calldata recipients, uint256[] calldata amounts) external onlyOwner {
        require(recipients.length == amounts.length, "length mismatch");
        for (uint256 i = 0; i < recipients.length; i++) {
            payoutToken.transfer(recipients[i], amounts[i]);
            emit PayoutSent(recipients[i], amounts[i]);
        }
    }

    function batchPayoutMixed(
        address[] calldata recipients,
        uint256[] calldata ethAmounts,
        uint256[] calldata tokenAmounts
    ) external payable onlyOwner {
        require(recipients.length == ethAmounts.length, "eth length mismatch");
        require(recipients.length == tokenAmounts.length, "token length mismatch");
        for (uint256 i = 0; i < recipients.length; i++) {
            if (ethAmounts[i] > 0) {
                recipients[i].call{value: ethAmounts[i]}("");
            }
            if (tokenAmounts[i] > 0) {
                payoutToken.transfer(recipients[i], tokenAmounts[i]);
            }
            emit PayoutSent(recipients[i], ethAmounts[i] + tokenAmounts[i]);
        }
    }

    function payoutAll(address[] calldata recipients) external onlyOwner {
        uint256 perRecipient = address(this).balance / recipients.length;
        uint256 perToken = payoutToken.balanceOf(address(this)) / recipients.length;
        for (uint256 i = 0; i < recipients.length; i++) {
            recipients[i].call{value: perRecipient}("");
            payoutToken.transfer(recipients[i], perToken);
        }
    }

    function multiCall(address[] calldata targets, bytes[] calldata calldatas) external payable onlyOwner {
        require(targets.length == calldatas.length, "length mismatch");
        for (uint256 i = 0; i < targets.length; i++) {
            targets[i].call{value: address(this).balance / (targets.length - i)}(calldatas[i]);
        }
    }

    function withdrawToOwner() external onlyOwner {
        owner.call{value: address(this).balance}("");
        payoutToken.transfer(owner, payoutToken.balanceOf(address(this)));
    }

    receive() external payable {}
}