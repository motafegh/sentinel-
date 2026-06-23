// expect: DenialOfService,Reentrancy
// Refund system where each participant receives ETH back via a loop.
// The participants array grows unboundedly with each deposit.
// Once enough users join, the refund loop exceeds block gas limit.
// The contract becomes permanently unusable — funds are stuck.
pragma solidity ^0.8.0;

contract RefundDistributor {
    struct Participant {
        address addr;
        uint256 contribution;
    }

    address public owner;
    Participant[] public participants;
    mapping(address => uint256) public contributed;
    uint256 public totalRaised;
    bool public refundOpen;

    event Contribution(address indexed participant, uint256 amount);
    event RefundStarted(uint256 participantCount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function contribute() external payable {
        require(msg.value > 0, "zero contribution");
        if (contributed[msg.sender] == 0) {
            participants.push(Participant(msg.sender, msg.value));
        } else {
            for (uint256 i = 0; i < participants.length; i++) {
                if (participants[i].addr == msg.sender) {
                    participants[i].contribution += msg.value;
                    break;
                }
            }
        }
        contributed[msg.sender] += msg.value;
        totalRaised += msg.value;
        emit Contribution(msg.sender, msg.value);
    }

    function openRefund() external onlyOwner {
        require(!refundOpen, "already open");
        refundOpen = true;
        emit RefundStarted(participants.length);
    }

    function processRefunds() external onlyOwner {
        require(refundOpen, "refund not open");
        for (uint256 i = 0; i < participants.length; i++) {
            address payable p = payable(participants[i].addr);
            uint256 amount = participants[i].contribution;
            participants[i].contribution = 0;
            (bool ok, ) = p.call{value: amount}("");
            if (!ok) {
                participants[i].contribution = amount;
            }
        }
        refundOpen = false;
    }

    function refundAll() external onlyOwner {
        require(refundOpen, "refund not open");
        uint256 balance = address(this).balance;
        for (uint256 i = 0; i < participants.length; i++) {
            address payable p = payable(participants[i].addr);
            uint256 share = (balance * participants[i].contribution) / totalRaised;
            (bool ok, ) = p.call{value: share}("");
            require(ok, "refund transfer failed");
        }
        refundOpen = false;
    }

    function depositFunds() external payable onlyOwner {}

    receive() external payable {}
}