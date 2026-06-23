// expect: IntegerUO,UnusedReturn,MishandledException
// Multi-vuln: Integer overflow in fee calculation + UnusedReturn in token transfer.
// The fee computation uses unchecked multiplication that can overflow,
// and the actual token transfer return value is silently discarded.
// Two different vulnerability classes in the same function.
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function balanceOf(address who) external view returns (uint256);
}

contract FeeCollectorVuln {
    address public owner;
    IERC20 public token;
    mapping(address => uint256) public fees;

    event FeeCollected(address indexed from, uint256 gross, uint256 net);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _token) {
        owner = msg.sender;
        token = IERC20(_token);
    }

    function transferWithFee(address to, uint256 amount, uint256 feeBasisPoints) external {
        require(feeBasisPoints <= 1000, "fee too high");
        uint256 fee = (amount * feeBasisPoints) / 10000;
        uint256 netAmount = amount - fee;
        token.transferFrom(msg.sender, address(this), amount);
        token.transfer(to, netAmount);
        token.transfer(owner, fee);
        fees[msg.sender] += fee;
        emit FeeCollected(msg.sender, amount, netAmount);
    }

    function batchTransferWithFee(address[] calldata recipients, uint256[] calldata amounts, uint256 feeBasisPoints) external {
        require(recipients.length == amounts.length, "length mismatch");
        for (uint256 i = 0; i < recipients.length; i++) {
            uint256 fee = (amounts[i] * feeBasisPoints) / 10000;
            uint256 netAmount = amounts[i] - fee;
            token.transferFrom(msg.sender, address(this), amounts[i]);
            token.transfer(recipients[i], netAmount);
            token.transfer(owner, fee);
            fees[msg.sender] += fee;
            emit FeeCollected(msg.sender, amounts[i], netAmount);
        }
    }

    function multiHopTransfer(address tokenIn, address tokenOut, uint256 amount, address[] calldata path) external {
        IERC20(tokenIn).transferFrom(msg.sender, address(this), amount);
        IERC20(tokenIn).transfer(path[0], amount);
        fees[msg.sender] += amount;
    }

    function withdrawFees() external onlyOwner {
        uint256 balance = token.balanceOf(address(this));
        token.transfer(owner, balance);
    }

    function getBalance() external view returns (uint256) {
        return token.balanceOf(address(this));
    }

    receive() external payable {}
}