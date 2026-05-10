// expect: IntegerUO
// Solidity 0.8.x unchecked blocks disable overflow protection explicitly.
// Equivalent pattern to pre-0.8 arithmetic — overflow/underflow can occur.
pragma solidity ^0.8.0;

contract UncheckedToken {
    mapping(address => uint256) public balanceOf;
    uint256 public totalSupply;
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    function mint(address to, uint256 amount) external {
        require(msg.sender == owner);
        unchecked {
            // VULNERABILITY: unchecked addition — totalSupply wraps on overflow
            totalSupply += amount;
            balanceOf[to] += amount;
        }
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        unchecked {
            // VULNERABILITY: unchecked subtraction — underflows if amount > balance
            balanceOf[msg.sender] -= amount;
            balanceOf[to] += amount;
        }
        return true;
    }

    function burnAll() external {
        unchecked {
            // VULNERABILITY: totalSupply can wrap below zero becoming uint max
            totalSupply -= balanceOf[msg.sender];
        }
        balanceOf[msg.sender] = 0;
    }
}
