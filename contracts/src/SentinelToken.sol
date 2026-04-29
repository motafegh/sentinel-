// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

// RECALL - ERC20 gives us transfer/approve/balanceOf for free.
// Ownable gives us onlyOwner modifier for slash() -
// only the protocol deployer can punish dishonest agents.
contract SentinelToken is ERC20, Ownable {

    // RECALL - 1000 tokens with 18 decimals.
    // AuditRegistry reads this constant to check agent eligibility.
    uint256 public constant MIN_STAKE = 1000 * 10 ** 18;

    mapping(address => uint256) private _stakedBalances;

    event Staked(address indexed agent, uint256 amount);
    event Unstaked(address indexed agent, uint256 amount);
    event Slashed(address indexed agent, uint256 amount);

    constructor() ERC20("Sentinel", "SNTL") Ownable(msg.sender) {
        _mint(msg.sender, 1_000_000 * 10 ** 18);
    }

    // RECALL - Locks tokens into this contract as collateral.
    function stake(uint256 amount) external {
        require(amount > 0, "SentinelToken: amount must be > 0");
        _stakedBalances[msg.sender] += amount;
        _transfer(msg.sender, address(this), amount);
        emit Staked(msg.sender, amount);
    }

    // RECALL - Returns locked tokens back to agent.
    function unstake(uint256 amount) external {
        require(
            _stakedBalances[msg.sender] >= amount,
            "SentinelToken: insufficient staked balance"
        );
        _stakedBalances[msg.sender] -= amount;
        _transfer(address(this), msg.sender, amount);
        emit Unstaked(msg.sender, amount);
    }

    // RECALL - onlyOwner = protocol controls slashing in MVP.
    // _burn removes slashed tokens permanently - economically meaningful.
    function slash(address agent, uint256 amount) external onlyOwner {
        require(
            _stakedBalances[agent] >= amount,
            "SentinelToken: insufficient staked balance to slash"
        );
        _stakedBalances[agent] -= amount;
        _burn(address(this), amount);
        emit Slashed(agent, amount);
    }

    // RECALL - AuditRegistry calls this to check agent eligibility.
    function stakedBalance(address agent) external view returns (uint256) {
        return _stakedBalances[agent];
    }
}
