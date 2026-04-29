// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

// RECALL — ERC20 gives us transfer/approve/balanceOf for free.
// Ownable gives us onlyOwner modifier for slash() —
// only the protocol deployer can punish dishonest agents.
contract SentinelToken is ERC20, Ownable {

    // RECALL — 1000 tokens with 18 decimals (ERC-20 standard precision).
    // AuditRegistry reads this constant to check agent eligibility.
    uint256 public constant MIN_STAKE = 1000 * 10 ** 18;

    // RECALL — Staked tokens are still "owned" by the agent but locked
    // in this contract. Separate from balanceOf — staked tokens cannot
    // be transferred until unstaked.
    mapping(address => uint256) private _stakedBalances;

    event Staked(address indexed agent, uint256 amount);
    event Unstaked(address indexed agent, uint256 amount);
    event Slashed(address indexed agent, uint256 amount);

    // RECALL — ERC20 constructor sets name + symbol.
    // Ownable(msg.sender) sets deployer as owner automatically.
    // _mint gives deployer initial supply to distribute to agents.
    constructor() ERC20("Sentinel", "SNTL") Ownable(msg.sender) {
        _mint(msg.sender, 1_000_000 * 10 ** 18);
    }

    // RECALL — Locks tokens into this contract as collateral.
    // _transfer moves tokens from agent wallet → contract address.
    // Agent must have enough balance — _transfer reverts if not.
    function stake(uint256 amount) external {
        require(amount > 0, "SentinelToken: amount must be > 0");
        _stakedBalances[msg.sender] += amount;
        _transfer(msg.sender, address(this), amount);
        emit Staked(msg.sender, amount);
    }

    // RECALL — Returns locked tokens back to agent.
    // Check staked balance first — can't unstake more than locked.
    // _transfer moves tokens from contract address → agent wallet.
    function unstake(uint256 amount) external {
        require(
            _stakedBalances[msg.sender] >= amount,
            "SentinelToken: insufficient staked balance"
        );
        _stakedBalances[msg.sender] -= amount;
        _transfer(address(this), msg.sender, amount);
        emit Unstaked(msg.sender, amount);
    }

    // RECALL — onlyOwner = protocol controls slashing in MVP.
    // Production: replace with DAO governance or multi-sig.
    // _burn removes slashed tokens from circulation permanently —
    // makes slashing economically meaningful, not just a transfer.
    function slash(address agent, uint256 amount) external onlyOwner {
        require(
            _stakedBalances[agent] >= amount,
            "SentinelToken: insufficient staked balance to slash"
        );
        _stakedBalances[agent] -= amount;
        _burn(address(this), amount);
        emit Slashed(agent, amount);
    }

    // RECALL — AuditRegistry calls this to check agent eligibility
    // before accepting an audit submission. Pure read, no gas cost
    // when called off-chain.
    function stakedBalance(address agent) external view returns (uint256) {
        return _stakedBalances[agent];
    }
}