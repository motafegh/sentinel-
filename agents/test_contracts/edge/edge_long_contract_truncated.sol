// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title ComplexDeFiProtocol - a multi-contract file designed to test the
///        debate's source-truncation behavior (WS3 / Finding #1).
/// @notice The vulnerable function (VulnerableVault.withdraw) is placed PAST
///        the 2000-character truncation cutoff used by cross_validator today
///        (nodes.py:1116). With the current raw-truncation mechanism, the
///        debate sees only the safe preamble (interfaces + safe contracts) and
///        the actual vulnerable function is silently truncated away. WS3's
///        hotspot-guided excerpt + sliding-window fallback is what makes the
///        debate see the real bug. This file must be >2000 chars before the
///        vulnerable function begins.

// ── IERC20 interface (preamble padding - safe) ─────────────────────────────
interface IERC20 {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 value) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}

// ── IERC20Metadata (preamble padding - safe) ───────────────────────────────
interface IERC20Metadata is IERC20 {
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function decimals() external view returns (uint8);
}

// ── Context (preamble padding - safe) ──────────────────────────────────────
abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }
    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}

// ── Ownable (preamble padding - safe) ──────────────────────────────────────
abstract contract Ownable is Context {
    address private _owner;
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        _transferOwnership(_msgSender());
    }

    modifier onlyOwner() {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
        _;
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        _transferOwnership(newOwner);
    }

    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

// ── SafeToken (preamble padding - safe, a simple ERC20) ────────────────────
contract SafeToken is Context, IERC20, IERC20Metadata, Ownable {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    uint256 private _totalSupply;
    string private _name;
    string private _symbol;

    constructor(string memory name_, string memory symbol_) {
        _name = name_;
        _symbol = symbol_;
        _mint(_msgSender(), 1_000_000 * 10 ** decimals());
    }

    function name() public view virtual override returns (string memory) { return _name; }
    function symbol() public view virtual override returns (string memory) { return _symbol; }
    function decimals() public view virtual override returns (uint8) { return 18; }
    function totalSupply() public view virtual override returns (uint256) { return _totalSupply; }
    function balanceOf(address account) public view virtual override returns (uint256) { return _balances[account]; }
    function allowance(address o, address s) public view virtual override returns (uint256) { return _allowances[o][s]; }

    function transfer(address to, uint256 amount) public virtual override returns (bool) {
        _transfer(_msgSender(), to, amount);
        return true;
    }
    function approve(address spender, uint256 amount) public virtual override returns (bool) {
        _approve(_msgSender(), spender, amount);
        return true;
    }
    function transferFrom(address from, address to, uint256 amount) public virtual override returns (bool) {
        _transfer(from, to, amount);
        uint256 currentAllowance = _allowances[from][_msgSender()];
        require(currentAllowance >= amount, "ERC20: insufficient allowance");
        _approve(from, _msgSender(), currentAllowance - amount);
        return true;
    }
    function _transfer(address from, address to, uint256 amount) internal virtual {
        require(from != address(0), "ERC20: transfer from zero");
        require(to != address(0), "ERC20: transfer to zero");
        uint256 fromBalance = _balances[from];
        require(fromBalance >= amount, "ERC20: insufficient balance");
        _balances[from] = fromBalance - amount;
        _balances[to] += amount;
        emit Transfer(from, to, amount);
    }
    function _approve(address o, address s, uint256 a) internal virtual {
        require(o != address(0), "ERC20: approve from zero");
        require(s != address(0), "ERC20: approve to zero");
        _allowances[o][s] = a;
        emit Approval(o, s, a);
    }
    function _mint(address account, uint256 amount) internal virtual {
        require(account != address(0), "ERC20: mint to zero");
        _totalSupply += amount;
        _balances[account] += amount;
        emit Transfer(address(0), account, amount);
    }
}

// ── VulnerableVault - THE BUG IS HERE (past char 2000) ─────────────────────
// The withdraw function below updates state AFTER the external call (CEI
// violation). A cross_validator that only sees the first 2000 chars of this
// file sees none of VulnerableVault - just the safe preamble above - and
// cannot reason about the actual bug. This is the WS3 truncation test.
contract VulnerableVault {
    using SafeMath for uint256;
    IERC20 public token;
    mapping(address => uint256) public deposits;

    constructor(address _token) {
        token = IERC20(_token);
    }

    function deposit(uint256 amount) external {
        require(token.transferFrom(msg.sender, address(this), amount), "tf fail");
        deposits[msg.sender] = deposits[msg.sender].add(amount);
    }

    function withdraw(uint256 amount) external {
        require(deposits[msg.sender] >= amount, "insufficient");
        // BUG: external call BEFORE state update - reentrancy.
        require(token.transfer(msg.sender, amount), "transfer failed");
        deposits[msg.sender] = deposits[msg.sender].sub(amount);
    }
}

library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "SafeMath: subtraction overflow");
        return a - b;
    }
}
