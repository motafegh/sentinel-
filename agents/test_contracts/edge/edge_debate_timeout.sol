// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title UpgradeableVault - a complex proxy+implementation vault with a
///        reentrancy bug buried in the implementation.
/// @notice This contract is intentionally long + multi-function to produce a
///        long debate in LLM-on mode (tests the debate-timeout -> INCONCLUSIVE
///        path that WS1.3 introduces). The reentrancy bug is in
///        UpgradeableVault.withdraw(); the proxy + admin + upgrade logic is
///        safe but adds surface area the debate must reason through.
contract ProxyAdmin {
    address public admin;
    mapping(address => bool) public authorized;

    event AdminChanged(address oldAdmin, address newAdmin);
    event Authorized(address who, bool ok);

    modifier onlyAdmin() {
        require(msg.sender == admin, "not admin");
        _;
    }

    constructor() {
        admin = msg.sender;
        emit AdminChanged(address(0), msg.sender);
    }

    function setAdmin(address newAdmin) external onlyAdmin {
        emit AdminChanged(admin, newAdmin);
        admin = newAdmin;
    }

    function authorize(address who, bool ok) external onlyAdmin {
        authorized[who] = ok;
        emit Authorized(who, ok);
    }
}

contract VaultProxy {
    address public admin;
    address public implementation;
    ProxyAdmin public proxyAdmin;

    event Upgraded(address impl);

    constructor(address _impl, address _admin) {
        implementation = _impl;
        admin = _admin;
        proxyAdmin = new ProxyAdmin();
    }

    fallback() external payable {
        address impl = implementation;
        assembly {
            calldatacopy(0, 0, calldatasize())
            let r := delegatecall(gas(), impl, 0, calldatasize(), 0, 0)
            returndatacopy(0, 0, returndatasize())
            switch r
            case 0 { revert(0, returndatasize()) }
            default { return(0, returndatasize()) }
        }
    }

    function upgradeTo(address newImpl) external {
        require(msg.sender == admin, "not admin");
        implementation = newImpl;
        emit Upgraded(newImpl);
    }
}

contract UpgradeableVault {
    mapping(address => uint256) public balances;
    address public owner;
    IERC20 public token;

    event Deposit(address indexed who, uint256 amount);
    event Withdraw(address indexed who, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _token) {
        owner = msg.sender;
        token = IERC20(_token);
    }

    function deposit(uint256 amount) external {
        require(token.transferFrom(msg.sender, address(this), amount), "tf fail");
        balances[msg.sender] += amount;
        emit Deposit(msg.sender, amount);
    }

    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "insufficient");
        // BUG: state updated AFTER the external call - classic CEI violation.
        // The token.transfer triggers a callback into the attacker's receive()
        // which re-enters withdraw() before balances is decremented.
        require(token.transfer(msg.sender, amount), "transfer failed");
        balances[msg.sender] -= amount;
        emit Withdraw(msg.sender, amount);
    }

    function balanceOf(address who) external view returns (uint256) {
        return balances[who];
    }

    function sweep(address to) external onlyOwner {
        uint256 bal = token.balanceOf(address(this));
        require(token.transfer(to, bal), "sweep fail");
    }
}

interface IERC20 {
    function transfer(address to, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}
