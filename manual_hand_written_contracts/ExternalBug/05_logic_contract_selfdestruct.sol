// expect: ExternalBug,CallToUnknown
// Upgradeable vault that stores user funds in the logic contract's storage.
// The proxy delegates to an implementation that has a selfdestruct function.
// If the implementation is swapped to a contract with SELFDESTRUCT and called,
// the proxy's storage is erased and all funds are lost.
// The vault also uses delegatecall for executing user-submitted calldata
// which can modify arbitrary storage slots in the proxy context.
pragma solidity ^0.8.0;

contract DelegateVault {
    address public implementation;
    address public owner;
    mapping(address => uint256) public balances;
    uint256 public totalFunds;
    bool public frozen;

    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);
    event ImplementationUpgraded(address indexed oldImpl, address indexed newImpl);
    event Frozen(address indexed freezer);

    modifier notFrozen() {
        require(!frozen, "contract frozen");
        _;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function upgradeTo(address newImpl) external onlyOwner {
        require(newImpl != address(0), "zero address");
        address old = implementation;
        implementation = newImpl;
        emit ImplementationUpgraded(old, newImpl);
    }

    function upgradeToAndCall(address newImpl, bytes calldata initData) external onlyOwner {
        address old = implementation;
        implementation = newImpl;
        emit ImplementationUpgraded(old, newImpl);
        if (initData.length > 0) {
            (bool ok, ) = address(this).delegatecall(initData);
            require(ok, "init failed");
        }
    }

    function deposit() external payable notFrozen {
        require(msg.value > 0, "zero deposit");
        balances[msg.sender] += msg.value;
        totalFunds += msg.value;
        emit Deposited(msg.sender, msg.value);
    }

    function withdraw(uint256 amount) external notFrozen {
        require(balances[msg.sender] >= amount, "insufficient");
        balances[msg.sender] -= amount;
        totalFunds -= amount;
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "withdraw failed");
        emit Withdrawn(msg.sender, amount);
    }

    function delegateCallToImpl(bytes calldata data) external onlyOwner returns (bytes memory) {
        require(implementation != address(0), "no impl");
        (bool ok, bytes memory ret) = implementation.delegatecall(data);
        if (!ok) {
            assembly {
                revert(add(ret, 32), mload(ret))
            }
        }
        return ret;
    }

    function executeArbitrary(bytes calldata data) external onlyOwner returns (bytes memory) {
        (bool ok, bytes memory ret) = address(this).delegatecall(data);
        if (!ok) {
            assembly {
                revert(add(ret, 32), mload(ret))
            }
        }
        return ret;
    }

    function freeze() external onlyOwner {
        frozen = true;
        emit Frozen(msg.sender);
    }

    function emergencyWithdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        (bool ok, ) = owner.call{value: balance}("");
        require(ok, "emergency withdraw failed");
    }

    function destroy() external onlyOwner {
        selfdestruct(payable(owner));
    }

    fallback(bytes calldata data) external payable returns (bytes memory) {
        address impl = implementation;
        require(impl != address(0), "no implementation");
        (bool ok, bytes memory ret) = impl.delegatecall(data);
        if (!ok) {
            assembly {
                revert(add(ret, 32), mload(ret))
            }
        }
        return ret;
    }

    receive() external payable {
        if (msg.value > 0) {
            balances[msg.sender] += msg.value;
            totalFunds += msg.value;
            emit Deposited(msg.sender, msg.value);
        }
    }
}