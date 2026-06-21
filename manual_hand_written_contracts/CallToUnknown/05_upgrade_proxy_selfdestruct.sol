// expect: CallToUnknown
// Upgradeable proxy pattern with a timelock governance mechanism.
// The proxy can delegatecall to any implementation address after timelock expires.
// A malicious or compromised governance can point to a contract with SELFDESTRUCT
// which destroys the proxy contract and all its funds permanently.
// Also: the fallback uses delegatecall to any address with no validation.
pragma solidity ^0.8.0;

contract TimelockProxy {
    address public implementation;
    address public governor;
    uint256 public timelockDuration;
    mapping(address => uint256) public pendingUpgrades;

    event UpgradeScheduled(address indexed newImpl, uint256 readyAt);
    event Upgraded(address indexed oldImpl, address indexed newImpl);
    event Destroyed(address indexed destroyer);

    modifier onlyGovernor() {
        require(msg.sender == governor, "not governor");
        _;
    }

    constructor(address _impl, address _gov, uint256 _delay) {
        implementation = _impl;
        governor = _gov;
        timelockDuration = _delay;
    }

    function transferGovernorship(address newGov) external onlyGovernor {
        require(newGov != address(0), "zero address");
        governor = newGov;
    }

    function scheduleUpgrade(address newImpl) external onlyGovernor {
        require(newImpl != address(0), "zero address");
        uint256 readyAt = block.timestamp + timelockDuration;
        pendingUpgrades[newImpl] = readyAt;
        emit UpgradeScheduled(newImpl, readyAt);
    }

    function cancelUpgrade(address impl) external onlyGovernor {
        delete pendingUpgrades[impl];
    }

    function executeUpgrade(address newImpl) external onlyGovernor {
        require(pendingUpgrades[newImpl] != 0, "not scheduled");
        require(block.timestamp >= pendingUpgrades[newImpl], "timelock active");
        address old = implementation;
        implementation = newImpl;
        delete pendingUpgrades[newImpl];
        emit Upgraded(old, newImpl);
    }

    function destroy() external onlyGovernor {
        emit Destroyed(msg.sender);
        selfdestruct(payable(governor));
    }

    function upgradeAndCall(address newImpl, bytes calldata initData) external onlyGovernor {
        require(pendingUpgrades[newImpl] != 0, "not scheduled");
        require(block.timestamp >= pendingUpgrades[newImpl], "timelock active");
        address old = implementation;
        implementation = newImpl;
        delete pendingUpgrades[newImpl];
        emit Upgraded(old, newImpl);
        if (initData.length > 0) {
            (bool ok, ) = newImpl.delegatecall(initData);
            require(ok, "init failed");
        }
    }

    function emergencyOverride(address newImpl) external onlyGovernor {
        address old = implementation;
        implementation = newImpl;
        emit Upgraded(old, newImpl);
    }

    function proxyCall(address target, bytes calldata data) external onlyGovernor returns (bytes memory) {
        (bool ok, bytes memory ret) = target.delegatecall(data);
        if (!ok) {
            assembly {
                revert(add(ret, 32), mload(ret))
            }
        }
        return ret;
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

    receive() external payable {}
}