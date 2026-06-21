// expect: CallToUnknown,Reentrancy,Timestamp
// Multi-vuln: CallToUnknown + Reentrancy + Timestamp in a single contract.
// The vault allows the owner to set arbitrary plugin addresses (CallToUnknown).
// The plugin execution uses delegatecall (Reentrancy via callback).
// The plugin selection algorithm uses block.timestamp (Timestamp).
// Three different vulnerability classes interacting through the same plugin system.
pragma solidity ^0.8.0;

contract PluginVault {
    struct Plugin {
        address implementation;
        uint256 version;
        uint256 activatedAt;
        bool active;
    }

    address public owner;
    mapping(bytes32 => Plugin) public plugins;
    bytes32[] public pluginIds;
    mapping(address => uint256) public balances;
    mapping(address => uint256) public nonces;

    event PluginRegistered(bytes32 indexed id, address indexed implementation);
    event PluginExecuted(bytes32 indexed id, address indexed executor);
    event DepositMade(address indexed user, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function registerPlugin(bytes32 id, address implementation) external onlyOwner {
        require(implementation != address(0), "zero address");
        plugins[id] = Plugin(implementation, 1, block.timestamp, true);
        pluginIds.push(id);
        emit PluginRegistered(id, implementation);
    }

    function deactivatePlugin(bytes32 id) external onlyOwner {
        require(plugins[id].active, "already inactive");
        plugins[id].active = false;
    }

    function executePlugin(bytes32 id, bytes calldata data) external returns (bytes memory) {
        Plugin storage p = plugins[id];
        require(p.active, "plugin not active");
        p.version++;
        (bool ok, bytes memory ret) = p.implementation.delegatecall(data);
        require(ok, "plugin execution failed");
        emit PluginExecuted(id, msg.sender);
        return ret;
    }

    function executeBestPlugin(bytes calldata data) external returns (bytes memory) {
        require(pluginIds.length > 0, "no plugins");
        uint256 bestIndex = 0;
        uint256 bestTime = 0;
        for (uint256 i = 0; i < pluginIds.length; i++) {
            Plugin storage p = plugins[pluginIds[i]];
            if (p.active && p.activatedAt > bestTime) {
                bestTime = p.activatedAt;
                bestIndex = i;
            }
        }
        Plugin storage best = plugins[pluginIds[bestIndex]];
        (bool ok, bytes memory ret) = best.implementation.delegatecall(data);
        require(ok, "best plugin failed");
        emit PluginExecuted(pluginIds[bestIndex], msg.sender);
        return ret;
    }

    function deposit() external payable {
        require(msg.value > 0, "zero deposit");
        balances[msg.sender] += msg.value;
        emit DepositMade(msg.sender, msg.value);
    }

    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "insufficient");
        balances[msg.sender] -= amount;
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "withdraw failed");
    }

    function executePluginAndWithdraw(bytes32 id, bytes calldata pluginData, uint256 withdrawAmount) external {
        Plugin storage p = plugins[id];
        require(p.active, "plugin not active");
        p.implementation.delegatecall(pluginData);
        balances[msg.sender] -= withdrawAmount;
        (bool ok, ) = msg.sender.call{value: withdrawAmount}("");
        require(ok, "withdraw failed");
    }

    function getPluginCount() external view returns (uint256) {
        return pluginIds.length;
    }

    receive() external payable {}
}