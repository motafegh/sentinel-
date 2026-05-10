// expect: CallToUnknown
// Contract calls an address whose code is unknown at compile time.
// The target is stored in state and can be changed — delegatecall to arbitrary code.
pragma solidity ^0.8.0;

contract PluginRunner {
    address public plugin;
    address public owner;

    constructor(address _plugin) {
        plugin = _plugin;
        owner = msg.sender;
    }

    function setPlugin(address _new) external {
        require(msg.sender == owner);
        plugin = _new;
    }

    // VULNERABILITY: delegatecall to user-controlled address
    // attacker sets plugin to malicious contract; execute() runs attacker code
    // in this contract's storage context
    function execute(bytes calldata data) external returns (bytes memory) {
        (bool ok, bytes memory ret) = plugin.delegatecall(data);
        require(ok, "plugin call failed");
        return ret;
    }

    // VULNERABILITY: plain call to unknown address
    function relay(address target, bytes calldata data) external {
        target.call(data);
    }
}
