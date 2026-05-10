// expect:
// Safe contract with zero external calls — pure state machine, no ETH, no call().
// Should produce near-zero probabilities across all classes.
pragma solidity ^0.8.0;
contract PureRegistry {
    mapping(address => string) public names;
    mapping(string => address) public lookup;
    event Registered(address indexed user, string name);
    function register(string calldata name) external {
        require(bytes(name).length > 0 && bytes(name).length <= 32);
        require(lookup[name] == address(0), "taken");
        require(bytes(names[msg.sender]).length == 0, "already registered");
        names[msg.sender] = name;
        lookup[name] = msg.sender;
        emit Registered(msg.sender, name);
    }
    function isAvailable(string calldata name) external view returns (bool) {
        return lookup[name] == address(0);
    }
}
