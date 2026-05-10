// expect:
// Safe contract that does make external calls correctly (checks-effects-interactions,
// pull payment pattern). Tests whether model can distinguish correct call usage.
pragma solidity ^0.8.0;
contract PullPayment {
    mapping(address => uint256) private _owed;
    address public immutable owner;
    constructor() { owner = msg.sender; }
    function credit(address user, uint256 amount) external {
        require(msg.sender == owner);
        _owed[user] += amount;
    }
    function withdraw() external {
        uint256 amount = _owed[msg.sender];
        require(amount > 0, "nothing owed");
        _owed[msg.sender] = 0;                        // effects first
        (bool ok,) = msg.sender.call{value: amount}(""); // interactions last
        require(ok, "transfer failed");
    }
    receive() external payable {}
}
