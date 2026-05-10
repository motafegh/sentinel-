// expect: Reentrancy
// Absolute minimal reentrancy — no helper, no logic, just the bare pattern.
pragma solidity ^0.8.0;
contract MinimalReentrancy {
    mapping(address => uint256) public bal;
    function deposit() external payable { bal[msg.sender] += msg.value; }
    function withdraw() external {
        uint256 a = bal[msg.sender];
        (bool ok,) = msg.sender.call{value: a}("");
        require(ok);
        bal[msg.sender] = 0;
    }
}
