// expect: TransactionOrderDependence
// Minimal front-running: state readable from mempool, next tx exploits it.
pragma solidity ^0.8.0;
contract TODMinimal {
    uint256 public reward;
    address public solver;
    function setReward() external payable { reward = msg.value; }
    // VULNERABILITY: solver can be front-run — anyone who sees this in mempool
    // can submit same solution with higher gas and steal the reward
    function solve(bytes32 solution) external {
        require(solver == address(0));
        solver = msg.sender;
        payable(msg.sender).transfer(reward);
        reward = 0;
    }
}
