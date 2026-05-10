// expect: IntegerUO
// Pre-0.8 style: no SafeMath, arithmetic on user input. Pure overflow.
// Use 0.8 with unchecked to replicate the training distribution pattern.
pragma solidity ^0.8.0;
contract SimpleOverflow {
    mapping(address => uint256) public points;
    function addPoints(address user, uint256 amount) external {
        unchecked { points[user] += amount; }
    }
    function removePoints(address user, uint256 amount) external {
        unchecked { points[user] -= amount; }
    }
    function multiplyPoints(address user, uint256 factor) external {
        unchecked { points[user] *= factor; }
    }
}
