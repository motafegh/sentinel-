// expect: GasException
// Minimal gas DoS via transfer() 2300-gas stipend to contract recipients.
pragma solidity ^0.8.0;
contract GasMinimal {
    address[] public winners;
    function addWinner(address w) external { winners.push(w); }
    // VULNERABILITY: transfer() only forwards 2300 gas; any winner with
    // a fallback that uses >2300 gas causes permanent revert for all
    function payAll() external {
        for (uint i = 0; i < winners.length; i++) {
            payable(winners[i]).transfer(1 ether);
        }
    }
}
