// expect: DenialOfService
// DoS via unbounded loop over growing array. Owner array grows with each join.
// Once large enough, distribute() runs out of gas permanently.
pragma solidity ^0.8.0;

contract Distributor {
    address[] public participants;
    uint256 public pot;

    function join() external payable {
        require(msg.value >= 0.01 ether);
        participants.push(msg.sender);
        pot += msg.value;
    }

    function distribute() external {
        // VULNERABILITY: unbounded loop — grows with each join()
        // attacker joins 10,000 times; distribute() hits block gas limit forever
        uint256 share = pot / participants.length;
        for (uint256 i = 0; i < participants.length; i++) {
            payable(participants[i]).transfer(share);
        }
        pot = 0;
        delete participants;
    }
}
