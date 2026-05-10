// expect: GasException
// Out-of-gas patterns: large struct copy, unbounded storage read in view,
// and stipend exhaustion on transfer to contract.
pragma solidity ^0.8.0;

contract GasVictim {
    struct Record {
        uint256[50] data;
        address owner;
        string description;
    }

    Record[] public records;
    mapping(address => uint256[]) public userRecordIds;

    function addRecord(string calldata desc) external {
        Record memory r;
        r.owner = msg.sender;
        r.description = desc;
        // VULNERABILITY: pushes a large struct — storage write gas cost can hit limit
        records.push(r);
        userRecordIds[msg.sender].push(records.length - 1);
    }

    // VULNERABILITY: iterates all user records — O(n) storage reads, unbounded
    function getUserTotal(address user) external view returns (uint256 total) {
        uint256[] storage ids = userRecordIds[user];
        for (uint256 i = 0; i < ids.length; i++) {
            total += records[ids[i]].data[0];
        }
    }

    // VULNERABILITY: transfer() sends only 2300 gas stipend
    // receiver is a contract — fallback with any logic reverts
    function pay(address payable recipient) external payable {
        recipient.transfer(msg.value);
    }
}
