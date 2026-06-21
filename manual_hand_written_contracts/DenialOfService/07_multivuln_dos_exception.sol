// expect: DenialOfService,GasException
// Multi-vuln: DoS via unbounded loop + GasException from storage expansion.
// The contract stores all transaction history in an unbounded array.
// The function that computes user stats iterates ALL transactions (DoS).
// Each new transaction costs more gas due to storage expansion (GasException).
// The two vulnerabilities compound each other.
pragma solidity ^0.8.0;

contract TransactionLog {
    struct TxEntry {
        address from;
        address to;
        uint256 amount;
        uint256 timestamp;
        string memo;
    }

    address public owner;
    TxEntry[] public ledger;
    mapping(address => uint256) public totalSent;
    mapping(address => uint256) public totalReceived;
    uint256 public entryCount;

    event TransactionLogged(address indexed from, address indexed to, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function logTransaction(address to, uint256 amount, string calldata memo) external payable {
        require(to != address(0), "zero address");
        require(bytes(memo).length <= 256, "memo too long");
        ledger.push(TxEntry(msg.sender, to, amount, block.timestamp, memo));
        totalSent[msg.sender] += amount;
        totalReceived[to] += amount;
        entryCount++;
        emit TransactionLogged(msg.sender, to, amount);
    }

    function batchLogTransactions(address[] calldata toList, uint256[] calldata amounts, string[] calldata memos) external {
        require(toList.length == amounts.length && amounts.length == memos.length, "length mismatch");
        for (uint256 i = 0; i < toList.length; i++) {
            ledger.push(TxEntry(msg.sender, toList[i], amounts[i], block.timestamp, memos[i]));
            totalSent[msg.sender] += amounts[i];
            totalReceived[toList[i]] += amounts[i];
            entryCount++;
            emit TransactionLogged(msg.sender, toList[i], amounts[i]);
        }
    }

    function computeUserStats() external view returns (uint256 sent, uint256 received, uint256 txCount, uint256 uniquePartners) {
        sent = totalSent[msg.sender];
        received = totalReceived[msg.sender];
        txCount = 0;
        address[] memory partners = new address[](entryCount);
        uint256 partnerCount = 0;
        for (uint256 i = 0; i < ledger.length; i++) {
            if (ledger[i].from == msg.sender || ledger[i].to == msg.sender) {
                txCount++;
                address partner = ledger[i].from == msg.sender ? ledger[i].to : ledger[i].from;
                bool found = false;
                for (uint256 j = 0; j < partnerCount; j++) {
                    if (partners[j] == partner) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    partners[partnerCount] = partner;
                    partnerCount++;
                }
            }
        }
        uniquePartners = partnerCount;
    }

    function getHistory(uint256 fromIndex, uint256 toIndex) external view returns (TxEntry[] memory) {
        require(toIndex > fromIndex && toIndex <= ledger.length, "invalid range");
        uint256 length = toIndex - fromIndex;
        TxEntry[] memory result = new TxEntry[](length);
        for (uint256 i = 0; i < length; i++) {
            result[i] = ledger[fromIndex + i];
        }
        return result;
    }

    function getLedgerSize() external view returns (uint256) {
        return ledger.length;
    }

    function prune(uint256 keepCount) external onlyOwner {
        require(keepCount < ledger.length, "keep count too high");
        uint256 removeCount = ledger.length - keepCount;
        for (uint256 i = 0; i < removeCount; i++) {
            totalSent[ledger[i].from] -= ledger[i].amount;
            totalReceived[ledger[i].to] -= ledger[i].amount;
        }
        for (uint256 j = 0; j < keepCount; j++) {
            ledger[j] = ledger[removeCount + j];
        }
        while (ledger.length > keepCount) {
            ledger.pop();
        }
    }

    receive() external payable {}
}