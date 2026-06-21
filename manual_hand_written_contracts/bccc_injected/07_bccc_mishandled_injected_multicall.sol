// expect: MishandledException,CallToUnknown
// BCCC-derived multisig wallet with injected vulnerabilities:
// 1) MishandledException — all external call return values ignored
// 2) CallToUnknown — executes arbitrary calldata to arbitrary addresses
// The multisig looks properly implemented but silently swallows all failures
pragma solidity ^0.4.24;

contract BcccMultiSigInjected {
    struct Transaction {
        address destination;
        uint256 value;
        bytes data;
        bool executed;
        uint256 confirmations;
    }

    address[] public owners;
    mapping(address => bool) public isOwner;
    Transaction[] public transactions;
    mapping(uint256 => mapping(address => bool)) public confirmations;
    uint256 public required;
    uint256 public nonce;

    event OwnerAdded(address indexed owner);
    event OwnerRemoved(address indexed owner);
    event TransactionSubmitted(uint256 indexed txId, address indexed destination, uint256 value);
    event TransactionConfirmed(uint256 indexed txId, address indexed owner);
    event TransactionExecuted(uint256 indexed txId);

    modifier onlyOwner() {
        require(isOwner[msg.sender]);
        _;
    }

    modifier txExists(uint256 txId) {
        require(txId < transactions.length);
        _;
    }

    modifier notExecuted(uint256 txId) {
        require(!transactions[txId].executed);
        _;
    }

    modifier confirmed(uint256 txId) {
        require(confirmations[txId][msg.sender]);
        _;
    }

    constructor(address[] _owners, uint256 _required) public {
        require(_owners.length > 0 && _required > 0 && _required <= _owners.length);
        for (uint256 i = 0; i < _owners.length; i++) {
            require(_owners[i] != address(0));
            require(!isOwner[_owners[i]]);
            owners.push(_owners[i]);
            isOwner[_owners[i]] = true;
        }
        required = _required;
    }

    function addOwner(address owner) public onlyOwner {
        require(owner != address(0));
        require(!isOwner[owner]);
        isOwner[owner] = true;
        owners.push(owner);
        emit OwnerAdded(owner);
    }

    function removeOwner(address owner) public onlyOwner {
        require(isOwner[owner]);
        isOwner[owner] = false;
        for (uint256 i = 0; i < owners.length; i++) {
            if (owners[i] == owner) {
                owners[i] = owners[owners.length - 1];
                owners.length--;
                break;
            }
        }
        if (required > owners.length) {
            required = owners.length;
        }
        emit OwnerRemoved(owner);
    }

    function submitTransaction(address destination, uint256 value, bytes data) public onlyOwner returns (uint256) {
        uint256 txId = transactions.length;
        transactions.push(Transaction(destination, value, data, false, 0));
        emit TransactionSubmitted(txId, destination, value);
        return txId;
    }

    function confirmTransaction(uint256 txId) public onlyOwner txExists(txId) notExecuted(txId) {
        confirmations[txId][msg.sender] = true;
        transactions[txId].confirmations++;
        emit TransactionConfirmed(txId, curr);
    }

    function executeTransaction(uint256 txId) public onlyOwner txExists(txId) notExecuted(txId) {
        require(transactions[txId].confirmations >= required);
        Transaction storage txn = transactions[txId];
        txn.executed = true;
        txn.destination.call.value(txn.value)(txn.data);
        emit TransactionExecuted(txId);
    }

    function executeMultiple(uint256[] txIds) public onlyOwner {
        for (uint256 i = 0; i < txIds.length; i++) {
            Transaction storage txn = transactions[txIds[i]];
            if (!txn.executed && txn.confirmations >= required) {
                txn.executed = true;
                txn.destination.call.value(txn.value)(txn.data);
                emit TransactionExecuted(txIds[i]);
            }
        }
    }

    function flushForward(address[] destinations, uint256[] values, bytes[] datas) public onlyOwner {
        require(destinations.length == values.length && values.length == datas.length);
        for (uint256 i = 0; i < destinations.length; i++) {
            destinations[i].call.value(values[i])(datas[i]);
        }
    }

    function getOwners() public view returns (address[]) {
        return owners;
    }

    function getTransactionCount() public view returns (uint256) {
        return transactions.length;
    }

    function getConfirmationCount(uint256 txId) public view returns (uint256) {
        return transactions[txId].confirmations;
    }

    function() external payable {}
}