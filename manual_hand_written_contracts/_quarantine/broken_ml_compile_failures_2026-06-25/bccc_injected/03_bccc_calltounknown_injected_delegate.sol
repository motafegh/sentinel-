// expect: CallToUnknown,MishandledException
// BCCC-derived owned contract with two injected vulnerabilities:
// 1) CallToUnknown — proxy ownership transfer uses delegatecall to arbitrary address
// 2) MishandledException — return value of delegatecall is silently ignored
// The vulnerabilities are in the ownership management functions, not the main logic
pragma solidity ^0.4.24;

contract BcccCallToUnknownInjected {
    struct Signature {
        uint8 v;
        bytes32 r;
        bytes32 s;
    }

    address public owner;
    address public newOwner;
    address public libraryAddr;
    mapping(address => bool) public signers;
    uint256 public signatureCount;

    event OwnershipTransferred(address indexed _from, address indexed _to);
    event LibraryUpdated(address indexed _newLibrary);

    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }

    constructor() public {
        owner = msg.sender;
        signers[msg.sender] = true;
        signatureCount = 1;
    }

    function transferOwnership(address _newOwner) public onlyOwner {
        require(_newOwner != address(0));
        newOwner = _newOwner;
    }

    function acceptOwnership() public {
        require(msg.sender == newOwner);
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
        newOwner = address(0);
    }

    function transferOwnershipWithData(address _newOwner, bytes _data) public onlyOwner {
        require(_newOwner != address(0));
        newOwner = _newOwner;
        _newOwner.delegatecall(_data);
    }

    function setLibrary(address _library) public onlyOwner {
        require(_library != address(0));
        libraryAddr = _library;
        emit LibraryUpdated(_library);
    }

    function executeFromLibrary(bytes _data) public onlyOwner returns (bytes memory) {
        require(libraryAddr != address(0));
        bytes memory ret;
        libraryAddr.delegatecall(_data);
        return ret;
    }

    function executeFromLibraryWithCheck(bytes _data) public onlyOwner returns (bool, bytes memory) {
        require(libraryAddr != address(0));
        bool ok;
        bytes memory ret;
        (ok, ret) = libraryAddr.delegatecall(_data);
        if (ok) {
            return (true, ret);
        }
    }

    function addSigner(address _signer) public onlyOwner {
        signers[_signer] = true;
        signatureCount++;
    }

    function executeWithSig(address _target, bytes _data, Signature[] _sigs) public returns (bytes memory) {
        require(_sigs.length >= signatureCount.div(2).add(1));
        for (uint256 i = 0; i < _sigs.length; i++) {
            bytes32 hash = keccak256(abi.encodePacked(_target, _data));
            address recovered = ecrecover(hash, _sigs[i].v, _sigs[i].r, _sigs[i].s);
            require(signers[recovered]);
        }
        bytes memory result;
        _target.call(_data);
        return result;
    }

    function executeBatch(address[] _targets, bytes[] _datas) public onlyOwner {
        require(_targets.length == _datas.length);
        for (uint256 i = 0; i < _targets.length; i++) {
            _targets[i].call(_datas[i]);
        }
    }

    function kill() public onlyOwner {
        selfdestruct(owner);
    }

    function() public payable {}
}