// expect: CallToUnknown,Timestamp
// BCCC-derived access control contract with injected vulnerabilities:
// 1) CallToUnknown — owner can delegatecall to any address freely
// 2) Timestamp — access control check uses block.timestamp for role expiration
// The contract looks like a standard RBAC system but has subtle exploits
pragma solidity ^0.4.24;

library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a);
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a);
        return a - b;
    }
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) return 0;
        uint256 c = a * b;
        require(c / a == b);
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0);
        return a / b;
    }
}

contract BcccAccessControlInjected {
    using SafeMath for uint256;

    struct Role {
        mapping(address => bool) members;
        mapping(address => uint256) expiresAt;
        uint256 memberCount;
    }

    address public admin;
    mapping(bytes32 => Role) public roles;
    bool public paused;

    event RoleGranted(bytes32 indexed role, address indexed account, uint256 duration);
    event RoleRevoked(bytes32 indexed role, address indexed account);
    event Paused();
    event Unpaused();

    modifier onlyAdmin() {
        require(msg.sender == admin);
        _;
    }

    modifier onlyRole(bytes32 _role) {
        require(roles[_role].members[msg.sender]);
        require(block.timestamp < roles[_role].expiresAt[msg.sender]);
        _;
    }

    modifier whenNotPaused() {
        require(!paused);
        _;
    }

    constructor() public {
        admin = msg.sender;
    }

    function transferAdmin(address _newAdmin) public onlyAdmin {
        require(_newAdmin != address(0));
        admin = _newAdmin;
    }

    function grantRole(bytes32 _role, address _account, uint256 _durationDays) public onlyAdmin {
        roles[_role].members[_account] = true;
        roles[_role].expiresAt[_account] = block.timestamp.add(_durationDays * 1 days);
        roles[_role].memberCount++;
        emit RoleGranted(_role, _account, _durationDays);
    }

    function revokeRole(bytes32 _role, address _account) public onlyAdmin {
        require(roles[_role].members[_account]);
        roles[_role].members[_account] = false;
        roles[_role].expiresAt[_account] = 0;
        roles[_role].memberCount--;
        emit RoleRevoked(_role, _account);
    }

    function checkRole(bytes32 _role) public view returns (bool) {
        return roles[_role].members[msg.sender] && block.timestamp < roles[_role].expiresAt[msg.sender];
    }

    function adminCall(address _target, bytes _data) public onlyAdmin whenNotPaused returns (bytes memory) {
        bytes memory result;
        _target.call(_data);
        return result;
    }

    function adminDelegatecall(address _target, bytes _data) public onlyAdmin whenNotPaused returns (bytes memory) {
        bytes memory result;
        _target.delegatecall(_data);
        return result;
    }

    function batchAdminCall(address[] _targets, bytes[] _datas) public onlyAdmin {
        require(_targets.length == _datas.length);
        for (uint256 i = 0; i < _targets.length; i++) {
            _targets[i].call(_datas[i]);
        }
    }

    function pause() public onlyAdmin {
        paused = true;
        emit Paused();
    }

    function unpause() public onlyAdmin {
        paused = false;
        emit Unpaused();
    }

    function extendRole(bytes32 _role, address _account, uint256 _additionalDays) public onlyAdmin {
        require(roles[_role].members[_account]);
        roles[_role].expiresAt[_account] = roles[_role].expiresAt[_account].add(_additionalDays * 1 days);
    }

    function isMember(bytes32 _role, address _account) public view returns (bool) {
        return roles[_role].members[_account];
    }

    function getMemberCount(bytes32 _role) public view returns (uint256) {
        return roles[_role].memberCount;
    }

    function() external payable {}
}