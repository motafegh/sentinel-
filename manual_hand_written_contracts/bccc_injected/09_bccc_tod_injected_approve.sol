// expect: TransactionOrderDependence,IntegerUO
// BCCC-derived token contract with two injected front-running vulnerabilities:
// 1) TransactionOrderDependence — classic approve front-running (allowance race)
// 2) IntegerUO — unchecked arithmetic in the batch allowance increase
// The allowance functions look like standard ERC20 but the race condition is live
pragma solidity ^0.4.24;

contract BcccTODTokenInjected {
    string public name = "TODInjected";
    string public symbol = "TODI";
    uint8 public decimals = 18;
    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    mapping(address => uint256) public nonces;
    mapping(address => uint256) public lockedUntil;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor(uint256 initialSupply) public {
        totalSupply = initialSupply * 10 ** uint256(decimals);
        balanceOf[msg.sender] = totalSupply;
    }

    function _transfer(address _from, address _to, uint256 _value) internal {
        require(_to != address(0));
        require(balanceOf[_from] >= _value);
        require(lock[_from] == 0);
        if (_from != msg.sender) {
            require(allowance[_from][msg.sender] >= _value);
            allowance[_from][msg.sender] -= _value;
        }
        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        emit Transfer(_from, _to, _value);
    }

    function transfer(address _to, uint256 _value) public returns (bool) {
        _transfer(msg.sender, _to, _value);
        return true;
    }

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool) {
        _transfer(_from, _to, _value);
        return true;
    }

    function approve(address _spender, uint256 _value) public returns (bool) {
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    function increaseAllowance(address _spender, uint256 _addedValue) public returns (bool) {
        allowance[msg.sender][_spender] += _addedValue;
        emit Approval(msg.sender, _spender, allowance[msg.sender][_spender]);
        return true;
    }

    function decreaseAllowance(address _spender, uint256 _subtractedValue) public returns (bool) {
        allowance[msg.sender][_spender] -= _subtractedValue;
        emit Approval(msg.sender, _spender, allowance[msg.sender][_spender]);
        return true;
    }

    function approveWithNonce(address _spender, uint256 _value, uint256 _nonce) public returns (bool) {
        require(nonces[msg.sender] == _nonce);
        nonces[msg.sender]++;
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    function batchApprove(address[] _spenders, uint256[] _values) public returns (bool) {
        require(_spenders.length == _values.length);
        for (uint256 i = 0; i < _spenders.length; i++) {
            allowance[msg.sender][_spenders[i]] = _values[i];
            emit Approval(msg.sender, _spenders[i], _values[i]);
        }
        return true;
    }

    function batchIncreaseAllowance(address[] _spenders, uint256[] _addedValues) public returns (bool) {
        require(_spenders.length == _addedValues.length);
        for (uint256 i = 0; i < _spenders.length; i++) {
            allowance[msg.sender][_spenders[i]] += _addedValues[i];
            emit Approval(msg.sender, _spenders[i], allowance[msg.sender][_spenders[i]]);
        }
        return true;
    }

    function lock(address user, uint256 duration) public {
        require(msg.sender == user || balanceOf[msg.sender] > 0);
        lockedUntil[user] = block.timestamp + duration;
    }

    function mint(uint256 amount) public {
        balanceOf[msg.sender] += amount;
        totalSupply += amount;
        emit Transfer(address(0), msg.sender, amount);
    }

    function burn(uint256 amount) public {
        balanceOf[msg.sender] -= amount;
        totalSupply -= amount;
        emit Transfer(msg.sender, address(0), amount);
    }

    function() external payable {}
}