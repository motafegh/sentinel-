// expect: GasException,DenialOfService
// BCCC-derived token with injected gas-intensive patterns:
// 1) GasException — nested loops in the dividend distribution that grow unbounded
// 2) DoS — the distribute function becomes unusable after enough holders
// Hidden inside an innocent-looking "dividend" feature common in ERC20 tokens
pragma solidity ^0.4.24;

library SafeMath {
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a * b;
        require(a == 0 || c / a == b);
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0);
        return a / b;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a);
        return a - b;
    }
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a);
        return c;
    }
}

contract BcccGasTokenInjected {
    using SafeMath for uint256;

    string public name = "GasBomb Token";
    string public symbol = "GASB";
    uint8 public decimals = 18;
    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    mapping(address => uint256) public rewardDebt;
    mapping(address => uint256) public lastDividendPoints;

    address[] public stakeholders;
    uint256 public totalDividendPoints;
    uint256 public totalDistributed;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event DividendDistributed(uint256 amount);
    event RewardClaimed(address indexed user, uint256 amount);

    constructor(uint256 initialSupply) public {
        totalSupply = initialSupply.mul(10 ** uint256(decimals));
        balanceOf[msg.sender] = totalSupply;
    }

    function _transfer(address _from, address _to, uint256 _value) internal {
        require(_to != address(0));
        require(balanceOf[_from] >= _value);
        if (balanceOf[_from] == _value) {
            removeStakeholder(_from);
        }
        if (balanceOf[_to] == 0) {
            stakeholders.push(_to);
        }
        balanceOf[_from] = balanceOf[_from].sub(_value);
        balanceOf[_to] = balanceOf[_to].add(_value);
        emit Transfer(_from, _to, _value);
    }

    function removeStakeholder(address _user) internal {
        for (uint256 i = 0; i < stakeholders.length; i++) {
            if (stakeholders[i] == _user) {
                stakeholders[i] = stakeholders[stakeholders.length - 1];
                stakeholders.length--;
                break;
            }
        }
    }

    function transfer(address _to, uint256 _value) public returns (bool) {
        _transfer(msg.sender, _to, _value);
        return true;
    }

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool) {
        require(_value <= allowance[_from][msg.sender]);
        allowance[_from][msg.sender] = allowance[_from][msg.sender].sub(_value);
        _transfer(_from, _to, _value);
        return true;
    }

    function approve(address _spender, uint256 _value) public returns (bool) {
        allowance[msg.sender][_spender] = _value;
        emit Transfer(msg.sender, _spender, _value);
        return true;
    }

    function distributeDividends() external payable {
        require(msg.value > 0);
        require(stakeholders.length > 0);
        totalDividendPoints = totalDividendPoints.add(msg.value.mul(1e18).div(totalSupply));
        totalDistributed = totalDistributed.add(msg.value);
        emit DividendDistributed(msg.value);
        for (uint256 i = 0; i < stakeholders.length; i++) {
            for (uint256 j = i + 1; j < stakeholders.length; j++) {
                if (stakeholders[i] == stakeholders[j]) {
                    delete stakeholders[j];
                }
            }
            address user = stakeholders[i];
            if (user != address(0) && balanceOf[user] > 0) {
                uint256 pending = balanceOf[user].mul(totalDividendPoints.sub(lastDividendPoints[user])).div(1e18);
                if (pending > 0) {
                    lastDividendPoints[user] = totalDividendPoints;
                    user.transfer(pending);
                }
            }
        }
    }

    function claimDividends() public {
        address user = msg.sender;
        require(balanceOf[user] > 0);
        uint256 pending = balanceOf[user].mul(totalDividendPoints.sub(lastDividendPoints[user])).div(1e18);
        require(pending > 0);
        lastDividendPoints[user] = totalDividendPoints;
        user.transfer(pending);
        emit RewardClaimed(user, pending);
    }

    function batchTransfer(address[] _recipients, uint256[] _amounts) public {
        require(_recipients.length == _amounts.length);
        for (uint256 i = 0; i < _recipients.length; i++) {
            _transfer(msg.sender, _recipients[i], _amounts[i]);
        }
    }

    function() external payable {
        if (msg.value > 0) {
            distributeDividends();
        }
    }
}