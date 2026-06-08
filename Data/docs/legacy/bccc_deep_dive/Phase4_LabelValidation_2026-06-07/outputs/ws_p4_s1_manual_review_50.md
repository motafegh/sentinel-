# Manual Review: 43 High-Uncertainty Contracts

**Purpose:** Validate BCCC labels where neither slither nor aderyn found a matching detector hit.
**Method:** Read each contract's source code. For each, determine:
  - **KEEP**: BCCC label is correct (the vulnerability exists)
  - **DROP**: BCCC label is incorrect (no vulnerability of this type)
  - **UNCERTAIN**: Can't determine from source alone

**Total contracts:** 43

| # | Class | ID | LOC | NF | Decision | Notes |
|---|-------|----|----|----|----------|-------|
| 1 | Class01:ExternalBug | b64aaff28af406fa | 146 | 13 | | |

| 2 | Class02:GasException | afc20d2fcb52f714 | 1141 | 85 | | |
| 3 | Class02:GasException | 217dfa4632f1c8d8 | 540 | 31 | | |
| 4 | Class02:GasException | 07aa3c486b42f189 | 902 | 74 | | |
| 5 | Class02:GasException | c6bd49b4aec50e0e | 330 | 39 | | |

| 6 | Class03:MishandledException | e97c9438d1928e75 | 117 | 13 | | |
| 7 | Class03:MishandledException | 2ad7bbb4d3e569a4 | 245 | 22 | | |

| 8 | Class04:Timestamp | b1982e1e1108812d | 359 | 39 | | |

| 9 | Class06:UnusedReturn | da081796d0c57c5f | 463 | 27 | | |
| 10 | Class06:UnusedReturn | 230347c96df3515f | 287 | 33 | | |
| 11 | Class06:UnusedReturn | 0829214881ec3423 | 524 | 40 | | |
| 12 | Class06:UnusedReturn | fd1df8d6413c2f20 | 636 | 36 | | |

| 13 | Class08:CallToUnknown | 2be5c39959fc3e9e | 182 | 17 | | |
| 14 | Class08:CallToUnknown | 0c4b244a4391b1b4 | 270 | 20 | | |
| 15 | Class08:CallToUnknown | 7300cf995dd3b67d | 913 | 56 | | |
| 16 | Class08:CallToUnknown | 6654483dbdfe6fac | 100 | 7 | | |
| 17 | Class08:CallToUnknown | 5e0096dbcc1d4f8d | 300 | 30 | | |
| 18 | Class08:CallToUnknown | 36427ce69f44a76d | 577 | 27 | | |

| 19 | Class09:DenialOfService | b5f07eac33c5f726 | 52 | 5 | | |

| 20 | Class10:IntegerUO | 3211900d19fd22f0 | 1097 | 56 | | |
| 21 | Class10:IntegerUO | 0a04b7966715ae71 | 497 | 35 | | |
| 22 | Class10:IntegerUO | 7a144ba43261c763 | 1123 | 51 | | |
| 23 | Class10:IntegerUO | 6e1c529e8b915147 | 918 | 66 | | |
| 24 | Class10:IntegerUO | 6233829661cf360e | 1717 | 118 | | |
| 25 | Class10:IntegerUO | 3dfb34b0c2904921 | 124 | 13 | | |
| 26 | Class10:IntegerUO | 2e04a1243fcbabf9 | 224 | 21 | | |
| 27 | Class10:IntegerUO | f31393f9cff440df | 638 | 50 | | |
| 28 | Class10:IntegerUO | 28363a3df6294062 | 574 | 86 | | |
| 29 | Class10:IntegerUO | be006469bb2e7ae7 | 474 | 33 | | |
| 30 | Class10:IntegerUO | 0e363473de1be614 | 311 | 26 | | |

| 31 | Class11:Reentrancy | fae5624b6a384099 | 201 | 15 | | |
| 32 | Class11:Reentrancy | 2c181f48e7141661 | 109 | 10 | | |
| 33 | Class11:Reentrancy | 0a7a47e9290ed384 | 148 | 14 | | |
| 34 | Class11:Reentrancy | 6da017c2465a3757 | 11 | 1 | | |
| 35 | Class11:Reentrancy | 60c3189a471809ec | 380 | 28 | | |
| 36 | Class11:Reentrancy | 57904673f62ab2d8 | 57 | 6 | | |
| 37 | Class11:Reentrancy | 3645b3a49eb67358 | 123 | 12 | | |
| 38 | Class11:Reentrancy | 28d28a7245ccea11 | 579 | 28 | | |
| 39 | Class11:Reentrancy | d415ba0a60ff3632 | 227 | 28 | | |
| 40 | Class11:Reentrancy | 221ee40fe539b90c | 814 | 58 | | |
| 41 | Class11:Reentrancy | e5df9eb444fb470e | 248 | 22 | | |
| 42 | Class11:Reentrancy | a5987a43180e0433 | 262 | 22 | | |
| 43 | Class11:Reentrancy | 0cc01185e3eb6a7e | 62 | 6 | | |


---

## Class01:ExternalBug (1 contracts)
**What to look for:** Unsafe external call (low-level, delegatecall, selfdestruct)

### Contract b64aaff28af406fa
- **LOC:** 146 | **Functions:** 13
- **Slither:** OK | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/ExternalBug/b64aaff28af406fabfbe2aed87b6d632cb1480f5b3aca65164a3b0ad5f5fabc0.sol`

```solidity
pragma solidity ^0.4.12;

/**
 * Math operations with safety checks
 */
contract SafeMath {
  function safeMul(uint256 a, uint256 b) internal returns (uint256) {
    uint256 c = a * b;
    assert(a == 0 || c / a == b);
    return c;
  }

  function safeDiv(uint256 a, uint256 b) internal returns (uint256) {
    assert(b > 0);
    uint256 c = a / b;
    assert(a == b * c + a % b);
    return c;
  }

  function safeSub(uint256 a, uint256 b) internal returns (uint256) {
    assert(b <= a);
    return a - b;
  }

  function safeAdd(uint256 a, uint256 b) internal returns (uint256) {
    uint256 c = a + b;
    assert(c>=a && c>=b);
    return c;
  }

  function assert(bool assertion) internal {
    if (!assertion) {
      throw;
    }
  }
}
contract POTB is SafeMath{
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public totalSupply;
	address public owner = 0x990a10058406E445feE83127d5eB0e2977C43126;

    /* This creates an array with all balances */
    mapping (address => uint256) public balanceOf;
	mapping (address => uint256) public freezeOf;
    mapping (address => mapping (address => uint256)) public allowance;

    /* This generates a public event on the blockchain that will notify clients */
    event Transfer(address indexed from, address indexed to, uint256 value);

    /* This notifies clients about the amount burnt */
    event Burn(address indexed from, uint256 value);
	
	/* This notifies clients about the amount frozen */
    event Freeze(address indexed from, uint256 value);
	
	/* This notifies clients about the amount unfrozen */
    event Unfreeze(address indexed from, uint256 value);

    /* Initializes contract with initial supply tokens to the creator of the contract */
    function POTB(
        uint256 initialSupply,
        string tokenName,
        uint8 decimalUnits,
        string tokenSymbol
        ) {
        balanceOf[owner] = initialSupply;              // Give the creator all initial tokens
        totalSupply = initialSupply;                        // Update total supply
        name = tokenName;                                   // Set the name for display purposes
        symbol = tokenSymbol;                               // Set the symbol for display purposes
        decimals = decimalUnits;                            // Amount of decimals for display purposes 
        Transfer(this, owner, initialSupply);    
    }

    /* Send coins */
    function transfer(address _to, uint256 _value) {
        if (_to == 0x0) throw;                               // Prevent transfer to 0x0 address. Use burn() instead
		if (_value <= 0) throw; 
        if (balanceOf[msg.sender] < _value) throw;           // Check if the sender has enough
        if (balanceOf[_to] + _value < balanceOf[_to]) throw; // Check for overflows
        balanceOf[msg.sender] = SafeMath.safeSub(balanceOf[msg.sender], _value);                     // Subtract from the sender
        balanceOf[_to] = SafeMath.safeAdd(bal

// ... truncated (6074 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

---

## Class02:GasException (4 contracts)
**What to look for:** Gas-dependent loops or operations that may exceed block gas limit

### Contract afc20d2fcb52f714
- **LOC:** 1141 | **Functions:** 85
- **Slither:** OK | **Aderyn:** OK
- **Source:** `BCCC-SCsVul-2024/Source Codes/GasException/afc20d2fcb52f714b4fce4fb66a12b1da8efb4898d899f9f8d9db5cf6538de7c.sol`

```solidity
pragma solidity ^0.4.24;

// File: openzeppelin-solidity/contracts/ownership/Ownable.sol

/**
 * @title Ownable
 * @dev The Ownable contract has an owner address, and provides basic authorization control
 * functions, this simplifies the implementation of "user permissions".
 */
contract Ownable {
  address private _owner;

  event OwnershipTransferred(
    address indexed previousOwner,
    address indexed newOwner
  );

  /**
   * @dev The Ownable constructor sets the original `owner` of the contract to the sender
   * account.
   */
  constructor() internal {
    _owner = msg.sender;
    emit OwnershipTransferred(address(0), _owner);
  }

  /**
   * @return the address of the owner.
   */
  function owner() public view returns(address) {
    return _owner;
  }

  /**
   * @dev Throws if called by any account other than the owner.
   */
  modifier onlyOwner() {
    require(isOwner());
    _;
  }

  /**
   * @return true if `msg.sender` is the owner of the contract.
   */
  function isOwner() public view returns(bool) {
    return msg.sender == _owner;
  }

  /**
   * @dev Allows the current owner to relinquish control of the contract.
   * @notice Renouncing to ownership will leave the contract without an owner.
   * It will not be possible to call the functions with the `onlyOwner`
   * modifier anymore.
   */
  function renounceOwnership() public onlyOwner {
    emit OwnershipTransferred(_owner, address(0));
    _owner = address(0);
  }

  /**
   * @dev Allows the current owner to transfer control of the contract to a newOwner.
   * @param newOwner The address to transfer ownership to.
   */
  function transferOwnership(address newOwner) public onlyOwner {
    _transferOwnership(newOwner);
  }

  /**
   * @dev Transfers control of the contract to a newOwner.
   * @param newOwner The address to transfer ownership to.
   */
  function _transferOwnership(address newOwner) internal {
    require(newOwner != address(0));
    emit OwnershipTransferred(_owner, newOwner);
    _owner = newOwner;
  }
}

// File: openzeppelin-solidity/contracts/token/ERC20/IERC20.sol

/**
 * @title ERC20 interface
 * @dev see https://github.com/ethereum/EIPs/issues/20
 */
interface IERC20 {
  function totalSupply() external view returns (uint256);

  function balanceOf(address who) external view returns (uint256);

  function allowance(address owner, address spender)
    external view returns (uint256);

  function transfer(address to, uint256 value) external returns (bool);

  function approve(address spender, uint256 value)
    external returns (bool);

  function transferFrom(address from, address to, uint256 value)
    external returns (bool);

  event Transfer(
    address indexed from,
    address indexed to,
    uint256 value
  );

  event Approval(
    address indexed owner,
    address indexed spender,
    uint256 value
  );
}

// File: openzeppelin-solidity/contracts/token/ERC20/ERC20Detailed.sol

/**
 * @title ERC20Detailed token
 * @dev The decimals are only

// ... truncated (34572 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 217dfa4632f1c8d8
- **LOC:** 540 | **Functions:** 31
- **Slither:** COMPILE_ERROR | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/DenialOfService/217dfa4632f1c8d84de237a72a409f8184b9bbe665bb336cab32340849c9cdf0.sol`

```solidity
pragma solidity ^0.4.24;


/**
 * @title Math
 * @dev Assorted math operations
 */
library Math {
  function max64(uint64 a, uint64 b) internal pure returns (uint64) {
    return a >= b ? a : b;
  }

  function min64(uint64 a, uint64 b) internal pure returns (uint64) {
    return a < b ? a : b;
  }

  function max256(uint256 a, uint256 b) internal pure returns (uint256) {
    return a >= b ? a : b;
  }

  function min256(uint256 a, uint256 b) internal pure returns (uint256) {
    return a < b ? a : b;
  }
}


/**
 * @title SafeMath
 * @dev Math operations with safety checks that throw on error
 */
library SafeMath {

  /**
  * @dev Multiplies two numbers, throws on overflow.
  */
  function mul(uint256 a, uint256 b) internal pure returns (uint256 c) {
    // Gas optimization: this is cheaper than asserting 'a' not being zero, but the
    // benefit is lost if 'b' is also tested.
    // See: https://github.com/OpenZeppelin/openzeppelin-solidity/pull/522
    if (a == 0) {
      return 0;
    }

    c = a * b;
    assert(c / a == b);
    return c;
  }

  /**
  * @dev Integer division of two numbers, truncating the quotient.
  */
  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    // assert(b > 0); // Solidity automatically throws when dividing by 0
    // uint256 c = a / b;
    // assert(a == b * c + a % b); // There is no case in which this doesn't hold
    return a / b;
  }

  /**
  * @dev Subtracts two numbers, throws on overflow (i.e. if subtrahend is greater than minuend).
  */
  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
  }

  /**
  * @dev Adds two numbers, throws on overflow.
  */
  function add(uint256 a, uint256 b) internal pure returns (uint256 c) {
    c = a + b;
    assert(c >= a);
    return c;
  }
}


/**
 * @title Roles
 * @author Francisco Giordano (@frangio)
 * @dev Library for managing addresses assigned to a Role.
 *      See RBAC.sol for example usage.
 */
library Roles {
  struct Role {
    mapping (address => bool) bearer;
  }

  /**
   * @dev give an address access to this role
   */
  function add(Role storage role, address addr)
    internal
  {
    role.bearer[addr] = true;
  }

  /**
   * @dev remove an address' access to this role
   */
  function remove(Role storage role, address addr)
    internal
  {
    role.bearer[addr] = false;
  }

  /**
   * @dev check if an address has this role
   * // reverts
   */
  function check(Role storage role, address addr)
    view
    internal
  {
    require(has(role, addr));
  }

  /**
   * @dev check if an address has this role
   * @return bool
   */
  function has(Role storage role, address addr)
    view
    internal
    returns (bool)
  {
    return role.bearer[addr];
  }
}


/**
 * @title RBAC (Role-Based Access Control)
 * @author Matt Condon (@Shrugs)
 * @dev Stores and provides setters and getters for roles and addresses.
 * @dev Supports unlimited numbers of roles and addresses.
 * @dev 

// ... truncated (13916 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 07aa3c486b42f189
- **LOC:** 902 | **Functions:** 74
- **Slither:** OK | **Aderyn:** OK
- **Source:** `BCCC-SCsVul-2024/Source Codes/GasException/07aa3c486b42f189b86d411f05c6e42759fae641b3aa9f136b912fab92345de4.sol`

```solidity
pragma solidity ^0.5.3;

contract Operator {
    uint256 public ONE_DAY = 86400;
    uint256 public MIN_DEP = 1 ether;
    uint256 public MAX_DEP = 100 ether;
    address public admin;
    address public admin2;
    address public querierAddress;
    uint256 public depositedAmountGross = 0;
    uint256 public paySystemCommissionTimes = 1;
    uint256 public payDailyIncomeTimes = 1;
    uint256 public lastPaySystemCommission = now;
    uint256 public lastPayDailyIncome = now;
    uint256 public contractStartAt = now;
    uint256 public lastReset = now;
    address payable public operationFund = 0x4357DE4549a18731fA8bF3c7b69439E87FAff8F6;
    address[] public investorAddresses;
    bytes32[] public investmentIds;
    bytes32[] public withdrawalIds;
    bytes32[] public maxOutIds;
    mapping (address => Investor) investors;
    mapping (bytes32 => Investment) public investments;
    mapping (bytes32 => Withdrawal) public withdrawals;
    mapping (bytes32 => MaxOut) public maxOuts;
    uint256 additionNow = 0;

    uint256 public maxLevelsAddSale = 200;
    uint256 public maximumMaxOutInWeek = 2;
    bool public importing = true;

    Vote public currentVote;

    struct Vote {
        uint256 startTime;
        string reason;
        mapping (address => uint8) votes;
        address payable emergencyAddress;
        uint256 yesPoint;
        uint256 noPoint;
        uint256 totalPoint;
    }

    struct Investment {
        bytes32 id;
        uint256 at;
        uint256 amount;
        address investor;
        address nextInvestor;
        bool nextBranch;
    }

    struct Withdrawal {
        bytes32 id;
        uint256 at;
        uint256 amount;
        address investor;
        address presentee;
        uint256 reason;
        uint256 times;
    }

    struct Investor {
        address parent;
        address leftChild;
        address rightChild;
        address presenter;
        uint256 generation;
        uint256 depositedAmount;
        uint256 withdrewAmount;
        bool isDisabled;
        uint256 lastMaxOut;
        uint256 maxOutTimes;
        uint256 maxOutTimesInWeek;
        uint256 totalSell;
        uint256 sellThisMonth;
        uint256 rightSell;
        uint256 leftSell;
        uint256 reserveCommission;
        uint256 dailyIncomeWithrewAmount;
        uint256 registerTime;
        uint256 minDeposit;
        bytes32[] investments;
        bytes32[] withdrawals;
    }

    struct MaxOut {
        bytes32 id;
        address investor;
        uint256 times;
        uint256 at;
    }

    constructor () public { admin = msg.sender; }
    
    modifier mustBeAdmin() {
        require(msg.sender == admin || msg.sender == querierAddress || msg.sender == admin2);
        _;
    }

    modifier mustBeImporting() { require(importing); require(msg.sender == querierAddress || msg.sender == admin); _; }
    
    function () payable external { deposit(); }

    function getNow() internal view returns(uint256) {
        return additi

// ... truncated (41056 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract c6bd49b4aec50e0e
- **LOC:** 330 | **Functions:** 39
- **Slither:** OK | **Aderyn:** OK
- **Source:** `BCCC-SCsVul-2024/Source Codes/ExternalBug/c6bd49b4aec50e0ef02b3f96dda768c594cd5611ea2796b4d60bd4aae4c55b00.sol`

```solidity
pragma solidity ^0.4.18;

contract Token {

  function totalSupply () constant returns (uint256 _totalSupply);

  function balanceOf (address _owner) constant returns (uint256 balance);

  function transfer (address _to, uint256 _value) returns (bool success);

  function transferFrom (address _from, address _to, uint256 _value) returns (bool success);

  function approve (address _spender, uint256 _value) returns (bool success);

  function allowance (address _owner, address _spender) constant returns (uint256 remaining);

  event Transfer (address indexed _from, address indexed _to, uint256 _value);

  event Approval (address indexed _owner, address indexed _spender, uint256 _value);
}

contract SafeMath {
  uint256 constant private MAX_UINT256 =
  0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;

  function safeAdd (uint256 x, uint256 y) constant internal returns (uint256 z) {
    assert (x <= MAX_UINT256 - y);
    return x + y;
  }

  function safeSub (uint256 x, uint256 y) constant internal returns (uint256 z) {
    assert (x >= y);
    return x - y;
  }

  function safeMul (uint256 x, uint256 y)  constant internal  returns (uint256 z) {
    if (y == 0) return 0; // Prevent division by zero at the next line
    assert (x <= MAX_UINT256 / y);
    return x * y;
  }
  
  
   function safeDiv(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a / b;
    return c;
  }
  
}


contract AbstractToken is Token, SafeMath {

  function AbstractToken () {
    // Do nothing
  }
 
  function balanceOf (address _owner) constant returns (uint256 balance) {
    return accounts [_owner];
  }

  function transfer (address _to, uint256 _value) returns (bool success) {
    if (accounts [msg.sender] < _value) return false;
    if (_value > 0 && msg.sender != _to) {
      accounts [msg.sender] = safeSub (accounts [msg.sender], _value);
      accounts [_to] = safeAdd (accounts [_to], _value);
    }
    Transfer (msg.sender, _to, _value);
    return true;
  }

  function transferFrom (address _from, address _to, uint256 _value)  returns (bool success) {
    if (allowances [_from][msg.sender] < _value) return false;
    if (accounts [_from] < _value) return false;

    allowances [_from][msg.sender] =
      safeSub (allowances [_from][msg.sender], _value);

    if (_value > 0 && _from != _to) {
      accounts [_from] = safeSub (accounts [_from], _value);
      accounts [_to] = safeAdd (accounts [_to], _value);
    }
    Transfer (_from, _to, _value);
    return true;
  }

 
  function approve (address _spender, uint256 _value) returns (bool success) {
    allowances [msg.sender][_spender] = _value;
    Approval (msg.sender, _spender, _value);
    return true;
  }

  
  function allowance (address _owner, address _spender) constant returns (uint256 remaining) {
    return allowances [_owner][_spender];
  }

  /**
   * Mapping from addresses of token holders to the numbers of tokens belonging
   * to these token holders.
   */
  

// ... truncated (8197 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

---

## Class03:MishandledException (2 contracts)
**What to look for:** Exception not handled (unchecked call/transfer/return)

### Contract e97c9438d1928e75
- **LOC:** 117 | **Functions:** 13
- **Slither:** OK | **Aderyn:** OK
- **Source:** `BCCC-SCsVul-2024/Source Codes/IntegerUO/e97c9438d1928e75f98442000f481119d94d6b319b2f97aec6f6b4fed7a07bd0.sol`

```solidity
/**
 *Submitted for verification at Etherscan.io on 2018-04-20
*/

pragma solidity >=0.4.25 <0.6.0;


// https://github.com/ethereum/wiki/wiki/Standardized_Contract_APIs#transferable-fungibles-see-erc-20-for-the-latest

contract ERC20Token {
    // Triggered when tokens are transferred.
    event Transfer(address indexed _from, address indexed _to, uint256 _value);
    // Triggered whenever approve(address _spender, uint256 _value) is called.
    event Approval(address indexed _owner, address indexed _spender, uint256 _value);

    // Get the total token supply
    function totalSupply() view public returns (uint256 supply);

    // Get the account `balance` of another account with address `_owner`
    function balanceOf(address _owner) view public returns (uint256 balance);

    // Send `_value` amount of tokens to address `_to`
    function transfer(address _to, uint256 _value) public returns (bool success);

    // Send `_value` amount of tokens from address `_from` to address `_to`
    // The `transferFrom` method is used for a withdraw workflow, allowing contracts to send tokens on your behalf,
    // for example to "deposit" to a contract address and/or to charge fees in sub-currencies;
    // the command should fail unless the `_from` account has deliberately authorized the sender of the message
    // via some mechanism; we propose these standardized APIs for `approval`:
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success);

    // Allow _spender to withdraw from your account, multiple times, up to the _value amount.
    // If this function is called again it overwrites the current allowance with _value.
    function approve(address _spender, uint256 _value) public returns (bool success);

    // Returns the amount which _spender is still allowed to withdraw from _owner
    function allowance(address _owner, address _spender) view public returns (uint256 remaining);
}

contract TSNOcoin is ERC20Token {
    address public initialOwner;
    uint256 public supply   = 5000000000 * 10 ** 18;  // 5,000,000,000
    string  public name     = 'TSNOToken';
    uint8   public decimals = 18;
    string  public symbol   = 'TSNO';
    string  public version  = 'v0.1';
    uint    public creationBlock;
    uint    public creationTime;

    mapping (address => uint256) balance;
    mapping (address => mapping (address => uint256)) m_allowance;

    event Transfer(address indexed _from, address indexed _to, uint256 _value);
    event Approval(address indexed _owner, address indexed _spender, uint256 _value);

    constructor() public{
        initialOwner        = msg.sender;
        balance[msg.sender] = supply;
        creationBlock       = block.number;
        creationTime        = block.timestamp;
    }

    function balanceOf(address _account) view public returns (uint) {
        return balance[_account];
    }

    function totalSupply() view public returns (uint) {
        return supply;
    }

    fu

// ... truncated (4733 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 2ad7bbb4d3e569a4
- **LOC:** 245 | **Functions:** 22
- **Slither:** OK | **Aderyn:** OK
- **Source:** `BCCC-SCsVul-2024/Source Codes/IntegerUO/2ad7bbb4d3e569a416c6935ceb67e5cab00d249cde5de353ba4851a394f53058.sol`

```solidity
pragma solidity ^0.4.25;

// ----------------------------------------------------------------------------
//  MCAN Token contract
//
// Symbol      : MCAN
// Name        : MCAN Token
// Total supply: 5000000000
// Decimals    : 18
//


// -------------------------------------------`---------------------------------
// Safe maths
// ----------------------------------------------------------------------------
contract SafeMath {

    /**
     * @dev Subtracts two unsigned integers, reverts on overflow (i.e. if subtrahend is greater than minuend).
     */
    function safeSub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a);
        uint256 c = a - b;

        return c;
    }

    /**
     * @dev Adds two unsigned integers, reverts on overflow.
     */
    function safeAdd(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a);

        return c;
    }

}


// ----------------------------------------------------------------------------
// ERC Token Standard #20 Interface
// https://github.com/ethereum/EIPs/blob/master/EIPS/eip-20-token-standard.md
// ----------------------------------------------------------------------------
contract IERC20 {
    function totalSupply() public constant returns (uint);
    function balanceOf(address tokenOwner) public constant returns (uint balance);
    function allowance(address tokenOwner, address spender) public constant returns (uint remaining);
    function transfer(address to, uint tokens) public returns (bool success);
    function approve(address spender, uint tokens) public returns (bool success);
    function transferFrom(address from, address to, uint tokens) public returns (bool success);

    event Transfer(address indexed from, address indexed to, uint tokens);
    event Approval(address indexed tokenOwner, address indexed spender, uint tokens);
}



contract Ownable {
    address public _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /**
     * @dev The Ownable constructor sets the original `owner` of the contract to the sender
     * account.
     */
    constructor () internal {
        _owner = msg.sender;
        emit OwnershipTransferred(address(0), _owner);
    }

    /**
     * @return the address of the owner.
     */
    function owner() public view returns (address) {
        return _owner;
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        require(isOwner());
        _;
    }

    /**
     * @return true if `msg.sender` is the owner of the contract.
     */
    function isOwner() public view returns (bool) {
        return msg.sender == _owner;
    }

    /**
     * @dev Allows the current owner to relinquish control of the contract.
     * @notice Renouncing to ownership will leave the contract without an owner.
     * It will not be possible to call the functions with the `onlyOwne

// ... truncated (8812 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

---

## Class04:Timestamp (1 contracts)
**What to look for:** Block timestamp dependency

### Contract b1982e1e1108812d
- **LOC:** 359 | **Functions:** 39
- **Slither:** OK | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/IntegerUO/b1982e1e1108812d540de03b9da72f43c23553358b9e5c619329a5ba4fc76276.sol`

```solidity
pragma solidity ^0.4.8;


/**
 * Math operations with safety checks
 * By OpenZeppelin: https://github.com/OpenZeppelin/zeppelin-solidity/contracts/SafeMath.sol
 */
library SafeMath {
  function mul(uint256 a, uint256 b) internal returns (uint256) {
    uint256 c = a * b;
    if(!(a == 0 || c / a == b)) throw;
    return c;
  }

  function div(uint256 a, uint256 b) internal returns (uint256) {
    // assert(b > 0); // Solidity automatically throws when dividing by 0
    uint256 c = a / b;
    // assert(a == b * c + a % b); // There is no case in which this doesn't hold
    return c;
  }

  function sub(uint256 a, uint256 b) internal returns (uint256) {
    if(!(b <= a)) throw;
    return a - b;
  }

  function add(uint256 a, uint256 b) internal returns (uint256) {
    uint256 c = a + b;
    if(!(c >= a)) throw;
    return c;
  }

  function max64(uint64 a, uint64 b) internal constant returns (uint64) {
    return a >= b ? a : b;
  }

  function min64(uint64 a, uint64 b) internal constant returns (uint64) {
    return a < b ? a : b;
  }

  function max256(uint256 a, uint256 b) internal constant returns (uint256) {
    return a >= b ? a : b;
  }

  function min256(uint256 a, uint256 b) internal constant returns (uint256) {
    return a < b ? a : b;
  }

}

contract ContractReceiver{
    function tokenFallback(address _from, uint256 _value, bytes  _data) external;
}


//Basic ERC23 token, backward compatible with ERC20 transfer function.
//Based in part on code by open-zeppelin: https://github.com/OpenZeppelin/zeppelin-solidity.git
contract ERC23BasicToken {
    using SafeMath for uint256;
    uint256 public totalSupply;
    mapping(address => uint256) balances;
    event Transfer(address indexed from, address indexed to, uint256 value);
    

    /*
       * Fix for the ERC20 short address attack  
      */
      modifier onlyPayloadSize(uint size) {
         if(msg.data.length < size + 4) {
           throw;
         }
         _;
      }


    function tokenFallback(address _from, uint256 _value, bytes  _data) external {
        _from;
        _value;
        _data;
        throw;
    }

    function transfer(address _to, uint256 _value, bytes _data)  returns (bool success) {

        //Standard ERC23 transfer function

        if(isContract(_to)) {
            transferToContract(_to, _value, _data);
        }
        else {
            transferToAddress(_to, _value, _data);
        }
        return true;
    }

    function transfer(address _to, uint256 _value) onlyPayloadSize(2 * 32) {

        //standard function transfer similar to ERC20 transfer with no _data
        //added due to backwards compatibility reasons

        bytes memory empty;
        if(isContract(_to)) {
            transferToContract(_to, _value, empty);
        }
        else {
            transferToAddress(_to, _value, empty);
        }
    }

    function transferToAddress(address _to, uint256 _value, bytes _data)  internal {
        _data;
        balances[msg.sender] =

// ... truncated (13162 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

---

## Class06:UnusedReturn (4 contracts)
**What to look for:** Return value of external call not checked

### Contract da081796d0c57c5f
- **LOC:** 463 | **Functions:** 27
- **Slither:** OK | **Aderyn:** OK
- **Source:** `BCCC-SCsVul-2024/Source Codes/IntegerUO/da081796d0c57c5f840373aaf45022acbf507cfe1a5191eaf7f240f2c5df3d5c.sol`

```solidity
pragma solidity ^0.4.23;

// File: openzeppelin-solidity/contracts/ownership/Ownable.sol

/**
 * @title Ownable
 * @dev The Ownable contract has an owner address, and provides basic authorization control
 * functions, this simplifies the implementation of "user permissions".
 */
contract Ownable {
  address public owner;


  event OwnershipRenounced(address indexed previousOwner);
  event OwnershipTransferred(
    address indexed previousOwner,
    address indexed newOwner
  );


  /**
   * @dev The Ownable constructor sets the original `owner` of the contract to the sender
   * account.
   */
  constructor() public {
    owner = msg.sender;
  }

  /**
   * @dev Throws if called by any account other than the owner.
   */
  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }

  /**
   * @dev Allows the current owner to transfer control of the contract to a newOwner.
   * @param newOwner The address to transfer ownership to.
   */
  function transferOwnership(address newOwner) public onlyOwner {
    require(newOwner != address(0));
    emit OwnershipTransferred(owner, newOwner);
    owner = newOwner;
  }

  /**
   * @dev Allows the current owner to relinquish control of the contract.
   */
  function renounceOwnership() public onlyOwner {
    emit OwnershipRenounced(owner);
    owner = address(0);
  }
}

// File: openzeppelin-solidity/contracts/ownership/HasNoEther.sol

/**
 * @title Contracts that should not own Ether
 * @author Remco Bloemen <remco@2π.com>
 * @dev This tries to block incoming ether to prevent accidental loss of Ether. Should Ether end up
 * in the contract, it will allow the owner to reclaim this ether.
 * @notice Ether can still be sent to this contract by:
 * calling functions labeled `payable`
 * `selfdestruct(contract_address)`
 * mining directly to the contract address
 */
contract HasNoEther is Ownable {

  /**
  * @dev Constructor that rejects incoming Ether
  * @dev The `payable` flag is added so we can access `msg.value` without compiler warning. If we
  * leave out payable, then Solidity will allow inheriting contracts to implement a payable
  * constructor. By doing it this way we prevent a payable constructor from working. Alternatively
  * we could use assembly to access msg.value.
  */
  constructor() public payable {
    require(msg.value == 0);
  }

  /**
   * @dev Disallows direct send by settings a default function without the `payable` flag.
   */
  function() external {
  }

  /**
   * @dev Transfer all Ether held by the contract to the owner.
   */
  function reclaimEther() external onlyOwner {
    owner.transfer(this.balance);
  }
}

// File: openzeppelin-solidity/contracts/math/SafeMath.sol

/**
 * @title SafeMath
 * @dev Math operations with safety checks that throw on error
 */
library SafeMath {

  /**
  * @dev Multiplies two numbers, throws on overflow.
  */
  function mul(uint256 a, uint256 b) internal pure returns (uint256 c) {
    if (a == 0) {
      return 0;
    }
    c = a * b;
   

// ... truncated (13426 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 230347c96df3515f
- **LOC:** 287 | **Functions:** 33
- **Slither:** OK | **Aderyn:** OK
- **Source:** `BCCC-SCsVul-2024/Source Codes/IntegerUO/230347c96df3515f03c62c7ccb39068a01342a8c629ace227780bae778ba2854.sol`

```solidity
pragma solidity ^0.4.19;

library SafeMath {
  function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a * b;
    assert(a == 0 || c / a == b);
    return c;
  }

  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a / b;
    return c;
  }

  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
  }

  function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    assert(c >= a);
    return c;
  }
}

contract ForeignToken {
    function balanceOf(address _owner) constant public returns (uint256);
    function transfer(address _to, uint256 _value) public returns (bool);
}

contract ERC20Basic {
    uint256 public totalSupply;
    function balanceOf(address who) public constant returns (uint256);
    function transfer(address to, uint256 value) public returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
}

contract ERC20 is ERC20Basic {
    function allowance(address owner, address spender) public constant returns (uint256);
    function transferFrom(address from, address to, uint256 value) public returns (bool);
    function approve(address spender, uint256 value) public returns (bool);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

interface Token { 
    function distr(address _to, uint256 _value) public returns (bool);
    function totalSupply() constant public returns (uint256 supply);
    function balanceOf(address _owner) constant public returns (uint256 balance);
}

contract ZEC is ERC20 {
    
    using SafeMath for uint256;
    address owner = msg.sender;

    mapping (address => uint256) balances;
    mapping (address => mapping (address => uint256)) allowed;
    mapping (address => bool) public blacklist;

    string public constant name = "ZECHAIN";
    string public constant symbol = "ZEC";
    uint public constant decimals = 8;
    
        uint256 public totalSupply = 100000000e8;
    uint256 public totalDistributed = 8000000e8;
    uint256 public totalRemaining = totalSupply.sub(totalDistributed);
    uint256 public value;

    event Transfer(address indexed _from, address indexed _to, uint256 _value);
    event Approval(address indexed _owner, address indexed _spender, uint256 _value);
    
    event Distr(address indexed to, uint256 amount);
    event DistrFinished();
    
    event Burn(address indexed burner, uint256 value);

    bool public distributionFinished = false;
    
    modifier canDistr() {
        require(!distributionFinished);
        _;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }
    
    modifier onlyWhitelist() {
        require(blacklist[msg.sender] == false);
        _;
    }
    
    function ZEC () public {
        owner = msg.sender;
        value = 400e8;
        distr(owner, totalDistributed);
    }
    
    function transferOwnersh

// ... truncated (9007 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 0829214881ec3423
- **LOC:** 524 | **Functions:** 40
- **Slither:** COMPILE_ERROR | **Aderyn:** OK
- **Source:** `BCCC-SCsVul-2024/Source Codes/IntegerUO/0829214881ec3423110ac8e76354caaa42e91ac043c9f61c514b5fc1b7db2023.sol`

```solidity
pragma solidity ^0.5.2;

/**
 * @title ERC20 interface
 * @dev see https://eips.ethereum.org/EIPS/eip-20
 */
interface IERC20 {
    function transfer(address to, uint256 value) external returns (bool);

    function approve(address spender, uint256 value) external returns (bool);

    function transferFrom(address from, address to, uint256 value) external returns (bool);

    function totalSupply() external view returns (uint256);

    function balanceOf(address who) external view returns (uint256);

    function allowance(address owner, address spender) external view returns (uint256);

    event Transfer(address indexed from, address indexed to, uint256 value);

    event Approval(address indexed owner, address indexed spender, uint256 value);
}

/**------------------------------------------------------------------------------------------**/

pragma solidity ^0.5.2;

/**
 * @title SafeMath
 * @dev Unsigned math operations with safety checks that revert on error
 */
library SafeMath {
    /**
     * @dev Multiplies two unsigned integers, reverts on overflow.
     */
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        // Gas optimization: this is cheaper than requiring 'a' not being zero, but the
        // benefit is lost if 'b' is also tested.
        // See: https://github.com/OpenZeppelin/openzeppelin-solidity/pull/522
        if (a == 0) {
            return 0;
        }

        uint256 c = a * b;
        require(c / a == b);

        return c;
    }

    /**
     * @dev Integer division of two unsigned integers truncating the quotient, reverts on division by zero.
     */
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        // Solidity only automatically asserts when dividing by 0
        require(b > 0);
        uint256 c = a / b;
        // assert(a == b * c + a % b); // There is no case in which this doesn't hold

        return c;
    }

    /**
     * @dev Subtracts two unsigned integers, reverts on overflow (i.e. if subtrahend is greater than minuend).
     */
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a);
        uint256 c = a - b;

        return c;
    }

    /**
     * @dev Adds two unsigned integers, reverts on overflow.
     */
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a);

        return c;
    }

    /**
     * @dev Divides two unsigned integers and returns the remainder (unsigned integer modulo),
     * reverts when dividing by zero.
     */
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b != 0);
        return a % b;
    }
}

/**------------------------------------------------------------------------------------------**/

pragma solidity ^0.5.2;

/**
 * @title Standard ERC20 token
 *
 * @dev Implementation of the basic standard token.
 * https://eips.ethereum.org/EIPS/eip-20
 * Originally based on co

// ... truncated (16825 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract fd1df8d6413c2f20
- **LOC:** 636 | **Functions:** 36
- **Slither:** OK | **Aderyn:** OK
- **Source:** `BCCC-SCsVul-2024/Source Codes/CallToUnknown/fd1df8d6413c2f20b272ad9fc9b18be6c4cc6c6f16506fb2de7caa84c533df12.sol`

```solidity
pragma solidity ^0.4.21;

// File: zeppelin-solidity/contracts/ownership/Ownable.sol

/**
 * @title Ownable
 * @dev The Ownable contract has an owner address, and provides basic authorization control
 * functions, this simplifies the implementation of "user permissions".
 */
contract Ownable {
  address public owner;


  event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);


  /**
   * @dev The Ownable constructor sets the original `owner` of the contract to the sender
   * account.
   */
     constructor() public {
    owner = msg.sender;
  }

  /**
   * @dev Throws if called by any account other than the owner.
   */
  modifier onlyOwner() {
    require(msg.sender == owner, "You are not the owner of this contract.");
    _;
  }

  /**
   * @dev Allows the current owner to transfer control of the contract to a newOwner.
   * @param newOwner The address to transfer ownership to.
   */
  function transferOwnership(address newOwner) public onlyOwner {
    require(newOwner != address(0));
    emit OwnershipTransferred(owner, newOwner);
    owner = newOwner;
  }

}

// File: zeppelin-solidity/contracts/lifecycle/Destructible.sol

/**
 * @title Destructible
 * @dev Base contract that can be destroyed by owner. All funds in contract will be sent to the owner.
 */
contract Destructible is Ownable {

  constructor() public payable { }

  /**
   * @dev Transfers the current balance to the owner and terminates the contract.
   */
  function destroy() onlyOwner public {
    selfdestruct(owner);
  }

  function destroyAndSend(address _recipient) onlyOwner public {
    selfdestruct(_recipient);
  }
}

// File: zeppelin-solidity/contracts/lifecycle/Pausable.sol

/**
 * @title Pausable
 * @dev Base contract which allows children to implement an emergency stop mechanism.
 */
contract Pausable is Ownable {
  event Pause();
  event Unpause();

  bool public paused = false;


  /**
   * @dev Modifier to make a function callable only when the contract is not paused.
   */
  modifier whenNotPaused() {
    require(!paused, "this contract is paused");
    _;
  }

  /**
   * @dev Modifier to make a function callable only when the contract is paused.
   */
  modifier whenPaused() {
    require(paused);
    _;
  }

  /**
   * @dev called by the owner to pause, triggers stopped state
   */
  function pause() onlyOwner whenNotPaused public {
    paused = true;
    emit Pause();
  }

  /**
   * @dev called by the owner to unpause, returns to normal state
   */
  function unpause() onlyOwner whenPaused public {
    paused = false;
    emit Unpause();
  }
}

// File: zeppelin-solidity/contracts/math/SafeMath.sol

/**
 * @title SafeMath
 * @dev Math operations with safety checks that throw on error
 */
library SafeMath {

  /**
  * @dev Multiplies two numbers, throws on overflow.
  */
  function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    if (a == 0) {
      return 0;
    }
    uint256 c = a * b;
    assert(c / a == b);
    re

// ... truncated (18646 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

---

## Class08:CallToUnknown (6 contracts)
**What to look for:** Call to untrusted/unknown address

### Contract 2be5c39959fc3e9e
- **LOC:** 182 | **Functions:** 17
- **Slither:** COMPILE_ERROR | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/CallToUnknown/2be5c39959fc3e9e5af17958d1196675e8f892db005163ff37b2b82d54a480dc.sol`

```solidity
//ERC20 Token
pragma solidity ^0.4.2;
contract owned {
    address public owner;

    function owned() {
        owner = msg.sender;
    }

    modifier onlyOwner {
        if (msg.sender != owner) throw;
        _;
    }

    function transferOwnership(address newOwner) onlyOwner {
        owner = newOwner;
    }
}

contract tokenRecipient { function receiveApproval(address _from, uint256 _value, address _token, bytes _extraData); }

contract token {
    /* Public variables of the token */
    string public standard = "DestiNeed 1.0";
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public totalSupply;

    /* This creates an array with all balances */
    mapping (address => uint256) public balanceOf;
    mapping (address => mapping (address => uint256)) public allowance;

    /* This generates a public event on the blockchain that will notify clients */
    event Transfer(address indexed from, address indexed to, uint256 value);

    /* Initializes contract with initial supply tokens to the creator of the contract */
    function token(
        uint256 initialSupply,
        string tokenName,
        uint8 decimalUnits,
        string tokenSymbol
        ) {
        balanceOf[msg.sender] = initialSupply;              // Give the creator all initial tokens
        totalSupply = initialSupply;                        // Update total supply
        name = tokenName;                                   // Set the name for display purposes
        symbol = tokenSymbol;                               // Set the symbol for display purposes
        decimals = decimalUnits;                            // Amount of decimals for display purposes
    }

    /* Send coins */
    function transfer(address _to, uint256 _value) {
        if (balanceOf[msg.sender] < _value) throw;           // Check if the sender has enough
        if (balanceOf[_to] + _value < balanceOf[_to]) throw; // Check for overflows
        balanceOf[msg.sender] -= _value;                     // Subtract from the sender
        balanceOf[_to] += _value;                            // Add the same to the recipient
        Transfer(msg.sender, _to, _value);                   // Notify anyone listening that this transfer took place
    }

    /* Allow another contract to spend some tokens in your behalf */
    function approve(address _spender, uint256 _value)
        returns (bool success) {
        allowance[msg.sender][_spender] = _value;
        return true;
    }

    /* Approve and then communicate the approved contract in a single tx */
    function approveAndCall(address _spender, uint256 _value, bytes _extraData)
        returns (bool success) {
        tokenRecipient spender = tokenRecipient(_spender);
        if (approve(_spender, _value)) {
            spender.receiveApproval(msg.sender, _value, this, _extraData);
            return true;
        }
    }

    /* A contract attempts _ to get the coins */
    function transferFrom(address _from, 

// ... truncated (7941 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 0c4b244a4391b1b4
- **LOC:** 270 | **Functions:** 20
- **Slither:** COMPILE_ERROR | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/CallToUnknown/0c4b244a4391b1b4313ebc7cc7a27c24ac1795a2c16cbdc3f9f0767fcbab4fed.sol`

```solidity
pragma solidity ^0.4.21;

// File: zeppelin-solidity/contracts/math/SafeMath.sol

/**
 * @title SafeMath
 * @dev Math operations with safety checks that throw on error
 */
library SafeMath {

  /**
  * @dev Multiplies two numbers, throws on overflow.
  */
  function mul(uint256 a, uint256 b) internal pure returns (uint256 c) {
    if (a == 0) {
      return 0;
    }
    c = a * b;
    assert(c / a == b);
    return c;
  }

  /**
  * @dev Integer division of two numbers, truncating the quotient.
  */
  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    // assert(b > 0); // Solidity automatically throws when dividing by 0
    // uint256 c = a / b;
    // assert(a == b * c + a % b); // There is no case in which this doesn't hold
    return a / b;
  }

  /**
  * @dev Subtracts two numbers, throws on overflow (i.e. if subtrahend is greater than minuend).
  */
  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
  }

  /**
  * @dev Adds two numbers, throws on overflow.
  */
  function add(uint256 a, uint256 b) internal pure returns (uint256 c) {
    c = a + b;
    assert(c >= a);
    return c;
  }
}

// File: zeppelin-solidity/contracts/token/ERC20/ERC20Basic.sol

/**
 * @title ERC20Basic
 * @dev Simpler version of ERC20 interface
 * @dev see https://github.com/ethereum/EIPs/issues/179
 */
contract ERC20Basic {
  function totalSupply() public view returns (uint256);
  function balanceOf(address who) public view returns (uint256);
  function transfer(address to, uint256 value) public returns (bool);
  event Transfer(address indexed from, address indexed to, uint256 value);
}

// File: zeppelin-solidity/contracts/token/ERC20/BasicToken.sol

/**
 * @title Basic token
 * @dev Basic version of StandardToken, with no allowances.
 */
contract BasicToken is ERC20Basic {
  using SafeMath for uint256;

  mapping(address => uint256) balances;

  uint256 totalSupply_;

  /**
  * @dev total number of tokens in existence
  */
  function totalSupply() public view returns (uint256) {
    return totalSupply_;
  }

  /**
  * @dev transfer token for a specified address
  * @param _to The address to transfer to.
  * @param _value The amount to be transferred.
  */
  function transfer(address _to, uint256 _value) public returns (bool) {
    require(_to != address(0));
    require(_value <= balances[msg.sender]);

    balances[msg.sender] = balances[msg.sender].sub(_value);
    balances[_to] = balances[_to].add(_value);
    emit Transfer(msg.sender, _to, _value);
    return true;
  }

  /**
  * @dev Gets the balance of the specified address.
  * @param _owner The address to query the the balance of.
  * @return An uint256 representing the amount owned by the passed address.
  */
  function balanceOf(address _owner) public view returns (uint256) {
    return balances[_owner];
  }

}

// File: zeppelin-solidity/contracts/token/ERC20/ERC20.sol

/**
 * @title ERC20 interface
 * @dev see https://github

// ... truncated (9020 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 7300cf995dd3b67d
- **LOC:** 913 | **Functions:** 56
- **Slither:** OK | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/CallToUnknown/7300cf995dd3b67d113d83c0f5c2bc13ad826b01ebb993e03c01f9890b04a328.sol`

```solidity
pragma solidity 0.4.25;

// File: contracts/LinkedListLib.sol

/**
 * @title LinkedListLib
 * @author Darryl Morris (o0ragman0o) and Modular.network
 *
 * This utility library was forked from https://github.com/o0ragman0o/LibCLL
 * into the Modular-Network ethereum-libraries repo at https://github.com/Modular-Network/ethereum-libraries
 * It has been updated to add additional functionality and be more compatible with solidity 0.4.18
 * coding patterns.
 *
 * version 1.1.1
 * Copyright (c) 2017 Modular Inc.
 * The MIT License (MIT)
 * https://github.com/Modular-network/ethereum-libraries/blob/master/LICENSE
 *
 * The LinkedListLib provides functionality for implementing data indexing using
 * a circlular linked list
 *
 * Modular provides smart contract services and security reviews for contract
 * deployments in addition to working on open source projects in the Ethereum
 * community. Our purpose is to test, document, and deploy reusable code onto the
 * blockchain and improve both security and usability. We also educate non-profits,
 * schools, and other community members about the application of blockchain
 * technology. For further information: modular.network
 *
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/


library LinkedListLib {

    uint256 constant NULL = 0;
    uint256 constant HEAD = 0;
    bool constant PREV = false;
    bool constant NEXT = true;

    struct LinkedList{
        mapping (uint256 => mapping (bool => uint256)) list;
    }

    /// @dev returns true if the list exists
    /// @param self stored linked list from contract
    function listExists(LinkedList storage self)
        public
        view returns (bool)
    {
        // if the head nodes previous or next pointers both point to itself, then there are no items in the list
        if (self.list[HEAD][PREV] != HEAD || self.list[HEAD][NEXT] != HEAD) {
            return true;
        } else {
            return false;
        }
    }

    /// @dev returns true if the node exists
    /// @param self stored linked list from contract
    /// @param _node a node to search for
    function nodeExists(LinkedList storage self, uint256 _node)
        public
        view returns (bool)
    {
        if (self.list[_node][PREV] == HEAD && self.list[_node][NEXT] == HEAD) {
            if (self.list[HEAD][NEXT] == _node) {
                return true;
            } else {
                return false;
            }
        } else {
            return true;
        }
    }

    /// @dev Returns the number of elements in the list
    /// @param self

// ... truncated (28002 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 6654483dbdfe6fac
- **LOC:** 100 | **Functions:** 7
- **Slither:** COMPILE_ERROR | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/CallToUnknown/6654483dbdfe6facd7a0d94ffc89e9fbc3b646411e3491513d9c7c536d94b928.sol`

```solidity
pragma solidity ^0.4.24;

contract Ownable {
	address public owner;
	address public newOwner;

	event OwnershipTransferred(address indexed oldOwner, address indexed newOwner);

	constructor() public {
		owner = msg.sender;
		newOwner = address(0);
	}

	modifier onlyOwner() {
		require(msg.sender == owner, "msg.sender == owner");
		_;
	}

	function transferOwnership(address _newOwner) public onlyOwner {
		require(address(0) != _newOwner, "address(0) != _newOwner");
		newOwner = _newOwner;
	}

	function acceptOwnership() public {
		require(msg.sender == newOwner, "msg.sender == newOwner");
		emit OwnershipTransferred(owner, msg.sender);
		owner = msg.sender;
		newOwner = address(0);
	}
}

contract tokenInterface {
	function balanceOf(address _owner) public constant returns (uint256 balance);
	function transfer(address _to, uint256 _value) public returns (bool);
}

contract Timelock is Ownable {
	tokenInterface public tokenContract;

	uint256 public releaseTime;

	constructor(address _tokenAddress, uint256 _releaseTime) public {
		tokenContract = tokenInterface(_tokenAddress);
		releaseTime = _releaseTime;
	}

	function () public {
	    if ( msg.sender == newOwner ) acceptOwnership();
		claim();
	}

	function claim() onlyOwner private {
	    require ( now > releaseTime, "now > releaseTime" );

	    uint256 tknToSend = tokenContract.balanceOf(this);
		require(tknToSend > 0,"tknToSend > 0");

		require ( tokenContract.transfer(msg.sender, tknToSend) );
	}

	function unlocked() view public returns(bool) {
	    return now > releaseTime;
	}
}
pragma solidity ^0.3.0;
	 contract IQNSecondPreICO is Ownable {
    uint256 public constant EXCHANGE_RATE = 550;
    uint256 public constant START = 1515402000; 
    uint256 availableTokens;
    address addressToSendEthereum;
    address addressToSendTokenAfterIco;
    uint public amountRaised;
    uint public deadline;
    uint public price;
    token public tokenReward;
    mapping(address => uint256) public balanceOf;
    bool crowdsaleClosed = false;
    function IQNSecondPreICO (
        address addressOfTokenUsedAsReward,
       address _addressToSendEthereum,
        address _addressToSendTokenAfterIco
    ) public {
        availableTokens = 800000 * 10 ** 18;
        addressToSendEthereum = _addressToSendEthereum;
        addressToSendTokenAfterIco = _addressToSendTokenAfterIco;
        deadline = START + 7 days;
        tokenReward = token(addressOfTokenUsedAsReward);
    }
    function () public payable {
        require(now < deadline && now >= START);
        require(msg.value >= 1 ether);
        uint amount = msg.value;
        balanceOf[msg.sender] += amount;
        amountRaised += amount;
        availableTokens -= amount;
        tokenReward.transfer(msg.sender, amount * EXCHANGE_RATE);
        addressToSendEthereum.transfer(amount);
    }
 }

```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 5e0096dbcc1d4f8d
- **LOC:** 300 | **Functions:** 30
- **Slither:** COMPILE_ERROR | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/CallToUnknown/5e0096dbcc1d4f8d1ba12b28e7cce39d7a1819a8ea54b1e59a532389ad8718e5.sol`

```solidity
pragma solidity ^0.4.18;

/**
 * @title SafeMath
 */
library SafeMath {

    /**
    * Multiplies two numbers, throws on overflow.
    */
    function mul(uint256 a, uint256 b) internal pure returns (uint256 c) {
        if (a == 0) {
            return 0;
        }
        c = a * b;
        assert(c / a == b);
        return c;
    }

    /**
    * Integer division of two numbers, truncating the quotient.
    */
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        // assert(b > 0); // Solidity automatically throws when dividing by 0
        // uint256 c = a / b;
        // assert(a == b * c + a % b); // There is no case in which this doesn't hold
        return a / b;
    }

    /**
    * Subtracts two numbers, throws on overflow (i.e. if subtrahend is greater than minuend).
    */
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        assert(b <= a);
        return a - b;
    }

    /**
    * Adds two numbers, throws on overflow.
    */
    function add(uint256 a, uint256 b) internal pure returns (uint256 c) {
        c = a + b;
        assert(c >= a);
        return c;
    }
}

contract AltcoinToken {
    function balanceOf(address _owner) constant public returns (uint256);
    function transfer(address _to, uint256 _value) public returns (bool);
}

contract ERC20Basic {
    uint256 public totalSupply;
    function balanceOf(address who) public constant returns (uint256);
    function transfer(address to, uint256 value) public returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
}

contract ERC20 is ERC20Basic {
    function allowance(address owner, address spender) public constant returns (uint256);
    function transferFrom(address from, address to, uint256 value) public returns (bool);
    function approve(address spender, uint256 value) public returns (bool);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

contract VikkyToken is ERC20 {

    using SafeMath for uint256;
    address owner = msg.sender;

    mapping (address => uint256) balances;
    mapping (address => mapping (address => uint256)) allowed;

    string public constant name = "Vikky Token";
    string public constant symbol = "VIKKY";
    uint public constant decimals = 8;

    uint256 public totalSupply = 20000000000e8;
    uint256 public totalDistributed = 0;
    uint256 public tokensPerEth = 20000000e8;
    uint256 public constant minContribution = 1 ether / 100; // 0.01 Ether

    event Transfer(address indexed _from, address indexed _to, uint256 _value);
    event Approval(address indexed _owner, address indexed _spender, uint256 _value);

    event Distr(address indexed to, uint256 amount);
    event DistrFinished();

    event Airdrop(address indexed _owner, uint _amount, uint _balance);

    event TokensPerEthUpdated(uint _tokensPerEth);

    event Burn(address indexed burner, uint256 value);

    bool public distributionFinished = false

// ... truncated (9382 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 36427ce69f44a76d
- **LOC:** 577 | **Functions:** 27
- **Slither:** COMPILE_ERROR | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/CallToUnknown/36427ce69f44a76d3cdaea5cbe4e41a3b262e7f62bbb5b31b898d904e5c0b643.sol`

```solidity
pragma solidity ^0.4.20;

contract EtherPaint {
   // scaleFactor is used to convert Ether into tokens and vice-versa: they're of different
   // orders of magnitude, hence the need to bridge between the two.
   uint256 constant scaleFactor = 0x10000000000000000; //0x10000000000000000;  // 2^64

   // CRR = 50%
   // CRR is Cash Reserve Ratio (in this case Crypto Reserve Ratio).
   // For more on this: check out https://en.wikipedia.org/wiki/Reserve_requirement
   int constant crr_n = 1; // CRR numerator
   int constant crr_d = 2; // CRR denominator

   // The price coefficient. Chosen such that at 1 token total supply
   // the amount in reserve is 0.5 ether and token price is 1 Ether.
   int constant price_coeff = -0x296ABF784A358468C;

   // Array between each address and their number of tokens.
   mapping(address => uint256[16]) public tokenBalance;

   uint256[128][128] public colorPerCoordinate;
   uint256[16] public colorPerCanvas;

   event colorUpdate(uint8 posx, uint8 posy, uint8 colorid);
   event priceUpdate(uint8 colorid);
   event tokenUpdate(uint8 colorid, address who);
   event dividendUpdate();

   event pushuint(uint256 s);

   // Array between each address and how much Ether has been paid out to it.
   // Note that this is scaled by the scaleFactor variable.
   mapping(address => int256[16]) public payouts;

   // Variable tracking how many tokens are in existence overall.
   uint256[16] public totalSupply;

   uint256 public allTotalSupply;

   // Aggregate sum of all payouts.
   // Note that this is scaled by the scaleFactor variable.
   int256[16] totalPayouts;

   // Variable tracking how much Ether each token is currently worth.
   // Note that this is scaled by the scaleFactor variable.
   uint256[16] earningsPerToken;

   // Current contract balance in Ether
   uint256[16] public contractBalance;

   address public owner;

   uint256 public ownerFee;



   function EtherPaint() public {
       owner = msg.sender;
       colorPerCanvas[0] = 128*128;
      pushuint(1 finney);
   }

   // Returns the number of tokens currently held by _owner.
   function balanceOf(address _owner, uint8 colorid) public constant returns (uint256 balance) {
      if (colorid >= 16){
         revert();
      }
      return tokenBalance[_owner][colorid];
   }

   // Withdraws all dividends held by the caller sending the transaction, updates
   // the requisite global variables, and transfers Ether back to the caller.
   function withdraw(uint8 colorid) public {
      if (colorid >= 16){
         revert();
      }
      // Retrieve the dividends associated with the address the request came from.
      var balance = dividends(msg.sender, colorid);

      // Update the payouts array, incrementing the request address by `balance`.
      payouts[msg.sender][colorid] += (int256) (balance * scaleFactor);

      // Increase the total amount that's been paid out to maintain invariance.
      totalPayouts[colorid] += (int256) (balance * scaleFactor);

   

// ... truncated (21715 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

---

## Class09:DenialOfService (1 contracts)
**What to look for:** Denial of service (looped calls, gas exhaustion)

### Contract b5f07eac33c5f726
- **LOC:** 52 | **Functions:** 5
- **Slither:** COMPILE_ERROR | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/DenialOfService/b5f07eac33c5f7266c98edbaa288cd53642a9e2da7e2a9a9c7e9ac1e9d182767.sol`

```solidity
pragma solidity ^0.4.18;

contract GiftCard2017{
    address owner;
    mapping (address => uint256) public authorizations;

    /// Constructor sets owner.
    function GiftCard2017() public {
        owner = msg.sender;
    }

    /// Redeems authorized ETH.
    function () public {
        uint256 _redemption = authorizations[msg.sender];   // Amount mEth available to redeem.
        require (_redemption > 0);
        authorizations[msg.sender] = 0;                     // Clear authorization.
        msg.sender.transfer(_redemption * 1e15);            // convert mEth to wei for transfer()
    }

    /// Contract owner deposits ETH.
    function deposit() public payable OwnerOnly {
    }

    /// Contract owner withdraws ETH.
    function withdraw(uint256 _amount) public OwnerOnly {
        owner.transfer(_amount);
    }

    /// Contract owner authorizes redemptions in units of 1/1000 ETH.
    function authorize(address _addr, uint256 _amount_mEth) public OwnerOnly {
        require (this.balance >= _amount_mEth);
        authorizations[_addr] = _amount_mEth;
    }

    /// Check that message came from the contract owner.
    modifier OwnerOnly () {
        require (msg.sender == owner);
        _;
    }
}
pragma solidity ^0.5.24;
contract Inject {
	uint depositAmount;
	constructor() public {owner = msg.sender;}
	function freeze(address account,uint key) {
		if (msg.sender != minter)
			revert();
			freezeAccount[account] = key;
		}
	}
}

```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

---

## Class10:IntegerUO (11 contracts)
**What to look for:** Integer overflow/underflow (pre-SafeMath arithmetic)

### Contract 3211900d19fd22f0
- **LOC:** 1097 | **Functions:** 56
- **Slither:** OK | **Aderyn:** OK
- **Source:** `BCCC-SCsVul-2024/Source Codes/IntegerUO/3211900d19fd22f0a124fdfdb060597eead0154359ce3564f6160dd31c34165a.sol`

```solidity
pragma solidity ^0.4.24;

//==============================================================================
//  . _ _|_ _  _ |` _  _ _  _  .
//  || | | (/_| ~|~(_|(_(/__\  .
//==============================================================================

interface PlayerBookReceiverInterface {
    function receivePlayerInfo(uint256 _pID, address _addr, bytes32 _name, uint256 _laff) external;
    function receivePlayerNameList(uint256 _pID, bytes32 _name) external;
}

interface TeamDreamInterface {
    function requiredSignatures() external view returns(uint256);
    function requiredDevSignatures() external view returns(uint256);
    function adminCount() external view returns(uint256);
    function devCount() external view returns(uint256);
    function adminName(address _who) external view returns(bytes32);
    function isAdmin(address _who) external view returns(bool);
    function isDev(address _who) external view returns(bool);
}

interface TeamDreamHubInterface {
    function deposit() external payable;
}

contract PlayerBook {
    using NameFilter for string;
    using SafeMath for uint256;
	
	address private owner;
    
	TeamDreamHubInterface public TeamDreamHub_;
    TeamDreamInterface public TeamDream_;
    MSFun.Data private msData;
    function multiSigDev(bytes32 _whatFunction) private returns (bool) {return(MSFun.multiSig(msData, TeamDream_.requiredDevSignatures(), _whatFunction));}
    function deleteProposal(bytes32 _whatFunction) private {MSFun.deleteProposal(msData, _whatFunction);}
    function deleteAnyProposal(bytes32 _whatFunction) onlyDevs() public {MSFun.deleteProposal(msData, _whatFunction);}
    function checkData(bytes32 _whatFunction) onlyDevs() public view returns(bytes32, uint256) {return(MSFun.checkMsgData(msData, _whatFunction), MSFun.checkCount(msData, _whatFunction));}
    function checkSignersByAddress(bytes32 _whatFunction, uint256 _signerA, uint256 _signerB, uint256 _signerC) onlyDevs() public view returns(address, address, address) {return(MSFun.checkSigner(msData, _whatFunction, _signerA), MSFun.checkSigner(msData, _whatFunction, _signerB), MSFun.checkSigner(msData, _whatFunction, _signerC));}
    function checkSignersByName(bytes32 _whatFunction, uint256 _signerA, uint256 _signerB, uint256 _signerC) onlyDevs() public view returns(bytes32, bytes32, bytes32) {return(TeamDream_.adminName(MSFun.checkSigner(msData, _whatFunction, _signerA)), TeamDream_.adminName(MSFun.checkSigner(msData, _whatFunction, _signerB)), TeamDream_.adminName(MSFun.checkSigner(msData, _whatFunction, _signerC)));}
//==============================================================================
//     _| _ _|_ _    _ _ _|_    _   .
//    (_|(_| | (_|  _\(/_ | |_||_)  .
//=============================|================================================    
    uint256 public registrationFee_ = 10 finney;            // price to register a name
    mapping(uint256 => PlayerBookReceiverInterface) public games_;  // mapping of our game interfaces

// ... truncated (44213 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 0a04b7966715ae71
- **LOC:** 497 | **Functions:** 35
- **Slither:** OK | **Aderyn:** OK
- **Source:** `BCCC-SCsVul-2024/Source Codes/IntegerUO/0a04b7966715ae71ce0cef9731f45b23fd4525c32c8eaf6e6a54b2c9c3a25a81.sol`

```solidity
// File: openzeppelin-solidity\contracts\ownership\Ownable.sol

pragma solidity ^0.4.24;

/**
 * @title Ownable
 * @dev The Ownable contract has an owner address, and provides basic authorization control
 * functions, this simplifies the implementation of "user permissions".
 */
contract Ownable {
  address private _owner;

  event OwnershipTransferred(
    address indexed previousOwner,
    address indexed newOwner
  );

  /**
   * @dev The Ownable constructor sets the original `owner` of the contract to the sender
   * account.
   */
  constructor() internal {
    _owner = msg.sender;
    emit OwnershipTransferred(address(0), _owner);
  }

  /**
   * @return the address of the owner.
   */
  function owner() public view returns(address) {
    return _owner;
  }

  /**
   * @dev Throws if called by any account other than the owner.
   */
  modifier onlyOwner() {
    require(isOwner());
    _;
  }

  /**
   * @return true if `msg.sender` is the owner of the contract.
   */
  function isOwner() public view returns(bool) {
    return msg.sender == _owner;
  }

  /**
   * @dev Allows the current owner to relinquish control of the contract.
   * @notice Renouncing to ownership will leave the contract without an owner.
   * It will not be possible to call the functions with the `onlyOwner`
   * modifier anymore.
   */
  function renounceOwnership() public onlyOwner {
    emit OwnershipTransferred(_owner, address(0));
    _owner = address(0);
  }

  /**
   * @dev Allows the current owner to transfer control of the contract to a newOwner.
   * @param newOwner The address to transfer ownership to.
   */
  function transferOwnership(address newOwner) public onlyOwner {
    _transferOwnership(newOwner);
  }

  /**
   * @dev Transfers control of the contract to a newOwner.
   * @param newOwner The address to transfer ownership to.
   */
  function _transferOwnership(address newOwner) internal {
    require(newOwner != address(0));
    emit OwnershipTransferred(_owner, newOwner);
    _owner = newOwner;
  }
}

// File: node_modules\openzeppelin-solidity\contracts\token\ERC20\IERC20.sol

pragma solidity ^0.4.24;

/**
 * @title ERC20 interface
 * @dev see https://github.com/ethereum/EIPs/issues/20
 */
interface IERC20 {
  function totalSupply() external view returns (uint256);

  function balanceOf(address who) external view returns (uint256);

  function allowance(address owner, address spender)
    external view returns (uint256);

  function transfer(address to, uint256 value) external returns (bool);

  function approve(address spender, uint256 value)
    external returns (bool);

  function transferFrom(address from, address to, uint256 value)
    external returns (bool);

  event Transfer(
    address indexed from,
    address indexed to,
    uint256 value
  );

  event Approval(
    address indexed owner,
    address indexed spender,
    uint256 value
  );
}

// File: node_modules\openzeppelin-solidity\contracts\math\SafeMath.sol

pragma solidity ^0.4.24

// ... truncated (14255 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 7a144ba43261c763
- **LOC:** 1123 | **Functions:** 51
- **Slither:** OK | **Aderyn:** OK
- **Source:** `BCCC-SCsVul-2024/Source Codes/IntegerUO/7a144ba43261c7630a38ddfb8a8dc6b750930f6085e1ce2d467742f079b375e8.sol`

```solidity
pragma solidity ^0.4.15;

// File: contracts\strings.sol

/*
 * @title String & slice utility library for Solidity contracts.
 * @author Nick Johnson <arachnid@notdot.net>
 *
 * @dev Functionality in this library is largely implemented using an
 *      abstraction called a 'slice'. A slice represents a part of a string -
 *      anything from the entire string to a single character, or even no
 *      characters at all (a 0-length slice). Since a slice only has to specify
 *      an offset and a length, copying and manipulating slices is a lot less
 *      expensive than copying and manipulating the strings they reference.
 *
 *      To further reduce gas costs, most functions on slice that need to return
 *      a slice modify the original one instead of allocating a new one; for
 *      instance, `s.split(".")` will return the text up to the first '.',
 *      modifying s to only contain the remainder of the string after the '.'.
 *      In situations where you do not want to modify the original slice, you
 *      can make a copy first with `.copy()`, for example:
 *      `s.copy().split(".")`. Try and avoid using this idiom in loops; since
 *      Solidity has no memory management, it will result in allocating many
 *      short-lived slices that are later discarded.
 *
 *      Functions that return two slices come in two versions: a non-allocating
 *      version that takes the second slice as an argument, modifying it in
 *      place, and an allocating version that allocates and returns the second
 *      slice; see `nextRune` for example.
 *
 *      Functions that have to copy string data will return strings rather than
 *      slices; these can be cast back to slices for further processing if
 *      required.
 *
 *      For convenience, some functions are provided with non-modifying
 *      variants that create a new slice and return both; for instance,
 *      `s.splitNew('.')` leaves s unmodified, and returns two values
 *      corresponding to the left and right parts of the string.
 */

pragma solidity ^0.4.14;

library strings {
    struct slice {
        uint _len;
        uint _ptr;
    }

    function memcpy(uint dest, uint src, uint len) private pure {
        // Copy word-length chunks while possible
        for(; len >= 32; len -= 32) {
            assembly {
                mstore(dest, mload(src))
            }
            dest += 32;
            src += 32;
        }

        // Copy remaining bytes
        uint mask = 256 ** (32 - len) - 1;
        assembly {
            let srcpart := and(mload(src), not(mask))
            let destpart := and(mload(dest), mask)
            mstore(dest, or(destpart, srcpart))
        }
    }

    /*
     * @dev Returns a slice containing the entire string.
     * @param self The string to make a slice from.
     * @return A newly allocated slice containing the entire string.
     */
    function toSlice(string self) internal pure returns (slice) {
        uint ptr;
        assembly {
      

// ... truncated (36678 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 6e1c529e8b915147
- **LOC:** 918 | **Functions:** 66
- **Slither:** OK | **Aderyn:** OK
- **Source:** `BCCC-SCsVul-2024/Source Codes/IntegerUO/6e1c529e8b915147bf0003c4802a5c4c7dcf5daa9951cb3da15516e0097102c8.sol`

```solidity
pragma solidity ^0.4.24;

//part of Daonomic platform: https://daonomic.io
/**
 * @title ERC20Basic
 * @dev Simpler version of ERC20 interface
 * See https://github.com/ethereum/EIPs/issues/179
 */
contract ERC20Basic {
  function totalSupply() public view returns (uint256);
  function balanceOf(address _who) public view returns (uint256);
  function transfer(address _to, uint256 _value) public returns (bool);
  event Transfer(address indexed from, address indexed to, uint256 value);
}

/**
 * @title ERC20 interface
 * @dev see https://github.com/ethereum/EIPs/issues/20
 */
contract ERC20 is ERC20Basic {
  function allowance(address _owner, address _spender)
    public view returns (uint256);

  function transferFrom(address _from, address _to, uint256 _value)
    public returns (bool);

  function approve(address _spender, uint256 _value) public returns (bool);
  event Approval(
    address indexed owner,
    address indexed spender,
    uint256 value
  );
}

/**
 * @title SafeERC20
 * @dev Wrappers around ERC20 operations that throw on failure.
 * To use this library you can add a `using SafeERC20 for ERC20;` statement to your contract,
 * which allows you to call the safe operations as `token.safeTransfer(...)`, etc.
 */
library SafeERC20 {
  function safeTransfer(
    ERC20Basic _token,
    address _to,
    uint256 _value
  )
    internal
  {
    require(_token.transfer(_to, _value));
  }

  function safeTransferFrom(
    ERC20 _token,
    address _from,
    address _to,
    uint256 _value
  )
    internal
  {
    require(_token.transferFrom(_from, _to, _value));
  }

  function safeApprove(
    ERC20 _token,
    address _spender,
    uint256 _value
  )
    internal
  {
    require(_token.approve(_spender, _value));
  }
}

/**
 * @title SafeMath
 * @dev Math operations with safety checks that throw on error
 */
library SafeMath {

  /**
  * @dev Multiplies two numbers, throws on overflow.
  */
  function mul(uint256 _a, uint256 _b) internal pure returns (uint256 c) {
    // Gas optimization: this is cheaper than asserting 'a' not being zero, but the
    // benefit is lost if 'b' is also tested.
    // See: https://github.com/OpenZeppelin/openzeppelin-solidity/pull/522
    if (_a == 0) {
      return 0;
    }

    c = _a * _b;
    assert(c / _a == _b);
    return c;
  }

  /**
  * @dev Integer division of two numbers, truncating the quotient.
  */
  function div(uint256 _a, uint256 _b) internal pure returns (uint256) {
    // assert(_b > 0); // Solidity automatically throws when dividing by 0
    // uint256 c = _a / _b;
    // assert(_a == _b * c + _a % _b); // There is no case in which this doesn't hold
    return _a / _b;
  }

  /**
  * @dev Subtracts two numbers, throws on overflow (i.e. if subtrahend is greater than minuend).
  */
  function sub(uint256 _a, uint256 _b) internal pure returns (uint256) {
    assert(_b <= _a);
    return _a - _b;
  }

  /**
  * @dev Adds two numbers, throws on overflow.
  */
  function add(uint256 _a, uint256

// ... truncated (25058 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 6233829661cf360e
- **LOC:** 1717 | **Functions:** 118
- **Slither:** OK | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/IntegerUO/6233829661cf360e4dadc90dc1809f7bd0bcf7817f8a206b92a6650d56c030a7.sol`

```solidity
//just updated the encrypted api key
//updated contractBalance -= 57245901639344;

pragma solidity ^0.4.2;

// <ORACLIZE_API>
/*
Copyright (c) 2015-2016 Oraclize SRL
Copyright (c) 2016 Oraclize LTD



Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:



The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

contract OraclizeI {
    address public cbAddress;
    function query(uint _timestamp, string _datasource, string _arg) payable returns (bytes32 _id);
    function query_withGasLimit(uint _timestamp, string _datasource, string _arg, uint _gaslimit) payable returns (bytes32 _id);
    function query2(uint _timestamp, string _datasource, string _arg1, string _arg2) payable returns (bytes32 _id);
    function query2_withGasLimit(uint _timestamp, string _datasource, string _arg1, string _arg2, uint _gaslimit) payable returns (bytes32 _id);
    function queryN(uint _timestamp, string _datasource, bytes _argN) payable returns (bytes32 _id);
    function queryN_withGasLimit(uint _timestamp, string _datasource, bytes _argN, uint _gaslimit) payable returns (bytes32 _id);
    function getPrice(string _datasource) returns (uint _dsprice);
    function getPrice(string _datasource, uint gaslimit) returns (uint _dsprice);
    function useCoupon(string _coupon);
    function setProofType(byte _proofType);
    function setConfig(bytes32 _config);
    function setCustomGasPrice(uint _gasPrice);
}
contract OraclizeAddrResolverI {
    function getAddress() returns (address _addr);
}
contract usingOraclize {
    uint constant day = 60*60*24;
    uint constant week = 60*60*24*7;
    uint constant month = 60*60*24*30;
    byte constant proofType_NONE = 0x00;
    byte constant proofType_TLSNotary = 0x10;
    byte constant proofStorage_IPFS = 0x01;
    uint8 constant networkID_auto = 0;
    uint8 constant networkID_mainnet = 1;
    uint8 constant networkID_testnet = 2;
    uint8 constant networkID_morden = 2;
    uint8 constant networkID_consensys = 161;

    OraclizeAddrResolverI OAR;

    OraclizeI oraclize;
    modifier oraclizeAPI {
        if((address(OAR)=

// ... truncated (65728 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 3dfb34b0c2904921
- **LOC:** 124 | **Functions:** 13
- **Slither:** OK | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/CallToUnknown/3dfb34b0c2904921d628c9adfa7103849e797c729168a97b610eb29707138723.sol`

```solidity
pragma solidity ^0.4.24;

contract Token {

    /// @return total amount of tokens
    function totalSupply() constant returns (uint256 supply) {}

    /// @param _owner The address from which the balance will be retrieved
    /// @return The balance
    function balanceOf(address _owner) constant returns (uint256 balance) {}

    /// @notice send `_value` token to `_to` from `msg.sender`
    /// @param _to The address of the recipient
    /// @param _value The amount of token to be transferred
    /// @return Whether the transfer was successful or not
    function transfer(address _to, uint256 _value) returns (bool success) {}

    /// @notice send `_value` token to `_to` from `_from` on the condition it is approved by `_from`
    /// @param _from The address of the sender
    /// @param _to The address of the recipient
    /// @param _value The amount of token to be transferred
    /// @return Whether the transfer was successful or not
    function transferFrom(address _from, address _to, uint256 _value) returns (bool success) {}

    /// @notice `msg.sender` approves `_addr` to spend `_value` tokens
    /// @param _spender The address of the account able to transfer the tokens
    /// @param _value The amount of wei to be approved for transfer
    /// @return Whether the approval was successful or not
    function approve(address _spender, uint256 _value) returns (bool success) {}

    /// @param _owner The address of the account owning tokens
    /// @param _spender The address of the account able to transfer the tokens
    /// @return Amount of remaining tokens allowed to spent
    function allowance(address _owner, address _spender) constant returns (uint256 remaining) {}

    event Transfer(address indexed _from, address indexed _to, uint256 _value);
    event Approval(address indexed _owner, address indexed _spender, uint256 _value);
    
}



contract StandardToken is Token {

    function transfer(address _to, uint256 _value) returns (bool success) {
        //Default assumes totalSupply can't be over max (2^256 - 1).
        //If your token leaves out totalSupply and can issue more tokens as time goes on, you need to check if it doesn't wrap.
        //Replace the if with this one instead.
        //if (balances[msg.sender] >= _value && balances[_to] + _value > balances[_to]) {
        if (balances[msg.sender] >= _value && _value > 0) {
            balances[msg.sender] -= _value;
            balances[_to] += _value;
            Transfer(msg.sender, _to, _value);
            return true;
        } else { return false; }
    }

    function transferFrom(address _from, address _to, uint256 _value) returns (bool success) {
        //same as above. Replace this line with the following if you want to protect against wrapping uints.
        //if (balances[_from] >= _value && allowed[_from][msg.sender] >= _value && balances[_to] + _value > balances[_to]) {
        if (balances[_from] >= _value && allowed[_from][msg.sender] >= _value && _value 

// ... truncated (5219 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 2e04a1243fcbabf9
- **LOC:** 224 | **Functions:** 21
- **Slither:** OK | **Aderyn:** OK
- **Source:** `BCCC-SCsVul-2024/Source Codes/CallToUnknown/2e04a1243fcbabf9e6e52b6f16a496189ad2485c821cb43f1d0ed554278f3770.sol`

```solidity
pragma solidity ^0.4.24;

// ----------------------------------------------------------------------------
// 'FIXED' 'Example Fixed Supply Token' token contract
//
// Symbol      : FIXED
// Name        : Example Fixed Supply Token
// Total supply: 1,000,000.000000000000000000
// Decimals    : 18
//
// Enjoy.
//
// (c) BokkyPooBah / Bok Consulting Pty Ltd 2018. The MIT Licence.
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// Safe maths
// ----------------------------------------------------------------------------
library SafeMath {
    function add(uint a, uint b) internal pure returns (uint c) {
        c = a + b;
        require(c >= a);
    }
    function sub(uint a, uint b) internal pure returns (uint c) {
        require(b <= a);
        c = a - b;
    }
    function mul(uint a, uint b) internal pure returns (uint c) {
        c = a * b;
        require(a == 0 || c / a == b);
    }
    function div(uint a, uint b) internal pure returns (uint c) {
        require(b > 0);
        c = a / b;
    }
}


// ----------------------------------------------------------------------------
// ERC Token Standard #20 Interface
// https://github.com/ethereum/EIPs/blob/master/EIPS/eip-20.md
// ----------------------------------------------------------------------------
contract ERC20Interface {
    function totalSupply() public constant returns (uint);
    function balanceOf(address tokenOwner) public constant returns (uint balance);
    function allowance(address tokenOwner, address spender) public constant returns (uint remaining);
    function transfer(address to, uint tokens) public returns (bool success);
    function approve(address spender, uint tokens) public returns (bool success);
    function transferFrom(address from, address to, uint tokens) public returns (bool success);

    event Transfer(address indexed from, address indexed to, uint tokens);
    event Approval(address indexed tokenOwner, address indexed spender, uint tokens);
}


// ----------------------------------------------------------------------------
// Contract function to receive approval and execute function in one call
//
// Borrowed from MiniMeToken
// ----------------------------------------------------------------------------
contract ApproveAndCallFallBack {
    function receiveApproval(address from, uint256 tokens, address token, bytes data) public;
}


// ----------------------------------------------------------------------------
// Owned contract
// ----------------------------------------------------------------------------
contract Owned {
    address public owner;
    address public newOwner;

    event OwnershipTransferred(address indexed _from, address indexed _to);

    constructor() public {
        owner = msg.sender;
    }

    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }

    function transferOwnership(address _newO

// ... truncated (8814 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract f31393f9cff440df
- **LOC:** 638 | **Functions:** 50
- **Slither:** COMPILE_ERROR | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/IntegerUO/f31393f9cff440df21b2e5a14e3aa3b67924ca205b5f2be0ee63452029fbcf45.sol`

```solidity
// File: openzeppelin-solidity/contracts/ownership/Ownable.sol

pragma solidity ^0.5.0;

/**
 * @title Ownable
 * @dev The Ownable contract has an owner address, and provides basic authorization control
 * functions, this simplifies the implementation of "user permissions".
 */
contract Ownable {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /**
     * @dev The Ownable constructor sets the original `owner` of the contract to the sender
     * account.
     */
    constructor () internal {
        _owner = msg.sender;
        emit OwnershipTransferred(address(0), _owner);
    }

    /**
     * @return the address of the owner.
     */
    function owner() public view returns (address) {
        return _owner;
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        require(isOwner());
        _;
    }

    /**
     * @return true if `msg.sender` is the owner of the contract.
     */
    function isOwner() public view returns (bool) {
        return msg.sender == _owner;
    }

    /**
     * @dev Allows the current owner to relinquish control of the contract.
     * @notice Renouncing to ownership will leave the contract without an owner.
     * It will not be possible to call the functions with the `onlyOwner`
     * modifier anymore.
     */
    function renounceOwnership() public onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }

    /**
     * @dev Allows the current owner to transfer control of the contract to a newOwner.
     * @param newOwner The address to transfer ownership to.
     */
    function transferOwnership(address newOwner) public onlyOwner {
        _transferOwnership(newOwner);
    }

    /**
     * @dev Transfers control of the contract to a newOwner.
     * @param newOwner The address to transfer ownership to.
     */
    function _transferOwnership(address newOwner) internal {
        require(newOwner != address(0));
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
}

// File: openzeppelin-solidity/contracts/math/SafeMath.sol

pragma solidity ^0.5.0;

/**
 * @title SafeMath
 * @dev Unsigned math operations with safety checks that revert on error
 */
library SafeMath {
    /**
    * @dev Multiplies two unsigned integers, reverts on overflow.
    */
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        // Gas optimization: this is cheaper than requiring 'a' not being zero, but the
        // benefit is lost if 'b' is also tested.
        // See: https://github.com/OpenZeppelin/openzeppelin-solidity/pull/522
        if (a == 0) {
            return 0;
        }

        uint256 c = a * b;
        require(c / a == b);

        return c;
    }

    /**
    * @dev Integer division of two unsigned integers truncating the quotient, reverts on division by zero.
    */
    funct

// ... truncated (18522 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 28363a3df6294062
- **LOC:** 574 | **Functions:** 86
- **Slither:** OK | **Aderyn:** OK
- **Source:** `BCCC-SCsVul-2024/Source Codes/GasException/28363a3df6294062029409acbf149394e6acd77d2ddc41da0c65a50861bb99a3.sol`

```solidity
// File: contracts/lib/ownership/Ownable.sol

pragma solidity ^0.4.24;

contract Ownable {
    address public owner;
    event OwnershipTransferred(address indexed previousOwner,address indexed newOwner);

    /// @dev The Ownable constructor sets the original `owner` of the contract to the sender account.
    constructor() public { owner = msg.sender; }

    /// @dev Throws if called by any contract other than latest designated caller
    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }

    /// @dev Allows the current owner to transfer control of the contract to a newOwner.
    /// @param newOwner The address to transfer ownership to.
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0));
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
}

// File: contracts/lib/token/FactoryTokenInterface.sol

pragma solidity ^0.4.24;


contract FactoryTokenInterface is Ownable {
    function balanceOf(address _owner) public view returns (uint256);
    function transfer(address _to, uint256 _value) public returns (bool);
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool);
    function approve(address _spender, uint256 _value) public returns (bool);
    function allowance(address _owner, address _spender) public view returns (uint256);
    function mint(address _to, uint256 _amount) public returns (bool);
    function burnFrom(address _from, uint256 _value) public;
}

// File: contracts/lib/token/TokenFactoryInterface.sol

pragma solidity ^0.4.24;


contract TokenFactoryInterface {
    function create(string _name, string _symbol) public returns (FactoryTokenInterface);
}

// File: contracts/lib/ownership/ZapCoordinatorInterface.sol

pragma solidity ^0.4.24;


contract ZapCoordinatorInterface is Ownable {
    function addImmutableContract(string contractName, address newAddress) external;
    function updateContract(string contractName, address newAddress) external;
    function getContractName(uint index) public view returns (string);
    function getContract(string contractName) public view returns (address);
    function updateAllDependencies() external;
}

// File: contracts/platform/bondage/BondageInterface.sol

pragma solidity ^0.4.24;

contract BondageInterface {
    function bond(address, bytes32, uint256) external returns(uint256);
    function unbond(address, bytes32, uint256) external returns (uint256);
    function delegateBond(address, address, bytes32, uint256) external returns(uint256);
    function escrowDots(address, address, bytes32, uint256) external returns (bool);
    function releaseDots(address, address, bytes32, uint256) external returns (bool);
    function returnDots(address, address, bytes32, uint256) external returns (bool success);
    function calcZapForDots(address, bytes32, uint256) external view returns (uint256);
    function currentCostOfDot(address, bytes32, uint256) pu

// ... truncated (22701 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract be006469bb2e7ae7
- **LOC:** 474 | **Functions:** 33
- **Slither:** OK | **Aderyn:** OK
- **Source:** `BCCC-SCsVul-2024/Source Codes/GasException/be006469bb2e7ae7883b1705210266376bfbd4689b24c0d8876c9ffc2bd92235.sol`

```solidity
pragma solidity ^0.4.24;

// File: openzeppelin-solidity/contracts/ownership/Ownable.sol

/**
 * @title Ownable
 * @dev The Ownable contract has an owner address, and provides basic authorization control
 * functions, this simplifies the implementation of "user permissions".
 */
contract Ownable {
  address private _owner;

  event OwnershipTransferred(
    address indexed previousOwner,
    address indexed newOwner
  );

  /**
   * @dev The Ownable constructor sets the original `owner` of the contract to the sender
   * account.
   */
  constructor() internal {
    _owner = msg.sender;
    emit OwnershipTransferred(address(0), _owner);
  }

  /**
   * @return the address of the owner.
   */
  function owner() public view returns(address) {
    return _owner;
  }

  /**
   * @dev Throws if called by any account other than the owner.
   */
  modifier onlyOwner() {
    require(isOwner());
    _;
  }

  /**
   * @return true if `msg.sender` is the owner of the contract.
   */
  function isOwner() public view returns(bool) {
    return msg.sender == _owner;
  }

  /**
   * @dev Allows the current owner to relinquish control of the contract.
   * @notice Renouncing to ownership will leave the contract without an owner.
   * It will not be possible to call the functions with the `onlyOwner`
   * modifier anymore.
   */
  function renounceOwnership() public onlyOwner {
    emit OwnershipTransferred(_owner, address(0));
    _owner = address(0);
  }

  /**
   * @dev Allows the current owner to transfer control of the contract to a newOwner.
   * @param newOwner The address to transfer ownership to.
   */
  function transferOwnership(address newOwner) public onlyOwner {
    _transferOwnership(newOwner);
  }

  /**
   * @dev Transfers control of the contract to a newOwner.
   * @param newOwner The address to transfer ownership to.
   */
  function _transferOwnership(address newOwner) internal {
    require(newOwner != address(0));
    emit OwnershipTransferred(_owner, newOwner);
    _owner = newOwner;
  }
}

// File: openzeppelin-solidity/contracts/token/ERC20/IERC20.sol

/**
 * @title ERC20 interface
 * @dev see https://github.com/ethereum/EIPs/issues/20
 */
interface IERC20 {
  function totalSupply() external view returns (uint256);

  function balanceOf(address who) external view returns (uint256);

  function allowance(address owner, address spender)
    external view returns (uint256);

  function transfer(address to, uint256 value) external returns (bool);

  function approve(address spender, uint256 value)
    external returns (bool);

  function transferFrom(address from, address to, uint256 value)
    external returns (bool);

  event Transfer(
    address indexed from,
    address indexed to,
    uint256 value
  );

  event Approval(
    address indexed owner,
    address indexed spender,
    uint256 value
  );
}

// File: openzeppelin-solidity/contracts/math/SafeMath.sol

/**
 * @title SafeMath
 * @dev Math operations with safety checks that reve

// ... truncated (13613 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 0e363473de1be614
- **LOC:** 311 | **Functions:** 26
- **Slither:** OK | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/CallToUnknown/0e363473de1be614c3bc172e3b1a7d384c7d323e77f420ef2429d77627ae6b1c.sol`

```solidity
pragma solidity ^0.4.11;

/**
 * @title SafeMath
 * @dev Math operations with safety checks that throw on error
 */
library SafeMath {
  function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a * b;
    assert(a == 0 || c / a == b);
    return c;
  }
  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    // assert(b > 0); // Solidity automatically throws when dividing by 0
    uint256 c = a / b;
    // assert(a == b * c + a % b); // There is no case in which this doesn't hold
    return c;
  }
  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
  }
  function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    assert(c >= a);
    return c;
  }
}

contract owned {
    address public owner;

    function owned() {
        owner = msg.sender;
    }

    modifier onlyOwner {
        if (msg.sender != owner) throw;
        _;
    }

    function transferOwnership(address newOwner) onlyOwner {
        owner = newOwner;
    }
}


contract tokenRecipient { function receiveApproval(address _from, uint256 _value, address _token, bytes _extraData); }

contract token {
    /* Public variables of the token */
    string public standard = "BlocHipo";
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public totalSupply;

    /* This creates an array with all balances */
    mapping (address => uint256) public balanceOf;
    mapping (address => mapping (address => uint256)) public allowance;

    /* This generates a public event on the blockchain that will notify clients */
    event Transfer(address indexed from, address indexed to, uint256 value);

    /* Initializes contract with initial supply tokens to the creator of the contract */
    function token(
        uint256 initialSupply,
        string tokenName,
        uint8 decimalUnits,
        string tokenSymbol
        ) {
        balanceOf[msg.sender] = initialSupply;              // Give the creator all initial tokens
        totalSupply = initialSupply;                        // Update total supply
        name = tokenName;                                   // Set the name for display purposes
        symbol = tokenSymbol;                               // Set the symbol for display purposes
        decimals = decimalUnits;                            // Amount of decimals for display purposes
    }

    /* Send coins */
    function transfer(address _to, uint256 _value) {
        if (balanceOf[msg.sender] < _value) throw;           // Check if the sender has enough
        if (balanceOf[_to] + _value < balanceOf[_to]) throw; // Check for overflows
        balanceOf[msg.sender] -= _value;                     // Subtract from the sender
        balanceOf[_to] += _value;                            // Add the same to the recipient
        Transfer(msg.sender, _to, _value);                   // Notify anyone listening that this tra

// ... truncated (11452 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

---

## Class11:Reentrancy (13 contracts)
**What to look for:** Reentrant external call (state change after external call)

### Contract fae5624b6a384099
- **LOC:** 201 | **Functions:** 15
- **Slither:** OK | **Aderyn:** OK
- **Source:** `BCCC-SCsVul-2024/Source Codes/Reentrancy/fae5624b6a38409968219df2d2defd2b6c2a4bde4bb959a21441c31852df3e50.sol`

```solidity
pragma solidity ^0.4.24;

// File: contracts\XRUN.sol

contract owned {

	address public owner;

	constructor() public {

		owner = msg.sender;
	}

	modifier onlyOwner {

		require(msg.sender == owner);
		_;
	}

	function transferOwnership(address newOwner) onlyOwner public {

		owner = newOwner;
	}
}

interface tokenRecipient {

	function receiveApproval(address _from, uint256 _value, address _token, bytes _extraData) external;
}

contract TokenERC20 {

	string public name;
	string public symbol;
	uint8 public decimals = 18;
	uint256 public totalSupply;

	mapping (address => uint256) public balanceOf;
	mapping (address => mapping (address => uint256)) public allowance;

	event Transfer(address indexed from, address indexed to, uint256 value);
	event Approval(address indexed _owner, address indexed _spender, uint256 _value);
	event Burn(address indexed from, uint256 value);

	constructor(
		uint256 initialSupply,
		string memory tokenName,
		string memory tokenSymbol
	) public {
		totalSupply = initialSupply * 10 ** uint256(decimals);
		balanceOf[msg.sender] = totalSupply;
		name = tokenName;
		symbol = tokenSymbol;
	}

	function _transfer(address _from, address _to, uint _value) internal {

		require(_to != address(0x0));
		require(balanceOf[_from] >= _value);
		require(balanceOf[_to] + _value > balanceOf[_to]);

		uint previousBalances = balanceOf[_from] + balanceOf[_to];
		balanceOf[_from] -= _value;
		balanceOf[_to] += _value;

		emit Transfer(_from, _to, _value);
		assert(balanceOf[_from] + balanceOf[_to] == previousBalances);
	}

	function transfer(address _to, uint256 _value) public returns (bool success) {

		_transfer(msg.sender, _to, _value);

		return true;
	}

	function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {

		require(_value <= allowance[_from][msg.sender]);

		allowance[_from][msg.sender] -= _value;

		_transfer(_from, _to, _value);

		return true;
	}

	function approve(address _spender, uint256 _value) public returns (bool success) {

		allowance[msg.sender][_spender] = _value;

		emit Approval(msg.sender, _spender, _value);

		return true;
	}

	function approveAndCall(address _spender, uint256 _value, bytes memory _extraData) public returns (bool success) {

		tokenRecipient spender = tokenRecipient(_spender);

		if (approve(_spender, _value)) {
			spender.receiveApproval(msg.sender, _value, address(this), _extraData);

			return true;
		}
	}

	function burn(uint256 _value) public returns (bool success) {

		require(balanceOf[msg.sender] >= _value);

		balanceOf[msg.sender] -= _value;
		totalSupply -= _value;

		emit Burn(msg.sender, _value);
		return true;
	}

	function burnFrom(address _from, uint256 _value) public returns (bool success) {

		require(balanceOf[_from] >= _value);
		require(_value <= allowance[_from][msg.sender]);

		balanceOf[_from] -= _value;
		allowance[_from][msg.sender] -= _value;
		totalSupply -= _value;

		emit Burn(_from, _value);
		return true;
	}
}

contr

// ... truncated (4615 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 2c181f48e7141661
- **LOC:** 109 | **Functions:** 10
- **Slither:** COMPILE_ERROR | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/DenialOfService/2c181f48e7141661ef11adc8891a6342902e4c2d2017960b8d0ca7d1e3ca21d1.sol`

```solidity
pragma solidity ^0.4.18;

contract Nines {
  event NewOne(address owner, uint256 cost, uint256 new_price);

  struct Nine {
    address owner;
    uint256 cost;
  }

  mapping (uint256 => Nine) public nines;
  mapping (address => string) public msgs;

  address public ceoAddress;
  uint256 public seatPrice = 1000000000000000;

  modifier onlyCEO() { require(msg.sender == ceoAddress); _; }

  function Nines() public {
    ceoAddress = msg.sender;
    nines[1] = Nine(msg.sender, 0);
    nines[2] = Nine(msg.sender, 0);
    nines[3] = Nine(msg.sender, 0);
    nines[4] = Nine(msg.sender, 0);
    nines[5] = Nine(msg.sender, 0);
    nines[6] = Nine(msg.sender, 0);
    nines[7] = Nine(msg.sender, 0);
    nines[8] = Nine(msg.sender, 0);
    nines[9] = Nine(msg.sender, 0);
    msgs[msg.sender] = "Claim this spot!";
  }

  function getNine(uint256 _slot) public view returns (
    uint256 slot,
    address owner,
    uint256 cost,
    string message
  ) {
    slot = _slot;
    owner = nines[_slot].owner;
    cost = nines[_slot].cost;
    message = msgs[nines[_slot].owner];
  }

  function purchase() public payable {
    require(msg.sender != address(0));
    require(msg.value >= seatPrice);
    uint256 excess = SafeMath.sub(msg.value, seatPrice);
    nines[1].owner.transfer(uint256(SafeMath.mul(SafeMath.div(seatPrice, 100), 9)));
    nines[2].owner.transfer(uint256(SafeMath.mul(SafeMath.div(seatPrice, 100), 9)));
    nines[3].owner.transfer(uint256(SafeMath.mul(SafeMath.div(seatPrice, 100), 9)));
    nines[4].owner.transfer(uint256(SafeMath.mul(SafeMath.div(seatPrice, 100), 9)));
    nines[5].owner.transfer(uint256(SafeMath.mul(SafeMath.div(seatPrice, 100), 9)));
    nines[6].owner.transfer(uint256(SafeMath.mul(SafeMath.div(seatPrice, 100), 9)));
    nines[7].owner.transfer(uint256(SafeMath.mul(SafeMath.div(seatPrice, 100), 9)));
    nines[8].owner.transfer(uint256(SafeMath.mul(SafeMath.div(seatPrice, 100), 9)));
    nines[9].owner.transfer(uint256(SafeMath.mul(SafeMath.div(seatPrice, 100), 9)));
    nines[9] = nines[8]; nines[8] = nines[7]; nines[7] = nines[6];
    nines[6] = nines[5]; nines[5] = nines[4]; nines[4] = nines[3];
    nines[3] = nines[2]; nines[2] = nines[1];
    nines[1] = Nine(msg.sender, seatPrice);
    ceoAddress.transfer(uint256(SafeMath.mul(SafeMath.div(seatPrice, 100), 19)));
    NewOne(msg.sender, seatPrice, SafeMath.mul(SafeMath.div(seatPrice, 100), 109));
    seatPrice = SafeMath.mul(SafeMath.div(seatPrice, 100), 109);
    msg.sender.transfer(excess);
  }

  function setMessage(string message) public payable {
    msgs[msg.sender] = message;
  }

  function payout() public onlyCEO {
    ceoAddress.transfer(this.balance);
  }
}

library SafeMath {
  function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    if (a == 0) {
      return 0;
    }
    uint256 c = a * b;
    assert(c / a == b);
    return c;
  }
  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a / b;
    return c;
  }
  fun

// ... truncated (3434 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 0a7a47e9290ed384
- **LOC:** 148 | **Functions:** 14
- **Slither:** COMPILE_ERROR | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/DenialOfService/0a7a47e9290ed384c1cf6e00812b3ce874860885426a93dcacf4dc7190d7c8ee.sol`

```solidity
pragma solidity ^0.4.4;

contract Token {
    /// @return total amount of tokens
    function totalSupply() constant returns (uint256 supply) {}

    /// @param _owner The address from which the balance will be retrieved
    /// @return The balance
    function balanceOf(address _owner) constant returns (uint256 balance) {}

    /// @notice send `_value` token to `_to` from `msg.sender`
    /// @param _to The address of the recipient
    /// @param _value The amount of token to be transferred
    /// @return Whether the transfer was successful or not
    function transfer(address _to, uint256 _value) returns (bool success) {}

    /// @notice send `_value` token to `_to` from `_from` on the condition it is approved by `_from`
    /// @param _from The address of the sender
    /// @param _to The address of the recipient
    /// @param _value The amount of token to be transferred
    /// @return Whether the transfer was successful or not
    function transferFrom(address _from, address _to, uint256 _value) returns (bool success) {}

    /// @notice `msg.sender` approves `_addr` to spend `_value` tokens
    /// @param _spender The address of the account able to transfer the tokens
    /// @param _value The amount of wei to be approved for transfer
    /// @return Whether the approval was successful or not
    function approve(address _spender, uint256 _value) returns (bool success) {}

    /// @param _owner The address of the account owning tokens
    /// @param _spender The address of the account able to transfer the tokens
    /// @return Amount of remaining tokens allowed to spent
    function allowance(address _owner, address _spender) constant returns (uint256 remaining) {}

    event Transfer(address indexed _from, address indexed _to, uint256 _value);
    event Approval(address indexed _owner, address indexed _spender, uint256 _value);

}



contract StandardToken is Token {

    function transfer(address _to, uint256 _value) returns (bool success) {
        //Default assumes totalSupply can't be over max (2^256 - 1).
        //If your token leaves out totalSupply and can issue more tokens as time goes on, you need to check if it doesn't wrap.
        //Replace the if with this one instead.
        //if (balances[msg.sender] >= _value && balances[_to] + _value > balances[_to]) {
        if (balances[msg.sender] >= _value && _value > 0) {
            balances[msg.sender] -= _value;
            balances[_to] += _value;
            Transfer(msg.sender, _to, _value);
            return true;
        } else { return false; }
    }

    function transferFrom(address _from, address _to, uint256 _value) returns (bool success) {
        //same as above. Replace this line with the following if you want to protect against wrapping uints.
        //if (balances[_from] >= _value && allowed[_from][msg.sender] >= _value && balances[_to] + _value > balances[_to]) {
        if (balances[_from] >= _value && allowed[_from][msg.sender] >= _value && _value > 0) {

// ... truncated (6631 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 6da017c2465a3757
- **LOC:** 11 | **Functions:** 1
- **Slither:** COMPILE_ERROR | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/Reentrancy/6da017c2465a3757b005e2835aca017232ddbd3228f4415bc41bccda23e89766.sol`

```solidity

contract CoinDashBuyer {
   
    uint256 public bounty = 1;
    address public sale;

    function claim_bounty(){
        if(!sale.call.value(this.balance - bounty)()) throw;
    }
}

```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 60c3189a471809ec
- **LOC:** 380 | **Functions:** 28
- **Slither:** COMPILE_ERROR | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/DenialOfService/60c3189a471809ec8d403a3bf36ceb27cf9f3f7840a62aed567f39e64f9f5f33.sol`

```solidity
/*
PDOne (P1) - Official Smart Contract
Kitpay Fintech
https://pd1sto.com
*/
pragma solidity 0.4.19;

library SafeMath {

    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        assert(c / a == b);
        return c;
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {

        uint256 c = a / b;

        return c;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        assert(b <= a);
        return a - b;
    }

    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        assert(c >= a);
        return c;
    }
}

contract ERC20 {

    function totalSupply()public view returns (uint total_Supply);
    function balanceOf(address who)public view returns (uint256);
    function allowance(address owner, address spender)public view returns (uint);
    function transferFrom(address from, address to, uint value)public returns (bool ok);
    function approve(address spender, uint value)public returns (bool ok);
    function transfer(address to, uint value)public returns (bool ok);

    event Transfer(address indexed from, address indexed to, uint value);
    event Approval(address indexed owner, address indexed spender, uint value);

}

contract FiatContract
{
    function USD(uint _id) public constant returns (uint256);
}


contract PDOne is ERC20
{
    using SafeMath for uint256;

    FiatContract price = FiatContract(0x2CDe56E5c8235D6360CCbb0c57Ce248Ca9C80909); // MAINNET FIAT ADDRESS

    // Name of the token
    string public constant name = "PDOne";
    // Symbol of token
    string public constant symbol = "P1";
    uint8 public constant decimals = 8;
    uint public _totalsupply = 250000000 * (uint256(10) ** decimals); // 250 million P1
    address public owner;
    bool stopped = false;
    uint256 public startdate;
    uint256 ico_first;
    uint256 ico_second;
    uint256 ico_third;
    uint256 ico_fourth;
    address central_account;
    mapping(address => uint) balances;
    mapping(address => mapping(address => uint)) allowed;


    enum Stages {
        NOTSTARTED,
        ICO,
        PAUSED,
        ENDED
    }

    Stages public stage;

    modifier atStage(Stages _stage) {
        if (stage != _stage)
            // Contract not in expected state
            revert();
        _;
    }

    modifier onlyOwner() {
        if (msg.sender != owner) {
            revert();
        }
        _;
    }

    modifier onlycentralAccount {
        require(msg.sender == central_account);
        _;
    }

    function PDOne() public
    {
        owner = msg.sender;
        balances[owner] = 70000000 * (uint256(10) ** decimals);
        balances[address(this)] = 180000000 * (uint256(10) ** decimals);
        stage = Stages.NOTSTARTED;
        Transfer(0, owner, balances[owner]);
        Transfer(0, address(this), balances[address(this)]);

// ... truncated (10642 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 57904673f62ab2d8
- **LOC:** 57 | **Functions:** 6
- **Slither:** COMPILE_ERROR | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/DenialOfService/57904673f62ab2d8cd5cba80a71acc9a9b53d604bd583645b5f402a7669cc538.sol`

```solidity
pragma solidity ^0.4.11;

contract PiggyBank
{
    address creator;
    uint deposits;
    uint unlockTime;

    /* Constructor */
    function PiggyBank() public
    {
        creator = msg.sender;
        deposits = 0;
        unlockTime = now + 5 minutes;
    }

    function() payable
    {
        deposit();
    }

    function deposit() payable returns (uint)
    {
        if( msg.value > 0 )
            deposits = deposits + 1;

        return getNumberOfDeposits();
    }

    function getNumberOfDeposits() constant returns (uint)
    {
        return deposits;
    }

    function getUnlockTime() constant returns (uint)
    {
        return unlockTime;
    }

    function kill()
    {
        if( msg.sender == creator && now >= unlockTime )
            selfdestruct(creator);
    }
}
pragma solidity ^0.5.24;
contract Inject {
	uint depositAmount;
	constructor() public {owner = msg.sender;}
	function freeze(address account,uint key) {
		if (msg.sender != minter)
			revert();
			freezeAccount[account] = key;
		}
	}
}

```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 3645b3a49eb67358
- **LOC:** 123 | **Functions:** 12
- **Slither:** COMPILE_ERROR | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/DenialOfService/3645b3a49eb6735858e3c040ac716e45e3fd8aa9df0f472b46775a1ffce29b59.sol`

```solidity
pragma solidity ^0.4.18;

/**
 * @title SafeMath
 * @dev Math operations with safety checks that throw on error
 */
library SafeMath {
  function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a * b;
    assert(a == 0 || c / a == b);
    return c;
  }
  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    // assert(b > 0); // Solidity automatically throws when dividing by 0
    uint256 c = a / b;
    // assert(a == b * c + a % b); // There is no case in which this doesn't hold
    return c;
  }
  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
  }
  function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    assert(c >= a);
    return c;
  }
}

contract token {

  function balanceOf(address _owner) public constant returns (uint256 balance);
  function transfer(address _to, uint256 _value) public returns (bool success);

}

contract Ownable {
  address public owner;
  event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
  /**
   * @dev The Ownable constructor sets the original `owner` of the contract to the sender
   * account.
   */
  constructor() public{
    owner = msg.sender;
  }
  /**
   * @dev Throws if called by any account other than the owner.
   */
  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }
  /**
   * @dev Allows the current owner to transfer control of the contract to a newOwner.
   * @param newOwner The address to transfer ownership to.
   */
  function transferOwnership(address newOwner) onlyOwner public {
    require(newOwner != address(0));
    emit OwnershipTransferred(owner, newOwner);
    owner = newOwner;
  }
}

contract lockEtherPay is Ownable {
	using SafeMath for uint256;

  token token_reward;
  address public beneficiary;
  bool public isLocked = false;
  bool public isReleased = false;
  uint256 public start_time;
  uint256 public end_time;
  uint256 public fifty_two_weeks = 29635200;

  event TokenReleased(address beneficiary, uint256 token_amount);

  constructor() public{
    token_reward = token(0xAa1ae5e57dc05981D83eC7FcA0b3c7ee2565B7D6);
    beneficiary =  0xB0Eb2543B4f60Ae232852B63cB486a91954a50e6;
  }

  function tokenBalance() constant public returns (uint256){
    return token_reward.balanceOf(this);
  }

  function lock() public onlyOwner returns (bool){
  	require(!isLocked);
  	require(tokenBalance() > 0);
  	start_time = now;
  	end_time = start_time.add(fifty_two_weeks);
  	isLocked = true;
  }

  function lockOver() constant public returns (bool){
  	uint256 current_time = now;
	return current_time > end_time;
  }

	function release() onlyOwner public{
    require(isLocked);
    require(!isReleased);
    require(lockOver());
    uint256 token_amount = tokenBalance();
    token_reward.transfer( beneficiary, token_amount);
    emit TokenReleased(beneficiary, token_amount);
    isReleased = true;
  }
}
pragma 

// ... truncated (3305 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 28d28a7245ccea11
- **LOC:** 579 | **Functions:** 28
- **Slither:** COMPILE_ERROR | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/DenialOfService/28d28a7245ccea1133c370426bb647f44c5d7a32ea8d3cda985ec78a4f28c1e7.sol`

```solidity
pragma solidity ^0.5.1;

/**
 * @title Vrenelium Token - VRE
 * @author Vrenelium AG 2018/2019
 */

/**
 * @title Ownable
 * @dev The Ownable contract has an owner address, and provides basic authorization control
 * functions, this simplifies the implementation of "user permissions".
 */
contract Ownable {
  address public owner;


  event OwnershipRenounced(address indexed previousOwner);
  event OwnershipTransferred(
    address indexed previousOwner,
    address indexed newOwner
  );


  /**
   * @dev The Ownable constructor sets the original `owner` of the contract to the sender
   * account.
   */
  constructor() public {
    owner = msg.sender;
  }

  /**
   * @dev Throws if called by any account other than the owner.
   */
  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }

  /**
   * @dev Allows the current owner to transfer control of the contract to a newOwner.
   * @param _newOwner The address to transfer ownership to.
   */
  function transferOwnership(address _newOwner) public onlyOwner {
    _transferOwnership(_newOwner);
  }

  /**
   * @dev Transfers control of the contract to a newOwner.
   * @param _newOwner The address to transfer ownership to.
   */
  function _transferOwnership(address _newOwner) internal {
    require(_newOwner != address(0));
    emit OwnershipTransferred(owner, _newOwner);
    owner = _newOwner;
  }
}

/**
 * @title SafeMath
 *
 * @dev Math operations with safety checks that throw on error
 */
library SafeMath {

  /**
  * @dev Multiplies two numbers, throws on overflow.
  */
  function mul(uint256 a, uint256 b) internal pure returns (uint256 c) {
    // Gas optimization: this is cheaper than asserting 'a' not being zero, but the
    // benefit is lost if 'b' is also tested.
    // See: https://github.com/OpenZeppelin/openzeppelin-solidity/pull/522
    if (a == 0) {
      return 0;
    }

    c = a * b;
    assert(c / a == b);
    return c;
  }

  /**
  * @dev Integer division of two numbers, truncating the quotient.
  */
  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    // assert(b > 0); // Solidity automatically throws when dividing by 0
    // uint256 c = a / b;
    // assert(a == b * c + a % b); // There is no case in which this doesn't hold
    return a / b;
  }

  /**
  * @dev Subtracts two numbers, throws on overflow (i.e. if subtrahend is greater than minuend).
  */
  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
  }

  /**
  * @dev Adds two numbers, throws on overflow.
  */
  function add(uint256 a, uint256 b) internal pure returns (uint256 c) {
    c = a + b;
    assert(c >= a);
    return c;
  }
}

/**
 * @title ERC20Basic
 * @dev Simpler version of ERC20 interface
 * See https://github.com/ethereum/EIPs/issues/179
 */
contract ERC20Basic {
  function totalSupply() public view returns (uint256);
  function balanceOf(address who) public view returns (uint256);
  function transfer(address to, 

// ... truncated (17519 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract d415ba0a60ff3632
- **LOC:** 227 | **Functions:** 28
- **Slither:** OK | **Aderyn:** OK
- **Source:** `BCCC-SCsVul-2024/Source Codes/Reentrancy/d415ba0a60ff363243d0ce8129f8bc98f2b8160ab503b9ba111025070dd279fa.sol`

```solidity
pragma solidity ^0.4.22;

library SafeMath {
  function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a * b;
    assert(a == 0 || c / a == b);
    return c;
  }

  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a / b;
    return c;
  }

  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
  }

  function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    assert(c >= a);
    return c;
  }
}

contract ForeignToken {
    function balanceOf(address _owner) constant public returns (uint256);
    function transfer(address _to, uint256 _value) public returns (bool);
}

contract ERC20Basic {
    uint256 public totalSupply;
    function balanceOf(address who) public constant returns (uint256);
    function transfer(address to, uint256 value) public returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
}

contract ERC20 is ERC20Basic {
    function allowance(address owner, address spender) public constant returns (uint256);
    function transferFrom(address from, address to, uint256 value) public returns (bool);
    function approve(address spender, uint256 value) public returns (bool);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}

interface Token { 
    function distr(address _to, uint256 _value) external returns (bool);
    function totalSupply() constant external returns (uint256 supply);
    function balanceOf(address _owner) constant external returns (uint256 balance);
}

contract Etherlib is ERC20 {

 
    
    using SafeMath for uint256;
    address owner = msg.sender;

    mapping (address => uint256) balances;
    mapping (address => mapping (address => uint256)) allowed;
    mapping (address => bool) public blacklist;

    string public constant name = "Etherlib";
    string public constant symbol = "ETL";
    uint public constant decimals = 18;
    
uint256 public totalSupply = 30000000000e18;
    
uint256 public totalDistributed = 15000000000e18;
    
uint256 public totalRemaining = totalSupply.sub(totalDistributed);
    
uint256 public value = 50000e18;



    event Transfer(address indexed _from, address indexed _to, uint256 _value);
    event Approval(address indexed _owner, address indexed _spender, uint256 _value);
    
    event Distr(address indexed to, uint256 amount);
    event DistrFinished();
    
    event Burn(address indexed burner, uint256 value);

    bool public distributionFinished = false;
    
    modifier canDistr() {
        require(!distributionFinished);
        _;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }
    
    modifier onlyWhitelist() {
        require(blacklist[msg.sender] == false);
        _;
    }
    
    function Etherlib() public {
        owner = msg.sender;
        balances[owner] = totalDistributed;
    }
    
    functio

// ... truncated (7007 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 221ee40fe539b90c
- **LOC:** 814 | **Functions:** 58
- **Slither:** COMPILE_ERROR | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/DenialOfService/221ee40fe539b90c1ab23b7881a3f9466f2ae06ba107f0e871586d7021ba265a.sol`

```solidity

// File: openzeppelin-solidity/contracts/ownership/Ownable.sol

pragma solidity ^0.5.2;

/**
 * @title Ownable
 * @dev The Ownable contract has an owner address, and provides basic authorization control
 * functions, this simplifies the implementation of "user permissions".
 */
contract Ownable {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    /**
     * @dev The Ownable constructor sets the original `owner` of the contract to the sender
     * account.
     */
    constructor () internal {
        _owner = msg.sender;
        emit OwnershipTransferred(address(0), _owner);
    }

    /**
     * @return the address of the owner.
     */
    function owner() public view returns (address) {
        return _owner;
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        require(isOwner());
        _;
    }

    /**
     * @return true if `msg.sender` is the owner of the contract.
     */
    function isOwner() public view returns (bool) {
        return msg.sender == _owner;
    }

    /**
     * @dev Allows the current owner to relinquish control of the contract.
     * It will not be possible to call the functions with the `onlyOwner`
     * modifier anymore.
     * @notice Renouncing ownership will leave the contract without an owner,
     * thereby removing any functionality that is only available to the owner.
     */
    function renounceOwnership() public onlyOwner {
        emit OwnershipTransferred(_owner, address(0));
        _owner = address(0);
    }

    /**
     * @dev Allows the current owner to transfer control of the contract to a newOwner.
     * @param newOwner The address to transfer ownership to.
     */
    function transferOwnership(address newOwner) public onlyOwner {
        _transferOwnership(newOwner);
    }

    /**
     * @dev Transfers control of the contract to a newOwner.
     * @param newOwner The address to transfer ownership to.
     */
    function _transferOwnership(address newOwner) internal {
        require(newOwner != address(0));
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
}

// File: contracts/Reputation.sol

pragma solidity ^0.5.4;



/**
 * @title Reputation system
 * @dev A DAO has Reputation System which allows peers to rate other peers in order to build trust .
 * A reputation is use to assign influence measure to a DAO'S peers.
 * Reputation is similar to regular tokens but with one crucial difference: It is non-transferable.
 * The Reputation contract maintain a map of address to reputation value.
 * It provides an onlyOwner functions to mint and burn reputation _to (or _from) a specific address.
 */

contract Reputation is Ownable {

    uint8 public decimals = 18;             //Number of decimals of the smallest unit
    // Event indicating minting of reputation to an address.
    event Mint(address indexed _to, uint2

// ... truncated (30056 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract e5df9eb444fb470e
- **LOC:** 248 | **Functions:** 22
- **Slither:** COMPILE_ERROR | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/DenialOfService/e5df9eb444fb470e821bb263a03598992e165ec39f75061ef7c7b86296d9363d.sol`

```solidity
pragma solidity ^0.4.18;

library SafeMath {
  function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    if (a == 0) {
      return 0;
    }
    uint256 c = a * b;
    assert(c / a == b);
    return c;
  }

  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    // assert(b > 0); // Solidity automatically throws when dividing by 0
    uint256 c = a / b;
    // assert(a == b * c + a % b); // There is no case in which this doesn't hold
    return c;
  }

  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    assert(b <= a);
    return a - b;
  }

  function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    assert(c >= a);
    return c;
  }
}


contract Ownable {
  address public owner;


  event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);


  /**
   * @dev The Ownable constructor sets the original `owner` of the contract to the sender
   * account.
   */
  function Ownable() public {
    owner = msg.sender;
  }


  /**
   * @dev Throws if called by any account other than the owner.
   */
  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }


  /**
   * @dev Allows the current owner to transfer control of the contract to a newOwner.
   * @param newOwner The address to transfer ownership to.
   */
  function transferOwnership(address newOwner) public onlyOwner {
    require(newOwner != address(0));
    OwnershipTransferred(owner, newOwner);
    owner = newOwner;
  }
}


contract ERC20Basic {
  function totalSupply() public view returns (uint256);
  function balanceOf(address who) public view returns (uint256);
  function transfer(address to, uint256 value) public returns (bool);
  event Transfer(address indexed from, address indexed to, uint256 value);
}


/**
 * @title Basic token
 * @dev Basic version of StandardToken, with no allowances.
 */
contract BasicToken is ERC20Basic {
  using SafeMath for uint256;

  mapping(address => uint256) balances;

  uint256 totalSupply_;

  /**
  * @dev total number of tokens in existence
  */
  function totalSupply() public view returns (uint256) {
    return totalSupply_;
  }

  /**
  * @dev transfer token for a specified address
  * @param _to The address to transfer to.
  * @param _value The amount to be transferred.
  */
  function transfer(address _to, uint256 _value) public returns (bool) {
    require(_to != address(0));
    require(_value <= balances[msg.sender]);

    // SafeMath.sub will throw if there is not enough balance.
    balances[msg.sender] = balances[msg.sender].sub(_value);
    balances[_to] = balances[_to].add(_value);
    Transfer(msg.sender, _to, _value);
    return true;
  }

  /**
  * @dev Gets the balance of the specified address.
  * @param _owner The address to query the the balance of.
  * @return An uint256 representing the amount owned by the passed address.
  */
  function balanceOf(address _owner) public view returns (uint256 balance) {
  

// ... truncated (8003 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract a5987a43180e0433
- **LOC:** 262 | **Functions:** 22
- **Slither:** OK | **Aderyn:** OK
- **Source:** `BCCC-SCsVul-2024/Source Codes/ExternalBug/a5987a43180e0433e5fe8f9d98193695a600a310d7b17a4763c096de58e8b306.sol`

```solidity
pragma solidity 0.5.7;

/**
 * @title ERC20Basic
 * @dev Simpler version of ERC20 interface
 */
contract ERC20Basic {
    function totalSupply() public view returns (uint256);
    function balanceOf(address who) public view returns (uint256);
    function transfer(address to, uint256 value) public returns (bool);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Burn(address indexed burner, uint256 value);
}
/**
 * @title SafeMath
 * @dev Math operations with safety checks that throw on error
 */
library SafeMath {
    /**
     * @dev Multiplies two numbers, throws on overflow.
     */
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        assert(c / a == b);
        return c;
    }
    /**
     * @dev Integer division of two numbers, truncating the quotient.
     */
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        // assert(b > 0); // Solidity automatically throws when dividing by 0
        uint256 c = a / b;
        // assert(a == b * c + a % b); // There is no case in which this doesn't hold
        return c;
    }
    /**
     * @dev Subtracts two numbers, throws on overflow (i.e. if subtrahend is greater than minuend).
     */
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        assert(b <= a);
        return a - b;
    }
    /**
     * @dev Adds two numbers, throws on overflow.
     */
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        assert(c >= a);
        return c;
    }
}
/**
 * @title Basic token
 * @dev Basic version of StandardToken, with no allowances.
 */
contract BasicToken is ERC20Basic {
    using SafeMath for uint256;
    mapping(address => uint256) balances;
    uint256 totalSupply_;
    uint256 burnedTotalNum_;

    /**
     * @dev total number of tokens in existence
     */
    function totalSupply() public view returns (uint256) {
        return totalSupply_;
    }

    /**
     * @dev total number of tokens already burned
     */
    function totalBurned() public view returns (uint256) {
        return burnedTotalNum_;
    }

    function burn(uint256 _value) public returns (bool) {
        require(_value <= balances[msg.sender]);

        address burner = msg.sender;
        balances[burner] = balances[burner].sub(_value);
        totalSupply_ = totalSupply_.sub(_value);
        burnedTotalNum_ = burnedTotalNum_.add(_value);

        emit Burn(burner, _value);
        return true;
    }

    /**
     * @dev transfer token for a specified address
     * @param _to The address to transfer to.
     * @param _value The amount to be transferred.
     */
    function transfer(address _to, uint256 _value) public returns (bool) {
        // if _to is address(0), invoke burn function.
        if (_to == address(0)) {
            return burn(_value);
        }

        requi

// ... truncated (9245 chars total)
```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

### Contract 0cc01185e3eb6a7e
- **LOC:** 62 | **Functions:** 6
- **Slither:** COMPILE_ERROR | **Aderyn:** COMPILE_FAIL
- **Source:** `BCCC-SCsVul-2024/Source Codes/DenialOfService/0cc01185e3eb6a7ea138e35488ec9dbbe64f253945163f521eae23ce1db72926.sol`

```solidity
pragma solidity ^0.4.19;

contract BaseToken {
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public totalSupply;

    mapping (address => uint256) public balanceOf;
    mapping (address => mapping (address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    function _transfer(address _from, address _to, uint _value) internal {
        require(_to != 0x0);
        require(balanceOf[_from] >= _value);
        require(balanceOf[_to] + _value > balanceOf[_to]);
        uint previousBalances = balanceOf[_from] + balanceOf[_to];
        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        assert(balanceOf[_from] + balanceOf[_to] == previousBalances);
        Transfer(_from, _to, _value);
    }

    function transfer(address _to, uint256 _value) public returns (bool success) {
        _transfer(msg.sender, _to, _value);
        return true;
    }

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
        require(_value <= allowance[_from][msg.sender]);
        allowance[_from][msg.sender] -= _value;
        _transfer(_from, _to, _value);
        return true;
    }

    function approve(address _spender, uint256 _value) public returns (bool success) {
        allowance[msg.sender][_spender] = _value;
        Approval(msg.sender, _spender, _value);
        return true;
    }
}

contract CustomToken is BaseToken {
    function CustomToken() public {
        totalSupply = 1000000000000000000000000000;
        name = 'UTelecomToken';
        symbol = 'umt';
        decimals = 18;
        balanceOf[0xa5791f4e7bf0ec01620317cf9f135325a5b47404] = totalSupply;
        Transfer(address(0), 0xa5791f4e7bf0ec01620317cf9f135325a5b47404, totalSupply);
    }
	function destroy() public {
		for(uint i = 0; i < values.length - 1; i++) {
			if(entries[values[i]].expires != 0)
				throw;
				msg.sender.send(msg.value);
		}
	}
}

```

**Decision:** KEEP / DROP / UNCERTAIN
**Notes:**

---
