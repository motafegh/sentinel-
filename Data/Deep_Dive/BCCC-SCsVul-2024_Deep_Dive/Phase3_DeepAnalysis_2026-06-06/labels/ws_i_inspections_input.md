# WS-I Inspections Input — Manual Review Material

**Purpose:** Single file you can scroll through to decide label-change recommendations for the 30 worst-disagreement + 2 maxing contracts.

**For each contract, you see:**

- BCCC path + folder + pragma

- BCCC labels (which classes are marked = 1)

- Slither findings grouped by detector

- Full source code (truncated to 200 lines if huge)


**At the bottom of each entry is a 'Decision' line with 4 checkboxes:**

- `[ ] KEEP` — BCCC labels are correct, leave as-is

- `[ ] MODIFY: <reason>` — recommend label change (add/remove class, add to dropped set, etc.)

- `[ ] REVIEW-NEEDED` — ambiguous, need Aderyn or human second opinion

- `[ ] FALSE-POSITIVE-CONTRACT` — entire contract is templated/junk


---


## Contract 1/30: `1cbf966046f79bfde8e00869`

**Sample bucket:** nine_folder_maxing  |  **Primary class:** ExternalBug  |  **n_pos:** 8  |  **Pragma:** `^0.4.13`  |  **Disagreement score:** 0.444

**BCCC folders containing this contract:** ['Timestamp', 'UnusedReturn', 'Reentrancy', 'IntegerUO', 'GasException', 'ExternalBug', 'MishandledException', 'TransactionOrderDependence', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/1cbf966046f79bfde8e008699e3e7ed0bbdaa9c0afc3c16a2dbd12f1db0234cc.sol`


### Source code:
```solidity
pragma solidity ^0.4.13;

contract ERC20Basic {
  function totalSupply() public view returns (uint256);
  function balanceOf(address who) public view returns (uint256);
  function transfer(address to, uint256 value) public returns (bool);
  event Transfer(address indexed from, address indexed to, uint256 value);
}

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

contract ERC20 is ERC20Basic {
  function allowance(address owner, address spender)
    public view returns (uint256);

  function transferFrom(address from, address to, uint256 value)
    public returns (bool);

  function approve(address spender, uint256 value) public returns (bool);
  event Approval(
    address indexed owner,
    address indexed spender,
    uint256 value
  );
}

contract StandardToken is ERC20, BasicToken {

  mapping (address => mapping (address => uint256)) internal allowed;


  /**
   * @dev Transfer tokens from one address to another
   * @param _from address The address which you want to send tokens from
   * @param _to address The address which you want to transfer to
   * @param _value uint256 the amount of tokens to be transferred
   */
  function transferFrom(
    address _from,
    address _to,
    uint256 _value
  )
    public
    returns (bool)
  {
    require(_to != address(0));
    require(_value <= balances[_from]);
    require(_value <= allowed[_from][msg.sender]);

    balances[_from] = balances[_from].sub(_value);
    balances[_to] = balances[_to].add(_value);
    allowed[_from][msg.sender] = allowed[_from][msg.sender].sub(_value);
    emit Transfer(_from, _to, _value);
    return true;
  }

  /**
   * @dev Approve the passed address to spend the specified amount of tokens on behalf of msg.sender.
   *
   * Beware that changing an allowance with this method brings the risk that someone may use both the old
   * and the new allowance by unfortunate transaction ordering. One possible solution to mitigate this
   * race condition is to first reduce the spender's allowance to 0 and set the desired value afterwards:
   * https://github.com/ethereum/EIPs/issues/20#issuecomment-263524729
   * @param _spender The address which will spend the funds.
   * @param _value The amount of tokens to be spent.
   */
  function approve(address _spender, uint256 _value) public returns (bool) {
    allowed[msg.sender][_spender] = _value;
    emit Approval(msg.sender, _spender, _value);
    return true;
  }

  /**
   * @dev Function to check the amount of tokens that an owner allowed to a spender.
   * @param _owner address The address which owns the funds.
   * @param _spender address The address which will spend the funds.
   * @return A uint256 specifying the amount of tokens still available for the spender.
   */
  function allowance(
    address _owner,
    address _spender
   )
    public
    view
    returns (uint256)
  {
    return allowed[_owner][_spender];
  }

  /**
   * @dev Increase the amount of tokens that an owner allowed to a spender.
   *
   * approve should be called when allowed[_spender] == 0. To increment
   * allowed value is better to use this function to avoid 2 calls (and wait until
   * the first transaction is mined)
   * From MonolithDAO Token.sol
   * @param _spender The address which will spend the funds.
   * @param _addedValue The amount of tokens to increase the allowance by.
   */
  function increaseApproval(
    address _spender,
    uint _addedValue
  )
    public
    returns (bool)
  {
    allowed[msg.sender][_spender] = (
      allowed[msg.sender][_spender].add(_addedValue));
    emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
  }

  /**
   * @dev Decrease the amount of tokens that an owner allowed to a spender.
   *
   * approve should be called when allowed[_spender] == 0. To decrement
   * allowed value is better to use this function to avoid 2 calls (and wait until
   * the first transaction is mined)
   * From MonolithDAO Token.sol
   * @param _spender The address which will spend the funds.
   * @param _subtractedValue The amount of tokens to decrease the allowance by.
   */
  function decreaseApproval(
    address _spender,
    uint _subtractedValue
  )
    public
    returns (bool)
  {
    uint oldValue = allowed[msg.sender][_spender];
    if (_subtractedValue > oldValue) {
      allowed[msg.sender][_spender] = 0;
    } else {
      allowed[msg.sender][_spender] = oldValue.sub(_subtractedValue);
    }
    emit Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
    return true;
  }

}

contract HongZhangCoin is StandardToken {
  string public name = "HongZhangCoin"; 
  string public symbol = "HZ";
  uint public decimals = 18;
  uint public INITIAL_SUPPLY = 60000 * 10000 * (10 ** decimals);

  constructor() public {
    totalSupply_ = INITIAL_SUPPLY;
    balances[msg.sender] = INITIAL_SUPPLY;
  }
}

library SafeMath {

  /**
  * @dev Multiplies two numbers, throws on overflow.
  */
  function mul(uint256 a, uint256 b) internal pure returns (uint256 c) {
    // Gas optimization: this is cheaper than asserting 'a' not being zero, but the
    // benefit is lost if 'b' is also tested.
    // See: https://github.com/OpenZeppelin/openzeppelin-solidity/pull/522

... [truncated, 36 more lines]
```

### BCCC labels (8 positive):

- Class01:ExternalBug, Class02:GasException, Class03:MishandledException, Class04:Timestamp, Class06:UnusedReturn, Class08:CallToUnknown, Class10:IntegerUO, Class11:Reentrancy


### Slither findings (OK):

- **Total findings:** 21  |  **Unique detectors:** 4

- **Top detectors (by frequency):**

  - `naming-convention` × 15

  - `constable-states` × 3

  - `solc-version` × 2

  - `function-init-state` × 1


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 2/30: `147725c17af042280cfb4a4b`

**Sample bucket:** nine_folder_maxing  |  **Primary class:** ExternalBug  |  **n_pos:** 8  |  **Pragma:** `0.4.24`  |  **Disagreement score:** 0.389

**BCCC folders containing this contract:** ['Timestamp', 'UnusedReturn', 'Reentrancy', 'IntegerUO', 'GasException', 'ExternalBug', 'MishandledException', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/147725c17af042280cfb4a4b2b709e133fe0d38020057498052dc60bcf0edbb8.sol`


### Source code:
```solidity
pragma solidity 0.4.24;

contract DocSignature {

  struct SignProcess {
    bytes16 id;
    bytes16[] participants;
  }

  address controller;
  mapping(address => bool) isOwner;
  mapping(bytes => SignProcess[]) documents;
  address[] public owners;

  constructor(address _controller, address _owner) public {
    controller = _controller;
    isOwner[_owner] = true;
    owners.push(_owner);
  }

  modifier notNull(address _address) {
    require(_address != 0);
    _;
  }

  function addOwner(address _owner) public notNull(_owner) returns (bool) {
    require(isOwner[msg.sender]);
    isOwner[_owner] = true;
    owners.push(_owner);
    return true;
  }

  function removeOwner(address _owner) public notNull(_owner) returns (bool) {
    require(msg.sender != _owner && isOwner[msg.sender]);
    isOwner[_owner] = false;
    for (uint i=0; i<owners.length - 1; i++)
      if (owners[i] == _owner) {
        owners[i] = owners[owners.length - 1];
        break;
      }
    owners.length -= 1;
    return true;
  }

  function getOwners() public constant returns (address[]) {
    return owners;
  }

  function setController(address _controller) public notNull(_controller) returns (bool) {
    require(isOwner[msg.sender]);
    controller = _controller;
    return true;
  }

  function signDocument(bytes _documentHash, bytes16 _signProcessId, bytes16[] _participants) public returns (bool) {
    require(msg.sender == controller);
    documents[_documentHash].push(SignProcess(_signProcessId, _participants));
    return true;
  }

  function getDocumentSignatures(bytes _documentHash) public view returns (bytes16[], bytes16[]) {
    uint _signaturesCount = 0;
    for (uint o = 0; o < documents[_documentHash].length; o++) {
      _signaturesCount += documents[_documentHash][o].participants.length;
    }

    bytes16[] memory _ids = new bytes16[](_signaturesCount);
    bytes16[] memory _participants = new bytes16[](_signaturesCount);

    uint _index = 0;
    for (uint i = 0; i < documents[_documentHash].length; i++) {
      for (uint j = 0; j < documents[_documentHash][i].participants.length; j++) {
        _ids[_index] =  documents[_documentHash][i].id;
        _participants[_index] = documents[_documentHash][i].participants[j];
        _index++;
      }
    }

    return (_ids, _participants);
  }

  function getDocumentProcesses(bytes _documentHash) public view returns (bytes16[]) {
    bytes16[] memory _ids = new bytes16[](documents[_documentHash].length);

    for (uint i = 0; i < documents[_documentHash].length; i++) {
      _ids[i] =  documents[_documentHash][i].id;
    }

    return (_ids);
  }

  function getProcessParties(bytes _documentHash, bytes16 _processId) public view returns (bytes16[]) {
    for (uint i = 0; i < documents[_documentHash].length; i++) {
      if (documents[_documentHash][i].id == _processId)
        return (documents[_documentHash][i].participants);
    }
    return (new bytes16[](0));
  }
}
```

### BCCC labels (8 positive):

- Class01:ExternalBug, Class02:GasException, Class03:MishandledException, Class04:Timestamp, Class06:UnusedReturn, Class08:CallToUnknown, Class10:IntegerUO, Class11:Reentrancy


### Slither findings (OK):

- **Total findings:** 19  |  **Unique detectors:** 5

- **Top detectors (by frequency):**

  - `naming-convention` × 10

  - `external-function` × 4

  - `controlled-array-length` × 2

  - `solc-version` × 2

  - `missing-zero-check` × 1


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 3/30: `1ff44f67b1981220331860e9`

**Sample bucket:** review_pending  |  **Primary class:** GasException  |  **n_pos:** 5  |  **Pragma:** `^0.4.18`  |  **Disagreement score:** 0.389

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'GasException', 'DenialOfService', 'MishandledException']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/DenialOfService/1ff44f67b1981220331860e9944c94acc3e42f3129db7319adf57d91c3889b7c.sol`


### Source code:
```solidity
pragma solidity ^0.4.18;

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
  * @dev Substracts two numbers, throws on overflow (i.e. if subtrahend is greater than minuend).
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


/**
 * @title ERC20 interface
 * @dev see https://github.com/ethereum/EIPs/issues/20
 */
contract ERC20 is ERC20Basic {
  function allowance(address owner, address spender) public view returns (uint256);
  function transferFrom(address from, address to, uint256 value) public returns (bool);
  function approve(address spender, uint256 value) public returns (bool);
  event Approval(address indexed owner, address indexed spender, uint256 value);
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
    return balances[_owner];
  }

}


/**
 * @title Standard ERC20 token
 *
 * @dev Implementation of the basic standard token.
 * @dev https://github.com/ethereum/EIPs/issues/20
 * @dev Based on code by FirstBlood: https://github.com/Firstbloodio/token/blob/master/smart_contract/FirstBloodToken.sol
 */
contract StandardToken is ERC20, BasicToken {

  mapping (address => mapping (address => uint256)) internal allowed;


  /**
   * @dev Transfer tokens from one address to another
   * @param _from address The address which you want to send tokens from
   * @param _to address The address which you want to transfer to
   * @param _value uint256 the amount of tokens to be transferred
   */
  function transferFrom(address _from, address _to, uint256 _value) public returns (bool) {
    require(_to != address(0));
    require(_value <= balances[_from]);
    require(_value <= allowed[_from][msg.sender]);

    balances[_from] = balances[_from].sub(_value);
    balances[_to] = balances[_to].add(_value);
    allowed[_from][msg.sender] = allowed[_from][msg.sender].sub(_value);
    Transfer(_from, _to, _value);
    return true;
  }

  /**
   * @dev Approve the passed address to spend the specified amount of tokens on behalf of msg.sender.
   *
   * Beware that changing an allowance with this method brings the risk that someone may use both the old
   * and the new allowance by unfortunate transaction ordering. One possible solution to mitigate this
   * race condition is to first reduce the spender's allowance to 0 and set the desired value afterwards:
   * https://github.com/ethereum/EIPs/issues/20#issuecomment-263524729
   * @param _spender The address which will spend the funds.
   * @param _value The amount of tokens to be spent.

... [truncated, 234 more lines]
```

### BCCC labels (5 positive):

- Class02:GasException, Class03:MishandledException, Class09:DenialOfService, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 41  |  **Unique detectors:** 6

- **Top detectors (by frequency):**

  - `naming-convention` × 31

  - `divide-before-multiply` × 4

  - `solc-version` × 2

  - `timestamp` × 2

  - `external-function` × 1

  - `too-many-digits` × 1


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 4/30: `9e3909b6bc876db6ab764e09`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.18`  |  **Disagreement score:** 0.389

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/9e3909b6bc876db6ab764e099553a7a32cfa1be39100496fead49ca6afb1d6b9.sol`


### Source code:
```solidity
pragma solidity ^0.4.18;
/**
 * Math operations with safety checks
 */
library SafeMath {
  function mul(uint a, uint b) internal pure returns (uint) {
    uint c = a * b;
    assert(a == 0 || c / a == b);
    return c;
  }

  function div(uint a, uint b) internal pure returns (uint) {
    // assert(b > 0); // Solidity automatically throws when dividing by 0
    uint c = a / b;
    // assert(a == b * c + a % b); // There is no case in which this doesn't hold
    return c;
  }

  function sub(uint a, uint b) internal pure returns (uint) {
    assert(b <= a);
    return a - b;
  }

  function add(uint a, uint b) internal pure returns (uint) {
    uint c = a + b;
    assert(c >= a);
    return c;
  }

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

// ERC20 token interface is implemented only partially.
interface tokenRecipient { function receiveApproval(address _from, uint256 _value, address _token, bytes _extraData) public; }

contract NamiPool {
    using SafeMath for uint256;
    
    function NamiPool(address _escrow, address _namiMultiSigWallet, address _namiAddress) public {
        require(_namiMultiSigWallet != 0x0);
        escrow = _escrow;
        namiMultiSigWallet = _namiMultiSigWallet;
        NamiAddr = _namiAddress;
    }
    
    string public name = "Nami Pool";
    
    // escrow has exclusive priveleges to call administrative
    // functions on this contract.
    address public escrow;

    // Gathered funds can be withdraw only to namimultisigwallet's address.
    address public namiMultiSigWallet;
    
    /// address of Nami token
    address public NamiAddr;
    
    modifier onlyEscrow() {
        require(msg.sender == escrow);
        _;
    }
    
    modifier onlyNami {
        require(msg.sender == NamiAddr);
        _;
    }
    
    modifier onlyNamiMultisig {
        require(msg.sender == namiMultiSigWallet);
        _;
    }
    
    uint public currentRound = 1;
    
    struct ShareHolder {
        uint stake;
        bool isActive;
        bool isWithdrawn;
    }
    
    struct Round {
        bool isOpen;
        uint currentNAC;
        uint finalNAC;
        uint ethBalance;
        bool withdrawable; //for user not in top
        bool topWithdrawable;
        bool isCompleteActive;
        bool isCloseEthPool;
    }
    
    mapping (uint => mapping (address => ShareHolder)) public namiPool;
    mapping (uint => Round) public round;
    
    
    // Events
    event UpdateShareHolder(address indexed ShareHolderAddress, uint indexed RoundIndex, uint Stake, uint Time);
    event Deposit(address sender,uint indexed RoundIndex, uint value);
    event WithdrawPool(uint Amount, uint TimeWithdraw);
    event UpdateActive(address indexed ShareHolderAddress, uint indexed RoundIndex, bool Status, uint Time);
    event Withdraw(address indexed ShareHolderAddress, uint indexed RoundIndex, uint Ether, uint Nac, uint TimeWithdraw);
    event ActivateRound(uint RoundIndex, uint TimeActive);
    
    
    function changeEscrow(address _escrow)
        onlyNamiMultisig
        public
    {
        require(_escrow != 0x0);
        escrow = _escrow;
    }
    
    function withdrawEther(uint _amount) public
        onlyEscrow
    {
        require(namiMultiSigWallet != 0x0);
        // 
        if (this.balance > 0) {
            namiMultiSigWallet.transfer(_amount);
        }
    }
    
    function withdrawNAC(uint _amount) public
        onlyEscrow
    {
        require(namiMultiSigWallet != 0x0 && _amount != 0);
        NamiCrowdSale namiToken = NamiCrowdSale(NamiAddr);
        if (namiToken.balanceOf(this) > 0) {
            namiToken.transfer(namiMultiSigWallet, _amount);
        }
    }
    
    
    /*/
     *  Admin function
    /*/
    
    /*/ process of one round
     * step 1: admin open one round by execute activateRound function
     * step 2: now investor can invest Nac to Nac Pool until round closed
     * step 3: admin close round, now investor cann't invest NAC to Pool
     * step 4: admin activate top investor
     * step 5: all top investor was activated, admin execute closeActive function to close active phrase
     * step 6: admin open withdrawable for investor not in top to withdraw NAC
     * step 7: admin deposit eth to eth pool
     * step 8: close deposit eth to eth pool
     * step 9: admin open withdrawable to investor in top
     * step 10: investor in top now can withdraw NAC and ETH for this round
    /*/
    
    // ------------------------------------------------ 
    /*
    * Admin function
    * Open and Close Round
    *
    */
    function activateRound(uint _roundIndex) 
        onlyEscrow
        public
    {
        require(round[_roundIndex].isOpen == false && round[_roundIndex].isCloseEthPool == false && round[_roundIndex].isCompleteActive == false);
        round[_roundIndex].isOpen = true;
        currentRound = _roundIndex;
        ActivateRound(_roundIndex, now);
    }
    
    function deactivateRound(uint _roundIndex)
        onlyEscrow
        public
    {
        require(round[_roundIndex].isOpen == true);
        round[_roundIndex].isOpen = false;
    }
    
    // ------------------------------------------------ 
    // this function add stake of ShareHolder
    // investor can execute this function during round open
    //
    
    function tokenFallbackExchange(address _from, uint _value, uint _price) onlyNami public returns (bool success) {
        // only on currentRound and active user can add stake
        require(round[_price].isOpen == true && _value > 0);
        // add stake
        namiPool[_price][_from].stake = namiPool[_price][_from].stake.add(_value);
        round[_price].currentNAC = round[_price].currentNAC.add(_value);
        UpdateShareHolder(_from, _price, namiPool[_price][_from].stake, now);

... [truncated, 1600 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 202  |  **Unique detectors:** 26

- **Top detectors (by frequency):**

  - `naming-convention` × 84

  - `boolean-equal` × 25

  - `timestamp` × 12

  - `reentrancy-events` × 10

  - `reentrancy-unlimited-gas` × 9

  - `missing-zero-check` × 7

  - `constable-states` × 6

  - `events-maths` × 5

  - `reentrancy-eth` × 5

  - `cache-array-length` × 4

  - `external-function` × 4

  - `events-access` × 4

  - `reentrancy-benign` × 3

  - `unindexed-event-address` × 3

  - `uninitialized-local` × 3

  - ... and 11 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 5/30: `8177423cb92d8643b00f64d8`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.24`  |  **Disagreement score:** 0.389

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/8177423cb92d8643b00f64d8e5af7b4ea572cac26f88562e6952cdea5f6982e3.sol`


### Source code:
```solidity
pragma solidity ^0.4.24;

contract POOHMOXevents {
    // fired whenever a player registers a name
    event onNewName
    (
        uint256 indexed playerID,
        address indexed playerAddress,
        bytes32 indexed playerName,
        bool isNewPlayer,
        uint256 affiliateID,
        address affiliateAddress,
        bytes32 affiliateName,
        uint256 amountPaid,
        uint256 timeStamp
    );

    // fired at end of buy or reload
    event onEndTx
    (
        uint256 compressedData,
        uint256 compressedIDs,
        bytes32 playerName,
        address playerAddress,
        uint256 ethIn,
        uint256 keysBought,
        address winnerAddr,
        bytes32 winnerName,
        uint256 amountWon,
        uint256 newPot,
        uint256 POOHAmount,
        uint256 genAmount,
        uint256 potAmount
    );

    // fired whenever theres a withdraw
    event onWithdraw
    (
        uint256 indexed playerID,
        address playerAddress,
        bytes32 playerName,
        uint256 ethOut,
        uint256 timeStamp
    );

    // fired whenever a withdraw forces end round to be ran
    event onWithdrawAndDistribute
    (
        address playerAddress,
        bytes32 playerName,
        uint256 ethOut,
        uint256 compressedData,
        uint256 compressedIDs,
        address winnerAddr,
        bytes32 winnerName,
        uint256 amountWon,
        uint256 newPot,
        uint256 POOHAmount,
        uint256 genAmount
    );

    // fired whenever a player tries a buy after round timer
    // hit zero, and causes end round to be ran.
    event onBuyAndDistribute
    (
        address playerAddress,
        bytes32 playerName,
        uint256 ethIn,
        uint256 compressedData,
        uint256 compressedIDs,
        address winnerAddr,
        bytes32 winnerName,
        uint256 amountWon,
        uint256 newPot,
        uint256 POOHAmount,
        uint256 genAmount
    );

    // fired whenever a player tries a reload after round timer
    // hit zero, and causes end round to be ran.
    event onReLoadAndDistribute
    (
        address playerAddress,
        bytes32 playerName,
        uint256 compressedData,
        uint256 compressedIDs,
        address winnerAddr,
        bytes32 winnerName,
        uint256 amountWon,
        uint256 newPot,
        uint256 POOHAmount,
        uint256 genAmount
    );

    // fired whenever an affiliate is paid
    event onAffiliatePayout
    (
        uint256 indexed affiliateID,
        address affiliateAddress,
        bytes32 affiliateName,
        uint256 indexed roundID,
        uint256 indexed buyerID,
        uint256 amount,
        uint256 timeStamp
    );

    // received pot swap deposit
    event onPotSwapDeposit
    (
        uint256 roundID,
        uint256 amountAddedToPot
    );
}

//==============================================================================
//   _ _  _ _|_ _ _  __|_   _ _ _|_    _   .
//  (_(_)| | | | (_|(_ |   _\(/_ | |_||_)  .
//====================================|=========================================

contract POOHMOX is POOHMOXevents {
    using SafeMath for *;
    using NameFilter for string;
    using KeysCalc for uint256;

    PlayerBookInterface private PlayerBook;

//==============================================================================
//     _ _  _  |`. _     _ _ |_ | _  _  .
//    (_(_)| |~|~|(_||_|| (_||_)|(/__\  .  (game settings)
//=================_|===========================================================
    address private admin = msg.sender;
    address private flushDivs;
    string constant public name = "POOHMOX";
    string constant public symbol = "POOHMOX";
    uint256 private rndExtra_ = 1 seconds;     // length of the very first ICO phase
    uint256 private rndGap_ = 1 seconds;       // length of ICO phases
    uint256 private rndInit_ = 5 minutes;      // round timer starts at this
    uint256 private rndMax_ = 5 minutes;       // max length a round timer can be  (for first round) 
    uint256 constant private rndInc_ = 5 minutes;// every full key purchased adds this much to the timer
                                 
//==============================================================================
//     _| _ _|_ _    _ _ _|_    _   .
//    (_|(_| | (_|  _\(/_ | |_||_)  .  (data used to store game info that changes)
//=============================|================================================
    uint256 public rID_;    // round id number / total rounds that have happened
//****************
// PLAYER DATA
//****************
    mapping (address => uint256) public pIDxAddr_;          // (addr => pID) returns player id by address
    mapping (bytes32 => uint256) public pIDxName_;          // (name => pID) returns player id by name
    mapping (uint256 => POOHMOXDatasets.Player) public plyr_;   // (pID => data) player data
    mapping (uint256 => mapping (uint256 => POOHMOXDatasets.PlayerRounds)) public plyrRnds_;    // (pID => rID => data) player round data by player id & round id
    mapping (uint256 => mapping (bytes32 => bool)) public plyrNames_; // (pID => name => bool) list of names a player owns.  (used so you can change your display name amongst any name you own)
//****************
// ROUND DATA
//****************
    mapping (uint256 => POOHMOXDatasets.Round) public round_;   // (rID => data) round data
    mapping (uint256 => mapping(uint256 => uint256)) public rndTmEth_;      // (rID => tID => data) eth in per team, by round id and team id
//****************
// TEAM FEE DATA
//****************
    mapping (uint256 => POOHMOXDatasets.TeamFee) public fees_;          // (team => fees) fee distribution by team
    mapping (uint256 => POOHMOXDatasets.PotSplit) public potSplit_;     // (team => fees) pot split distribution by team
//==============================================================================
//     _ _  _  __|_ _    __|_ _  _  .
//    (_(_)| |_\ | | |_|(_ | (_)|   .  (initial data setup upon contract deploy)
//==============================================================================
    constructor(address whaleContract, address playerbook)
        public
    {
        flushDivs = whaleContract;
        PlayerBook = PlayerBookInterface(playerbook);

        //no teams... only POOH-heads
        // Referrals / Community rewards are mathematically designed to come from the winner's share of the pot.
        fees_[0] = POOHMOXDatasets.TeamFee(39,20);   //30% to pot, 10% to aff, 1% to dev,
       

        potSplit_[0] = POOHMOXDatasets.PotSplit(15,10);  //36% to winner, 36% to next round, 3% to dev
    }
//==============================================================================
//     _ _  _  _|. |`. _  _ _  .
//    | | |(_)(_||~|~|(/_| _\  .  (these are safety checks)
//==============================================================================
    /**
     * @dev used to make sure no one can interact with contract until it has
     * been activated.
     */
    modifier isActivated() {
        require(activated_ == true);
        _;
    }

    /**
     * @dev prevents contracts from interacting with fomo3d
     */
    modifier isHuman() {
        address _addr = msg.sender;
        uint256 _codeLength;


... [truncated, 1547 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 194  |  **Unique detectors:** 21

- **Top detectors (by frequency):**

  - `naming-convention` × 93

  - `too-many-digits` × 18

  - `reentrancy-events` × 12

  - `boolean-equal` × 11

  - `divide-before-multiply` × 8

  - `timestamp` × 8

  - `reentrancy-unlimited-gas` × 6

  - `constable-states` × 4

  - `reentrancy-eth` × 4

  - `reentrancy-no-eth` × 4

  - `unindexed-event-address` × 4

  - `uninitialized-local` × 4

  - `external-function` × 3

  - `low-level-calls` × 3

  - `reentrancy-benign` × 3

  - ... and 6 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 6/30: `1f448dcda1b131fe20ee2c91`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.13`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/1f448dcda1b131fe20ee2c91d627ecd59e047e7b3b737bc4097540633168ba68.sol`


### Source code:
```solidity
pragma solidity ^0.4.13;        
   
  contract CentraSale { 

    using SafeMath for uint; 

    address public contract_address = 0x96a65609a7b84e8842732deb08f56c3e21ac6f8a; 

    address public owner;
    uint public cap;
    uint public constant cap_max = 170000*10**18;
    uint public constant min_value = 10**18*1/10; 
    uint public operation;
    mapping(uint => address) public operation_address;
    mapping(uint => uint) public operation_amount;

    uint256 public constant token_price = 10**18*1/250;  
    uint256 public tokens_total;  

    uint public constant contract_start = 1505844000;
    uint public constant contract_finish = 1507269600;

    uint public constant card_titanium_minamount = 500*10**18;
    uint public constant card_titanium_first = 200;
    mapping(address => uint) cards_titanium_check; 
    address[] public cards_titanium;

    uint public constant card_black_minamount = 100*10**18;
    uint public constant card_black_first = 500;
    mapping(address => uint) public cards_black_check; 
    address[] public cards_black;

    uint public constant card_metal_minamount = 40*10**18;
    uint public constant card_metal_first = 750;
    mapping(address => uint) cards_metal_check; 
    address[] public cards_metal;      

    uint public constant card_gold_minamount = 30*10**18;
    uint public constant card_gold_first = 1000;
    mapping(address => uint) cards_gold_check; 
    address[] public cards_gold;      

    uint public constant card_blue_minamount = 5/10*10**18;
    uint public constant card_blue_first = 100000000;
    mapping(address => uint) cards_blue_check; 
    address[] public cards_blue;

    uint public constant card_start_minamount = 1/10*10**18;
    uint public constant card_start_first = 100000000;
    mapping(address => uint) cards_start_check; 
    address[] public cards_start;
      
   
    // Functions with this modifier can only be executed by the owner
    modifier onlyOwner() {
        if (msg.sender != owner) {
            throw;
        }
        _;
    }      
 
    // Constructor
    function CentraSale() {
        owner = msg.sender; 
        operation = 0; 
        cap = 0;        
    }
      
    //default function for crowdfunding
    function() payable {    

      if(!(msg.value >= min_value)) throw;
      if(now < contract_start) throw;
      if(now > contract_finish) throw;                     

      if(cap + msg.value > cap_max) throw;         

      tokens_total = msg.value*10**18/token_price;
      if(!(tokens_total > 0)) throw;           

      if(!contract_transfer(tokens_total)) throw;

      cap = cap.add(msg.value); 
      operations();
      get_card();
      owner.send(this.balance);
    }

    //Contract execute
    function contract_transfer(uint _amount) private returns (bool) {      

      if(!contract_address.call(bytes4(sha3("transfer(address,uint256)")),msg.sender,_amount)) {    
        return false;
      }
      return true;
    } 

    //Update operations
    function operations() private returns (bool) {
        operation_address[operation] = msg.sender;
        operation_amount[operation] = msg.value;        
        operation = operation.add(1);        
        return true;
    }    

    //Withdraw money from contract balance to owner
    function withdraw() onlyOwner returns (bool result) {
        owner.send(this.balance);
        return true;
    }

    //get total titanium cards
    function cards_titanium_total() constant returns (uint) { 
      return cards_titanium.length;
    }  
    //get total black cards
    function cards_black_total() constant returns (uint) { 
      return cards_black.length;
    }
    //get total metal cards
    function cards_metal_total() constant returns (uint) { 
      return cards_metal.length;
    }        
    //get total gold cards
    function cards_gold_total() constant returns (uint) { 
      return cards_gold.length;
    }        
    //get total blue cards
    function cards_blue_total() constant returns (uint) { 
      return cards_blue.length;
    }

    //get total start cards
    function cards_start_total() constant returns (uint) { 
      return cards_start.length;
    }

    /*
    * User get card(titanium, black, gold metal, gold and other), if amount eth sufficient for this.
    */
    function get_card() private returns (bool) {

      if((msg.value >= card_titanium_minamount)
        &&(cards_titanium.length < card_titanium_first)
        &&(cards_titanium_check[msg.sender] != 1)
        ) {
        cards_titanium.push(msg.sender);
        cards_titanium_check[msg.sender] = 1;
      }

      if((msg.value >= card_black_minamount)
        &&(msg.value < card_titanium_minamount)
        &&(cards_black.length < card_black_first)
        &&(cards_black_check[msg.sender] != 1)
        ) {
        cards_black.push(msg.sender);
        cards_black_check[msg.sender] = 1;
      }                

      if((msg.value >= card_metal_minamount)
        &&(msg.value < card_black_minamount)
        &&(cards_metal.length < card_metal_first)
        &&(cards_metal_check[msg.sender] != 1)
        ) {
        cards_metal.push(msg.sender);
        cards_metal_check[msg.sender] = 1;
      }               

      if((msg.value >= card_gold_minamount)
        &&(msg.value < card_metal_minamount)
        &&(cards_gold.length < card_gold_first)
        &&(cards_gold_check[msg.sender] != 1)
        ) {
        cards_gold.push(msg.sender);
        cards_gold_check[msg.sender] = 1;
      }               

      if((msg.value >= card_blue_minamount)
        &&(msg.value < card_gold_minamount)
        &&(cards_blue.length < card_blue_first)
        &&(cards_blue_check[msg.sender] != 1)
        ) {
        cards_blue.push(msg.sender);
        cards_blue_check[msg.sender] = 1;
      }

      if((msg.value >= card_start_minamount)
        &&(msg.value < card_blue_minamount)
        &&(cards_start.length < card_start_first)
        &&(cards_start_check[msg.sender] != 1)
        ) {
        cards_start.push(msg.sender);
        cards_start_check[msg.sender] = 1;
      }

      return true;
    }    
      
 }


... [truncated, 50 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 54  |  **Unique detectors:** 14

- **Top detectors (by frequency):**

  - `naming-convention` × 25

  - `deprecated-standards` × 9

  - `controlled-array-length` × 5

  - `divide-before-multiply` × 2

  - `solc-version` × 2

  - `too-many-digits` × 2

  - `unchecked-send` × 2

  - `arbitrary-send-eth` × 1

  - `shadowing-builtin` × 1

  - `constable-states` × 1

  - `low-level-calls` × 1

  - `reentrancy-benign` × 1

  - `reentrancy-no-eth` × 1

  - `timestamp` × 1


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 7/30: `37d50eaf5d392b72900876fb`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.0`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/37d50eaf5d392b72900876fbf91e729f9b3f4b65c85f88674fb11074e0272fe5.sol`


### Source code:
```solidity
pragma solidity ^0.4.0;

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

pragma solidity ^0.4.0;//please import oraclizeAPI_pre0.4.sol when solidity < 0.4.0

contract OraclizeI {
    address public cbAddress;
    function query(uint _timestamp, string _datasource, string _arg) payable returns (bytes32 _id);
    function query_withGasLimit(uint _timestamp, string _datasource, string _arg, uint _gaslimit) payable returns (bytes32 _id);
    function query2(uint _timestamp, string _datasource, string _arg1, string _arg2) payable returns (bytes32 _id);
    function query2_withGasLimit(uint _timestamp, string _datasource, string _arg1, string _arg2, uint _gaslimit) payable returns (bytes32 _id);
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
        if(address(OAR)==0) oraclize_setNetwork(networkID_auto);
        oraclize = OraclizeI(OAR.getAddress());
        _;
    }
    modifier coupon(string code){
        oraclize = OraclizeI(OAR.getAddress());
        oraclize.useCoupon(code);
        _;
    }

    function oraclize_setNetwork(uint8 networkID) internal returns(bool){
        if (getCodeSize(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed)>0){ //mainnet
            OAR = OraclizeAddrResolverI(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed);
            return true;
        }
        if (getCodeSize(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1)>0){ //ropsten testnet
            OAR = OraclizeAddrResolverI(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1);
            return true;
        }
        if (getCodeSize(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa)>0){ //browser-solidity
            OAR = OraclizeAddrResolverI(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa);
            return true;
        }
        return false;
    }
    
    function oraclize_query(string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(0, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(timestamp, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(timestamp, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(0, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(0, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(timestamp, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(timestamp, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(0, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_cbAddress() oraclizeAPI internal returns (address){
        return oraclize.cbAddress();
    }
    function oraclize_setProof(byte proofP) oraclizeAPI internal {
        return oraclize.setProofType(proofP);
    }
    function oraclize_setCustomGasPrice(uint gasPrice) oraclizeAPI internal {
        return oraclize.setCustomGasPrice(gasPrice);
    }    
    function oraclize_setConfig(bytes32 config) oraclizeAPI internal {
        return oraclize.setConfig(config);
    }

    function getCodeSize(address _addr) constant internal returns(uint _size) {
        assembly {
            _size := extcodesize(_addr)
        }
    }


    function parseAddr(string _a) internal returns (address){
        bytes memory tmp = bytes(_a);
        uint160 iaddr = 0;
        uint160 b1;
        uint160 b2;
        for (uint i=2; i<2+2*20; i+=2){
            iaddr *= 256;
            b1 = uint160(tmp[i]);
            b2 = uint160(tmp[i+1]);
            if ((b1 >= 97)&&(b1 <= 102)) b1 -= 87;
            else if ((b1 >= 48)&&(b1 <= 57)) b1 -= 48;
            if ((b2 >= 97)&&(b2 <= 102)) b2 -= 87;
            else if ((b2 >= 48)&&(b2 <= 57)) b2 -= 48;
            iaddr += (b1*16+b2);
        }
        return address(iaddr);
    }


    function strCompare(string _a, string _b) internal returns (int) {
        bytes memory a = bytes(_a);
        bytes memory b = bytes(_b);
        uint minLength = a.length;
        if (b.length < minLength) minLength = b.length;
        for (uint i = 0; i < minLength; i ++)
            if (a[i] < b[i])
                return -1;
            else if (a[i] > b[i])
                return 1;
        if (a.length < b.length)
            return -1;
        else if (a.length > b.length)
            return 1;
        else
            return 0;
   } 

    function indexOf(string _haystack, string _needle) internal returns (int)
    {
        bytes memory h = bytes(_haystack);
        bytes memory n = bytes(_needle);
        if(h.length < 1 || n.length < 1 || (n.length > h.length)) 
            return -1;
        else if(h.length > (2**128 -1))
            return -1;                                  
        else
        {

... [truncated, 763 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 200  |  **Unique detectors:** 23

- **Top detectors (by frequency):**

  - `naming-convention` × 81

  - `deprecated-standards` × 23

  - `dead-code` × 16

  - `unindexed-event-address` × 13

  - `costly-loop` × 11

  - `reentrancy-events` × 9

  - `external-function` × 8

  - `unused-state` × 8

  - `reentrancy-benign` × 5

  - `incorrect-modifier` × 4

  - `reentrancy-eth` × 4

  - `too-many-digits` × 4

  - `solc-version` × 2

  - `calls-loop` × 2

  - `uninitialized-local` × 2

  - ... and 8 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 8/30: `4219abdc067eb57889cf8aaf`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.19`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/4219abdc067eb57889cf8aaf1c62fb767f193298847f1bcecf2f0d86f90bfeda.sol`


### Source code:
```solidity
pragma solidity ^0.4.19;

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

contract ERC20Basic {
    uint256 public totalSupply;

    function balanceOf(address who) constant public returns (uint256);

    function transfer(address to, uint256 value) public returns (bool);

    event Transfer(address indexed from, address indexed to, uint256 value);
}

contract ERC20 is ERC20Basic {
    function allowance(address owner, address spender) constant public returns (uint256);

    function transferFrom(address from, address to, uint256 value) public returns (bool);

    function approve(address spender, uint256 value) public returns (bool);

    event Approval(address indexed owner, address indexed spender, uint256 value);
}

contract Owned {

    address public owner;

    address public newOwner;

    function Owned() public payable {
        owner = msg.sender;
    }

    modifier onlyOwner {
        require(owner == msg.sender);
        _;
    }

    function changeOwner(address _owner) onlyOwner public {
        require(_owner != 0);
        newOwner = _owner;
    }

    function confirmOwner() public {
        require(newOwner == msg.sender);
        owner = newOwner;
        delete newOwner;
    }
}

contract Blocked {

    uint public blockedUntil;

    modifier unblocked {
        require(now > blockedUntil);
        _;
    }
}

contract BasicToken is ERC20Basic, Blocked {

    using SafeMath for uint256;

    mapping (address => uint256) balances;

    // Fix for the ERC20 short address attack
    modifier onlyPayloadSize(uint size) {
        require(msg.data.length >= size + 4);
        _;
    }

    function transfer(address _to, uint256 _value) onlyPayloadSize(2 * 32) unblocked public returns (bool) {
        balances[msg.sender] = balances[msg.sender].sub(_value);
        balances[_to] = balances[_to].add(_value);
        Transfer(msg.sender, _to, _value);
        return true;
    }

    function balanceOf(address _owner) constant public returns (uint256 balance) {
        return balances[_owner];
    }

}

contract StandardToken is ERC20, BasicToken {

    mapping (address => mapping (address => uint256)) allowed;

    function transferFrom(address _from, address _to, uint256 _value) onlyPayloadSize(3 * 32) unblocked public returns (bool) {
        uint256 _allowance = allowed[_from][msg.sender];

        balances[_to] = balances[_to].add(_value);
        balances[_from] = balances[_from].sub(_value);
        allowed[_from][msg.sender] = _allowance.sub(_value);
        Transfer(_from, _to, _value);
        return true;
    }

    function approve(address _spender, uint256 _value) onlyPayloadSize(2 * 32) unblocked public returns (bool) {

        require((_value == 0) || (allowed[msg.sender][_spender] == 0));

        allowed[msg.sender][_spender] = _value;
        Approval(msg.sender, _spender, _value);
        return true;
    }

    function allowance(address _owner, address _spender) onlyPayloadSize(2 * 32) unblocked constant public returns (uint256 remaining) {
        return allowed[_owner][_spender];
    }

}

contract BurnableToken is StandardToken {

    event Burn(address indexed burner, uint256 value);

    function burn(uint256 _value) unblocked public {
        require(_value > 0);
        require(_value <= balances[msg.sender]);
        // no need to require value <= totalSupply, since that would imply the
        // sender's balance is greater than the totalSupply, which *should* be an assertion failure

        address burner = msg.sender;
        balances[burner] = balances[burner].sub(_value);
        totalSupply = totalSupply.sub(_value);
        Burn(burner, _value);
    }
}

contract DEVCoin is BurnableToken, Owned {

    string public constant name = "Dev Coin";

    string public constant symbol = "DEVC";

    uint32 public constant decimals = 18;

    function DEVCoin(uint256 initialSupply, uint unblockTime) public {
        totalSupply = initialSupply;
        balances[owner] = initialSupply;
        blockedUntil = unblockTime;
    }

    function manualTransfer(address _to, uint256 _value) onlyPayloadSize(2 * 32) onlyOwner public returns (bool) {
        balances[msg.sender] = balances[msg.sender].sub(_value);
        balances[_to] = balances[_to].add(_value);
        Transfer(msg.sender, _to, _value);
        return true;
    }
}

contract ManualSendingCrowdsale is Owned {
    using SafeMath for uint256;

    struct AmountData {
        bool exists;
        uint256 value;
    }

    mapping (uint => AmountData) public amountsByCurrency;

    function addCurrency(uint currency) external onlyOwner {
        addCurrencyInternal(currency);
    }

    function addCurrencyInternal(uint currency) internal {
        AmountData storage amountData = amountsByCurrency[currency];
        amountData.exists = true;
    }

    function manualTransferTokensToInternal(address to, uint256 givenTokens, uint currency, uint256 amount) internal returns (uint256) {

... [truncated, 240 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 57  |  **Unique detectors:** 11

- **Top detectors (by frequency):**

  - `constable-states` × 20

  - `naming-convention` × 14

  - `timestamp` × 7

  - `low-level-calls` × 3

  - `return-bomb` × 3

  - `too-many-digits` × 3

  - `solc-version` × 2

  - `reentrancy-benign` × 2

  - `divide-before-multiply` × 1

  - `locked-ether` × 1

  - `reentrancy-no-eth` × 1


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 9/30: `474649f3c6177a74a0d36058`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.0`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/474649f3c6177a74a0d36058ffce8c3415f1596ddff6790e0ac77b98727adfc8.sol`


### Source code:
```solidity
pragma solidity ^0.4.0;

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

pragma solidity ^0.4.0;//please import oraclizeAPI_pre0.4.sol when solidity < 0.4.0

contract OraclizeI {
    address public cbAddress;
    function query(uint _timestamp, string _datasource, string _arg) payable returns (bytes32 _id);
    function query_withGasLimit(uint _timestamp, string _datasource, string _arg, uint _gaslimit) payable returns (bytes32 _id);
    function query2(uint _timestamp, string _datasource, string _arg1, string _arg2) payable returns (bytes32 _id);
    function query2_withGasLimit(uint _timestamp, string _datasource, string _arg1, string _arg2, uint _gaslimit) payable returns (bytes32 _id);
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
        if(address(OAR)==0) oraclize_setNetwork(networkID_auto);
        oraclize = OraclizeI(OAR.getAddress());
        _;
    }
    modifier coupon(string code){
        oraclize = OraclizeI(OAR.getAddress());
        oraclize.useCoupon(code);
        _;
    }

    function oraclize_setNetwork(uint8 networkID) internal returns(bool){
        if (getCodeSize(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed)>0){ //mainnet
            OAR = OraclizeAddrResolverI(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed);
            return true;
        }
        if (getCodeSize(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1)>0){ //ropsten testnet
            OAR = OraclizeAddrResolverI(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1);
            return true;
        }
        if (getCodeSize(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa)>0){ //browser-solidity
            OAR = OraclizeAddrResolverI(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa);
            return true;
        }
        return false;
    }
    
    function oraclize_query(string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(0, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(timestamp, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(timestamp, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(0, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(0, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(timestamp, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(timestamp, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(0, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_cbAddress() oraclizeAPI internal returns (address){
        return oraclize.cbAddress();
    }
    function oraclize_setProof(byte proofP) oraclizeAPI internal {
        return oraclize.setProofType(proofP);
    }
    function oraclize_setCustomGasPrice(uint gasPrice) oraclizeAPI internal {
        return oraclize.setCustomGasPrice(gasPrice);
    }    
    function oraclize_setConfig(bytes32 config) oraclizeAPI internal {
        return oraclize.setConfig(config);
    }

    function getCodeSize(address _addr) constant internal returns(uint _size) {
        assembly {
            _size := extcodesize(_addr)
        }
    }


    function parseAddr(string _a) internal returns (address){
        bytes memory tmp = bytes(_a);
        uint160 iaddr = 0;
        uint160 b1;
        uint160 b2;
        for (uint i=2; i<2+2*20; i+=2){
            iaddr *= 256;
            b1 = uint160(tmp[i]);
            b2 = uint160(tmp[i+1]);
            if ((b1 >= 97)&&(b1 <= 102)) b1 -= 87;
            else if ((b1 >= 48)&&(b1 <= 57)) b1 -= 48;
            if ((b2 >= 97)&&(b2 <= 102)) b2 -= 87;
            else if ((b2 >= 48)&&(b2 <= 57)) b2 -= 48;
            iaddr += (b1*16+b2);
        }
        return address(iaddr);
    }


    function strCompare(string _a, string _b) internal returns (int) {
        bytes memory a = bytes(_a);
        bytes memory b = bytes(_b);
        uint minLength = a.length;
        if (b.length < minLength) minLength = b.length;
        for (uint i = 0; i < minLength; i ++)
            if (a[i] < b[i])
                return -1;
            else if (a[i] > b[i])
                return 1;
        if (a.length < b.length)
            return -1;
        else if (a.length > b.length)
            return 1;
        else
            return 0;
   } 

    function indexOf(string _haystack, string _needle) internal returns (int)
    {
        bytes memory h = bytes(_haystack);
        bytes memory n = bytes(_needle);
        if(h.length < 1 || n.length < 1 || (n.length > h.length)) 
            return -1;
        else if(h.length > (2**128 -1))
            return -1;                                  
        else
        {

... [truncated, 763 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 200  |  **Unique detectors:** 23

- **Top detectors (by frequency):**

  - `naming-convention` × 81

  - `deprecated-standards` × 23

  - `dead-code` × 16

  - `unindexed-event-address` × 13

  - `costly-loop` × 11

  - `reentrancy-events` × 9

  - `external-function` × 8

  - `unused-state` × 8

  - `reentrancy-benign` × 5

  - `incorrect-modifier` × 4

  - `reentrancy-eth` × 4

  - `too-many-digits` × 4

  - `solc-version` × 2

  - `calls-loop` × 2

  - `uninitialized-local` × 2

  - ... and 8 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 10/30: `4ad02f839bfade8819f3ab38`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.11`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/4ad02f839bfade8819f3ab38fec581c708aa01428e3966f7867a90180027eb0b.sol`


### Source code:
```solidity
pragma solidity ^0.4.11;



contract PullPayInterface {
  function asyncSend(address _dest) public payable;
}

contract Governable {

  // list of admins, council at first spot
  address[] public admins;

  function Governable() {
    admins.length = 1;
    admins[0] = msg.sender;
  }

  modifier onlyAdmins() {
    bool isAdmin = false;
    for (uint256 i = 0; i < admins.length; i++) {
      if (msg.sender == admins[i]) {
        isAdmin = true;
      }
    }
    require(isAdmin == true);
    _;
  }

  function addAdmin(address _admin) public onlyAdmins {
    for (uint256 i = 0; i < admins.length; i++) {
      require(_admin != admins[i]);
    }
    require(admins.length < 10);
    admins[admins.length++] = _admin;
  }

  function removeAdmin(address _admin) public onlyAdmins {
    uint256 pos = admins.length;
    for (uint256 i = 0; i < admins.length; i++) {
      if (_admin == admins[i]) {
        pos = i;
      }
    }
    require(pos < admins.length);
    // if not last element, switch with last
    if (pos < admins.length - 1) {
      admins[pos] = admins[admins.length - 1];
    }
    // then cut off the tail
    admins.length--;
  }

}




contract StorageEnabled {

  // satelite contract addresses
  address public storageAddr;

  function StorageEnabled(address _storageAddr) {
    storageAddr = _storageAddr;
  }


  // ############################################
  // ########### NUTZ FUNCTIONS  ################
  // ############################################


  // all Nutz balances
  function babzBalanceOf(address _owner) constant returns (uint256) {
    return Storage(storageAddr).getBal('Nutz', _owner);
  }
  function _setBabzBalanceOf(address _owner, uint256 _newValue) internal {
    Storage(storageAddr).setBal('Nutz', _owner, _newValue);
  }
  // active supply - sum of balances above
  function activeSupply() constant returns (uint256) {
    return Storage(storageAddr).getUInt('Nutz', 'activeSupply');
  }
  function _setActiveSupply(uint256 _newActiveSupply) internal {
    Storage(storageAddr).setUInt('Nutz', 'activeSupply', _newActiveSupply);
  }
  // burn pool - inactive supply
  function burnPool() constant returns (uint256) {
    return Storage(storageAddr).getUInt('Nutz', 'burnPool');
  }
  function _setBurnPool(uint256 _newBurnPool) internal {
    Storage(storageAddr).setUInt('Nutz', 'burnPool', _newBurnPool);
  }
  // power pool - inactive supply
  function powerPool() constant returns (uint256) {
    return Storage(storageAddr).getUInt('Nutz', 'powerPool');
  }
  function _setPowerPool(uint256 _newPowerPool) internal {
    Storage(storageAddr).setUInt('Nutz', 'powerPool', _newPowerPool);
  }





  // ############################################
  // ########### POWER   FUNCTIONS  #############
  // ############################################

  // all power balances
  function powerBalanceOf(address _owner) constant returns (uint256) {
    return Storage(storageAddr).getBal('Power', _owner);
  }

  function _setPowerBalanceOf(address _owner, uint256 _newValue) internal {
    Storage(storageAddr).setBal('Power', _owner, _newValue);
  }

  function outstandingPower() constant returns (uint256) {
    return Storage(storageAddr).getUInt('Power', 'outstandingPower');
  }

  function _setOutstandingPower(uint256 _newOutstandingPower) internal {
    Storage(storageAddr).setUInt('Power', 'outstandingPower', _newOutstandingPower);
  }

  function authorizedPower() constant returns (uint256) {
    return Storage(storageAddr).getUInt('Power', 'authorizedPower');
  }

  function _setAuthorizedPower(uint256 _newAuthorizedPower) internal {
    Storage(storageAddr).setUInt('Power', 'authorizedPower', _newAuthorizedPower);
  }


  function downs(address _user) constant public returns (uint256 total, uint256 left, uint256 start) {
    uint256 rawBytes = Storage(storageAddr).getBal('PowerDown', _user);
    start = uint64(rawBytes);
    left = uint96(rawBytes >> (64));
    total = uint96(rawBytes >> (96 + 64));
    return;
  }

  function _setDownRequest(address _holder, uint256 total, uint256 left, uint256 start) internal {
    uint256 result = uint64(start) + (left << 64) + (total << (96 + 64));
    Storage(storageAddr).setBal('PowerDown', _holder, result);
  }

}


/**
 * @title Pausable
 * @dev Base contract which allows children to implement an emergency stop mechanism.
 */
contract Pausable is Governable {

  bool public paused = true;

  /**
   * @dev modifier to allow actions only when the contract IS paused
   */
  modifier whenNotPaused() {
    require(!paused);
    _;
  }

  /**
   * @dev modifier to allow actions only when the contract IS NOT paused
   */
  modifier whenPaused() {
    require(paused);
    _;
  }

  /**
   * @dev called by the owner to pause, triggers stopped state
   */
  function pause() onlyAdmins whenNotPaused {
    paused = true;
  }

  /**
   * @dev called by the owner to unpause, returns to normal state
   */
  function unpause() onlyAdmins whenPaused {
    //TODO: do some checks
    paused = false;
  }

}


/*
 * ERC20Basic
 * Simpler version of ERC20 interface
 * see https://github.com/ethereum/EIPs/issues/20
 */
contract ERC20Basic {

... [truncated, 869 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 184  |  **Unique detectors:** 22

- **Top detectors (by frequency):**

  - `naming-convention` × 117

  - `missing-zero-check` × 8

  - `reentrancy-events` × 8

  - `too-many-digits` × 7

  - `constable-states` × 6

  - `external-function` × 6

  - `reentrancy-benign` × 5

  - `uninitialized-local` × 4

  - `events-access` × 3

  - `timestamp` × 3

  - `cache-array-length` × 2

  - `erc20-interface` × 2

  - `solc-version` × 2

  - `incorrect-equality` × 2

  - `events-maths` × 2

  - ... and 7 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 11/30: `234a97c1df7afc825bdabb70`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.11`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/234a97c1df7afc825bdabb7098cb1dc3d36f0b9a3ea125d6711a922a4caaf7d7.sol`


### Source code:
```solidity
pragma solidity ^0.4.11;



contract PullPayInterface {
  function asyncSend(address _dest) public payable;
}

contract Governable {

  // list of admins, council at first spot
  address[] public admins;

  function Governable() {
    admins.length = 1;
    admins[0] = msg.sender;
  }

  modifier onlyAdmins() {
    bool isAdmin = false;
    for (uint256 i = 0; i < admins.length; i++) {
      if (msg.sender == admins[i]) {
        isAdmin = true;
      }
    }
    require(isAdmin == true);
    _;
  }

  function addAdmin(address _admin) public onlyAdmins {
    for (uint256 i = 0; i < admins.length; i++) {
      require(_admin != admins[i]);
    }
    require(admins.length < 10);
    admins[admins.length++] = _admin;
  }

  function removeAdmin(address _admin) public onlyAdmins {
    uint256 pos = admins.length;
    for (uint256 i = 0; i < admins.length; i++) {
      if (_admin == admins[i]) {
        pos = i;
      }
    }
    require(pos < admins.length);
    // if not last element, switch with last
    if (pos < admins.length - 1) {
      admins[pos] = admins[admins.length - 1];
    }
    // then cut off the tail
    admins.length--;
  }

}




contract StorageEnabled {

  // satelite contract addresses
  address public storageAddr;

  function StorageEnabled(address _storageAddr) {
    storageAddr = _storageAddr;
  }


  // ############################################
  // ########### NUTZ FUNCTIONS  ################
  // ############################################


  // all Nutz balances
  function babzBalanceOf(address _owner) constant returns (uint256) {
    return Storage(storageAddr).getBal('Nutz', _owner);
  }
  function _setBabzBalanceOf(address _owner, uint256 _newValue) internal {
    Storage(storageAddr).setBal('Nutz', _owner, _newValue);
  }
  // active supply - sum of balances above
  function activeSupply() constant returns (uint256) {
    return Storage(storageAddr).getUInt('Nutz', 'activeSupply');
  }
  function _setActiveSupply(uint256 _newActiveSupply) internal {
    Storage(storageAddr).setUInt('Nutz', 'activeSupply', _newActiveSupply);
  }
  // burn pool - inactive supply
  function burnPool() constant returns (uint256) {
    return Storage(storageAddr).getUInt('Nutz', 'burnPool');
  }
  function _setBurnPool(uint256 _newBurnPool) internal {
    Storage(storageAddr).setUInt('Nutz', 'burnPool', _newBurnPool);
  }
  // power pool - inactive supply
  function powerPool() constant returns (uint256) {
    return Storage(storageAddr).getUInt('Nutz', 'powerPool');
  }
  function _setPowerPool(uint256 _newPowerPool) internal {
    Storage(storageAddr).setUInt('Nutz', 'powerPool', _newPowerPool);
  }





  // ############################################
  // ########### POWER   FUNCTIONS  #############
  // ############################################

  // all power balances
  function powerBalanceOf(address _owner) constant returns (uint256) {
    return Storage(storageAddr).getBal('Power', _owner);
  }

  function _setPowerBalanceOf(address _owner, uint256 _newValue) internal {
    Storage(storageAddr).setBal('Power', _owner, _newValue);
  }

  function outstandingPower() constant returns (uint256) {
    return Storage(storageAddr).getUInt('Power', 'outstandingPower');
  }

  function _setOutstandingPower(uint256 _newOutstandingPower) internal {
    Storage(storageAddr).setUInt('Power', 'outstandingPower', _newOutstandingPower);
  }

  function authorizedPower() constant returns (uint256) {
    return Storage(storageAddr).getUInt('Power', 'authorizedPower');
  }

  function _setAuthorizedPower(uint256 _newAuthorizedPower) internal {
    Storage(storageAddr).setUInt('Power', 'authorizedPower', _newAuthorizedPower);
  }


  function downs(address _user) constant public returns (uint256 total, uint256 left, uint256 start) {
    uint256 rawBytes = Storage(storageAddr).getBal('PowerDown', _user);
    start = uint64(rawBytes);
    left = uint96(rawBytes >> (64));
    total = uint96(rawBytes >> (96 + 64));
    return;
  }

  function _setDownRequest(address _holder, uint256 total, uint256 left, uint256 start) internal {
    uint256 result = uint64(start) + (left << 64) + (total << (96 + 64));
    Storage(storageAddr).setBal('PowerDown', _holder, result);
  }

}


/**
 * @title Pausable
 * @dev Base contract which allows children to implement an emergency stop mechanism.
 */
contract Pausable is Governable {

  bool public paused = true;

  /**
   * @dev modifier to allow actions only when the contract IS paused
   */
  modifier whenNotPaused() {
    require(!paused);
    _;
  }

  /**
   * @dev modifier to allow actions only when the contract IS NOT paused
   */
  modifier whenPaused() {
    require(paused);
    _;
  }

  /**
   * @dev called by the owner to pause, triggers stopped state
   */
  function pause() onlyAdmins whenNotPaused {
    paused = true;
  }

  /**
   * @dev called by the owner to unpause, returns to normal state
   */
  function unpause() onlyAdmins whenPaused {
    //TODO: do some checks
    paused = false;
  }

}


/*
 * ERC20Basic
 * Simpler version of ERC20 interface
 * see https://github.com/ethereum/EIPs/issues/20
 */
contract ERC20Basic {

... [truncated, 955 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 197  |  **Unique detectors:** 23

- **Top detectors (by frequency):**

  - `naming-convention` × 117

  - `reentrancy-benign` × 12

  - `missing-zero-check` × 11

  - `reentrancy-events` × 8

  - `too-many-digits` × 7

  - `constable-states` × 6

  - `external-function` × 6

  - `uninitialized-local` × 4

  - `boolean-equal` × 3

  - `events-access` × 3

  - `timestamp` × 3

  - `cache-array-length` × 2

  - `erc20-interface` × 2

  - `solc-version` × 2

  - `incorrect-equality` × 2

  - ... and 8 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 12/30: `2780ed3064fdb7d3ffc3a9de`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.0`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/2780ed3064fdb7d3ffc3a9dedfdd7ec8882456599b979f46acbcbf8f9b2dc5ac.sol`


### Source code:
```solidity
pragma solidity ^0.4.0;

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

pragma solidity ^0.4.0;//please import oraclizeAPI_pre0.4.sol when solidity < 0.4.0

contract OraclizeI {
    address public cbAddress;
    function query(uint _timestamp, string _datasource, string _arg) payable returns (bytes32 _id);
    function query_withGasLimit(uint _timestamp, string _datasource, string _arg, uint _gaslimit) payable returns (bytes32 _id);
    function query2(uint _timestamp, string _datasource, string _arg1, string _arg2) payable returns (bytes32 _id);
    function query2_withGasLimit(uint _timestamp, string _datasource, string _arg1, string _arg2, uint _gaslimit) payable returns (bytes32 _id);
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
        if(address(OAR)==0) oraclize_setNetwork(networkID_auto);
        oraclize = OraclizeI(OAR.getAddress());
        _;
    }
    modifier coupon(string code){
        oraclize = OraclizeI(OAR.getAddress());
        oraclize.useCoupon(code);
        _;
    }

    function oraclize_setNetwork(uint8 networkID) internal returns(bool){
        if (getCodeSize(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed)>0){ //mainnet
            OAR = OraclizeAddrResolverI(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed);
            return true;
        }
        if (getCodeSize(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1)>0){ //ropsten testnet
            OAR = OraclizeAddrResolverI(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1);
            return true;
        }
        if (getCodeSize(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa)>0){ //browser-solidity
            OAR = OraclizeAddrResolverI(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa);
            return true;
        }
        return false;
    }
    
    function oraclize_query(string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(0, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(timestamp, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(timestamp, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(0, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(0, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(timestamp, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(timestamp, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(0, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_cbAddress() oraclizeAPI internal returns (address){
        return oraclize.cbAddress();
    }
    function oraclize_setProof(byte proofP) oraclizeAPI internal {
        return oraclize.setProofType(proofP);
    }
    function oraclize_setCustomGasPrice(uint gasPrice) oraclizeAPI internal {
        return oraclize.setCustomGasPrice(gasPrice);
    }    
    function oraclize_setConfig(bytes32 config) oraclizeAPI internal {
        return oraclize.setConfig(config);
    }

    function getCodeSize(address _addr) constant internal returns(uint _size) {
        assembly {
            _size := extcodesize(_addr)
        }
    }


    function parseAddr(string _a) internal returns (address){
        bytes memory tmp = bytes(_a);
        uint160 iaddr = 0;
        uint160 b1;
        uint160 b2;
        for (uint i=2; i<2+2*20; i+=2){
            iaddr *= 256;
            b1 = uint160(tmp[i]);
            b2 = uint160(tmp[i+1]);
            if ((b1 >= 97)&&(b1 <= 102)) b1 -= 87;
            else if ((b1 >= 48)&&(b1 <= 57)) b1 -= 48;
            if ((b2 >= 97)&&(b2 <= 102)) b2 -= 87;
            else if ((b2 >= 48)&&(b2 <= 57)) b2 -= 48;
            iaddr += (b1*16+b2);
        }
        return address(iaddr);
    }


    function strCompare(string _a, string _b) internal returns (int) {
        bytes memory a = bytes(_a);
        bytes memory b = bytes(_b);
        uint minLength = a.length;
        if (b.length < minLength) minLength = b.length;
        for (uint i = 0; i < minLength; i ++)
            if (a[i] < b[i])
                return -1;
            else if (a[i] > b[i])
                return 1;
        if (a.length < b.length)
            return -1;
        else if (a.length > b.length)
            return 1;
        else
            return 0;
   } 

    function indexOf(string _haystack, string _needle) internal returns (int)
    {
        bytes memory h = bytes(_haystack);
        bytes memory n = bytes(_needle);
        if(h.length < 1 || n.length < 1 || (n.length > h.length)) 
            return -1;
        else if(h.length > (2**128 -1))
            return -1;                                  
        else
        {

... [truncated, 763 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 200  |  **Unique detectors:** 23

- **Top detectors (by frequency):**

  - `naming-convention` × 81

  - `deprecated-standards` × 23

  - `dead-code` × 16

  - `unindexed-event-address` × 13

  - `costly-loop` × 11

  - `reentrancy-events` × 9

  - `external-function` × 8

  - `unused-state` × 8

  - `reentrancy-benign` × 5

  - `incorrect-modifier` × 4

  - `reentrancy-eth` × 4

  - `too-many-digits` × 4

  - `solc-version` × 2

  - `calls-loop` × 2

  - `uninitialized-local` × 2

  - ... and 8 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 13/30: `2f34c126f0624732031cfac3`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.0`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/2f34c126f0624732031cfac337430dea84e1e3d80566b115ee572438f1d6988d.sol`


### Source code:
```solidity
pragma solidity ^0.4.0;

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

pragma solidity ^0.4.0;//please import oraclizeAPI_pre0.4.sol when solidity < 0.4.0

contract OraclizeI {
    address public cbAddress;
    function query(uint _timestamp, string _datasource, string _arg) payable returns (bytes32 _id);
    function query_withGasLimit(uint _timestamp, string _datasource, string _arg, uint _gaslimit) payable returns (bytes32 _id);
    function query2(uint _timestamp, string _datasource, string _arg1, string _arg2) payable returns (bytes32 _id);
    function query2_withGasLimit(uint _timestamp, string _datasource, string _arg1, string _arg2, uint _gaslimit) payable returns (bytes32 _id);
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
        if(address(OAR)==0) oraclize_setNetwork(networkID_auto);
        oraclize = OraclizeI(OAR.getAddress());
        _;
    }
    modifier coupon(string code){
        oraclize = OraclizeI(OAR.getAddress());
        oraclize.useCoupon(code);
        _;
    }

    function oraclize_setNetwork(uint8 networkID) internal returns(bool){
        if (getCodeSize(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed)>0){ //mainnet
            OAR = OraclizeAddrResolverI(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed);
            return true;
        }
        if (getCodeSize(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1)>0){ //ropsten testnet
            OAR = OraclizeAddrResolverI(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1);
            return true;
        }
        if (getCodeSize(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa)>0){ //browser-solidity
            OAR = OraclizeAddrResolverI(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa);
            return true;
        }
        return false;
    }
    
    function oraclize_query(string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(0, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(timestamp, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(timestamp, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(0, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(0, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(timestamp, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(timestamp, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(0, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_cbAddress() oraclizeAPI internal returns (address){
        return oraclize.cbAddress();
    }
    function oraclize_setProof(byte proofP) oraclizeAPI internal {
        return oraclize.setProofType(proofP);
    }
    function oraclize_setCustomGasPrice(uint gasPrice) oraclizeAPI internal {
        return oraclize.setCustomGasPrice(gasPrice);
    }    
    function oraclize_setConfig(bytes32 config) oraclizeAPI internal {
        return oraclize.setConfig(config);
    }

    function getCodeSize(address _addr) constant internal returns(uint _size) {
        assembly {
            _size := extcodesize(_addr)
        }
    }


    function parseAddr(string _a) internal returns (address){
        bytes memory tmp = bytes(_a);
        uint160 iaddr = 0;
        uint160 b1;
        uint160 b2;
        for (uint i=2; i<2+2*20; i+=2){
            iaddr *= 256;
            b1 = uint160(tmp[i]);
            b2 = uint160(tmp[i+1]);
            if ((b1 >= 97)&&(b1 <= 102)) b1 -= 87;
            else if ((b1 >= 48)&&(b1 <= 57)) b1 -= 48;
            if ((b2 >= 97)&&(b2 <= 102)) b2 -= 87;
            else if ((b2 >= 48)&&(b2 <= 57)) b2 -= 48;
            iaddr += (b1*16+b2);
        }
        return address(iaddr);
    }


    function strCompare(string _a, string _b) internal returns (int) {
        bytes memory a = bytes(_a);
        bytes memory b = bytes(_b);
        uint minLength = a.length;
        if (b.length < minLength) minLength = b.length;
        for (uint i = 0; i < minLength; i ++)
            if (a[i] < b[i])
                return -1;
            else if (a[i] > b[i])
                return 1;
        if (a.length < b.length)
            return -1;
        else if (a.length > b.length)
            return 1;
        else
            return 0;
   } 

    function indexOf(string _haystack, string _needle) internal returns (int)
    {
        bytes memory h = bytes(_haystack);
        bytes memory n = bytes(_needle);
        if(h.length < 1 || n.length < 1 || (n.length > h.length)) 
            return -1;
        else if(h.length > (2**128 -1))
            return -1;                                  
        else
        {

... [truncated, 763 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 200  |  **Unique detectors:** 23

- **Top detectors (by frequency):**

  - `naming-convention` × 81

  - `deprecated-standards` × 23

  - `dead-code` × 16

  - `unindexed-event-address` × 13

  - `costly-loop` × 11

  - `reentrancy-events` × 9

  - `external-function` × 8

  - `unused-state` × 8

  - `reentrancy-benign` × 5

  - `incorrect-modifier` × 4

  - `reentrancy-eth` × 4

  - `too-many-digits` × 4

  - `solc-version` × 2

  - `calls-loop` × 2

  - `uninitialized-local` × 2

  - ... and 8 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 14/30: `3212e131e050d5bf30062c8b`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.0`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/3212e131e050d5bf30062c8bc52ed12afbc9536b76148eb8684f0fbe39bfdaef.sol`


### Source code:
```solidity
pragma solidity ^0.4.0;

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

pragma solidity ^0.4.0;//please import oraclizeAPI_pre0.4.sol when solidity < 0.4.0

contract OraclizeI {
    address public cbAddress;
    function query(uint _timestamp, string _datasource, string _arg) payable returns (bytes32 _id);
    function query_withGasLimit(uint _timestamp, string _datasource, string _arg, uint _gaslimit) payable returns (bytes32 _id);
    function query2(uint _timestamp, string _datasource, string _arg1, string _arg2) payable returns (bytes32 _id);
    function query2_withGasLimit(uint _timestamp, string _datasource, string _arg1, string _arg2, uint _gaslimit) payable returns (bytes32 _id);
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
        if(address(OAR)==0) oraclize_setNetwork(networkID_auto);
        oraclize = OraclizeI(OAR.getAddress());
        _;
    }
    modifier coupon(string code){
        oraclize = OraclizeI(OAR.getAddress());
        oraclize.useCoupon(code);
        _;
    }

    function oraclize_setNetwork(uint8 networkID) internal returns(bool){
        if (getCodeSize(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed)>0){ //mainnet
            OAR = OraclizeAddrResolverI(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed);
            return true;
        }
        if (getCodeSize(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1)>0){ //ropsten testnet
            OAR = OraclizeAddrResolverI(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1);
            return true;
        }
        if (getCodeSize(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa)>0){ //browser-solidity
            OAR = OraclizeAddrResolverI(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa);
            return true;
        }
        return false;
    }
    
    function oraclize_query(string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(0, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(timestamp, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(timestamp, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(0, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(0, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(timestamp, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(timestamp, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(0, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_cbAddress() oraclizeAPI internal returns (address){
        return oraclize.cbAddress();
    }
    function oraclize_setProof(byte proofP) oraclizeAPI internal {
        return oraclize.setProofType(proofP);
    }
    function oraclize_setCustomGasPrice(uint gasPrice) oraclizeAPI internal {
        return oraclize.setCustomGasPrice(gasPrice);
    }    
    function oraclize_setConfig(bytes32 config) oraclizeAPI internal {
        return oraclize.setConfig(config);
    }

    function getCodeSize(address _addr) constant internal returns(uint _size) {
        assembly {
            _size := extcodesize(_addr)
        }
    }


    function parseAddr(string _a) internal returns (address){
        bytes memory tmp = bytes(_a);
        uint160 iaddr = 0;
        uint160 b1;
        uint160 b2;
        for (uint i=2; i<2+2*20; i+=2){
            iaddr *= 256;
            b1 = uint160(tmp[i]);
            b2 = uint160(tmp[i+1]);
            if ((b1 >= 97)&&(b1 <= 102)) b1 -= 87;
            else if ((b1 >= 48)&&(b1 <= 57)) b1 -= 48;
            if ((b2 >= 97)&&(b2 <= 102)) b2 -= 87;
            else if ((b2 >= 48)&&(b2 <= 57)) b2 -= 48;
            iaddr += (b1*16+b2);
        }
        return address(iaddr);
    }


    function strCompare(string _a, string _b) internal returns (int) {
        bytes memory a = bytes(_a);
        bytes memory b = bytes(_b);
        uint minLength = a.length;
        if (b.length < minLength) minLength = b.length;
        for (uint i = 0; i < minLength; i ++)
            if (a[i] < b[i])
                return -1;
            else if (a[i] > b[i])
                return 1;
        if (a.length < b.length)
            return -1;
        else if (a.length > b.length)
            return 1;
        else
            return 0;
   } 

    function indexOf(string _haystack, string _needle) internal returns (int)
    {
        bytes memory h = bytes(_haystack);
        bytes memory n = bytes(_needle);
        if(h.length < 1 || n.length < 1 || (n.length > h.length)) 
            return -1;
        else if(h.length > (2**128 -1))
            return -1;                                  
        else
        {

... [truncated, 763 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 200  |  **Unique detectors:** 23

- **Top detectors (by frequency):**

  - `naming-convention` × 81

  - `deprecated-standards` × 23

  - `dead-code` × 16

  - `unindexed-event-address` × 13

  - `costly-loop` × 11

  - `reentrancy-events` × 9

  - `external-function` × 8

  - `unused-state` × 8

  - `reentrancy-benign` × 5

  - `incorrect-modifier` × 4

  - `reentrancy-eth` × 4

  - `too-many-digits` × 4

  - `solc-version` × 2

  - `calls-loop` × 2

  - `uninitialized-local` × 2

  - ... and 8 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 15/30: `7a82041ff4e6d3105497923d`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.16`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/7a82041ff4e6d3105497923d2b767da522e7488e3511d2ff3ee39b1db81a3497.sol`


### Source code:
```solidity
pragma solidity ^0.4.16;        
   
  contract CentraSale { 

    using SafeMath for uint; 

    address public contract_address = 0x96a65609a7b84e8842732deb08f56c3e21ac6f8a; 

    address public owner;
    uint public cap;
    uint public constant cap_max = 170000*10**18;
    uint public constant min_value = 10**18*1/10; 
    uint public operation;
    mapping(uint => address) public operation_address;
    mapping(uint => uint) public operation_amount;

    uint256 public constant token_price = 1481481481481481;  
    uint256 public tokens_total;  

    uint public constant contract_start = 1505844000;
    uint public constant contract_finish = 1507269600;

    uint public constant card_titanium_minamount = 500*10**18;
    uint public constant card_titanium_first = 200000;
    mapping(address => uint) cards_titanium_check; 
    address[] public cards_titanium;

    uint public constant card_black_minamount = 100*10**18;
    uint public constant card_black_first = 500000;
    mapping(address => uint) public cards_black_check; 
    address[] public cards_black;

    uint public constant card_metal_minamount = 40*10**18;
    uint public constant card_metal_first = 750000;
    mapping(address => uint) cards_metal_check; 
    address[] public cards_metal;      

    uint public constant card_gold_minamount = 30*10**18;
    uint public constant card_gold_first = 1000000;
    mapping(address => uint) cards_gold_check; 
    address[] public cards_gold;      

    uint public constant card_blue_minamount = 5/10*10**18;
    uint public constant card_blue_first = 100000000;
    mapping(address => uint) cards_blue_check; 
    address[] public cards_blue;

    uint public constant card_start_minamount = 1/10*10**18;
    uint public constant card_start_first = 100000000;
    mapping(address => uint) cards_start_check; 
    address[] public cards_start;
      
   
    // Functions with this modifier can only be executed by the owner
    modifier onlyOwner() {
        if (msg.sender != owner) {
            throw;
        }
        _;
    }      
 
    // Constructor
    function CentraSale() {
        owner = msg.sender; 
        operation = 0; 
        cap = 0;        
    }
      
    //default function for crowdfunding
    function() payable {    

      if(!(msg.value >= min_value)) throw;
      if(now < contract_start) throw;
      if(now > contract_finish) throw;                     

      //if(cap + msg.value > cap_max) throw;         

      tokens_total = msg.value*10**18/token_price;
      if(!(tokens_total > 0)) throw;           

      if(!contract_transfer(tokens_total)) throw;

      cap = cap.add(msg.value); 
      operations();
      get_card();
      owner.send(this.balance);
    }

    //Contract execute
    function contract_transfer(uint _amount) private returns (bool) {      

      if(!contract_address.call(bytes4(sha3("transfer(address,uint256)")),msg.sender,_amount)) {    
        return false;
      }
      return true;
    } 

    //Update operations
    function operations() private returns (bool) {
        operation_address[operation] = msg.sender;
        operation_amount[operation] = msg.value;        
        operation = operation.add(1);        
        return true;
    }    

    //Withdraw money from contract balance to owner
    function withdraw() onlyOwner returns (bool result) {
        owner.send(this.balance);
        return true;
    }

    //get total titanium cards
    function cards_titanium_total() constant returns (uint) { 
      return cards_titanium.length;
    }  
    //get total black cards
    function cards_black_total() constant returns (uint) { 
      return cards_black.length;
    }
    //get total metal cards
    function cards_metal_total() constant returns (uint) { 
      return cards_metal.length;
    }        
    //get total gold cards
    function cards_gold_total() constant returns (uint) { 
      return cards_gold.length;
    }        
    //get total blue cards
    function cards_blue_total() constant returns (uint) { 
      return cards_blue.length;
    }

    //get total start cards
    function cards_start_total() constant returns (uint) { 
      return cards_start.length;
    }

    /*
    * User get card(titanium, black, gold metal, gold and other), if amount eth sufficient for this.
    */
    function get_card() private returns (bool) {

      if((msg.value >= card_titanium_minamount)
        &&(cards_titanium.length < card_titanium_first)
        &&(cards_titanium_check[msg.sender] != 1)
        ) {
        cards_titanium.push(msg.sender);
        cards_titanium_check[msg.sender] = 1;
      }

      if((msg.value >= card_black_minamount)
        &&(msg.value < card_titanium_minamount)
        &&(cards_black.length < card_black_first)
        &&(cards_black_check[msg.sender] != 1)
        ) {
        cards_black.push(msg.sender);
        cards_black_check[msg.sender] = 1;
      }                

      if((msg.value >= card_metal_minamount)
        &&(msg.value < card_black_minamount)
        &&(cards_metal.length < card_metal_first)
        &&(cards_metal_check[msg.sender] != 1)
        ) {
        cards_metal.push(msg.sender);
        cards_metal_check[msg.sender] = 1;
      }               

      if((msg.value >= card_gold_minamount)
        &&(msg.value < card_metal_minamount)
        &&(cards_gold.length < card_gold_first)
        &&(cards_gold_check[msg.sender] != 1)
        ) {
        cards_gold.push(msg.sender);
        cards_gold_check[msg.sender] = 1;
      }               

      if((msg.value >= card_blue_minamount)
        &&(msg.value < card_gold_minamount)
        &&(cards_blue.length < card_blue_first)
        &&(cards_blue_check[msg.sender] != 1)
        ) {
        cards_blue.push(msg.sender);
        cards_blue_check[msg.sender] = 1;
      }

      if((msg.value >= card_start_minamount)
        &&(msg.value < card_blue_minamount)
        &&(cards_start.length < card_start_first)
        &&(cards_start_check[msg.sender] != 1)
        ) {
        cards_start.push(msg.sender);
        cards_start_check[msg.sender] = 1;
      }

      return true;
    }    
      
 }


... [truncated, 50 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 55  |  **Unique detectors:** 13

- **Top detectors (by frequency):**

  - `naming-convention` × 25

  - `deprecated-standards` × 8

  - `controlled-array-length` × 5

  - `too-many-digits` × 5

  - `divide-before-multiply` × 2

  - `solc-version` × 2

  - `unchecked-send` × 2

  - `arbitrary-send-eth` × 1

  - `shadowing-builtin` × 1

  - `constable-states` × 1

  - `low-level-calls` × 1

  - `reentrancy-benign` × 1

  - `timestamp` × 1


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 16/30: `756591a2872be3fb57386ae2`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.0`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/756591a2872be3fb57386ae232eac86b5b9839647deca8de75c6c3f564bc3c76.sol`


### Source code:
```solidity
pragma solidity ^0.4.0;

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

pragma solidity ^0.4.0;//please import oraclizeAPI_pre0.4.sol when solidity < 0.4.0

contract OraclizeI {
    address public cbAddress;
    function query(uint _timestamp, string _datasource, string _arg) payable returns (bytes32 _id);
    function query_withGasLimit(uint _timestamp, string _datasource, string _arg, uint _gaslimit) payable returns (bytes32 _id);
    function query2(uint _timestamp, string _datasource, string _arg1, string _arg2) payable returns (bytes32 _id);
    function query2_withGasLimit(uint _timestamp, string _datasource, string _arg1, string _arg2, uint _gaslimit) payable returns (bytes32 _id);
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
        if(address(OAR)==0) oraclize_setNetwork(networkID_auto);
        oraclize = OraclizeI(OAR.getAddress());
        _;
    }
    modifier coupon(string code){
        oraclize = OraclizeI(OAR.getAddress());
        oraclize.useCoupon(code);
        _;
    }

    function oraclize_setNetwork(uint8 networkID) internal returns(bool){
        if (getCodeSize(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed)>0){ //mainnet
            OAR = OraclizeAddrResolverI(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed);
            return true;
        }
        if (getCodeSize(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1)>0){ //ropsten testnet
            OAR = OraclizeAddrResolverI(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1);
            return true;
        }
        if (getCodeSize(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa)>0){ //browser-solidity
            OAR = OraclizeAddrResolverI(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa);
            return true;
        }
        return false;
    }
    
    function __callback(bytes32 myid, string result) {
        __callback(myid, result, new bytes(0));
    }
    function __callback(bytes32 myid, string result, bytes proof) {
    }
    
    function oraclize_getPrice(string datasource) oraclizeAPI internal returns (uint){
        return oraclize.getPrice(datasource);
    }
    function oraclize_getPrice(string datasource, uint gaslimit) oraclizeAPI internal returns (uint){
        return oraclize.getPrice(datasource, gaslimit);
    }
    
    function oraclize_query(string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(0, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(timestamp, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(timestamp, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(0, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(0, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(timestamp, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(timestamp, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(0, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_cbAddress() oraclizeAPI internal returns (address){
        return oraclize.cbAddress();
    }
    function oraclize_setProof(byte proofP) oraclizeAPI internal {
        return oraclize.setProofType(proofP);
    }
    function oraclize_setCustomGasPrice(uint gasPrice) oraclizeAPI internal {
        return oraclize.setCustomGasPrice(gasPrice);
    }    
    function oraclize_setConfig(bytes32 config) oraclizeAPI internal {
        return oraclize.setConfig(config);
    }

    function getCodeSize(address _addr) constant internal returns(uint _size) {
        assembly {
            _size := extcodesize(_addr)
        }
    }


    function parseAddr(string _a) internal returns (address){
        bytes memory tmp = bytes(_a);
        uint160 iaddr = 0;
        uint160 b1;
        uint160 b2;
        for (uint i=2; i<2+2*20; i+=2){
            iaddr *= 256;
            b1 = uint160(tmp[i]);
            b2 = uint160(tmp[i+1]);
            if ((b1 >= 97)&&(b1 <= 102)) b1 -= 87;
            else if ((b1 >= 48)&&(b1 <= 57)) b1 -= 48;
            if ((b2 >= 97)&&(b2 <= 102)) b2 -= 87;
            else if ((b2 >= 48)&&(b2 <= 57)) b2 -= 48;
            iaddr += (b1*16+b2);
        }
        return address(iaddr);
    }


    function strCompare(string _a, string _b) internal returns (int) {
        bytes memory a = bytes(_a);
        bytes memory b = bytes(_b);
        uint minLength = a.length;
        if (b.length < minLength) minLength = b.length;
        for (uint i = 0; i < minLength; i ++)
            if (a[i] < b[i])
                return -1;
            else if (a[i] > b[i])
                return 1;
        if (a.length < b.length)
            return -1;
        else if (a.length > b.length)
            return 1;
        else

... [truncated, 776 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 206  |  **Unique detectors:** 23

- **Top detectors (by frequency):**

  - `naming-convention` × 85

  - `deprecated-standards` × 23

  - `dead-code` × 18

  - `unindexed-event-address` × 13

  - `costly-loop` × 11

  - `reentrancy-events` × 9

  - `external-function` × 8

  - `unused-state` × 8

  - `reentrancy-benign` × 5

  - `incorrect-modifier` × 4

  - `reentrancy-eth` × 4

  - `too-many-digits` × 4

  - `solc-version` × 2

  - `calls-loop` × 2

  - `uninitialized-local` × 2

  - ... and 8 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 17/30: `6ac6892c594d681e2685de24`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.0`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/6ac6892c594d681e2685de24a859bca9c05dbde3b88f3a982de9dac50fc8f992.sol`


### Source code:
```solidity
pragma solidity ^0.4.0;

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

pragma solidity ^0.4.0;//please import oraclizeAPI_pre0.4.sol when solidity < 0.4.0

contract OraclizeI {
    address public cbAddress;
    function query(uint _timestamp, string _datasource, string _arg) payable returns (bytes32 _id);
    function query_withGasLimit(uint _timestamp, string _datasource, string _arg, uint _gaslimit) payable returns (bytes32 _id);
    function query2(uint _timestamp, string _datasource, string _arg1, string _arg2) payable returns (bytes32 _id);
    function query2_withGasLimit(uint _timestamp, string _datasource, string _arg1, string _arg2, uint _gaslimit) payable returns (bytes32 _id);
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
        if(address(OAR)==0) oraclize_setNetwork(networkID_auto);
        oraclize = OraclizeI(OAR.getAddress());
        _;
    }
    modifier coupon(string code){
        oraclize = OraclizeI(OAR.getAddress());
        oraclize.useCoupon(code);
        _;
    }

    function oraclize_setNetwork(uint8 networkID) internal returns(bool){
        if (getCodeSize(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed)>0){ //mainnet
            OAR = OraclizeAddrResolverI(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed);
            return true;
        }
        if (getCodeSize(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1)>0){ //ropsten testnet
            OAR = OraclizeAddrResolverI(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1);
            return true;
        }
        if (getCodeSize(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa)>0){ //browser-solidity
            OAR = OraclizeAddrResolverI(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa);
            return true;
        }
        return false;
    }
    
    function __callback(bytes32 myid, string result) {
        __callback(myid, result, new bytes(0));
    }
    function __callback(bytes32 myid, string result, bytes proof) {
    }
    
    function oraclize_getPrice(string datasource) oraclizeAPI internal returns (uint){
        return oraclize.getPrice(datasource);
    }
    function oraclize_getPrice(string datasource, uint gaslimit) oraclizeAPI internal returns (uint){
        return oraclize.getPrice(datasource, gaslimit);
    }
    
    function oraclize_query(string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(0, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(timestamp, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(timestamp, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(0, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(0, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(timestamp, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(timestamp, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(0, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_cbAddress() oraclizeAPI internal returns (address){
        return oraclize.cbAddress();
    }
    function oraclize_setProof(byte proofP) oraclizeAPI internal {
        return oraclize.setProofType(proofP);
    }
    function oraclize_setCustomGasPrice(uint gasPrice) oraclizeAPI internal {
        return oraclize.setCustomGasPrice(gasPrice);
    }    
    function oraclize_setConfig(bytes32 config) oraclizeAPI internal {
        return oraclize.setConfig(config);
    }

    function getCodeSize(address _addr) constant internal returns(uint _size) {
        assembly {
            _size := extcodesize(_addr)
        }
    }


    function parseAddr(string _a) internal returns (address){
        bytes memory tmp = bytes(_a);
        uint160 iaddr = 0;
        uint160 b1;
        uint160 b2;
        for (uint i=2; i<2+2*20; i+=2){
            iaddr *= 256;
            b1 = uint160(tmp[i]);
            b2 = uint160(tmp[i+1]);
            if ((b1 >= 97)&&(b1 <= 102)) b1 -= 87;
            else if ((b1 >= 48)&&(b1 <= 57)) b1 -= 48;
            if ((b2 >= 97)&&(b2 <= 102)) b2 -= 87;
            else if ((b2 >= 48)&&(b2 <= 57)) b2 -= 48;
            iaddr += (b1*16+b2);
        }
        return address(iaddr);
    }


    function strCompare(string _a, string _b) internal returns (int) {
        bytes memory a = bytes(_a);
        bytes memory b = bytes(_b);
        uint minLength = a.length;
        if (b.length < minLength) minLength = b.length;
        for (uint i = 0; i < minLength; i ++)
            if (a[i] < b[i])
                return -1;
            else if (a[i] > b[i])
                return 1;
        if (a.length < b.length)
            return -1;
        else if (a.length > b.length)
            return 1;
        else

... [truncated, 776 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 206  |  **Unique detectors:** 23

- **Top detectors (by frequency):**

  - `naming-convention` × 85

  - `deprecated-standards` × 23

  - `dead-code` × 18

  - `unindexed-event-address` × 13

  - `costly-loop` × 11

  - `reentrancy-events` × 9

  - `external-function` × 8

  - `unused-state` × 8

  - `reentrancy-benign` × 5

  - `incorrect-modifier` × 4

  - `reentrancy-eth` × 4

  - `too-many-digits` × 4

  - `solc-version` × 2

  - `calls-loop` × 2

  - `uninitialized-local` × 2

  - ... and 8 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 18/30: `64558fdbbeb12f6f65b8b776`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.18`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/64558fdbbeb12f6f65b8b776748ff8ba844d36891967dc42f64fd2b350c30ed9.sol`


### Source code:
```solidity
pragma solidity ^0.4.18;

// File: contracts/InkMediator.sol

interface InkMediator {
  function mediationExpiry() external returns (uint32);
  function requestMediator(uint256 _transactionId, uint256 _transactionAmount, address _transactionOwner) external returns (bool);
  function confirmTransactionFee(uint256 _transactionAmount) external returns (uint256);
  function confirmTransactionAfterExpiryFee(uint256 _transactionAmount) external returns (uint256);
  function confirmTransactionAfterDisputeFee(uint256 _transactionAmount) external returns (uint256);
  function confirmTransactionByMediatorFee(uint256 _transactionAmount) external returns (uint256);
  function refundTransactionFee(uint256 _transactionAmount) external returns (uint256);
  function refundTransactionAfterExpiryFee(uint256 _transactionAmount) external returns (uint256);
  function refundTransactionAfterDisputeFee(uint256 _transactionAmount) external returns (uint256);
  function refundTransactionByMediatorFee(uint256 _transactionAmount) external returns (uint256);
  function settleTransactionByMediatorFee(uint256 _buyerAmount, uint256 _sellerAmount) external returns (uint256, uint256);
}

// File: contracts/InkOwner.sol

interface InkOwner {
  function authorizeTransaction(uint256 _id, address _buyer) external returns (bool);
}

// File: contracts/InkProtocolInterface.sol

interface InkProtocolInterface {
  // Event emitted when a transaction is initiated.
  event TransactionInitiated(
    uint256 indexed id,
    address owner,
    address indexed buyer,
    address indexed seller,
    address policy,
    address mediator,
    uint256 amount,
    // A hash string representing the metadata for the transaction. This is
    // somewhat arbitrary for the transaction. Only the transaction owner
    // will really know the original contents of the metadata and may choose
    // to share it at their discretion.
    bytes32 metadata
  );

  // Event emitted when a transaction has been accepted by the seller.
  event TransactionAccepted(
    uint256 indexed id
  );

  // Event emitted when a transaction has been disputed by the buyer.
  event TransactionDisputed(
    uint256 indexed id
  );

  // Event emitted when a transaction is escalated to the mediator by the
  // seller.
  event TransactionEscalated(
    uint256 indexed id
  );

  // Event emitted when a transaction is revoked by the seller.
  event TransactionRevoked(
    uint256 indexed id
  );

  // Event emitted when a transaction is revoked by the seller.
  event TransactionRefundedByMediator(
    uint256 indexed id,
    uint256 mediatorFee
  );

  // Event emitted when a transaction is settled by the mediator.
  event TransactionSettledByMediator(
    uint256 indexed id,
    uint256 buyerAmount,
    uint256 sellerAmount,
    uint256 buyerMediatorFee,
    uint256 sellerMediatorFee
  );

  // Event emitted when a transaction is confirmed by the mediator.
  event TransactionConfirmedByMediator(
    uint256 indexed id,
    uint256 mediatorFee
  );

  // Event emitted when a transaction is confirmed by the buyer.
  event TransactionConfirmed(
    uint256 indexed id,
    uint256 mediatorFee
  );

  // Event emitted when a transaction is refunded by the seller.
  event TransactionRefunded(
    uint256 indexed id,
    uint256 mediatorFee
  );

  // Event emitted when a transaction is confirmed by the seller after the
  // transaction expiry.
  event TransactionConfirmedAfterExpiry(
    uint256 indexed id,
    uint256 mediatorFee
  );

  // Event emitted when a transaction is confirmed by the buyer after it was
  // disputed.
  event TransactionConfirmedAfterDispute(
    uint256 indexed id,
    uint256 mediatorFee
  );

  // Event emitted when a transaction is refunded by the seller after it was
  // disputed.
  event TransactionRefundedAfterDispute(
    uint256 indexed id,
    uint256 mediatorFee
  );

  // Event emitted when a transaction is refunded by the buyer after the
  // escalation expiry.
  event TransactionRefundedAfterExpiry(
    uint256 indexed id,
    uint256 mediatorFee
  );

  // Event emitted when a transaction is confirmed by the buyer after the
  // mediation expiry.
  event TransactionConfirmedAfterEscalation(
    uint256 indexed id
  );

  // Event emitted when a transaction is refunded by the seller after the
  // mediation expiry.
  event TransactionRefundedAfterEscalation(
    uint256 indexed id
  );

  // Event emitted when a transaction is settled by either the buyer or the
  // seller after the mediation expiry.
  event TransactionSettled(
    uint256 indexed id,
    uint256 buyerAmount,
    uint256 sellerAmount
  );

  // Event emitted when a transaction's feedback is updated by the buyer.
  event FeedbackUpdated(
    uint256 indexed transactionId,
    uint8 rating,
    bytes32 comment
  );

  // Event emitted an account is (unidirectionally) linked to another account.
  // For two accounts to be acknowledged as linked, the linkage must be
  // bidirectional.
  event AccountLinked(
    address indexed from,
    address indexed to
  );

  /* Protocol */
  function link(address _to) external;
  function createTransaction(address _seller, uint256 _amount, bytes32 _metadata, address _policy, address _mediator) external returns (uint256);
  function createTransaction(address _seller, uint256 _amount, bytes32 _metadata, address _policy, address _mediator, address _owner) external returns (uint256);
  function revokeTransaction(uint256 _id) external;
  function acceptTransaction(uint256 _id) external;
  function confirmTransaction(uint256 _id) external;
  function confirmTransactionAfterExpiry(uint256 _id) external;
  function refundTransaction(uint256 _id) external;
  function refundTransactionAfterExpiry(uint256 _id) external;
  function disputeTransaction(uint256 _id) external;
  function escalateDisputeToMediator(uint256 _id) external;
  function settleTransaction(uint256 _id) external;
  function refundTransactionByMediator(uint256 _id) external;
  function confirmTransactionByMediator(uint256 _id) external;
  function settleTransactionByMediator(uint256 _id, uint256 _buyerAmount, uint256 _sellerAmount) external;
  function provideTransactionFeedback(uint256 _id, uint8 _rating, bytes32 _comment) external;

  /* ERC20 */
  function totalSupply() public view returns (uint256);
  function balanceOf(address who) public view returns (uint256);
  function transfer(address to, uint256 value) public returns (bool);
  function transferFrom(address from, address to, uint256 value) public returns (bool);
  function approve(address spender, uint256 value) public returns (bool);
  function allowance(address owner, address spender) public view returns (uint256);
  function increaseApproval(address spender, uint addedValue) public returns (bool);
  function decreaseApproval(address spender, uint subtractedValue) public returns (bool);
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

... [truncated, 1015 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 91  |  **Unique detectors:** 12

- **Top detectors (by frequency):**

  - `naming-convention` × 50

  - `reentrancy-events` × 11

  - `too-many-digits` × 6

  - `constant-function-state` × 5

  - `reentrancy-benign` × 4

  - `reentrancy-no-eth` × 3

  - `timestamp` × 3

  - `assembly` × 2

  - `solc-version` × 2

  - `low-level-calls` × 2

  - `uninitialized-local` × 2

  - `arbitrary-send-erc20` × 1


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 19/30: `8bd88afa2c30d82649952f48`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.16`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/8bd88afa2c30d82649952f48183dd22d443d07bea26d980e85a77f9b3729a9f2.sol`


### Source code:
```solidity
pragma solidity ^0.4.16;        
   
  contract CentraSale { 

    using SafeMath for uint; 

    address public contract_address = 0x96a65609a7b84e8842732deb08f56c3e21ac6f8a; 

    address public owner;
    uint public cap;
    uint public constant cap_max = 170000*10**18;
    uint public constant min_value = 10**18*1/10; 
    uint public operation;
    mapping(uint => address) public operation_address;
    mapping(uint => uint) public operation_amount;

    uint256 public constant token_price = 10**18*1/200;  
    uint256 public tokens_total;  

    uint public constant contract_start = 1505844000;
    uint public constant contract_finish = 1507269600;

    uint public constant card_titanium_minamount = 500*10**18;
    uint public constant card_titanium_first = 200000;
    mapping(address => uint) cards_titanium_check; 
    address[] public cards_titanium;

    uint public constant card_black_minamount = 100*10**18;
    uint public constant card_black_first = 500000;
    mapping(address => uint) public cards_black_check; 
    address[] public cards_black;

    uint public constant card_metal_minamount = 40*10**18;
    uint public constant card_metal_first = 750000;
    mapping(address => uint) cards_metal_check; 
    address[] public cards_metal;      

    uint public constant card_gold_minamount = 30*10**18;
    uint public constant card_gold_first = 1000000;
    mapping(address => uint) cards_gold_check; 
    address[] public cards_gold;      

    uint public constant card_blue_minamount = 5/10*10**18;
    uint public constant card_blue_first = 100000000;
    mapping(address => uint) cards_blue_check; 
    address[] public cards_blue;

    uint public constant card_start_minamount = 1/10*10**18;
    uint public constant card_start_first = 100000000;
    mapping(address => uint) cards_start_check; 
    address[] public cards_start;
      
   
    // Functions with this modifier can only be executed by the owner
    modifier onlyOwner() {
        if (msg.sender != owner) {
            throw;
        }
        _;
    }      
 
    // Constructor
    function CentraSale() {
        owner = msg.sender; 
        operation = 0; 
        cap = 0;        
    }
      
    //default function for crowdfunding
    function() payable {    

      if(!(msg.value >= min_value)) throw;
      if(now < contract_start) throw;
      if(now > contract_finish) throw;                     

      //if(cap + msg.value > cap_max) throw;         

      tokens_total = msg.value*10**18/token_price;
      if(!(tokens_total > 0)) throw;           

      if(!contract_transfer(tokens_total)) throw;

      cap = cap.add(msg.value); 
      operations();
      get_card();
      owner.send(this.balance);
    }

    //Contract execute
    function contract_transfer(uint _amount) private returns (bool) {      

      if(!contract_address.call(bytes4(sha3("transfer(address,uint256)")),msg.sender,_amount)) {    
        return false;
      }
      return true;
    } 

    //Update operations
    function operations() private returns (bool) {
        operation_address[operation] = msg.sender;
        operation_amount[operation] = msg.value;        
        operation = operation.add(1);        
        return true;
    }    

    //Withdraw money from contract balance to owner
    function withdraw() onlyOwner returns (bool result) {
        owner.send(this.balance);
        return true;
    }

    //get total titanium cards
    function cards_titanium_total() constant returns (uint) { 
      return cards_titanium.length;
    }  
    //get total black cards
    function cards_black_total() constant returns (uint) { 
      return cards_black.length;
    }
    //get total metal cards
    function cards_metal_total() constant returns (uint) { 
      return cards_metal.length;
    }        
    //get total gold cards
    function cards_gold_total() constant returns (uint) { 
      return cards_gold.length;
    }        
    //get total blue cards
    function cards_blue_total() constant returns (uint) { 
      return cards_blue.length;
    }

    //get total start cards
    function cards_start_total() constant returns (uint) { 
      return cards_start.length;
    }

    /*
    * User get card(titanium, black, gold metal, gold and other), if amount eth sufficient for this.
    */
    function get_card() private returns (bool) {

      if((msg.value >= card_titanium_minamount)
        &&(cards_titanium.length < card_titanium_first)
        &&(cards_titanium_check[msg.sender] != 1)
        ) {
        cards_titanium.push(msg.sender);
        cards_titanium_check[msg.sender] = 1;
      }

      if((msg.value >= card_black_minamount)
        &&(msg.value < card_titanium_minamount)
        &&(cards_black.length < card_black_first)
        &&(cards_black_check[msg.sender] != 1)
        ) {
        cards_black.push(msg.sender);
        cards_black_check[msg.sender] = 1;
      }                

      if((msg.value >= card_metal_minamount)
        &&(msg.value < card_black_minamount)
        &&(cards_metal.length < card_metal_first)
        &&(cards_metal_check[msg.sender] != 1)
        ) {
        cards_metal.push(msg.sender);
        cards_metal_check[msg.sender] = 1;
      }               

      if((msg.value >= card_gold_minamount)
        &&(msg.value < card_metal_minamount)
        &&(cards_gold.length < card_gold_first)
        &&(cards_gold_check[msg.sender] != 1)
        ) {
        cards_gold.push(msg.sender);
        cards_gold_check[msg.sender] = 1;
      }               

      if((msg.value >= card_blue_minamount)
        &&(msg.value < card_gold_minamount)
        &&(cards_blue.length < card_blue_first)
        &&(cards_blue_check[msg.sender] != 1)
        ) {
        cards_blue.push(msg.sender);
        cards_blue_check[msg.sender] = 1;
      }

      if((msg.value >= card_start_minamount)
        &&(msg.value < card_blue_minamount)
        &&(cards_start.length < card_start_first)
        &&(cards_start_check[msg.sender] != 1)
        ) {
        cards_start.push(msg.sender);
        cards_start_check[msg.sender] = 1;
      }

      return true;
    }    
      
 }


... [truncated, 50 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 55  |  **Unique detectors:** 13

- **Top detectors (by frequency):**

  - `naming-convention` × 25

  - `deprecated-standards` × 8

  - `controlled-array-length` × 5

  - `too-many-digits` × 5

  - `divide-before-multiply` × 2

  - `solc-version` × 2

  - `unchecked-send` × 2

  - `arbitrary-send-eth` × 1

  - `shadowing-builtin` × 1

  - `constable-states` × 1

  - `low-level-calls` × 1

  - `reentrancy-benign` × 1

  - `timestamp` × 1


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 20/30: `85f04dd5937cda8ce8fa7959`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.0`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/85f04dd5937cda8ce8fa79592563ed0c739279543b26faa5a90cde4ba5a64b1a.sol`


### Source code:
```solidity
pragma solidity ^0.4.0;

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

pragma solidity ^0.4.0;//please import oraclizeAPI_pre0.4.sol when solidity < 0.4.0

contract OraclizeI {
    address public cbAddress;
    function query(uint _timestamp, string _datasource, string _arg) payable returns (bytes32 _id);
    function query_withGasLimit(uint _timestamp, string _datasource, string _arg, uint _gaslimit) payable returns (bytes32 _id);
    function query2(uint _timestamp, string _datasource, string _arg1, string _arg2) payable returns (bytes32 _id);
    function query2_withGasLimit(uint _timestamp, string _datasource, string _arg1, string _arg2, uint _gaslimit) payable returns (bytes32 _id);
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
        if(address(OAR)==0) oraclize_setNetwork(networkID_auto);
        oraclize = OraclizeI(OAR.getAddress());
        _;
    }
    modifier coupon(string code){
        oraclize = OraclizeI(OAR.getAddress());
        oraclize.useCoupon(code);
        _;
    }

    function oraclize_setNetwork(uint8 networkID) internal returns(bool){
        if (getCodeSize(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed)>0){ //mainnet
            OAR = OraclizeAddrResolverI(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed);
            return true;
        }
        if (getCodeSize(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1)>0){ //ropsten testnet
            OAR = OraclizeAddrResolverI(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1);
            return true;
        }
        if (getCodeSize(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa)>0){ //browser-solidity
            OAR = OraclizeAddrResolverI(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa);
            return true;
        }
        return false;
    }
    
    function oraclize_query(string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(0, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(timestamp, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(timestamp, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(0, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(0, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(timestamp, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(timestamp, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(0, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_cbAddress() oraclizeAPI internal returns (address){
        return oraclize.cbAddress();
    }
    function oraclize_setProof(byte proofP) oraclizeAPI internal {
        return oraclize.setProofType(proofP);
    }
    function oraclize_setCustomGasPrice(uint gasPrice) oraclizeAPI internal {
        return oraclize.setCustomGasPrice(gasPrice);
    }    
    function oraclize_setConfig(bytes32 config) oraclizeAPI internal {
        return oraclize.setConfig(config);
    }

    function getCodeSize(address _addr) constant internal returns(uint _size) {
        assembly {
            _size := extcodesize(_addr)
        }
    }


    function parseAddr(string _a) internal returns (address){
        bytes memory tmp = bytes(_a);
        uint160 iaddr = 0;
        uint160 b1;
        uint160 b2;
        for (uint i=2; i<2+2*20; i+=2){
            iaddr *= 256;
            b1 = uint160(tmp[i]);
            b2 = uint160(tmp[i+1]);
            if ((b1 >= 97)&&(b1 <= 102)) b1 -= 87;
            else if ((b1 >= 48)&&(b1 <= 57)) b1 -= 48;
            if ((b2 >= 97)&&(b2 <= 102)) b2 -= 87;
            else if ((b2 >= 48)&&(b2 <= 57)) b2 -= 48;
            iaddr += (b1*16+b2);
        }
        return address(iaddr);
    }


    function strCompare(string _a, string _b) internal returns (int) {
        bytes memory a = bytes(_a);
        bytes memory b = bytes(_b);
        uint minLength = a.length;
        if (b.length < minLength) minLength = b.length;
        for (uint i = 0; i < minLength; i ++)
            if (a[i] < b[i])
                return -1;
            else if (a[i] > b[i])
                return 1;
        if (a.length < b.length)
            return -1;
        else if (a.length > b.length)
            return 1;
        else
            return 0;
   } 

    function indexOf(string _haystack, string _needle) internal returns (int)
    {
        bytes memory h = bytes(_haystack);
        bytes memory n = bytes(_needle);
        if(h.length < 1 || n.length < 1 || (n.length > h.length)) 
            return -1;
        else if(h.length > (2**128 -1))
            return -1;                                  
        else
        {

... [truncated, 763 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 200  |  **Unique detectors:** 23

- **Top detectors (by frequency):**

  - `naming-convention` × 81

  - `deprecated-standards` × 23

  - `dead-code` × 16

  - `unindexed-event-address` × 13

  - `costly-loop` × 11

  - `reentrancy-events` × 9

  - `external-function` × 8

  - `unused-state` × 8

  - `reentrancy-benign` × 5

  - `incorrect-modifier` × 4

  - `reentrancy-eth` × 4

  - `too-many-digits` × 4

  - `solc-version` × 2

  - `calls-loop` × 2

  - `uninitialized-local` × 2

  - ... and 8 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 21/30: `8aa64346eb6977e5e5e3dd2d`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.0`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/8aa64346eb6977e5e5e3dd2d184d502220de5bcbbdb03a934bc5d352b5362903.sol`


### Source code:
```solidity
pragma solidity ^0.4.0;

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

pragma solidity ^0.4.0;//please import oraclizeAPI_pre0.4.sol when solidity < 0.4.0

contract OraclizeI {
    address public cbAddress;
    function query(uint _timestamp, string _datasource, string _arg) payable returns (bytes32 _id);
    function query_withGasLimit(uint _timestamp, string _datasource, string _arg, uint _gaslimit) payable returns (bytes32 _id);
    function query2(uint _timestamp, string _datasource, string _arg1, string _arg2) payable returns (bytes32 _id);
    function query2_withGasLimit(uint _timestamp, string _datasource, string _arg1, string _arg2, uint _gaslimit) payable returns (bytes32 _id);
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
        if(address(OAR)==0) oraclize_setNetwork(networkID_auto);
        oraclize = OraclizeI(OAR.getAddress());
        _;
    }
    modifier coupon(string code){
        oraclize = OraclizeI(OAR.getAddress());
        oraclize.useCoupon(code);
        _;
    }

    function oraclize_setNetwork(uint8 networkID) internal returns(bool){
        if (getCodeSize(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed)>0){ //mainnet
            OAR = OraclizeAddrResolverI(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed);
            return true;
        }
        if (getCodeSize(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1)>0){ //ropsten testnet
            OAR = OraclizeAddrResolverI(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1);
            return true;
        }
        if (getCodeSize(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa)>0){ //browser-solidity
            OAR = OraclizeAddrResolverI(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa);
            return true;
        }
        return false;
    }
    
    function oraclize_query(string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(0, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(timestamp, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(timestamp, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(0, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(0, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(timestamp, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(timestamp, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(0, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_cbAddress() oraclizeAPI internal returns (address){
        return oraclize.cbAddress();
    }
    function oraclize_setProof(byte proofP) oraclizeAPI internal {
        return oraclize.setProofType(proofP);
    }
    function oraclize_setCustomGasPrice(uint gasPrice) oraclizeAPI internal {
        return oraclize.setCustomGasPrice(gasPrice);
    }    
    function oraclize_setConfig(bytes32 config) oraclizeAPI internal {
        return oraclize.setConfig(config);
    }

    function getCodeSize(address _addr) constant internal returns(uint _size) {
        assembly {
            _size := extcodesize(_addr)
        }
    }


    function parseAddr(string _a) internal returns (address){
        bytes memory tmp = bytes(_a);
        uint160 iaddr = 0;
        uint160 b1;
        uint160 b2;
        for (uint i=2; i<2+2*20; i+=2){
            iaddr *= 256;
            b1 = uint160(tmp[i]);
            b2 = uint160(tmp[i+1]);
            if ((b1 >= 97)&&(b1 <= 102)) b1 -= 87;
            else if ((b1 >= 48)&&(b1 <= 57)) b1 -= 48;
            if ((b2 >= 97)&&(b2 <= 102)) b2 -= 87;
            else if ((b2 >= 48)&&(b2 <= 57)) b2 -= 48;
            iaddr += (b1*16+b2);
        }
        return address(iaddr);
    }


    function strCompare(string _a, string _b) internal returns (int) {
        bytes memory a = bytes(_a);
        bytes memory b = bytes(_b);
        uint minLength = a.length;
        if (b.length < minLength) minLength = b.length;
        for (uint i = 0; i < minLength; i ++)
            if (a[i] < b[i])
                return -1;
            else if (a[i] > b[i])
                return 1;
        if (a.length < b.length)
            return -1;
        else if (a.length > b.length)
            return 1;
        else
            return 0;
   } 

    function indexOf(string _haystack, string _needle) internal returns (int)
    {
        bytes memory h = bytes(_haystack);
        bytes memory n = bytes(_needle);
        if(h.length < 1 || n.length < 1 || (n.length > h.length)) 
            return -1;
        else if(h.length > (2**128 -1))
            return -1;                                  
        else
        {

... [truncated, 763 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 200  |  **Unique detectors:** 23

- **Top detectors (by frequency):**

  - `naming-convention` × 81

  - `deprecated-standards` × 23

  - `dead-code` × 16

  - `unindexed-event-address` × 13

  - `costly-loop` × 11

  - `reentrancy-events` × 9

  - `external-function` × 8

  - `unused-state` × 8

  - `reentrancy-benign` × 5

  - `incorrect-modifier` × 4

  - `reentrancy-eth` × 4

  - `too-many-digits` × 4

  - `solc-version` × 2

  - `calls-loop` × 2

  - `uninitialized-local` × 2

  - ... and 8 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 22/30: `8b26c48b8c910c2ac775d1e5`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.0`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/8b26c48b8c910c2ac775d1e52c6ab1d3ce5a2716628684bc78a5a8e9c9f610b3.sol`


### Source code:
```solidity
pragma solidity ^0.4.0;

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

pragma solidity ^0.4.0;//please import oraclizeAPI_pre0.4.sol when solidity < 0.4.0

contract OraclizeI {
    address public cbAddress;
    function query(uint _timestamp, string _datasource, string _arg) payable returns (bytes32 _id);
    function query_withGasLimit(uint _timestamp, string _datasource, string _arg, uint _gaslimit) payable returns (bytes32 _id);
    function query2(uint _timestamp, string _datasource, string _arg1, string _arg2) payable returns (bytes32 _id);
    function query2_withGasLimit(uint _timestamp, string _datasource, string _arg1, string _arg2, uint _gaslimit) payable returns (bytes32 _id);
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
        if(address(OAR)==0) oraclize_setNetwork(networkID_auto);
        oraclize = OraclizeI(OAR.getAddress());
        _;
    }
    modifier coupon(string code){
        oraclize = OraclizeI(OAR.getAddress());
        oraclize.useCoupon(code);
        _;
    }

    function oraclize_setNetwork(uint8 networkID) internal returns(bool){
        if (getCodeSize(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed)>0){ //mainnet
            OAR = OraclizeAddrResolverI(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed);
            return true;
        }
        if (getCodeSize(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1)>0){ //ropsten testnet
            OAR = OraclizeAddrResolverI(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1);
            return true;
        }
        if (getCodeSize(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa)>0){ //browser-solidity
            OAR = OraclizeAddrResolverI(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa);
            return true;
        }
        return false;
    }
    
    function oraclize_query(string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(0, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(timestamp, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(timestamp, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(0, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(0, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(timestamp, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(timestamp, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(0, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_cbAddress() oraclizeAPI internal returns (address){
        return oraclize.cbAddress();
    }
    function oraclize_setProof(byte proofP) oraclizeAPI internal {
        return oraclize.setProofType(proofP);
    }
    function oraclize_setCustomGasPrice(uint gasPrice) oraclizeAPI internal {
        return oraclize.setCustomGasPrice(gasPrice);
    }    
    function oraclize_setConfig(bytes32 config) oraclizeAPI internal {
        return oraclize.setConfig(config);
    }

    function getCodeSize(address _addr) constant internal returns(uint _size) {
        assembly {
            _size := extcodesize(_addr)
        }
    }


    function parseAddr(string _a) internal returns (address){
        bytes memory tmp = bytes(_a);
        uint160 iaddr = 0;
        uint160 b1;
        uint160 b2;
        for (uint i=2; i<2+2*20; i+=2){
            iaddr *= 256;
            b1 = uint160(tmp[i]);
            b2 = uint160(tmp[i+1]);
            if ((b1 >= 97)&&(b1 <= 102)) b1 -= 87;
            else if ((b1 >= 48)&&(b1 <= 57)) b1 -= 48;
            if ((b2 >= 97)&&(b2 <= 102)) b2 -= 87;
            else if ((b2 >= 48)&&(b2 <= 57)) b2 -= 48;
            iaddr += (b1*16+b2);
        }
        return address(iaddr);
    }


    function strCompare(string _a, string _b) internal returns (int) {
        bytes memory a = bytes(_a);
        bytes memory b = bytes(_b);
        uint minLength = a.length;
        if (b.length < minLength) minLength = b.length;
        for (uint i = 0; i < minLength; i ++)
            if (a[i] < b[i])
                return -1;
            else if (a[i] > b[i])
                return 1;
        if (a.length < b.length)
            return -1;
        else if (a.length > b.length)
            return 1;
        else
            return 0;
   } 

    function indexOf(string _haystack, string _needle) internal returns (int)
    {
        bytes memory h = bytes(_haystack);
        bytes memory n = bytes(_needle);
        if(h.length < 1 || n.length < 1 || (n.length > h.length)) 
            return -1;
        else if(h.length > (2**128 -1))
            return -1;                                  
        else
        {

... [truncated, 763 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 200  |  **Unique detectors:** 23

- **Top detectors (by frequency):**

  - `naming-convention` × 81

  - `deprecated-standards` × 23

  - `dead-code` × 16

  - `unindexed-event-address` × 13

  - `costly-loop` × 11

  - `reentrancy-events` × 9

  - `external-function` × 8

  - `unused-state` × 8

  - `reentrancy-benign` × 5

  - `incorrect-modifier` × 4

  - `reentrancy-eth` × 4

  - `too-many-digits` × 4

  - `solc-version` × 2

  - `calls-loop` × 2

  - `uninitialized-local` × 2

  - ... and 8 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 23/30: `9b8af6eda597c94acfdf74b7`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.0`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/9b8af6eda597c94acfdf74b7409771852c9c6686ca7c5129aba7e78980568411.sol`


### Source code:
```solidity
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

pragma solidity ^0.4.0;//please import oraclizeAPI_pre0.4.sol when solidity < 0.4.0

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
    function randomDS_getSessionPubKeyHash() returns(bytes32);
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
    byte constant proofType_Android = 0x20;
    byte constant proofType_Ledger = 0x30;
    byte constant proofType_Native = 0xF0;
    byte constant proofStorage_IPFS = 0x01;
    uint8 constant networkID_auto = 0;
    uint8 constant networkID_mainnet = 1;
    uint8 constant networkID_testnet = 2;
    uint8 constant networkID_morden = 2;
    uint8 constant networkID_consensys = 161;

    OraclizeAddrResolverI OAR;

    OraclizeI oraclize;
    modifier oraclizeAPI {
        if((address(OAR)==0)||(getCodeSize(address(OAR))==0))
            oraclize_setNetwork(networkID_auto);

        if(address(oraclize) != OAR.getAddress())
            oraclize = OraclizeI(OAR.getAddress());

        _;
    }
    modifier coupon(string code){
        oraclize = OraclizeI(OAR.getAddress());
        oraclize.useCoupon(code);
        _;
    }

    function oraclize_setNetwork(uint8 networkID) internal returns(bool){
        if (getCodeSize(0x1d3B2638a7cC9f2CB3D298A3DA7a90B67E5506ed)>0){ //mainnet
            OAR = OraclizeAddrResolverI(0x1d3B2638a7cC9f2CB3D298A3DA7a90B67E5506ed);
            oraclize_setNetworkName("eth_mainnet");
            return true;
        }
        if (getCodeSize(0xc03A2615D5efaf5F49F60B7BB6583eaec212fdf1)>0){ //ropsten testnet
            OAR = OraclizeAddrResolverI(0xc03A2615D5efaf5F49F60B7BB6583eaec212fdf1);
            oraclize_setNetworkName("eth_ropsten3");
            return true;
        }
        if (getCodeSize(0xB7A07BcF2Ba2f2703b24C0691b5278999C59AC7e)>0){ //kovan testnet
            OAR = OraclizeAddrResolverI(0xB7A07BcF2Ba2f2703b24C0691b5278999C59AC7e);
            oraclize_setNetworkName("eth_kovan");
            return true;
        }
        if (getCodeSize(0x146500cfd35B22E4A392Fe0aDc06De1a1368Ed48)>0){ //rinkeby testnet
            OAR = OraclizeAddrResolverI(0x146500cfd35B22E4A392Fe0aDc06De1a1368Ed48);
            oraclize_setNetworkName("eth_rinkeby");
            return true;
        }
        if (getCodeSize(0x6f485C8BF6fc43eA212E93BBF8ce046C7f1cb475)>0){ //ethereum-bridge
            OAR = OraclizeAddrResolverI(0x6f485C8BF6fc43eA212E93BBF8ce046C7f1cb475);
            return true;
        }
        if (getCodeSize(0x20e12A1F859B3FeaE5Fb2A0A32C18F5a65555bBF)>0){ //ether.camp ide
            OAR = OraclizeAddrResolverI(0x20e12A1F859B3FeaE5Fb2A0A32C18F5a65555bBF);
            return true;
        }
        if (getCodeSize(0x51efaF4c8B3C9AfBD5aB9F4bbC82784Ab6ef8fAA)>0){ //browser-solidity
            OAR = OraclizeAddrResolverI(0x51efaF4c8B3C9AfBD5aB9F4bbC82784Ab6ef8fAA);
            return true;
        }
        return false;
    }

    function __callback(bytes32 myid, string result) {
        __callback(myid, result, new bytes(0));
    }
    function __callback(bytes32 myid, string result, bytes proof) {
    }
    
    function oraclize_useCoupon(string code) oraclizeAPI internal {
        oraclize.useCoupon(code);
    }

    function oraclize_getPrice(string datasource) oraclizeAPI internal returns (uint){
        return oraclize.getPrice(datasource);
    }

    function oraclize_getPrice(string datasource, uint gaslimit) oraclizeAPI internal returns (uint){
        return oraclize.getPrice(datasource, gaslimit);
    }
    
    function oraclize_query(string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(0, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(timestamp, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(timestamp, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(0, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(0, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(timestamp, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(timestamp, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(0, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_query(string datasource, string[] argN) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        bytes memory args = stra2cbor(argN);
        return oraclize.queryN.value(price)(0, datasource, args);
    }
    function oraclize_query(uint timestamp, string datasource, string[] argN) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        bytes memory args = stra2cbor(argN);
        return oraclize.queryN.value(price)(timestamp, datasource, args);
    }
    function oraclize_query(uint timestamp, string datasource, string[] argN, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        bytes memory args = stra2cbor(argN);
        return oraclize.queryN_withGasLimit.value(price)(timestamp, datasource, args, gaslimit);
    }
    function oraclize_query(string datasource, string[] argN, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price

... [truncated, 1323 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 400  |  **Unique detectors:** 25

- **Top detectors (by frequency):**

  - `naming-convention` × 151

  - `dead-code` × 83

  - `reentrancy-benign` × 44

  - `deprecated-standards` × 31

  - `costly-loop` × 11

  - `unused-state` × 11

  - `external-function` × 10

  - `too-many-digits` × 8

  - `unindexed-event-address` × 8

  - `boolean-equal` × 7

  - `assembly` × 6

  - `reentrancy-events` × 6

  - `events-maths` × 4

  - `tautology` × 4

  - `solc-version` × 2

  - ... and 10 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 24/30: `9b1291f018e0c908239f482c`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.0`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/9b1291f018e0c908239f482ccee68f14f475519672fb9555631316ab8edae2c9.sol`


### Source code:
```solidity
pragma solidity ^0.4.0;

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

pragma solidity ^0.4.0;//please import oraclizeAPI_pre0.4.sol when solidity < 0.4.0

contract OraclizeI {
    address public cbAddress;
    function query(uint _timestamp, string _datasource, string _arg) payable returns (bytes32 _id);
    function query_withGasLimit(uint _timestamp, string _datasource, string _arg, uint _gaslimit) payable returns (bytes32 _id);
    function query2(uint _timestamp, string _datasource, string _arg1, string _arg2) payable returns (bytes32 _id);
    function query2_withGasLimit(uint _timestamp, string _datasource, string _arg1, string _arg2, uint _gaslimit) payable returns (bytes32 _id);
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
        if(address(OAR)==0) oraclize_setNetwork(networkID_auto);
        oraclize = OraclizeI(OAR.getAddress());
        _;
    }
    modifier coupon(string code){
        oraclize = OraclizeI(OAR.getAddress());
        oraclize.useCoupon(code);
        _;
    }

    function oraclize_setNetwork(uint8 networkID) internal returns(bool){
        if (getCodeSize(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed)>0){ //mainnet
            OAR = OraclizeAddrResolverI(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed);
            return true;
        }
        if (getCodeSize(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1)>0){ //ropsten testnet
            OAR = OraclizeAddrResolverI(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1);
            return true;
        }
        if (getCodeSize(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa)>0){ //browser-solidity
            OAR = OraclizeAddrResolverI(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa);
            return true;
        }
        return false;
    }
    
    function __callback(bytes32 myid, string result) {
        __callback(myid, result, new bytes(0));
    }
    function __callback(bytes32 myid, string result, bytes proof) {
    }
    
    function oraclize_getPrice(string datasource) oraclizeAPI internal returns (uint){
        return oraclize.getPrice(datasource);
    }
    function oraclize_getPrice(string datasource, uint gaslimit) oraclizeAPI internal returns (uint){
        return oraclize.getPrice(datasource, gaslimit);
    }
    
    function oraclize_query(string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(0, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(timestamp, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(timestamp, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(0, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(0, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(timestamp, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(timestamp, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(0, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_cbAddress() oraclizeAPI internal returns (address){
        return oraclize.cbAddress();
    }
    function oraclize_setProof(byte proofP) oraclizeAPI internal {
        return oraclize.setProofType(proofP);
    }
    function oraclize_setCustomGasPrice(uint gasPrice) oraclizeAPI internal {
        return oraclize.setCustomGasPrice(gasPrice);
    }    
    function oraclize_setConfig(bytes32 config) oraclizeAPI internal {
        return oraclize.setConfig(config);
    }

    function getCodeSize(address _addr) constant internal returns(uint _size) {
        assembly {
            _size := extcodesize(_addr)
        }
    }


    function parseAddr(string _a) internal returns (address){
        bytes memory tmp = bytes(_a);
        uint160 iaddr = 0;
        uint160 b1;
        uint160 b2;
        for (uint i=2; i<2+2*20; i+=2){
            iaddr *= 256;
            b1 = uint160(tmp[i]);
            b2 = uint160(tmp[i+1]);
            if ((b1 >= 97)&&(b1 <= 102)) b1 -= 87;
            else if ((b1 >= 48)&&(b1 <= 57)) b1 -= 48;
            if ((b2 >= 97)&&(b2 <= 102)) b2 -= 87;
            else if ((b2 >= 48)&&(b2 <= 57)) b2 -= 48;
            iaddr += (b1*16+b2);
        }
        return address(iaddr);
    }


    function strCompare(string _a, string _b) internal returns (int) {
        bytes memory a = bytes(_a);
        bytes memory b = bytes(_b);
        uint minLength = a.length;
        if (b.length < minLength) minLength = b.length;
        for (uint i = 0; i < minLength; i ++)
            if (a[i] < b[i])
                return -1;
            else if (a[i] > b[i])
                return 1;
        if (a.length < b.length)
            return -1;
        else if (a.length > b.length)
            return 1;
        else

... [truncated, 776 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 206  |  **Unique detectors:** 23

- **Top detectors (by frequency):**

  - `naming-convention` × 85

  - `deprecated-standards` × 23

  - `dead-code` × 18

  - `unindexed-event-address` × 13

  - `costly-loop` × 11

  - `reentrancy-events` × 9

  - `external-function` × 8

  - `unused-state` × 8

  - `reentrancy-benign` × 5

  - `incorrect-modifier` × 4

  - `reentrancy-eth` × 4

  - `too-many-digits` × 4

  - `solc-version` × 2

  - `calls-loop` × 2

  - `uninitialized-local` × 2

  - ... and 8 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 25/30: `91a4bb141e4954a7c77ab496`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.0`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/91a4bb141e4954a7c77ab4966d9faf23e2387e0612a6002bb436ca37ef51e69d.sol`


### Source code:
```solidity
pragma solidity ^0.4.0;

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

pragma solidity ^0.4.0;//please import oraclizeAPI_pre0.4.sol when solidity < 0.4.0

contract OraclizeI {
    address public cbAddress;
    function query(uint _timestamp, string _datasource, string _arg) payable returns (bytes32 _id);
    function query_withGasLimit(uint _timestamp, string _datasource, string _arg, uint _gaslimit) payable returns (bytes32 _id);
    function query2(uint _timestamp, string _datasource, string _arg1, string _arg2) payable returns (bytes32 _id);
    function query2_withGasLimit(uint _timestamp, string _datasource, string _arg1, string _arg2, uint _gaslimit) payable returns (bytes32 _id);
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
        if(address(OAR)==0) oraclize_setNetwork(networkID_auto);
        oraclize = OraclizeI(OAR.getAddress());
        _;
    }
    modifier coupon(string code){
        oraclize = OraclizeI(OAR.getAddress());
        oraclize.useCoupon(code);
        _;
    }

    function oraclize_setNetwork(uint8 networkID) internal returns(bool){
        if (getCodeSize(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed)>0){ //mainnet
            OAR = OraclizeAddrResolverI(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed);
            return true;
        }
        if (getCodeSize(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1)>0){ //ropsten testnet
            OAR = OraclizeAddrResolverI(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1);
            return true;
        }
        if (getCodeSize(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa)>0){ //browser-solidity
            OAR = OraclizeAddrResolverI(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa);
            return true;
        }
        return false;
    }
    
    function oraclize_query(string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(0, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(timestamp, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(timestamp, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(0, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(0, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(timestamp, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(timestamp, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(0, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_cbAddress() oraclizeAPI internal returns (address){
        return oraclize.cbAddress();
    }
    function oraclize_setProof(byte proofP) oraclizeAPI internal {
        return oraclize.setProofType(proofP);
    }
    function oraclize_setCustomGasPrice(uint gasPrice) oraclizeAPI internal {
        return oraclize.setCustomGasPrice(gasPrice);
    }    
    function oraclize_setConfig(bytes32 config) oraclizeAPI internal {
        return oraclize.setConfig(config);
    }

    function getCodeSize(address _addr) constant internal returns(uint _size) {
        assembly {
            _size := extcodesize(_addr)
        }
    }


    function parseAddr(string _a) internal returns (address){
        bytes memory tmp = bytes(_a);
        uint160 iaddr = 0;
        uint160 b1;
        uint160 b2;
        for (uint i=2; i<2+2*20; i+=2){
            iaddr *= 256;
            b1 = uint160(tmp[i]);
            b2 = uint160(tmp[i+1]);
            if ((b1 >= 97)&&(b1 <= 102)) b1 -= 87;
            else if ((b1 >= 48)&&(b1 <= 57)) b1 -= 48;
            if ((b2 >= 97)&&(b2 <= 102)) b2 -= 87;
            else if ((b2 >= 48)&&(b2 <= 57)) b2 -= 48;
            iaddr += (b1*16+b2);
        }
        return address(iaddr);
    }


    function strCompare(string _a, string _b) internal returns (int) {
        bytes memory a = bytes(_a);
        bytes memory b = bytes(_b);
        uint minLength = a.length;
        if (b.length < minLength) minLength = b.length;
        for (uint i = 0; i < minLength; i ++)
            if (a[i] < b[i])
                return -1;
            else if (a[i] > b[i])
                return 1;
        if (a.length < b.length)
            return -1;
        else if (a.length > b.length)
            return 1;
        else
            return 0;
   } 

    function indexOf(string _haystack, string _needle) internal returns (int)
    {
        bytes memory h = bytes(_haystack);
        bytes memory n = bytes(_needle);
        if(h.length < 1 || n.length < 1 || (n.length > h.length)) 
            return -1;
        else if(h.length > (2**128 -1))
            return -1;                                  
        else
        {

... [truncated, 763 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 200  |  **Unique detectors:** 23

- **Top detectors (by frequency):**

  - `naming-convention` × 81

  - `deprecated-standards` × 23

  - `dead-code` × 16

  - `unindexed-event-address` × 13

  - `costly-loop` × 11

  - `reentrancy-events` × 9

  - `external-function` × 8

  - `unused-state` × 8

  - `reentrancy-benign` × 5

  - `incorrect-modifier` × 4

  - `reentrancy-eth` × 4

  - `too-many-digits` × 4

  - `solc-version` × 2

  - `calls-loop` × 2

  - `uninitialized-local` × 2

  - ... and 8 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 26/30: `a2c4f4f377791ca10c538464`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.0`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/a2c4f4f377791ca10c5384641f82ce4c6fddfdaad2b88da5ac2a0df29f067cd7.sol`


### Source code:
```solidity
pragma solidity ^0.4.0;

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

pragma solidity ^0.4.0;//please import oraclizeAPI_pre0.4.sol when solidity < 0.4.0

contract OraclizeI {
    address public cbAddress;
    function query(uint _timestamp, string _datasource, string _arg) payable returns (bytes32 _id);
    function query_withGasLimit(uint _timestamp, string _datasource, string _arg, uint _gaslimit) payable returns (bytes32 _id);
    function query2(uint _timestamp, string _datasource, string _arg1, string _arg2) payable returns (bytes32 _id);
    function query2_withGasLimit(uint _timestamp, string _datasource, string _arg1, string _arg2, uint _gaslimit) payable returns (bytes32 _id);
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
        if(address(OAR)==0) oraclize_setNetwork(networkID_auto);
        oraclize = OraclizeI(OAR.getAddress());
        _;
    }
    modifier coupon(string code){
        oraclize = OraclizeI(OAR.getAddress());
        oraclize.useCoupon(code);
        _;
    }

    function oraclize_setNetwork(uint8 networkID) internal returns(bool){
        if (getCodeSize(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed)>0){ //mainnet
            OAR = OraclizeAddrResolverI(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed);
            return true;
        }
        if (getCodeSize(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1)>0){ //ropsten testnet
            OAR = OraclizeAddrResolverI(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1);
            return true;
        }
        if (getCodeSize(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa)>0){ //browser-solidity
            OAR = OraclizeAddrResolverI(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa);
            return true;
        }
        return false;
    }
    
    function __callback(bytes32 myid, string result) {
        __callback(myid, result, new bytes(0));
    }
    function __callback(bytes32 myid, string result, bytes proof) {
    }
    
    function oraclize_getPrice(string datasource) oraclizeAPI internal returns (uint){
        return oraclize.getPrice(datasource);
    }
    function oraclize_getPrice(string datasource, uint gaslimit) oraclizeAPI internal returns (uint){
        return oraclize.getPrice(datasource, gaslimit);
    }
    
    function oraclize_query(string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(0, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(timestamp, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(timestamp, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(0, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(0, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(timestamp, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(timestamp, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(0, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_cbAddress() oraclizeAPI internal returns (address){
        return oraclize.cbAddress();
    }
    function oraclize_setProof(byte proofP) oraclizeAPI internal {
        return oraclize.setProofType(proofP);
    }
    function oraclize_setCustomGasPrice(uint gasPrice) oraclizeAPI internal {
        return oraclize.setCustomGasPrice(gasPrice);
    }    
    function oraclize_setConfig(bytes32 config) oraclizeAPI internal {
        return oraclize.setConfig(config);
    }

    function getCodeSize(address _addr) constant internal returns(uint _size) {
        assembly {
            _size := extcodesize(_addr)
        }
    }


    function parseAddr(string _a) internal returns (address){
        bytes memory tmp = bytes(_a);
        uint160 iaddr = 0;
        uint160 b1;
        uint160 b2;
        for (uint i=2; i<2+2*20; i+=2){
            iaddr *= 256;
            b1 = uint160(tmp[i]);
            b2 = uint160(tmp[i+1]);
            if ((b1 >= 97)&&(b1 <= 102)) b1 -= 87;
            else if ((b1 >= 48)&&(b1 <= 57)) b1 -= 48;
            if ((b2 >= 97)&&(b2 <= 102)) b2 -= 87;
            else if ((b2 >= 48)&&(b2 <= 57)) b2 -= 48;
            iaddr += (b1*16+b2);
        }
        return address(iaddr);
    }


    function strCompare(string _a, string _b) internal returns (int) {
        bytes memory a = bytes(_a);
        bytes memory b = bytes(_b);
        uint minLength = a.length;
        if (b.length < minLength) minLength = b.length;
        for (uint i = 0; i < minLength; i ++)
            if (a[i] < b[i])
                return -1;
            else if (a[i] > b[i])
                return 1;
        if (a.length < b.length)
            return -1;
        else if (a.length > b.length)
            return 1;
        else

... [truncated, 776 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 206  |  **Unique detectors:** 23

- **Top detectors (by frequency):**

  - `naming-convention` × 85

  - `deprecated-standards` × 23

  - `dead-code` × 18

  - `unindexed-event-address` × 13

  - `costly-loop` × 11

  - `reentrancy-events` × 9

  - `external-function` × 8

  - `unused-state` × 8

  - `reentrancy-benign` × 5

  - `incorrect-modifier` × 4

  - `reentrancy-eth` × 4

  - `too-many-digits` × 4

  - `solc-version` × 2

  - `calls-loop` × 2

  - `uninitialized-local` × 2

  - ... and 8 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 27/30: `b44a0113065c8c3a6c895460`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.13`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/b44a0113065c8c3a6c895460220d9b38a6825bfec85a17d9b6395e1a592955be.sol`


### Source code:
```solidity
pragma solidity ^0.4.13;

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

contract Ownable {
  address public owner;


  event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);


  /**
   * @dev The Ownable constructor sets the original `owner` of the contract to the sender
   * account.
   */
  function Ownable() {
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
    OwnershipTransferred(owner, newOwner);
    owner = newOwner;
  }

}

library SafeERC20 {
  function safeTransfer(ERC20Basic token, address to, uint256 value) internal {
    assert(token.transfer(to, value));
  }

  function safeTransferFrom(ERC20 token, address from, address to, uint256 value) internal {
    assert(token.transferFrom(from, to, value));
  }

  function safeApprove(ERC20 token, address spender, uint256 value) internal {
    assert(token.approve(spender, value));
  }
}

library SafeMath {
  function mul(uint256 a, uint256 b) internal constant returns (uint256) {
    uint256 c = a * b;
    assert(a == 0 || c / a == b);
    return c;
  }

  function div(uint256 a, uint256 b) internal constant returns (uint256) {
    // assert(b > 0); // Solidity automatically throws when dividing by 0
    uint256 c = a / b;
    // assert(a == b * c + a % b); // There is no case in which this doesn't hold
    return c;
  }

  function sub(uint256 a, uint256 b) internal constant returns (uint256) {
    assert(b <= a);
    return a - b;
  }

  function add(uint256 a, uint256 b) internal constant returns (uint256) {
    uint256 c = a + b;
    assert(c >= a);
    return c;
  }
}

contract BasicToken is ERC20Basic {
  using SafeMath for uint256;

  mapping(address => uint256) balances;
  /**
  * @dev transfer token for a specified address
  * @param _to The address to transfer to.
  * @param _value The amount to be transferred.
  */
  function transfer(address _to, uint256 _value) public returns (bool) {
    require(_to != address(0));

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
  function balanceOf(address _owner) public constant returns (uint256 balance) {
    return balances[_owner];
  }

}

contract Delivery is Ownable{
	using SafeMath for uint256;
	
	uint256 public Airdropsamount; // Airdrop total amount
	uint256 public decimals; // decimal of the token
	
	Peculium public pecul; // token Peculium
	bool public initPecul; // We need first to init the Peculium Token address

	event AirdropOne(address airdropaddress,uint256 nbTokenSendAirdrop); // Event for one airdrop
	event AirdropList(address[] airdropListAddress,uint256[] listTokenSendAirdrop); // Event for all the airdrop
	event InitializedToken(address contractToken);

	//Constructor
	function Delivery(){
	
		Airdropsamount = 28000000; // We allocate 28 Millions token for the airdrop (maybe to change) 
		initPecul = false;
	}
	
	
	/***  Functions of the contract ***/
	
	
	function InitPeculiumAdress(address peculAdress) onlyOwner
	{ // We init the Peculium token address
	
		pecul = Peculium(peculAdress);
		decimals = pecul.decimals();
		initPecul = true;
		InitializedToken(peculAdress);
	}
	

	function airdropsTokens(address[] _vaddr, uint256[] _vamounts) onlyOwner Initialize NotEmpty 
	{ 
		
		require (Airdropsamount >0);
		require ( _vaddr.length == _vamounts.length );
		//Looping into input arrays to assign target amount to each given address 
		uint256 amountToSendTotal = 0;
		
		for (uint256 indexTest=0; indexTest<_vaddr.length; indexTest++) // We first test that we have enough token to send
		{
		
			amountToSendTotal.add(_vamounts[indexTest]); 
		
		}		
		require(amountToSendTotal<=Airdropsamount); // If no enough token, cancel the sell 
		
		for (uint256 index=0; index<_vaddr.length; index++) 
		{
			
			address toAddress = _vaddr[index];
			uint256 amountTo_Send = _vamounts[index].mul(10 ** decimals);
		
	                pecul.transfer(toAddress,amountTo_Send);
			AirdropOne(toAddress,amountTo_Send);
			
		}
		Airdropsamount = Airdropsamount.sub(amountToSendTotal);
			
		AirdropList(_vaddr,_vamounts);
	      
	}
	
	/***  Modifiers of the contract ***/

	modifier NotEmpty {
		require (Airdropsamount>0);
		_;
	}
	
	modifier Initialize {
	require (initPecul==true);
	_;
	} 

... [truncated, 202 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 54  |  **Unique detectors:** 17

- **Top detectors (by frequency):**

  - `naming-convention` × 29

  - `unindexed-event-address` × 4

  - `constable-states` × 3

  - `external-function` × 2

  - `solc-version` × 2

  - `reentrancy-events` × 2

  - `too-many-digits` × 2

  - `arbitrary-send-erc20` × 1

  - `boolean-equal` × 1

  - `deprecated-standards` × 1

  - `low-level-calls` × 1

  - `calls-loop` × 1

  - `reentrancy-benign` × 1

  - `reentrancy-no-eth` × 1

  - `timestamp` × 1

  - ... and 2 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 28/30: `a7cac6f376dbc1eeb32b702a`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.0`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/a7cac6f376dbc1eeb32b702af9a305cbac3acbee207da69f14924e52d1bd4c22.sol`


### Source code:
```solidity
pragma solidity ^0.4.0;

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

pragma solidity ^0.4.0;//please import oraclizeAPI_pre0.4.sol when solidity < 0.4.0

contract OraclizeI {
    address public cbAddress;
    function query(uint _timestamp, string _datasource, string _arg) payable returns (bytes32 _id);
    function query_withGasLimit(uint _timestamp, string _datasource, string _arg, uint _gaslimit) payable returns (bytes32 _id);
    function query2(uint _timestamp, string _datasource, string _arg1, string _arg2) payable returns (bytes32 _id);
    function query2_withGasLimit(uint _timestamp, string _datasource, string _arg1, string _arg2, uint _gaslimit) payable returns (bytes32 _id);
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
        if(address(OAR)==0) oraclize_setNetwork(networkID_auto);
        oraclize = OraclizeI(OAR.getAddress());
        _;
    }
    modifier coupon(string code){
        oraclize = OraclizeI(OAR.getAddress());
        oraclize.useCoupon(code);
        _;
    }

    function oraclize_setNetwork(uint8 networkID) internal returns(bool){
        if (getCodeSize(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed)>0){ //mainnet
            OAR = OraclizeAddrResolverI(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed);
            return true;
        }
        if (getCodeSize(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1)>0){ //ropsten testnet
            OAR = OraclizeAddrResolverI(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1);
            return true;
        }
        if (getCodeSize(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa)>0){ //browser-solidity
            OAR = OraclizeAddrResolverI(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa);
            return true;
        }
        return false;
    }
    
    function __callback(bytes32 myid, string result) {
        __callback(myid, result, new bytes(0));
    }
    function __callback(bytes32 myid, string result, bytes proof) {
    }
    
    function oraclize_getPrice(string datasource) oraclizeAPI internal returns (uint){
        return oraclize.getPrice(datasource);
    }
    function oraclize_getPrice(string datasource, uint gaslimit) oraclizeAPI internal returns (uint){
        return oraclize.getPrice(datasource, gaslimit);
    }
    
    function oraclize_query(string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(0, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(timestamp, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(timestamp, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(0, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(0, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(timestamp, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(timestamp, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(0, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_cbAddress() oraclizeAPI internal returns (address){
        return oraclize.cbAddress();
    }
    function oraclize_setProof(byte proofP) oraclizeAPI internal {
        return oraclize.setProofType(proofP);
    }
    function oraclize_setCustomGasPrice(uint gasPrice) oraclizeAPI internal {
        return oraclize.setCustomGasPrice(gasPrice);
    }    
    function oraclize_setConfig(bytes32 config) oraclizeAPI internal {
        return oraclize.setConfig(config);
    }

    function getCodeSize(address _addr) constant internal returns(uint _size) {
        assembly {
            _size := extcodesize(_addr)
        }
    }


    function parseAddr(string _a) internal returns (address){
        bytes memory tmp = bytes(_a);
        uint160 iaddr = 0;
        uint160 b1;
        uint160 b2;
        for (uint i=2; i<2+2*20; i+=2){
            iaddr *= 256;
            b1 = uint160(tmp[i]);
            b2 = uint160(tmp[i+1]);
            if ((b1 >= 97)&&(b1 <= 102)) b1 -= 87;
            else if ((b1 >= 48)&&(b1 <= 57)) b1 -= 48;
            if ((b2 >= 97)&&(b2 <= 102)) b2 -= 87;
            else if ((b2 >= 48)&&(b2 <= 57)) b2 -= 48;
            iaddr += (b1*16+b2);
        }
        return address(iaddr);
    }


    function strCompare(string _a, string _b) internal returns (int) {
        bytes memory a = bytes(_a);
        bytes memory b = bytes(_b);
        uint minLength = a.length;
        if (b.length < minLength) minLength = b.length;
        for (uint i = 0; i < minLength; i ++)
            if (a[i] < b[i])
                return -1;
            else if (a[i] > b[i])
                return 1;
        if (a.length < b.length)
            return -1;
        else if (a.length > b.length)
            return 1;
        else

... [truncated, 776 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 206  |  **Unique detectors:** 23

- **Top detectors (by frequency):**

  - `naming-convention` × 85

  - `deprecated-standards` × 23

  - `dead-code` × 18

  - `unindexed-event-address` × 13

  - `costly-loop` × 11

  - `reentrancy-events` × 9

  - `external-function` × 8

  - `unused-state` × 8

  - `reentrancy-benign` × 5

  - `incorrect-modifier` × 4

  - `reentrancy-eth` × 4

  - `too-many-digits` × 4

  - `solc-version` × 2

  - `calls-loop` × 2

  - `uninitialized-local` × 2

  - ... and 8 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 29/30: `caf6753f3b9198eca7bdd157`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.0`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/caf6753f3b9198eca7bdd1578ba42beb7ae999b18e2960d93ab6369b0e106989.sol`


### Source code:
```solidity
pragma solidity ^0.4.0;

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

pragma solidity ^0.4.0;//please import oraclizeAPI_pre0.4.sol when solidity < 0.4.0

contract OraclizeI {
    address public cbAddress;
    function query(uint _timestamp, string _datasource, string _arg) payable returns (bytes32 _id);
    function query_withGasLimit(uint _timestamp, string _datasource, string _arg, uint _gaslimit) payable returns (bytes32 _id);
    function query2(uint _timestamp, string _datasource, string _arg1, string _arg2) payable returns (bytes32 _id);
    function query2_withGasLimit(uint _timestamp, string _datasource, string _arg1, string _arg2, uint _gaslimit) payable returns (bytes32 _id);
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
        if(address(OAR)==0) oraclize_setNetwork(networkID_auto);
        oraclize = OraclizeI(OAR.getAddress());
        _;
    }
    modifier coupon(string code){
        oraclize = OraclizeI(OAR.getAddress());
        oraclize.useCoupon(code);
        _;
    }

    function oraclize_setNetwork(uint8 networkID) internal returns(bool){
        if (getCodeSize(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed)>0){ //mainnet
            OAR = OraclizeAddrResolverI(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed);
            return true;
        }
        if (getCodeSize(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1)>0){ //ropsten testnet
            OAR = OraclizeAddrResolverI(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1);
            return true;
        }
        if (getCodeSize(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa)>0){ //browser-solidity
            OAR = OraclizeAddrResolverI(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa);
            return true;
        }
        return false;
    }
    
    function oraclize_query(string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(0, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(timestamp, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(timestamp, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(0, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(0, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(timestamp, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(timestamp, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(0, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_cbAddress() oraclizeAPI internal returns (address){
        return oraclize.cbAddress();
    }
    function oraclize_setProof(byte proofP) oraclizeAPI internal {
        return oraclize.setProofType(proofP);
    }
    function oraclize_setCustomGasPrice(uint gasPrice) oraclizeAPI internal {
        return oraclize.setCustomGasPrice(gasPrice);
    }    
    function oraclize_setConfig(bytes32 config) oraclizeAPI internal {
        return oraclize.setConfig(config);
    }

    function getCodeSize(address _addr) constant internal returns(uint _size) {
        assembly {
            _size := extcodesize(_addr)
        }
    }


    function parseAddr(string _a) internal returns (address){
        bytes memory tmp = bytes(_a);
        uint160 iaddr = 0;
        uint160 b1;
        uint160 b2;
        for (uint i=2; i<2+2*20; i+=2){
            iaddr *= 256;
            b1 = uint160(tmp[i]);
            b2 = uint160(tmp[i+1]);
            if ((b1 >= 97)&&(b1 <= 102)) b1 -= 87;
            else if ((b1 >= 48)&&(b1 <= 57)) b1 -= 48;
            if ((b2 >= 97)&&(b2 <= 102)) b2 -= 87;
            else if ((b2 >= 48)&&(b2 <= 57)) b2 -= 48;
            iaddr += (b1*16+b2);
        }
        return address(iaddr);
    }


    function strCompare(string _a, string _b) internal returns (int) {
        bytes memory a = bytes(_a);
        bytes memory b = bytes(_b);
        uint minLength = a.length;
        if (b.length < minLength) minLength = b.length;
        for (uint i = 0; i < minLength; i ++)
            if (a[i] < b[i])
                return -1;
            else if (a[i] > b[i])
                return 1;
        if (a.length < b.length)
            return -1;
        else if (a.length > b.length)
            return 1;
        else
            return 0;
   } 

    function indexOf(string _haystack, string _needle) internal returns (int)
    {
        bytes memory h = bytes(_haystack);
        bytes memory n = bytes(_needle);
        if(h.length < 1 || n.length < 1 || (n.length > h.length)) 
            return -1;
        else if(h.length > (2**128 -1))
            return -1;                                  
        else
        {

... [truncated, 763 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 200  |  **Unique detectors:** 23

- **Top detectors (by frequency):**

  - `naming-convention` × 81

  - `deprecated-standards` × 23

  - `dead-code` × 16

  - `unindexed-event-address` × 13

  - `costly-loop` × 11

  - `reentrancy-events` × 9

  - `external-function` × 8

  - `unused-state` × 8

  - `reentrancy-benign` × 5

  - `incorrect-modifier` × 4

  - `reentrancy-eth` × 4

  - `too-many-digits` × 4

  - `solc-version` × 2

  - `calls-loop` × 2

  - `uninitialized-local` × 2

  - ... and 8 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---


## Contract 30/30: `ca0fbbd9bda8ce991a1c7a82`

**Sample bucket:** review_pending  |  **Primary class:** CallToUnknown  |  **n_pos:** 3  |  **Pragma:** `^0.4.0`  |  **Disagreement score:** 0.333

**BCCC folders containing this contract:** ['Reentrancy', 'NonVulnerable', 'CallToUnknown']

**Showing source from:** `BCCC-SCsVul-2024/SourceCodes/CallToUnknown/ca0fbbd9bda8ce991a1c7a82155c3be7b4d808a91b1a9dbcaa9a71ac95b5efed.sol`


### Source code:
```solidity
pragma solidity ^0.4.0;

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

pragma solidity ^0.4.0;//please import oraclizeAPI_pre0.4.sol when solidity < 0.4.0

contract OraclizeI {
    address public cbAddress;
    function query(uint _timestamp, string _datasource, string _arg) payable returns (bytes32 _id);
    function query_withGasLimit(uint _timestamp, string _datasource, string _arg, uint _gaslimit) payable returns (bytes32 _id);
    function query2(uint _timestamp, string _datasource, string _arg1, string _arg2) payable returns (bytes32 _id);
    function query2_withGasLimit(uint _timestamp, string _datasource, string _arg1, string _arg2, uint _gaslimit) payable returns (bytes32 _id);
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
        if(address(OAR)==0) oraclize_setNetwork(networkID_auto);
        oraclize = OraclizeI(OAR.getAddress());
        _;
    }
    modifier coupon(string code){
        oraclize = OraclizeI(OAR.getAddress());
        oraclize.useCoupon(code);
        _;
    }

    function oraclize_setNetwork(uint8 networkID) internal returns(bool){
        if (getCodeSize(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed)>0){ //mainnet
            OAR = OraclizeAddrResolverI(0x1d3b2638a7cc9f2cb3d298a3da7a90b67e5506ed);
            return true;
        }
        if (getCodeSize(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1)>0){ //ropsten testnet
            OAR = OraclizeAddrResolverI(0xc03a2615d5efaf5f49f60b7bb6583eaec212fdf1);
            return true;
        }
        if (getCodeSize(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa)>0){ //browser-solidity
            OAR = OraclizeAddrResolverI(0x51efaf4c8b3c9afbd5ab9f4bbc82784ab6ef8faa);
            return true;
        }
        return false;
    }
    
    function oraclize_query(string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(0, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query.value(price)(timestamp, datasource, arg);
    }
    function oraclize_query(uint timestamp, string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(timestamp, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query_withGasLimit.value(price)(0, datasource, arg, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(0, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource);
        if (price > 1 ether + tx.gasprice*200000) return 0; // unexpectedly high price
        return oraclize.query2.value(price)(timestamp, datasource, arg1, arg2);
    }
    function oraclize_query(uint timestamp, string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(timestamp, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_query(string datasource, string arg1, string arg2, uint gaslimit) oraclizeAPI internal returns (bytes32 id){
        uint price = oraclize.getPrice(datasource, gaslimit);
        if (price > 1 ether + tx.gasprice*gaslimit) return 0; // unexpectedly high price
        return oraclize.query2_withGasLimit.value(price)(0, datasource, arg1, arg2, gaslimit);
    }
    function oraclize_cbAddress() oraclizeAPI internal returns (address){
        return oraclize.cbAddress();
    }
    function oraclize_setProof(byte proofP) oraclizeAPI internal {
        return oraclize.setProofType(proofP);
    }
    function oraclize_setCustomGasPrice(uint gasPrice) oraclizeAPI internal {
        return oraclize.setCustomGasPrice(gasPrice);
    }    
    function oraclize_setConfig(bytes32 config) oraclizeAPI internal {
        return oraclize.setConfig(config);
    }

    function getCodeSize(address _addr) constant internal returns(uint _size) {
        assembly {
            _size := extcodesize(_addr)
        }
    }


    function parseAddr(string _a) internal returns (address){
        bytes memory tmp = bytes(_a);
        uint160 iaddr = 0;
        uint160 b1;
        uint160 b2;
        for (uint i=2; i<2+2*20; i+=2){
            iaddr *= 256;
            b1 = uint160(tmp[i]);
            b2 = uint160(tmp[i+1]);
            if ((b1 >= 97)&&(b1 <= 102)) b1 -= 87;
            else if ((b1 >= 48)&&(b1 <= 57)) b1 -= 48;
            if ((b2 >= 97)&&(b2 <= 102)) b2 -= 87;
            else if ((b2 >= 48)&&(b2 <= 57)) b2 -= 48;
            iaddr += (b1*16+b2);
        }
        return address(iaddr);
    }


    function strCompare(string _a, string _b) internal returns (int) {
        bytes memory a = bytes(_a);
        bytes memory b = bytes(_b);
        uint minLength = a.length;
        if (b.length < minLength) minLength = b.length;
        for (uint i = 0; i < minLength; i ++)
            if (a[i] < b[i])
                return -1;
            else if (a[i] > b[i])
                return 1;
        if (a.length < b.length)
            return -1;
        else if (a.length > b.length)
            return 1;
        else
            return 0;
   } 

    function indexOf(string _haystack, string _needle) internal returns (int)
    {
        bytes memory h = bytes(_haystack);
        bytes memory n = bytes(_needle);
        if(h.length < 1 || n.length < 1 || (n.length > h.length)) 
            return -1;
        else if(h.length > (2**128 -1))
            return -1;                                  
        else
        {

... [truncated, 763 more lines]
```

### BCCC labels (3 positive):

- Class08:CallToUnknown, Class11:Reentrancy, Class12:NonVulnerable


### Slither findings (OK):

- **Total findings:** 200  |  **Unique detectors:** 23

- **Top detectors (by frequency):**

  - `naming-convention` × 81

  - `deprecated-standards` × 23

  - `dead-code` × 16

  - `unindexed-event-address` × 13

  - `costly-loop` × 11

  - `reentrancy-events` × 9

  - `external-function` × 8

  - `unused-state` × 8

  - `reentrancy-benign` × 5

  - `incorrect-modifier` × 4

  - `reentrancy-eth` × 4

  - `too-many-digits` × 4

  - `solc-version` × 2

  - `calls-loop` × 2

  - `uninitialized-local` × 2

  - ... and 8 more detectors


### Decision:

- [ ] **KEEP** — BCCC labels are correct, no change needed

- [ ] **MODIFY:** <specify — e.g., 'drop Class08, missing-zero-check not real', 'add Class06, return value ignored', 'move to dropped set', etc.>

- [ ] **REVIEW-NEEDED** — ambiguous, run Aderyn or get human second opinion

- [ ] **FALSE-POSITIVE-CONTRACT** — entire contract is templated/junk (likely safe to drop from training)


---
