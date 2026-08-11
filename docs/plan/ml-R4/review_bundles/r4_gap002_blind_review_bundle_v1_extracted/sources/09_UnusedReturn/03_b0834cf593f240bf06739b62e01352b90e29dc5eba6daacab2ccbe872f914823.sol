pragma solidity ^0.8.0;

library EnumerableSet {

    struct Set {

        bytes32[] _values;

        mapping(bytes32 => uint256) _indexes;
    }

        function _add(Set storage set, bytes32 value) private returns (bool) {
        if (!_contains(set, value)) {
            set._values.push(value);

            set._indexes[value] = set._values.length;
            return true;
        } else {
            return false;
        }
    }

        function _remove(Set storage set, bytes32 value) private returns (bool) {

        uint256 valueIndex = set._indexes[value];

        if (valueIndex != 0) {

            uint256 toDeleteIndex = valueIndex - 1;
            uint256 lastIndex = set._values.length - 1;

            if (lastIndex != toDeleteIndex) {
                bytes32 lastValue = set._values[lastIndex];

                set._values[toDeleteIndex] = lastValue;

                set._indexes[lastValue] = valueIndex;
            }

            set._values.pop();

            delete set._indexes[value];

            return true;
        } else {
            return false;
        }
    }

        function _contains(Set storage set, bytes32 value) private view returns (bool) {
        return set._indexes[value] != 0;
    }

        function _length(Set storage set) private view returns (uint256) {
        return set._values.length;
    }

        function _at(Set storage set, uint256 index) private view returns (bytes32) {
        return set._values[index];
    }

        function _values(Set storage set) private view returns (bytes32[] memory) {
        return set._values;
    }

    struct Bytes32Set {
        Set _inner;
    }

        function add(Bytes32Set storage set, bytes32 value) internal returns (bool) {
        return _add(set._inner, value);
    }

        function remove(Bytes32Set storage set, bytes32 value) internal returns (bool) {
        return _remove(set._inner, value);
    }

        function contains(Bytes32Set storage set, bytes32 value) internal view returns (bool) {
        return _contains(set._inner, value);
    }

        function length(Bytes32Set storage set) internal view returns (uint256) {
        return _length(set._inner);
    }

        function at(Bytes32Set storage set, uint256 index) internal view returns (bytes32) {
        return _at(set._inner, index);
    }

        function values(Bytes32Set storage set) internal view returns (bytes32[] memory) {
        bytes32[] memory store = _values(set._inner);
        bytes32[] memory result;

        assembly {
            result := store
        }

        return result;
    }

    struct AddressSet {
        Set _inner;
    }

        function add(AddressSet storage set, address value) internal returns (bool) {
        return _add(set._inner, bytes32(uint256(uint160(value))));
    }

        function remove(AddressSet storage set, address value) internal returns (bool) {
        return _remove(set._inner, bytes32(uint256(uint160(value))));
    }

        function contains(AddressSet storage set, address value) internal view returns (bool) {
        return _contains(set._inner, bytes32(uint256(uint160(value))));
    }

        function length(AddressSet storage set) internal view returns (uint256) {
        return _length(set._inner);
    }

        function at(AddressSet storage set, uint256 index) internal view returns (address) {
        return address(uint160(uint256(_at(set._inner, index))));
    }

        function values(AddressSet storage set) internal view returns (address[] memory) {
        bytes32[] memory store = _values(set._inner);
        address[] memory result;

        assembly {
            result := store
        }

        return result;
    }

    struct UintSet {
        Set _inner;
    }

        function add(UintSet storage set, uint256 value) internal returns (bool) {
        return _add(set._inner, bytes32(value));
    }

        function remove(UintSet storage set, uint256 value) internal returns (bool) {
        return _remove(set._inner, bytes32(value));
    }

        function contains(UintSet storage set, uint256 value) internal view returns (bool) {
        return _contains(set._inner, bytes32(value));
    }

        function length(UintSet storage set) internal view returns (uint256) {
        return _length(set._inner);
    }

        function at(UintSet storage set, uint256 index) internal view returns (uint256) {
        return uint256(_at(set._inner, index));
    }

        function values(UintSet storage set) internal view returns (uint256[] memory) {
        bytes32[] memory store = _values(set._inner);
        uint256[] memory result;

        assembly {
            result := store
        }

        return result;
    }
}

pragma solidity ^0.8.1;

library AddressUpgradeable {
        function isContract(address account) internal view returns (bool) {

        return account.code.length > 0;
    }

        function sendValue(address payable recipient, uint256 amount) internal {
        require(address(this).balance >= amount, "Address: insufficient balance");

        (bool success, ) = recipient.call{value: amount}("");
        require(success, "Address: unable to send value, recipient may have reverted");
    }

        function functionCall(address target, bytes memory data) internal returns (bytes memory) {
        return functionCallWithValue(target, data, 0, "Address: low-level call failed");
    }

        function functionCall(
        address target,
        bytes memory data,
        string memory errorMessage
    ) internal returns (bytes memory) {
        return functionCallWithValue(target, data, 0, errorMessage);
    }

        function functionCallWithValue(
        address target,
        bytes memory data,
        uint256 value
    ) internal returns (bytes memory) {
        return functionCallWithValue(target, data, value, "Address: low-level call with value failed");
    }

        function functionCallWithValue(
        address target,
        bytes memory data,
        uint256 value,
        string memory errorMessage
    ) internal returns (bytes memory) {
        require(address(this).balance >= value, "Address: insufficient balance for call");
        (bool success, bytes memory returndata) = target.call{value: value}(data);
        return verifyCallResultFromTarget(target, success, returndata, errorMessage);
    }

        function functionStaticCall(address target, bytes memory data) internal view returns (bytes memory) {
        return functionStaticCall(target, data, "Address: low-level static call failed");
    }

        function functionStaticCall(
        address target,
        bytes memory data,
        string memory errorMessage
    ) internal view returns (bytes memory) {
        (bool success, bytes memory returndata) = target.staticcall(data);
        return verifyCallResultFromTarget(target, success, returndata, errorMessage);
    }

        function verifyCallResultFromTarget(
        address target,
        bool success,
        bytes memory returndata,
        string memory errorMessage
    ) internal view returns (bytes memory) {
        if (success) {
            if (returndata.length == 0) {

                require(isContract(target), "Address: call to non-contract");
            }
            return returndata;
        } else {
            _revert(returndata, errorMessage);
        }
    }

        function verifyCallResult(
        bool success,
        bytes memory returndata,
        string memory errorMessage
    ) internal pure returns (bytes memory) {
        if (success) {
            return returndata;
        } else {
            _revert(returndata, errorMessage);
        }
    }

    function _revert(bytes memory returndata, string memory errorMessage) private pure {

        if (returndata.length > 0) {

            assembly {
                let returndata_size := mload(returndata)
                revert(add(32, returndata), returndata_size)
            }
        } else {
            revert(errorMessage);
        }
    }
}

pragma solidity ^0.8.2;

abstract contract Initializable {
        uint8 private _initialized;

        bool private _initializing;

        event Initialized(uint8 version);

        modifier initializer() {
        bool isTopLevelCall = !_initializing;
        require(
            (isTopLevelCall && _initialized < 1) || (!AddressUpgradeable.isContract(address(this)) && _initialized == 1),
            "Initializable: contract is already initialized"
        );
        _initialized = 1;
        if (isTopLevelCall) {
            _initializing = true;
        }
        _;
        if (isTopLevelCall) {
            _initializing = false;
            emit Initialized(1);
        }
    }

        modifier reinitializer(uint8 version) {
        require(!_initializing && _initialized < version, "Initializable: contract is already initialized");
        _initialized = version;
        _initializing = true;
        _;
        _initializing = false;
        emit Initialized(version);
    }

        modifier onlyInitializing() {
        require(_initializing, "Initializable: contract is not initializing");
        _;
    }

        function _disableInitializers() internal virtual {
        require(!_initializing, "Initializable: contract is initializing");
        if (_initialized < type(uint8).max) {
            _initialized = type(uint8).max;
            emit Initialized(type(uint8).max);
        }
    }

        function _getInitializedVersion() internal view returns (uint8) {
        return _initialized;
    }

        function _isInitializing() internal view returns (bool) {
        return _initializing;
    }
}

pragma solidity ^0.8.0;

abstract contract ContextUpgradeable is Initializable {
    function __Context_init() internal onlyInitializing {
    }

    function __Context_init_unchained() internal onlyInitializing {
    }
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }

        uint256[50] private __gap;
}

pragma solidity ^0.8.0;

abstract contract OwnableUpgradeable is Initializable, ContextUpgradeable {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

        function __Ownable_init() internal onlyInitializing {
        __Ownable_init_unchained();
    }

    function __Ownable_init_unchained() internal onlyInitializing {
        _transferOwnership(_msgSender());
    }

        modifier onlyOwner() {
        _checkOwner();
        _;
    }

        function owner() public view virtual returns (address) {
        return _owner;
    }

        function _checkOwner() internal view virtual {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
    }

        function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

        function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        _transferOwnership(newOwner);
    }

        function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }

        uint256[49] private __gap;
}

pragma solidity ^0.8.0;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}

pragma solidity ^0.8.0;

abstract contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

        constructor() {
        _transferOwnership(_msgSender());
    }

        modifier onlyOwner() {
        _checkOwner();
        _;
    }

        function owner() public view virtual returns (address) {
        return _owner;
    }

        function _checkOwner() internal view virtual {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
    }

        function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

        function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        _transferOwnership(newOwner);
    }

        function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

pragma solidity ^0.8.1;

library Address {
        function isContract(address account) internal view returns (bool) {

        return account.code.length > 0;
    }

        function sendValue(address payable recipient, uint256 amount) internal {
        require(address(this).balance >= amount, "Address: insufficient balance");

        (bool success, ) = recipient.call{value: amount}("");
        require(success, "Address: unable to send value, recipient may have reverted");
    }

        function functionCall(address target, bytes memory data) internal returns (bytes memory) {
        return functionCallWithValue(target, data, 0, "Address: low-level call failed");
    }

        function functionCall(
        address target,
        bytes memory data,
        string memory errorMessage
    ) internal returns (bytes memory) {
        return functionCallWithValue(target, data, 0, errorMessage);
    }

        function functionCallWithValue(
        address target,
        bytes memory data,
        uint256 value
    ) internal returns (bytes memory) {
        return functionCallWithValue(target, data, value, "Address: low-level call with value failed");
    }

        function functionCallWithValue(
        address target,
        bytes memory data,
        uint256 value,
        string memory errorMessage
    ) internal returns (bytes memory) {
        require(address(this).balance >= value, "Address: insufficient balance for call");
        (bool success, bytes memory returndata) = target.call{value: value}(data);
        return verifyCallResultFromTarget(target, success, returndata, errorMessage);
    }

        function functionStaticCall(address target, bytes memory data) internal view returns (bytes memory) {
        return functionStaticCall(target, data, "Address: low-level static call failed");
    }

        function functionStaticCall(
        address target,
        bytes memory data,
        string memory errorMessage
    ) internal view returns (bytes memory) {
        (bool success, bytes memory returndata) = target.staticcall(data);
        return verifyCallResultFromTarget(target, success, returndata, errorMessage);
    }

        function functionDelegateCall(address target, bytes memory data) internal returns (bytes memory) {
        return functionDelegateCall(target, data, "Address: low-level delegate call failed");
    }

        function functionDelegateCall(
        address target,
        bytes memory data,
        string memory errorMessage
    ) internal returns (bytes memory) {
        (bool success, bytes memory returndata) = target.delegatecall(data);
        return verifyCallResultFromTarget(target, success, returndata, errorMessage);
    }

        function verifyCallResultFromTarget(
        address target,
        bool success,
        bytes memory returndata,
        string memory errorMessage
    ) internal view returns (bytes memory) {
        if (success) {
            if (returndata.length == 0) {

                require(isContract(target), "Address: call to non-contract");
            }
            return returndata;
        } else {
            _revert(returndata, errorMessage);
        }
    }

        function verifyCallResult(
        bool success,
        bytes memory returndata,
        string memory errorMessage
    ) internal pure returns (bytes memory) {
        if (success) {
            return returndata;
        } else {
            _revert(returndata, errorMessage);
        }
    }

    function _revert(bytes memory returndata, string memory errorMessage) private pure {

        if (returndata.length > 0) {

            assembly {
                let returndata_size := mload(returndata)
                revert(add(32, returndata), returndata_size)
            }
        } else {
            revert(errorMessage);
        }
    }
}

pragma solidity ^0.8.0;

interface IERC20Permit {
        function permit(
        address owner,
        address spender,
        uint256 value,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external;

        function nonces(address owner) external view returns (uint256);

    function DOMAIN_SEPARATOR() external view returns (bytes32);
}

pragma solidity ^0.8.0;

interface IERC20 {
        event Transfer(address indexed from, address indexed to, uint256 value);

        event Approval(address indexed owner, address indexed spender, uint256 value);

        function totalSupply() external view returns (uint256);

        function balanceOf(address account) external view returns (uint256);

        function transfer(address to, uint256 amount) external returns (bool);

        function allowance(address owner, address spender) external view returns (uint256);

        function approve(address spender, uint256 amount) external returns (bool);

        function transferFrom(
        address from,
        address to,
        uint256 amount
    ) external returns (bool);
}

pragma solidity ^0.8.0;

library SafeERC20 {
    using Address for address;

    function safeTransfer(
        IERC20 token,
        address to,
        uint256 value
    ) internal {
        _callOptionalReturn(token, abi.encodeWithSelector(token.transfer.selector, to, value));
    }

    function safeTransferFrom(
        IERC20 token,
        address from,
        address to,
        uint256 value
    ) internal {
        _callOptionalReturn(token, abi.encodeWithSelector(token.transferFrom.selector, from, to, value));
    }

        function safeApprove(
        IERC20 token,
        address spender,
        uint256 value
    ) internal {

        require(
            (value == 0) || (token.allowance(address(this), spender) == 0),
            "SafeERC20: approve from non-zero to non-zero allowance"
        );
        _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, value));
    }

    function safeIncreaseAllowance(
        IERC20 token,
        address spender,
        uint256 value
    ) internal {
        uint256 newAllowance = token.allowance(address(this), spender) + value;
        _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, newAllowance));
    }

    function safeDecreaseAllowance(
        IERC20 token,
        address spender,
        uint256 value
    ) internal {
        unchecked {
            uint256 oldAllowance = token.allowance(address(this), spender);
            require(oldAllowance >= value, "SafeERC20: decreased allowance below zero");
            uint256 newAllowance = oldAllowance - value;
            _callOptionalReturn(token, abi.encodeWithSelector(token.approve.selector, spender, newAllowance));
        }
    }

    function safePermit(
        IERC20Permit token,
        address owner,
        address spender,
        uint256 value,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) internal {
        uint256 nonceBefore = token.nonces(owner);
        token.permit(owner, spender, value, deadline, v, r, s);
        uint256 nonceAfter = token.nonces(owner);
        require(nonceAfter == nonceBefore + 1, "SafeERC20: permit did not succeed");
    }

        function _callOptionalReturn(IERC20 token, bytes memory data) private {

        bytes memory returndata = address(token).functionCall(data, "SafeERC20: low-level call failed");
        if (returndata.length > 0) {

            require(abi.decode(returndata, (bool)), "SafeERC20: ERC20 operation did not succeed");
        }
    }
}

pragma solidity ^0.8.0;

contract LiquidityLocker is OwnableUpgradeable {
    using SafeERC20 for IERC20;
    using EnumerableSet for EnumerableSet.UintSet;

    struct Lock {
        address owner;
        IERC20 token;
        uint256 amount;
        uint256 releaseTime;
        bool withdrawn;
    }

    uint256 private lockCount;
    EnumerableSet.UintSet private lockSet;
    mapping(uint256 => Lock) public allLocks;

    uint256 public fee;
    address public feeAddress;
    mapping(address => mapping(IERC20 => Lock[])) public locks;
    mapping(address => bool) private whitelisted;

    event Deposit(address indexed owner, IERC20 token, uint256 amount, uint256 releaseTime);
    event Withdraw(address indexed owner, IERC20 token, uint256 amount, uint256 unlockTime);
    event Extend(address owner, IERC20 token, uint256 lockTime);

    function initialize(address _feeAddress, uint256 _fee) public virtual initializer {
        __Ownable_init();
        feeAddress = _feeAddress;
        fee = _fee;
    }

    function deposit(IERC20 _token, uint256 _amount, uint256 _releaseTime) public payable {
        require(_token.transferFrom(msg.sender, address(this), _amount), "Transfer failed.");
        require(msg.value >= fee || whitelisted[msg.sender], "Fee must be paid.");
        require(_releaseTime > block.timestamp, "New lock time must be in the future");

        Lock memory _lock = Lock(msg.sender, _token, _amount, _releaseTime, false);
        locks[msg.sender][_token].push(_lock);
        lockSet.add(lockCount);
        allLocks[lockCount] = _lock;

        lockCount = lockCount + 1;
        payable(feeAddress).transfer(msg.value);
        emit Deposit(msg.sender, _token, _amount, _releaseTime);
    }

    function withdraw(IERC20 _token, uint256 _lockId) public {
        Lock storage lock = locks[msg.sender][_token][_lockId];

        require(block.timestamp >= lock.releaseTime, "Lock has not expired yet.");
        require(lock.owner == msg.sender, "Sender must be the owner of the lock.");

        uint256 _amount = lock.amount;
        lock.amount = 0;
        lock.withdrawn = true;
        lock.token.transfer(lock.owner, _amount);

        emit Withdraw(msg.sender, lock.token, lock.amount, block.timestamp);
    }

    function extendLock(IERC20 _token, uint256 _lockId, uint256 _newLockTime) public {
        Lock storage lock = locks[msg.sender][_token][_lockId];

        require(!lock.withdrawn, "Lock has been terminated");
        require(lock.owner == msg.sender, "Only the lock owner can extend the lock");
        require(_newLockTime > block.timestamp, "New lock time must be in the future");

        lock.releaseTime = _newLockTime;
        emit Extend(lock.owner, _token, lock.releaseTime);
    }

    function setFeeAddress(address _newFeeAddress) public onlyOwner {
        feeAddress = _newFeeAddress;
    }

    function setFee(uint256 _newFeeAmount) public onlyOwner {
        fee = _newFeeAmount;
    }

    function whitelistFromFee(address _whitelistedAddress, bool _isWhitelisted) public onlyOwner  {
        whitelisted[_whitelistedAddress] = _isWhitelisted;
    }

    function getLocks(address _owner, IERC20 _token) public view returns (Lock[] memory) {
        return locks[_owner][_token];
    }

    function getAllLocks() public view returns (Lock[] memory) {
        Lock[] memory result = new Lock[](lockSet.length());
        for (uint256 i = 0; i < lockSet.length(); i++) {
            result[i] = allLocks[lockSet.at(i)];
        }
        return result;
    }
}
