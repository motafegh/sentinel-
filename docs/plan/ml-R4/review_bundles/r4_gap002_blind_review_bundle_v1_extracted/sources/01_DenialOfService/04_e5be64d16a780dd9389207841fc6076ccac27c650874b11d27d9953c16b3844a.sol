pragma solidity 0.8.16;

abstract contract ERC20 {

    event Transfer(address indexed from, address indexed to, uint256 amount);

    event Approval(address indexed owner, address indexed spender, uint256 amount);

    string public name;

    string public symbol;

    uint8 public decimals;

    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;

    mapping(address => mapping(address => uint256)) public allowance;

    uint256 internal INITIAL_CHAIN_ID;

    bytes32 internal INITIAL_DOMAIN_SEPARATOR;

    mapping(address => uint256) public nonces;

    constructor(string memory _name, string memory _symbol, uint8 _decimals) {
        name = _name;
        symbol = _symbol;
        decimals = _decimals;

        INITIAL_CHAIN_ID = block.chainid;
        INITIAL_DOMAIN_SEPARATOR = computeDomainSeparator();
    }

    function approve(address spender, uint256 amount) public virtual returns (bool) {
        allowance[msg.sender][spender] = amount;

        emit Approval(msg.sender, spender, amount);

        return true;
    }

    function transfer(address to, uint256 amount) public virtual returns (bool) {
        balanceOf[msg.sender] -= amount;

        unchecked {
            balanceOf[to] += amount;
        }

        emit Transfer(msg.sender, to, amount);

        return true;
    }

    function transferFrom(address from, address to, uint256 amount) public virtual returns (bool) {
        uint256 allowed = allowance[from][msg.sender];

        if (allowed != type(uint256).max) allowance[from][msg.sender] = allowed - amount;

        balanceOf[from] -= amount;

        unchecked {
            balanceOf[to] += amount;
        }

        emit Transfer(from, to, amount);

        return true;
    }

    function permit(
        address owner,
        address spender,
        uint256 value,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) public virtual {
        require(deadline >= block.timestamp, "PERMIT_DEADLINE_EXPIRED");

        unchecked {
            address recoveredAddress = ecrecover(
                keccak256(
                    abi.encodePacked(
                        "\x19\x01",
                        DOMAIN_SEPARATOR(),
                        keccak256(
                            abi.encode(
                                keccak256(
                                    "Permit(address owner,address spender,uint256 value,uint256 nonce,uint256 deadline)"
                                ),
                                owner,
                                spender,
                                value,
                                nonces[owner]++,
                                deadline
                            )
                        )
                    )
                ),
                v,
                r,
                s
            );

            require(recoveredAddress != address(0) && recoveredAddress == owner, "INVALID_SIGNER");

            allowance[recoveredAddress][spender] = value;
        }

        emit Approval(owner, spender, value);
    }

    function DOMAIN_SEPARATOR() public view virtual returns (bytes32) {
        return block.chainid == INITIAL_CHAIN_ID ? INITIAL_DOMAIN_SEPARATOR : computeDomainSeparator();
    }

    function computeDomainSeparator() internal view virtual returns (bytes32) {
        return
            keccak256(
                abi.encode(
                    keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"),
                    keccak256(bytes(name)),
                    keccak256("1"),
                    block.chainid,
                    address(this)
                )
            );
    }

    function _mint(address to, uint256 amount) internal virtual {
        totalSupply += amount;

        unchecked {
            balanceOf[to] += amount;
        }

        emit Transfer(address(0), to, amount);
    }

    function _burn(address from, uint256 amount) internal virtual {
        balanceOf[from] -= amount;

        unchecked {
            totalSupply -= amount;
        }

        emit Transfer(from, address(0), amount);
    }
}

library SafeTransferLib {

    function safeTransferETH(address to, uint256 amount) internal {
        bool success;

        assembly {

            success := call(gas(), to, amount, 0, 0, 0, 0)
        }

        require(success, "ETH_TRANSFER_FAILED");
    }

    function safeTransferFrom(ERC20 token, address from, address to, uint256 amount) internal {
        bool success;

        assembly {

            let freeMemoryPointer := mload(0x40)

            mstore(freeMemoryPointer, 0x23b872dd00000000000000000000000000000000000000000000000000000000)
            mstore(add(freeMemoryPointer, 4), from)
            mstore(add(freeMemoryPointer, 36), to)
            mstore(add(freeMemoryPointer, 68), amount)

            success := and(

                or(and(eq(mload(0), 1), gt(returndatasize(), 31)), iszero(returndatasize())),

                call(gas(), token, 0, freeMemoryPointer, 100, 0, 32)
            )
        }

        require(success, "TRANSFER_FROM_FAILED");
    }

    function safeTransfer(ERC20 token, address to, uint256 amount) internal {
        bool success;

        assembly {

            let freeMemoryPointer := mload(0x40)

            mstore(freeMemoryPointer, 0xa9059cbb00000000000000000000000000000000000000000000000000000000)
            mstore(add(freeMemoryPointer, 4), to)
            mstore(add(freeMemoryPointer, 36), amount)

            success := and(

                or(and(eq(mload(0), 1), gt(returndatasize(), 31)), iszero(returndatasize())),

                call(gas(), token, 0, freeMemoryPointer, 68, 0, 32)
            )
        }

        require(success, "TRANSFER_FAILED");
    }

    function safeApprove(ERC20 token, address to, uint256 amount) internal {
        bool success;

        assembly {

            let freeMemoryPointer := mload(0x40)

            mstore(freeMemoryPointer, 0x095ea7b300000000000000000000000000000000000000000000000000000000)
            mstore(add(freeMemoryPointer, 4), to)
            mstore(add(freeMemoryPointer, 36), amount)

            success := and(

                or(and(eq(mload(0), 1), gt(returndatasize(), 31)), iszero(returndatasize())),

                call(gas(), token, 0, freeMemoryPointer, 68, 0, 32)
            )
        }

        require(success, "APPROVE_FAILED");
    }
}

library Math {

    function subMinZero(uint256 x, uint256 y) internal pure returns (uint256) {
        return x > y ? x - y : 0;
    }

    function changeDecimals(uint256 amount, uint8 fromDecimals, uint8 toDecimals) internal pure returns (uint256) {
        if (fromDecimals == toDecimals) {
            return amount;
        } else if (fromDecimals < toDecimals) {
            return amount * 10 ** (toDecimals - fromDecimals);
        } else {
            return amount / 10 ** (fromDecimals - toDecimals);
        }
    }

    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }

    uint256 public constant WAD = 1e18;

    function mulWadDown(uint256 x, uint256 y) internal pure returns (uint256) {
        return mulDivDown(x, y, WAD);
    }

    function mulDivDown(uint256 x, uint256 y, uint256 denominator) internal pure returns (uint256 z) {
        assembly {

            z := mul(x, y)

            if iszero(and(iszero(iszero(denominator)), or(iszero(x), eq(div(z, x), y)))) {
                revert(0, 0)
            }

            z := div(z, denominator)
        }
    }

    function mulDivUp(uint256 x, uint256 y, uint256 denominator) internal pure returns (uint256 z) {
        assembly {

            z := mul(x, y)

            if iszero(and(iszero(iszero(denominator)), or(iszero(x), eq(div(z, x), y)))) {
                revert(0, 0)
            }

            z := mul(iszero(iszero(z)), add(div(sub(z, 1), denominator), 1))
        }
    }
}

abstract contract ERC4626 is ERC20 {
    using SafeTransferLib for ERC20;
    using Math for uint256;

    event Deposit(address indexed caller, address indexed owner, uint256 assets, uint256 shares);

    event Withdraw(
        address indexed caller,
        address indexed receiver,
        address indexed owner,
        uint256 assets,
        uint256 shares
    );

    ERC20 public asset;

    constructor(
        ERC20 _asset,
        string memory _name,
        string memory _symbol,
        uint8 _decimals
    ) ERC20(_name, _symbol, _decimals) {
        asset = _asset;
    }

    function deposit(uint256 assets, address receiver) public virtual returns (uint256 shares) {

        require((shares = previewDeposit(assets)) != 0, "ZERO_SHARES");

        beforeDeposit(assets, shares, receiver);

        asset.safeTransferFrom(msg.sender, address(this), assets);

        _mint(receiver, shares);

        emit Deposit(msg.sender, receiver, assets, shares);

        afterDeposit(assets, shares, receiver);
    }

    function mint(uint256 shares, address receiver) public virtual returns (uint256 assets) {
        assets = previewMint(shares);

        beforeDeposit(assets, shares, receiver);

        asset.safeTransferFrom(msg.sender, address(this), assets);

        _mint(receiver, shares);

        emit Deposit(msg.sender, receiver, assets, shares);

        afterDeposit(assets, shares, receiver);
    }

    function withdraw(uint256 assets, address receiver, address owner) public virtual returns (uint256 shares) {
        shares = previewWithdraw(assets);

        if (msg.sender != owner) {
            uint256 allowed = allowance[owner][msg.sender];

            if (allowed != type(uint256).max) allowance[owner][msg.sender] = allowed - shares;
        }

        beforeWithdraw(assets, shares, receiver, owner);

        _burn(owner, shares);

        emit Withdraw(msg.sender, receiver, owner, assets, shares);

        asset.safeTransfer(receiver, assets);

        afterWithdraw(assets, shares, receiver, owner);
    }

    function redeem(uint256 shares, address receiver, address owner) public virtual returns (uint256 assets) {
        if (msg.sender != owner) {
            uint256 allowed = allowance[owner][msg.sender];

            if (allowed != type(uint256).max) allowance[owner][msg.sender] = allowed - shares;
        }

        require((assets = previewRedeem(shares)) != 0, "ZERO_ASSETS");

        beforeWithdraw(assets, shares, receiver, owner);

        _burn(owner, shares);

        emit Withdraw(msg.sender, receiver, owner, assets, shares);

        asset.safeTransfer(receiver, assets);

        afterWithdraw(assets, shares, receiver, owner);
    }

    function totalAssets() public view virtual returns (uint256);

    function convertToShares(uint256 assets) public view virtual returns (uint256) {
        uint256 supply = totalSupply;

        return supply == 0 ? assets : assets.mulDivDown(supply, totalAssets());
    }

    function convertToAssets(uint256 shares) public view virtual returns (uint256) {
        uint256 supply = totalSupply;

        return supply == 0 ? shares : shares.mulDivDown(totalAssets(), supply);
    }

    function previewDeposit(uint256 assets) public view virtual returns (uint256) {
        return convertToShares(assets);
    }

    function previewMint(uint256 shares) public view virtual returns (uint256) {
        uint256 supply = totalSupply;

        return supply == 0 ? shares : shares.mulDivUp(totalAssets(), supply);
    }

    function previewWithdraw(uint256 assets) public view virtual returns (uint256) {
        uint256 supply = totalSupply;

        return supply == 0 ? assets : assets.mulDivUp(supply, totalAssets());
    }

    function previewRedeem(uint256 shares) public view virtual returns (uint256) {
        return convertToAssets(shares);
    }

    function maxDeposit(address) public view virtual returns (uint256) {
        return type(uint256).max;
    }

    function maxMint(address) public view virtual returns (uint256) {
        return type(uint256).max;
    }

    function maxWithdraw(address owner) public view virtual returns (uint256) {
        return convertToAssets(balanceOf[owner]);
    }

    function maxRedeem(address owner) public view virtual returns (uint256) {
        return balanceOf[owner];
    }

    function beforeDeposit(uint256 assets, uint256 shares, address receiver) internal virtual {}

    function afterDeposit(uint256 assets, uint256 shares, address receiver) internal virtual {}

    function beforeWithdraw(uint256 assets, uint256 shares, address receiver, address owner) internal virtual {}

    function afterWithdraw(uint256 assets, uint256 shares, address receiver, address owner) internal virtual {}
}

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}

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

interface AutomationCompatibleInterface {

    function checkUpkeep(bytes calldata checkData) external returns (bool upkeepNeeded, bytes memory performData);

    function performUpkeep(bytes calldata performData) external;
}

interface AggregatorInterface {
    function latestAnswer() external view returns (int256);

    function latestTimestamp() external view returns (uint256);

    function latestRound() external view returns (uint256);

    function getAnswer(uint256 roundId) external view returns (int256);

    function getTimestamp(uint256 roundId) external view returns (uint256);

    event AnswerUpdated(int256 indexed current, uint256 indexed roundId, uint256 updatedAt);

    event NewRound(uint256 indexed roundId, address indexed startedBy, uint256 startedAt);
}

interface AggregatorV3Interface {
    function decimals() external view returns (uint8);

    function description() external view returns (string memory);

    function version() external view returns (uint256);

    function getRoundData(
        uint80 _roundId
    )
        external
        view
        returns (uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound);

    function latestRoundData()
        external
        view
        returns (uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound);
}

interface AggregatorV2V3Interface is AggregatorInterface, AggregatorV3Interface {}

interface IChainlinkAggregator is AggregatorV2V3Interface {
    function maxAnswer() external view returns (int192);

    function minAnswer() external view returns (int192);

    function aggregator() external view returns (address);
}

library SafeCast {

    function toUint248(uint256 value) internal pure returns (uint248) {
        require(value <= type(uint248).max, "SafeCast: value doesn't fit in 248 bits");
        return uint248(value);
    }

    function toUint240(uint256 value) internal pure returns (uint240) {
        require(value <= type(uint240).max, "SafeCast: value doesn't fit in 240 bits");
        return uint240(value);
    }

    function toUint232(uint256 value) internal pure returns (uint232) {
        require(value <= type(uint232).max, "SafeCast: value doesn't fit in 232 bits");
        return uint232(value);
    }

    function toUint224(uint256 value) internal pure returns (uint224) {
        require(value <= type(uint224).max, "SafeCast: value doesn't fit in 224 bits");
        return uint224(value);
    }

    function toUint216(uint256 value) internal pure returns (uint216) {
        require(value <= type(uint216).max, "SafeCast: value doesn't fit in 216 bits");
        return uint216(value);
    }

    function toUint208(uint256 value) internal pure returns (uint208) {
        require(value <= type(uint208).max, "SafeCast: value doesn't fit in 208 bits");
        return uint208(value);
    }

    function toUint200(uint256 value) internal pure returns (uint200) {
        require(value <= type(uint200).max, "SafeCast: value doesn't fit in 200 bits");
        return uint200(value);
    }

    function toUint192(uint256 value) internal pure returns (uint192) {
        require(value <= type(uint192).max, "SafeCast: value doesn't fit in 192 bits");
        return uint192(value);
    }

    function toUint184(uint256 value) internal pure returns (uint184) {
        require(value <= type(uint184).max, "SafeCast: value doesn't fit in 184 bits");
        return uint184(value);
    }

    function toUint176(uint256 value) internal pure returns (uint176) {
        require(value <= type(uint176).max, "SafeCast: value doesn't fit in 176 bits");
        return uint176(value);
    }

    function toUint168(uint256 value) internal pure returns (uint168) {
        require(value <= type(uint168).max, "SafeCast: value doesn't fit in 168 bits");
        return uint168(value);
    }

    function toUint160(uint256 value) internal pure returns (uint160) {
        require(value <= type(uint160).max, "SafeCast: value doesn't fit in 160 bits");
        return uint160(value);
    }

    function toUint152(uint256 value) internal pure returns (uint152) {
        require(value <= type(uint152).max, "SafeCast: value doesn't fit in 152 bits");
        return uint152(value);
    }

    function toUint144(uint256 value) internal pure returns (uint144) {
        require(value <= type(uint144).max, "SafeCast: value doesn't fit in 144 bits");
        return uint144(value);
    }

    function toUint136(uint256 value) internal pure returns (uint136) {
        require(value <= type(uint136).max, "SafeCast: value doesn't fit in 136 bits");
        return uint136(value);
    }

    function toUint128(uint256 value) internal pure returns (uint128) {
        require(value <= type(uint128).max, "SafeCast: value doesn't fit in 128 bits");
        return uint128(value);
    }

    function toUint120(uint256 value) internal pure returns (uint120) {
        require(value <= type(uint120).max, "SafeCast: value doesn't fit in 120 bits");
        return uint120(value);
    }

    function toUint112(uint256 value) internal pure returns (uint112) {
        require(value <= type(uint112).max, "SafeCast: value doesn't fit in 112 bits");
        return uint112(value);
    }

    function toUint104(uint256 value) internal pure returns (uint104) {
        require(value <= type(uint104).max, "SafeCast: value doesn't fit in 104 bits");
        return uint104(value);
    }

    function toUint96(uint256 value) internal pure returns (uint96) {
        require(value <= type(uint96).max, "SafeCast: value doesn't fit in 96 bits");
        return uint96(value);
    }

    function toUint88(uint256 value) internal pure returns (uint88) {
        require(value <= type(uint88).max, "SafeCast: value doesn't fit in 88 bits");
        return uint88(value);
    }

    function toUint80(uint256 value) internal pure returns (uint80) {
        require(value <= type(uint80).max, "SafeCast: value doesn't fit in 80 bits");
        return uint80(value);
    }

    function toUint72(uint256 value) internal pure returns (uint72) {
        require(value <= type(uint72).max, "SafeCast: value doesn't fit in 72 bits");
        return uint72(value);
    }

    function toUint64(uint256 value) internal pure returns (uint64) {
        require(value <= type(uint64).max, "SafeCast: value doesn't fit in 64 bits");
        return uint64(value);
    }

    function toUint56(uint256 value) internal pure returns (uint56) {
        require(value <= type(uint56).max, "SafeCast: value doesn't fit in 56 bits");
        return uint56(value);
    }

    function toUint48(uint256 value) internal pure returns (uint48) {
        require(value <= type(uint48).max, "SafeCast: value doesn't fit in 48 bits");
        return uint48(value);
    }

    function toUint40(uint256 value) internal pure returns (uint40) {
        require(value <= type(uint40).max, "SafeCast: value doesn't fit in 40 bits");
        return uint40(value);
    }

    function toUint32(uint256 value) internal pure returns (uint32) {
        require(value <= type(uint32).max, "SafeCast: value doesn't fit in 32 bits");
        return uint32(value);
    }

    function toUint24(uint256 value) internal pure returns (uint24) {
        require(value <= type(uint24).max, "SafeCast: value doesn't fit in 24 bits");
        return uint24(value);
    }

    function toUint16(uint256 value) internal pure returns (uint16) {
        require(value <= type(uint16).max, "SafeCast: value doesn't fit in 16 bits");
        return uint16(value);
    }

    function toUint8(uint256 value) internal pure returns (uint8) {
        require(value <= type(uint8).max, "SafeCast: value doesn't fit in 8 bits");
        return uint8(value);
    }

    function toUint256(int256 value) internal pure returns (uint256) {
        require(value >= 0, "SafeCast: value must be positive");
        return uint256(value);
    }

    function toInt248(int256 value) internal pure returns (int248 downcasted) {
        downcasted = int248(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 248 bits");
    }

    function toInt240(int256 value) internal pure returns (int240 downcasted) {
        downcasted = int240(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 240 bits");
    }

    function toInt232(int256 value) internal pure returns (int232 downcasted) {
        downcasted = int232(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 232 bits");
    }

    function toInt224(int256 value) internal pure returns (int224 downcasted) {
        downcasted = int224(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 224 bits");
    }

    function toInt216(int256 value) internal pure returns (int216 downcasted) {
        downcasted = int216(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 216 bits");
    }

    function toInt208(int256 value) internal pure returns (int208 downcasted) {
        downcasted = int208(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 208 bits");
    }

    function toInt200(int256 value) internal pure returns (int200 downcasted) {
        downcasted = int200(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 200 bits");
    }

    function toInt192(int256 value) internal pure returns (int192 downcasted) {
        downcasted = int192(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 192 bits");
    }

    function toInt184(int256 value) internal pure returns (int184 downcasted) {
        downcasted = int184(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 184 bits");
    }

    function toInt176(int256 value) internal pure returns (int176 downcasted) {
        downcasted = int176(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 176 bits");
    }

    function toInt168(int256 value) internal pure returns (int168 downcasted) {
        downcasted = int168(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 168 bits");
    }

    function toInt160(int256 value) internal pure returns (int160 downcasted) {
        downcasted = int160(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 160 bits");
    }

    function toInt152(int256 value) internal pure returns (int152 downcasted) {
        downcasted = int152(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 152 bits");
    }

    function toInt144(int256 value) internal pure returns (int144 downcasted) {
        downcasted = int144(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 144 bits");
    }

    function toInt136(int256 value) internal pure returns (int136 downcasted) {
        downcasted = int136(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 136 bits");
    }

    function toInt128(int256 value) internal pure returns (int128 downcasted) {
        downcasted = int128(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 128 bits");
    }

    function toInt120(int256 value) internal pure returns (int120 downcasted) {
        downcasted = int120(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 120 bits");
    }

    function toInt112(int256 value) internal pure returns (int112 downcasted) {
        downcasted = int112(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 112 bits");
    }

    function toInt104(int256 value) internal pure returns (int104 downcasted) {
        downcasted = int104(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 104 bits");
    }

    function toInt96(int256 value) internal pure returns (int96 downcasted) {
        downcasted = int96(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 96 bits");
    }

    function toInt88(int256 value) internal pure returns (int88 downcasted) {
        downcasted = int88(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 88 bits");
    }

    function toInt80(int256 value) internal pure returns (int80 downcasted) {
        downcasted = int80(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 80 bits");
    }

    function toInt72(int256 value) internal pure returns (int72 downcasted) {
        downcasted = int72(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 72 bits");
    }

    function toInt64(int256 value) internal pure returns (int64 downcasted) {
        downcasted = int64(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 64 bits");
    }

    function toInt56(int256 value) internal pure returns (int56 downcasted) {
        downcasted = int56(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 56 bits");
    }

    function toInt48(int256 value) internal pure returns (int48 downcasted) {
        downcasted = int48(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 48 bits");
    }

    function toInt40(int256 value) internal pure returns (int40 downcasted) {
        downcasted = int40(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 40 bits");
    }

    function toInt32(int256 value) internal pure returns (int32 downcasted) {
        downcasted = int32(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 32 bits");
    }

    function toInt24(int256 value) internal pure returns (int24 downcasted) {
        downcasted = int24(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 24 bits");
    }

    function toInt16(int256 value) internal pure returns (int16 downcasted) {
        downcasted = int16(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 16 bits");
    }

    function toInt8(int256 value) internal pure returns (int8 downcasted) {
        downcasted = int8(value);
        require(downcasted == value, "SafeCast: value doesn't fit in 8 bits");
    }

    function toInt256(uint256 value) internal pure returns (int256) {

        require(value <= uint256(type(int256).max), "SafeCast: value doesn't fit in an int256");
        return int256(value);
    }
}

library Address {

    function isContract(address account) internal view returns (bool) {

        return account.code.length > 0;
    }

    function sendValue(address payable recipient, uint256 amount) internal {
        require(address(this).balance >= amount, "Address: insufficient balance");

        (bool success, ) = recipient.call{ value: amount }("");
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

    function functionCallWithValue(address target, bytes memory data, uint256 value) internal returns (bytes memory) {
        return functionCallWithValue(target, data, value, "Address: low-level call with value failed");
    }

    function functionCallWithValue(
        address target,
        bytes memory data,
        uint256 value,
        string memory errorMessage
    ) internal returns (bytes memory) {
        require(address(this).balance >= value, "Address: insufficient balance for call");
        (bool success, bytes memory returndata) = target.call{ value: value }(data);
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

interface ICurvePool {
    function coins(uint256 i) external view returns (address);

    function get_virtual_price() external view returns (uint256);

    function claim_admin_fees() external;

    function withdraw_admin_fees() external;

    function gamma() external view returns (uint256);

    function A() external view returns (uint256);

    function lp_price() external view returns (uint256);

    function price_oracle() external view returns (uint256);

    function price_oracle(uint256 i) external view returns (uint256);
}

interface IERC20 {

    event Transfer(address indexed from, address indexed to, uint256 value);

    event Approval(address indexed owner, address indexed spender, uint256 value);

    function totalSupply() external view returns (uint256);

    function balanceOf(address account) external view returns (uint256);

    function transfer(address to, uint256 amount) external returns (bool);

    function allowance(address owner, address spender) external view returns (uint256);

    function approve(address spender, uint256 amount) external returns (bool);

    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

interface IAaveToken {
    function UNDERLYING_ASSET_ADDRESS() external view returns (address);
}

contract PriceRouter is Ownable, AutomationCompatibleInterface {
    using SafeTransferLib for ERC20;
    using SafeCast for int256;
    using Math for uint256;
    using Address for address;

    event AddAsset(address indexed asset);

    struct AssetSettings {
        uint8 derivative;
        address source;
    }

    mapping(ERC20 => AssetSettings) public getAssetSettings;

    error PriceRouter__InvalidMinPrice(uint256 minPrice, uint256 bufferedMinPrice);

    error PriceRouter__InvalidMaxPrice(uint256 maxPrice, uint256 bufferedMaxPrice);

    error PriceRouter__InvalidAsset(address asset);

    error PriceRouter__BadAnswer(uint256 answer, uint256 expectedAnswer);

    error PriceRouter__UnkownDerivative(uint8 unkownDerivative);

    error PriceRouter__MinPriceGreaterThanMaxPrice(uint256 min, uint256 max);

    uint256 public constant EXPECTED_ANSWER_DEVIATION = 0.02e18;

    struct PriceCache {
        address asset;
        uint96 price;
    }

    uint8 private constant PRICE_CACHE_SIZE = 8;

    function addAsset(
        ERC20 _asset,
        AssetSettings memory _settings,
        bytes memory _storage,
        uint256 _expectedAnswer
    ) external onlyOwner {
        if (address(_asset) == address(0)) revert PriceRouter__InvalidAsset(address(_asset));

        if (_settings.derivative == 0) revert PriceRouter__UnkownDerivative(_settings.derivative);

        if (_settings.derivative == 1) {
            _setupPriceForChainlinkDerivative(_asset, _settings.source, _storage);
        } else if (_settings.derivative == 2) {
            _setupPriceForCurveDerivative(_asset, _settings.source, _storage);
        } else if (_settings.derivative == 3) {
            _setupPriceForCurveV2Derivative(_asset, _settings.source, _storage);
        } else if (_settings.derivative == 4) {
            _setupPriceForAaveDerivative(_asset, _settings.source, _storage);
        } else revert PriceRouter__UnkownDerivative(_settings.derivative);

        uint256 minAnswer = _expectedAnswer.mulWadDown((1e18 - EXPECTED_ANSWER_DEVIATION));
        uint256 maxAnswer = _expectedAnswer.mulWadDown((1e18 + EXPECTED_ANSWER_DEVIATION));

        PriceCache[PRICE_CACHE_SIZE] memory cache;
        getAssetSettings[_asset] = _settings;
        uint256 answer = _getPriceInUSD(_asset, _settings, cache);
        if (answer < minAnswer || answer > maxAnswer) revert PriceRouter__BadAnswer(answer, _expectedAnswer);

        emit AddAsset(address(_asset));
    }

    function isSupported(ERC20 asset) external view returns (bool) {
        return getAssetSettings[asset].derivative > 0;
    }

    function checkUpkeep(bytes calldata checkData) external view returns (bool upkeepNeeded, bytes memory performData) {
        (uint8 derivative, bytes memory derivativeCheckData) = abi.decode(checkData, (uint8, bytes));

        if (derivative == 2) {
            (upkeepNeeded, performData) = _checkVirtualPriceBound(derivativeCheckData);
        } else if (derivative == 3) {
            (upkeepNeeded, performData) = _checkVirtualPriceBound(derivativeCheckData);
        } else revert PriceRouter__UnkownDerivative(derivative);
    }

    function performUpkeep(bytes calldata performData) external {
        (uint8 derivative, bytes memory derivativePerformData) = abi.decode(performData, (uint8, bytes));

        if (derivative == 2) {
            _updateVirtualPriceBound(derivativePerformData);
        } else if (derivative == 3) {
            _updateVirtualPriceBound(derivativePerformData);
        } else revert PriceRouter__UnkownDerivative(derivative);
    }

    function getPriceInUSD(ERC20 asset) external view returns (uint256) {
        AssetSettings memory assetSettings = getAssetSettings[asset];

        PriceCache[PRICE_CACHE_SIZE] memory cache;
        return _getPriceInUSD(asset, assetSettings, cache);
    }

    function getValue(ERC20 baseAsset, uint256 amount, ERC20 quoteAsset) external view returns (uint256 value) {
        AssetSettings memory baseSettings = getAssetSettings[baseAsset];
        AssetSettings memory quoteSettings = getAssetSettings[quoteAsset];
        if (baseSettings.derivative == 0) revert PriceRouter__UnsupportedAsset(address(baseAsset));
        if (quoteSettings.derivative == 0) revert PriceRouter__UnsupportedAsset(address(quoteAsset));
        PriceCache[PRICE_CACHE_SIZE] memory cache;
        uint256 priceBaseUSD = _getPriceInUSD(baseAsset, baseSettings, cache);
        uint256 priceQuoteUSD = _getPriceInUSD(quoteAsset, quoteSettings, cache);
        value = _getValueInQuote(priceBaseUSD, priceQuoteUSD, baseAsset.decimals(), quoteAsset.decimals(), amount);
    }

    function getValuesDelta(
        ERC20[] calldata baseAssets0,
        uint256[] calldata amounts0,
        ERC20[] calldata baseAssets1,
        uint256[] calldata amounts1,
        ERC20 quoteAsset
    ) external view returns (uint256) {

        PriceCache[PRICE_CACHE_SIZE] memory cache;

        uint256 value0 = _getValues(baseAssets0, amounts0, quoteAsset, cache);
        uint256 value1 = _getValues(baseAssets1, amounts1, quoteAsset, cache);
        return value0 - value1;
    }

    function getValues(
        ERC20[] calldata baseAssets,
        uint256[] calldata amounts,
        ERC20 quoteAsset
    ) external view returns (uint256) {

        PriceCache[PRICE_CACHE_SIZE] memory cache;

        return _getValues(baseAssets, amounts, quoteAsset, cache);
    }

    function getExchangeRate(ERC20 baseAsset, ERC20 quoteAsset) public view returns (uint256 exchangeRate) {
        AssetSettings memory baseSettings = getAssetSettings[baseAsset];
        AssetSettings memory quoteSettings = getAssetSettings[quoteAsset];
        if (baseSettings.derivative == 0) revert PriceRouter__UnsupportedAsset(address(baseAsset));
        if (quoteSettings.derivative == 0) revert PriceRouter__UnsupportedAsset(address(quoteAsset));

        PriceCache[PRICE_CACHE_SIZE] memory cache;

        exchangeRate = _getExchangeRate(
            baseAsset,
            baseSettings,
            quoteAsset,
            quoteSettings,
            quoteAsset.decimals(),
            cache
        );
    }

    function getExchangeRates(
        ERC20[] memory baseAssets,
        ERC20 quoteAsset
    ) external view returns (uint256[] memory exchangeRates) {
        uint8 quoteAssetDecimals = quoteAsset.decimals();
        AssetSettings memory quoteSettings = getAssetSettings[quoteAsset];
        if (quoteSettings.derivative == 0) revert PriceRouter__UnsupportedAsset(address(quoteAsset));

        PriceCache[PRICE_CACHE_SIZE] memory cache;

        uint256 numOfAssets = baseAssets.length;
        exchangeRates = new uint256[](numOfAssets);
        for (uint256 i; i < numOfAssets; i++) {
            AssetSettings memory baseSettings = getAssetSettings[baseAssets[i]];
            if (baseSettings.derivative == 0) revert PriceRouter__UnsupportedAsset(address(baseAssets[i]));
            exchangeRates[i] = _getExchangeRate(
                baseAssets[i],
                baseSettings,
                quoteAsset,
                quoteSettings,
                quoteAssetDecimals,
                cache
            );
        }
    }

    ERC20 private constant WETH = ERC20(0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2);

    error PriceRouter__UnsupportedAsset(address asset);

    function _getExchangeRate(
        ERC20 baseAsset,
        AssetSettings memory baseSettings,
        ERC20 quoteAsset,
        AssetSettings memory quoteSettings,
        uint8 quoteAssetDecimals,
        PriceCache[PRICE_CACHE_SIZE] memory cache
    ) internal view returns (uint256) {
        uint256 basePrice = _getPriceInUSD(baseAsset, baseSettings, cache);
        uint256 quotePrice = _getPriceInUSD(quoteAsset, quoteSettings, cache);
        uint256 exchangeRate = basePrice.mulDivDown(10 ** quoteAssetDecimals, quotePrice);
        return exchangeRate;
    }

    function _getPriceInUSD(
        ERC20 asset,
        AssetSettings memory settings,
        PriceCache[PRICE_CACHE_SIZE] memory cache
    ) internal view returns (uint256) {

        uint8 lastIndex = PRICE_CACHE_SIZE;
        for (uint8 i; i < PRICE_CACHE_SIZE; ++i) {

            if (cache[i].asset == address(0)) {

                lastIndex = i;
                break;
            }

            if (cache[i].asset == address(asset)) return cache[i].price;
        }

        uint256 price;
        if (settings.derivative == 1) {
            price = _getPriceForChainlinkDerivative(asset, settings.source, cache);
        } else if (settings.derivative == 2) {
            price = _getPriceForCurveDerivative(asset, settings.source, cache);
        } else if (settings.derivative == 3) {
            price = _getPriceForCurveV2Derivative(asset, settings.source, cache);
        } else if (settings.derivative == 4) {
            price = _getPriceForAaveDerivative(asset, settings.source, cache);
        } else revert PriceRouter__UnkownDerivative(settings.derivative);

        if (lastIndex < PRICE_CACHE_SIZE && price <= type(uint96).max) {
            for (uint8 i = lastIndex; i < PRICE_CACHE_SIZE; ++i) {

                if (cache[i].asset == address(0)) {
                    cache[i] = PriceCache(address(asset), uint96(price));
                    break;
                }
            }
        }

        return price;
    }

    function _getValueInQuote(
        uint256 priceBaseUSD,
        uint256 priceQuoteUSD,
        uint8 baseDecimals,
        uint8 quoteDecimals,
        uint256 amountBase
    ) internal pure returns (uint256 valueInQuote) {

        valueInQuote = amountBase.mulDivDown(
            (priceBaseUSD * 10 ** quoteDecimals),
            (10 ** baseDecimals * priceQuoteUSD)
        );
    }

    error PriceRouter__LengthMismatch();

    function _getValues(
        ERC20[] calldata baseAssets,
        uint256[] calldata amounts,
        ERC20 quoteAsset,
        PriceCache[PRICE_CACHE_SIZE] memory cache
    ) internal view returns (uint256) {
        if (baseAssets.length != amounts.length) revert PriceRouter__LengthMismatch();
        uint256 quotePrice;
        {
            AssetSettings memory quoteSettings = getAssetSettings[quoteAsset];
            if (quoteSettings.derivative == 0) revert PriceRouter__UnsupportedAsset(address(quoteAsset));
            quotePrice = _getPriceInUSD(quoteAsset, quoteSettings, cache);
        }
        uint256 valueInQuote;

        uint8 quoteDecimals = quoteAsset.decimals();

        for (uint8 i = 0; i < baseAssets.length; i++) {

            if (amounts[i] == 0) continue;
            ERC20 baseAsset = baseAssets[i];
            if (baseAsset == quoteAsset) valueInQuote += amounts[i];
            else {
                uint256 basePrice;
                {
                    AssetSettings memory baseSettings = getAssetSettings[baseAsset];
                    if (baseSettings.derivative == 0) revert PriceRouter__UnsupportedAsset(address(baseAsset));
                    basePrice = _getPriceInUSD(baseAsset, baseSettings, cache);
                }
                valueInQuote += _getValueInQuote(
                    basePrice,
                    quotePrice,
                    baseAsset.decimals(),
                    quoteDecimals,
                    amounts[i]
                );

            }
        }
        return valueInQuote;
    }

    struct ChainlinkDerivativeStorage {
        uint144 max;
        uint80 min;
        uint24 heartbeat;
        bool inETH;
    }

    mapping(ERC20 => ChainlinkDerivativeStorage) public getChainlinkDerivativeStorage;

    uint24 public constant DEFAULT_HEART_BEAT = 1 days;

    function _setupPriceForChainlinkDerivative(ERC20 _asset, address _source, bytes memory _storage) internal {
        ChainlinkDerivativeStorage memory parameters = abi.decode(_storage, (ChainlinkDerivativeStorage));

        IChainlinkAggregator aggregator = IChainlinkAggregator(IChainlinkAggregator(_source).aggregator());
        uint256 minFromChainklink = uint256(uint192(aggregator.minAnswer()));
        uint256 maxFromChainlink = uint256(uint192(aggregator.maxAnswer()));

        uint256 bufferedMinPrice = (minFromChainklink * 1.1e18) / 1e18;
        uint256 bufferedMaxPrice = (maxFromChainlink * 0.9e18) / 1e18;

        if (parameters.min == 0) {

            if (bufferedMinPrice > type(uint80).max) revert("Buffered Min Overflow");
            parameters.min = uint80(bufferedMinPrice);
        } else {
            if (parameters.min < bufferedMinPrice)
                revert PriceRouter__InvalidMinPrice(parameters.min, bufferedMinPrice);
        }

        if (parameters.max == 0) {

            parameters.max = bufferedMaxPrice > type(uint144).max ? type(uint144).max : uint144(bufferedMaxPrice);
        } else {
            if (parameters.max > bufferedMaxPrice)
                revert PriceRouter__InvalidMaxPrice(parameters.max, bufferedMaxPrice);
        }

        if (parameters.min >= parameters.max)
            revert PriceRouter__MinPriceGreaterThanMaxPrice(parameters.min, parameters.max);

        parameters.heartbeat = parameters.heartbeat != 0 ? parameters.heartbeat : DEFAULT_HEART_BEAT;

        getChainlinkDerivativeStorage[_asset] = parameters;
    }

    function _getPriceForChainlinkDerivative(
        ERC20 _asset,
        address _source,
        PriceCache[PRICE_CACHE_SIZE] memory cache
    ) internal view returns (uint256) {
        ChainlinkDerivativeStorage memory parameters = getChainlinkDerivativeStorage[_asset];
        IChainlinkAggregator aggregator = IChainlinkAggregator(_source);
        (, int256 _price, , uint256 _timestamp, ) = aggregator.latestRoundData();
        uint256 price = _price.toUint256();
        _checkPriceFeed(address(_asset), price, _timestamp, parameters.max, parameters.min, parameters.heartbeat);

        if (parameters.inETH) {
            uint256 _ethToUsd = _getPriceInUSD(WETH, getAssetSettings[WETH], cache);
            price = price.mulWadDown(_ethToUsd);
        }
        return price;
    }

    error PriceRouter__AssetBelowMinPrice(address asset, uint256 price, uint256 minPrice);

    error PriceRouter__AssetAboveMaxPrice(address asset, uint256 price, uint256 maxPrice);

    error PriceRouter__StalePrice(address asset, uint256 timeSinceLastUpdate, uint256 heartbeat);

    function _checkPriceFeed(
        address asset,
        uint256 value,
        uint256 timestamp,
        uint144 max,
        uint88 min,
        uint24 heartbeat
    ) internal view {
        if (value < min) revert PriceRouter__AssetBelowMinPrice(address(asset), value, min);

        if (value > max) revert PriceRouter__AssetAboveMaxPrice(address(asset), value, max);

        uint256 timeSinceLastUpdate = block.timestamp - timestamp;
        if (timeSinceLastUpdate > heartbeat)
            revert PriceRouter__StalePrice(address(asset), timeSinceLastUpdate, heartbeat);
    }

    struct VirtualPriceBound {
        uint96 datum;
        uint64 timeLastUpdated;
        uint32 posDelta;
        uint32 negDelta;
        uint32 rateLimit;
    }

    mapping(address => VirtualPriceBound) public getVirtualPriceBound;

    uint32 public constant DEFAULT_RATE_LIMIT = 1 days;

    address public ETH_FAST_GAS_FEED = 0x169E633A2D1E6c10dD91238Ba11c4A708dfEF37C;

    function setGasFeed(address gasFeed) external onlyOwner {
        ETH_FAST_GAS_FEED = gasFeed;
    }

    uint256 public gasConstant = 200e9;

    function setGasConstant(uint256 newConstant) external onlyOwner {
        gasConstant = newConstant;
    }

    uint256 public minDelta = 0.05e18;

    function setMinDelta(uint256 newMinDelta) external onlyOwner {
        minDelta = newMinDelta;
    }

    address[] public curveAssets;

    function updateVirtualPriceBound(
        address _asset,
        uint32 _posDelta,
        uint32 _negDelta,
        uint32 _rateLimit
    ) external onlyOwner {
        VirtualPriceBound storage vpBound = getVirtualPriceBound[_asset];
        vpBound.posDelta = _posDelta;
        vpBound.negDelta = _negDelta;
        vpBound.rateLimit = _rateLimit == 0 ? DEFAULT_RATE_LIMIT : _rateLimit;
    }

    function _checkVirtualPriceBound(
        bytes memory checkData
    ) internal view returns (bool upkeepNeeded, bytes memory performData) {

        (uint256 start, uint256 end) = abi.decode(checkData, (uint256, uint256));
        if (end == 0 || end > curveAssets.length) end = curveAssets.length;

        uint256 maxDelta;
        uint256 targetIndex;
        for (uint256 i = start; i < end; i++) {
            address asset = curveAssets[i];
            VirtualPriceBound memory vpBound = getVirtualPriceBound[asset];

            if ((block.timestamp - vpBound.timeLastUpdated) < vpBound.rateLimit) continue;

            uint256 currentVirtualPrice = ICurvePool(getAssetSettings[ERC20(asset)].source).get_virtual_price();
            currentVirtualPrice = currentVirtualPrice.changeDecimals(18, 8);
            uint256 delta;
            if (currentVirtualPrice > vpBound.datum) {
                uint256 upper = uint256(vpBound.datum).mulDivDown(vpBound.posDelta, 1e8);
                uint256 ceiling = upper - vpBound.datum;
                uint256 current = currentVirtualPrice - vpBound.datum;
                delta = _getDelta(ceiling, current);
            } else {
                uint256 lower = uint256(vpBound.datum).mulDivDown(vpBound.negDelta, 1e8);
                uint256 ceiling = vpBound.datum - lower;
                uint256 current = vpBound.datum - currentVirtualPrice;
                delta = _getDelta(ceiling, current);
            }

            if (delta > maxDelta) {
                maxDelta = delta;
                targetIndex = i;
            }
        }

        if (maxDelta >= minDelta) {

            if (ETH_FAST_GAS_FEED == address(0)) {

                upkeepNeeded = true;
                performData = abi.encode(targetIndex);
            } else {

                uint256 gasPriceLimit = gasConstant.mulDivDown(maxDelta ** 3, 1e54);
                uint256 currentGasPrice = uint256(IChainlinkAggregator(ETH_FAST_GAS_FEED).latestAnswer());
                if (currentGasPrice <= gasPriceLimit) {
                    upkeepNeeded = true;
                    performData = abi.encode(targetIndex);
                }
            }
        }
    }

    error PriceRouter__OnlyAutomationRegistry();

    error PriceRouter__VirtualPriceRateLimiter();

    error PriceRouter__NothingToUpdate();

    address public automationRegistry = 0x02777053d6764996e594c3E88AF1D58D5363a2e6;

    function setAutomationRegistry(address newRegistry) external onlyOwner {
        automationRegistry = newRegistry;
    }

    function _updateVirtualPriceBound(bytes memory performData) internal {

        if (msg.sender != automationRegistry) revert PriceRouter__OnlyAutomationRegistry();

        uint256 index = abi.decode(performData, (uint256));
        address asset = curveAssets[index];
        VirtualPriceBound storage vpBound = getVirtualPriceBound[asset];

        if ((block.timestamp - vpBound.timeLastUpdated) < vpBound.rateLimit)
            revert PriceRouter__VirtualPriceRateLimiter();

        uint256 currentVirtualPrice = ICurvePool(getAssetSettings[ERC20(asset)].source).get_virtual_price();
        currentVirtualPrice = currentVirtualPrice.changeDecimals(18, 8);
        if (currentVirtualPrice > vpBound.datum) {
            uint256 upper = uint256(vpBound.datum).mulDivDown(vpBound.posDelta, 1e8);
            vpBound.datum = uint96(currentVirtualPrice > upper ? upper : currentVirtualPrice);
        } else if (currentVirtualPrice < vpBound.datum) {
            uint256 lower = uint256(vpBound.datum).mulDivDown(vpBound.negDelta, 1e8);
            vpBound.datum = uint96(currentVirtualPrice < lower ? lower : currentVirtualPrice);
        } else {
            revert PriceRouter__NothingToUpdate();
        }

        vpBound.timeLastUpdated = uint64(block.timestamp);
    }

    function _getDelta(uint256 ceiling, uint256 current) internal pure returns (uint256) {
        return current.mulDivDown(1e18, ceiling);
    }

    error PriceRouter__CurrentBelowLowerBound(uint256 current, uint256 lower);

    error PriceRouter__CurrentAboveUpperBound(uint256 current, uint256 upper);

    function _checkBounds(uint256 lower, uint256 upper, uint256 current) internal pure {
        if (current < lower) revert PriceRouter__CurrentBelowLowerBound(current, lower);
        if (current > upper) revert PriceRouter__CurrentAboveUpperBound(current, upper);
    }

    mapping(ERC20 => address[]) public getCurveDerivativeStorage;

    function _setupPriceForCurveDerivative(ERC20 _asset, address _source, bytes memory _storage) internal {
        ICurvePool pool = ICurvePool(_source);
        uint8 coinsLength = 0;

        while (true) {
            try pool.coins(coinsLength) {
                coinsLength++;
            } catch {
                break;
            }
        }

        address[] memory coins = new address[](coinsLength);
        for (uint256 i = 0; i < coinsLength; i++) {
            coins[i] = pool.coins(i);
        }

        getCurveDerivativeStorage[_asset] = coins;

        curveAssets.push(address(_asset));

        VirtualPriceBound memory vpBound = abi.decode(_storage, (VirtualPriceBound));
        uint256 upper = uint256(vpBound.datum).mulDivDown(vpBound.posDelta, 1e8);
        upper = upper.changeDecimals(8, 18);
        uint256 lower = uint256(vpBound.datum).mulDivDown(vpBound.negDelta, 1e8);
        lower = lower.changeDecimals(8, 18);
        _checkBounds(lower, upper, pool.get_virtual_price());
        if (vpBound.rateLimit == 0) vpBound.rateLimit = DEFAULT_RATE_LIMIT;
        vpBound.timeLastUpdated = uint64(block.timestamp);
        getVirtualPriceBound[address(_asset)] = vpBound;
    }

    function _getPriceForCurveDerivative(
        ERC20 asset,
        address _source,
        PriceCache[PRICE_CACHE_SIZE] memory cache
    ) internal view returns (uint256 price) {
        ICurvePool pool = ICurvePool(_source);

        address[] memory coins = getCurveDerivativeStorage[asset];

        uint256 minPrice = type(uint256).max;
        for (uint256 i = 0; i < coins.length; i++) {
            ERC20 poolAsset = ERC20(coins[i]);
            uint256 tokenPrice = _getPriceInUSD(poolAsset, getAssetSettings[poolAsset], cache);
            if (tokenPrice < minPrice) minPrice = tokenPrice;
        }

        if (minPrice == type(uint256).max) revert("Min price not found.");

        uint256 virtualPrice = pool.get_virtual_price();
        VirtualPriceBound memory vpBound = getVirtualPriceBound[address(asset)];
        uint256 upper = uint256(vpBound.datum).mulDivDown(vpBound.posDelta, 1e8);
        upper = upper.changeDecimals(8, 18);
        uint256 lower = uint256(vpBound.datum).mulDivDown(vpBound.negDelta, 1e8);
        lower = lower.changeDecimals(8, 18);
        _checkBounds(lower, upper, virtualPrice);

        uint256 curveTokenDecimals = ERC20(asset).decimals();
        price = minPrice.mulDivDown(virtualPrice, 10 ** curveTokenDecimals);
    }

    function _setupPriceForCurveV2Derivative(ERC20 _asset, address _source, bytes memory _storage) internal {
        ICurvePool pool = ICurvePool(_source);
        uint8 coinsLength = 0;

        while (true) {
            try pool.coins(coinsLength) {
                coinsLength++;
            } catch {
                break;
            }
        }
        address[] memory coins = new address[](coinsLength);
        for (uint256 i = 0; i < coinsLength; i++) {
            coins[i] = pool.coins(i);
        }

        getCurveDerivativeStorage[_asset] = coins;

        curveAssets.push(address(_asset));

        VirtualPriceBound memory vpBound = abi.decode(_storage, (VirtualPriceBound));
        uint256 upper = uint256(vpBound.datum).mulDivDown(vpBound.posDelta, 1e8);
        upper = upper.changeDecimals(8, 18);
        uint256 lower = uint256(vpBound.datum).mulDivDown(vpBound.negDelta, 1e8);
        lower = lower.changeDecimals(8, 18);
        _checkBounds(lower, upper, pool.get_virtual_price());
        if (vpBound.rateLimit == 0) vpBound.rateLimit = DEFAULT_RATE_LIMIT;
        vpBound.timeLastUpdated = uint64(block.timestamp);
        getVirtualPriceBound[address(_asset)] = vpBound;
    }

    uint256 private constant GAMMA0 = 28000000000000;
    uint256 private constant A0 = 2 * 3 ** 3 * 10000;
    uint256 private constant DISCOUNT0 = 1087460000000000;

    function _cubicRoot(uint256 x) internal pure returns (uint256) {
        uint256 D = x / 1e18;
        for (uint8 i; i < 256; i++) {
            uint256 diff;
            uint256 D_prev = D;
            D = (D * (2 * 1e18 + ((((x / D) * 1e18) / D) * 1e18) / D)) / (3 * 1e18);
            if (D > D_prev) diff = D - D_prev;
            else diff = D_prev - D;
            if (diff <= 1 || diff * 10 ** 18 < D) return D;
        }
        revert("Did not converge");
    }

    function _getPriceForCurveV2Derivative(
        ERC20 asset,
        address _source,
        PriceCache[PRICE_CACHE_SIZE] memory cache
    ) internal view returns (uint256) {
        ICurvePool pool = ICurvePool(_source);

        uint256 virtualPrice = pool.get_virtual_price();
        VirtualPriceBound memory vpBound = getVirtualPriceBound[address(asset)];
        uint256 upper = uint256(vpBound.datum).mulDivDown(vpBound.posDelta, 1e8);
        upper = upper.changeDecimals(8, 18);
        uint256 lower = uint256(vpBound.datum).mulDivDown(vpBound.negDelta, 1e8);
        lower = lower.changeDecimals(8, 18);
        _checkBounds(lower, upper, virtualPrice);

        address[] memory coins = getCurveDerivativeStorage[asset];
        ERC20 token0 = ERC20(coins[0]);
        if (coins.length == 2) {
            return pool.lp_price().mulDivDown(_getPriceInUSD(token0, getAssetSettings[token0], cache), 1e18);
        } else if (coins.length == 3) {
            uint256 t1Price = pool.price_oracle(0);
            uint256 t2Price = pool.price_oracle(1);

            uint256 maxPrice = (3 * virtualPrice * _cubicRoot(t1Price * t2Price)) / 1e18;
            {
                uint256 g = pool.gamma().mulDivDown(1e18, GAMMA0);
                uint256 a = pool.A().mulDivDown(1e18, A0);
                uint256 coefficient = (g ** 2 / 1e18) * a;
                uint256 discount = coefficient > 1e34 ? coefficient : 1e34;
                discount = _cubicRoot(discount).mulDivDown(DISCOUNT0, 1e18);

                maxPrice -= maxPrice.mulDivDown(discount, 1e18);
            }
            return maxPrice.mulDivDown(_getPriceInUSD(token0, getAssetSettings[token0], cache), 1e18);
        } else revert("Unsupported Pool");
    }

    mapping(ERC20 => ERC20) public getAaveDerivativeStorage;

    function _setupPriceForAaveDerivative(ERC20 _asset, address _source, bytes memory) internal {
        IAaveToken aToken = IAaveToken(_source);
        getAaveDerivativeStorage[_asset] = ERC20(aToken.UNDERLYING_ASSET_ADDRESS());
    }

    function _getPriceForAaveDerivative(
        ERC20 asset,
        address,
        PriceCache[PRICE_CACHE_SIZE] memory cache
    ) internal view returns (uint256) {
        asset = getAaveDerivativeStorage[asset];
        return _getPriceInUSD(asset, getAssetSettings[asset], cache);
    }
}

interface IGravity {
    function sendToCosmos(address _tokenContract, bytes32 _destination, uint256 _amount) external;
}

library Uint32Array {

    function add(uint32[] storage array, uint32 index, uint32 value) internal {
        uint256 len = array.length;

        if (len > 0) {
            array.push(array[len - 1]);

            for (uint256 i = len - 1; i > index; i--) array[i] = array[i - 1];

            array[index] = value;
        } else {
            array.push(value);
        }
    }

    function remove(uint32[] storage array, uint32 index) internal {
        uint256 len = array.length;

        require(index < len, "Index out of bounds");

        for (uint256 i = index; i < len - 1; i++) array[i] = array[i + 1];

        array.pop();
    }

    function contains(uint32[] storage array, uint32 value) internal view returns (bool) {
        for (uint256 i; i < array.length; i++) if (value == array[i]) return true;

        return false;
    }
}

interface IERC721Receiver {

    function onERC721Received(
        address operator,
        address from,
        uint256 tokenId,
        bytes calldata data
    ) external returns (bytes4);
}

contract ERC721Holder is IERC721Receiver {

    function onERC721Received(address, address, uint256, bytes memory) public virtual override returns (bytes4) {
        return this.onERC721Received.selector;
    }
}

abstract contract Owned {

    event OwnershipTransferred(address indexed user, address indexed newOwner);

    address public owner;

    modifier onlyOwner() virtual {
        require(msg.sender == owner, "UNAUTHORIZED");

        _;
    }

    constructor(address _owner) {
        owner = _owner;

        emit OwnershipTransferred(address(0), _owner);
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        owner = newOwner;

        emit OwnershipTransferred(msg.sender, newOwner);
    }
}

contract Cellar is ERC4626, Owned, ERC721Holder {
    using Uint32Array for uint32[];
    using SafeTransferLib for ERC20;
    using Math for uint256;
    using Address for address;

    uint256 public locked = 1;

    modifier nonReentrant() {
        require(locked == 1, "REENTRANCY");

        locked = 2;

        _;

        locked = 1;
    }

    event PositionAdded(uint32 position, uint256 index);

    event PositionRemoved(uint32 position, uint256 index);

    event PositionSwapped(uint32 newPosition1, uint32 newPosition2, uint256 index1, uint256 index2);

    error Cellar__PositionAlreadyUsed(uint32 position);

    error Cellar__PositionNotUsed(uint32 position);

    error Cellar__PositionNotEmpty(uint32 position, uint256 sharesRemaining);

    error Cellar__AssetMismatch(address asset, address expectedAsset);

    error Cellar__PositionArrayFull(uint256 maxPositions);

    error Cellar__DebtMismatch(uint32 position);

    error Cellar__RemovingHoldingPosition();

    error Cellar__InvalidHoldingPosition(uint32 positionId);

    uint32[] public creditPositions;

    uint32[] public debtPositions;

    mapping(uint256 => bool) public isPositionUsed;

    mapping(uint32 => Registry.PositionData) public getPositionData;

    function getCreditPositions() external view returns (uint32[] memory) {
        return creditPositions;
    }

    function getDebtPositions() external view returns (uint32[] memory) {
        return debtPositions;
    }

    uint256 public constant MAX_POSITIONS = 16;

    uint32 public holdingPosition;

    function setHoldingPosition(uint32 positionId) external onlyOwner {
        _setHoldingPosition(positionId);
    }

    function _setHoldingPosition(uint32 positionId) internal {
        if (!isPositionUsed[positionId]) revert Cellar__PositionNotUsed(positionId);
        if (_assetOf(positionId) != asset) revert Cellar__AssetMismatch(address(asset), address(_assetOf(positionId)));
        if (getPositionData[positionId].isDebt) revert Cellar__InvalidHoldingPosition(positionId);
        holdingPosition = positionId;
    }

    function addPosition(
        uint32 index,
        uint32 positionId,
        bytes memory configurationData,
        bool inDebtArray
    ) external onlyOwner whenNotShutdown {
        _addPosition(index, positionId, configurationData, inDebtArray);
    }

    function _addPosition(uint32 index, uint32 positionId, bytes memory configurationData, bool inDebtArray) internal {

        if (isPositionUsed[positionId]) revert Cellar__PositionAlreadyUsed(positionId);

        (address adaptor, bool isDebt, bytes memory adaptorData) = registry.cellarAddPosition(
            positionId,
            assetRiskTolerance,
            protocolRiskTolerance
        );

        if (isDebt != inDebtArray) revert Cellar__DebtMismatch(positionId);

        getPositionData[positionId] = Registry.PositionData({
            adaptor: adaptor,
            isDebt: isDebt,
            adaptorData: adaptorData,
            configurationData: configurationData
        });

        if (isDebt) {
            if (debtPositions.length >= MAX_POSITIONS) revert Cellar__PositionArrayFull(MAX_POSITIONS);

            debtPositions.add(index, positionId);
        } else {
            if (creditPositions.length >= MAX_POSITIONS) revert Cellar__PositionArrayFull(MAX_POSITIONS);

            creditPositions.add(index, positionId);
        }

        isPositionUsed[positionId] = true;

        emit PositionAdded(positionId, index);
    }

    function removePosition(uint32 index, bool inDebtArray) external onlyOwner {

        uint32 positionId = inDebtArray ? debtPositions[index] : creditPositions[index];

        if (positionId == holdingPosition) revert Cellar__RemovingHoldingPosition();

        uint256 positionBalance = _balanceOf(positionId);
        if (positionBalance > 0) revert Cellar__PositionNotEmpty(positionId, positionBalance);

        if (inDebtArray) {

            debtPositions.remove(index);
        } else {
            creditPositions.remove(index);
        }

        isPositionUsed[positionId] = false;
        delete getPositionData[positionId];

        emit PositionRemoved(positionId, index);
    }

    function swapPositions(uint32 index1, uint32 index2, bool inDebtArray) external onlyOwner {

        uint32 newPosition1;
        uint32 newPosition2;

        if (inDebtArray) {
            newPosition1 = debtPositions[index2];
            newPosition2 = debtPositions[index1];

            (debtPositions[index1], debtPositions[index2]) = (newPosition1, newPosition2);
        } else {
            newPosition1 = creditPositions[index2];
            newPosition2 = creditPositions[index1];

            (creditPositions[index1], creditPositions[index2]) = (newPosition1, newPosition2);
        }

        emit PositionSwapped(newPosition1, newPosition2, index1, index2);
    }

    event PlatformFeeChanged(uint64 oldPlatformFee, uint64 newPlatformFee);

    event StrategistPlatformCutChanged(uint64 oldPlatformCut, uint64 newPlatformCut);

    event StrategistPayoutAddressChanged(address oldPayoutAddress, address newPayoutAddress);

    error Cellar__InvalidFeeCut();

    error Cellar__InvalidFee();

    struct FeeData {
        uint64 strategistPlatformCut;
        uint64 platformFee;
        uint64 lastAccrual;
        address strategistPayoutAddress;
    }

    FeeData public feeData =
        FeeData({
            strategistPlatformCut: 0.75e18,
            platformFee: 0.01e18,
            lastAccrual: 0,
            strategistPayoutAddress: address(0)
        });

    uint64 public constant MAX_PLATFORM_FEE = 0.2e18;

    uint64 public constant MAX_FEE_CUT = 1e18;

    function setPlatformFee(uint64 newPlatformFee) external onlyOwner {
        if (newPlatformFee > MAX_PLATFORM_FEE) revert Cellar__InvalidFee();
        emit PlatformFeeChanged(feeData.platformFee, newPlatformFee);

        feeData.platformFee = newPlatformFee;
    }

    function setStrategistPlatformCut(uint64 cut) external onlyOwner {
        if (cut > MAX_FEE_CUT) revert Cellar__InvalidFeeCut();
        emit StrategistPlatformCutChanged(feeData.strategistPlatformCut, cut);

        feeData.strategistPlatformCut = cut;
    }

    function setStrategistPayoutAddress(address payout) external onlyOwner {
        emit StrategistPayoutAddressChanged(feeData.strategistPayoutAddress, payout);

        feeData.strategistPayoutAddress = payout;
    }

    event ShutdownChanged(bool isShutdown);

    error Cellar__ContractShutdown();

    error Cellar__ContractNotShutdown();

    bool public isShutdown;

    modifier whenNotShutdown() {
        if (isShutdown) revert Cellar__ContractShutdown();

        _;
    }

    function initiateShutdown() external whenNotShutdown onlyOwner {
        isShutdown = true;

        emit ShutdownChanged(true);
    }

    function liftShutdown() external onlyOwner {
        if (!isShutdown) revert Cellar__ContractNotShutdown();
        isShutdown = false;

        emit ShutdownChanged(false);
    }

    uint256 public constant GRAVITY_BRIDGE_REGISTRY_SLOT = 0;

    uint256 public constant PRICE_ROUTER_REGISTRY_SLOT = 2;

    Registry public registry;

    uint128 public assetRiskTolerance;

    uint128 public protocolRiskTolerance;

    constructor(
        Registry _registry,
        ERC20 _asset,
        string memory _name,
        string memory _symbol,
        bytes memory params
    ) ERC4626(_asset, _name, _symbol, 18) Owned(_registry.getAddress(GRAVITY_BRIDGE_REGISTRY_SLOT)) {
        registry = _registry;

        {
            (
                uint32[] memory _creditPositions,
                uint32[] memory _debtPositions,
                bytes[] memory _creditConfigurationData,
                bytes[] memory _debtConfigurationData,
                uint32 _holdingPosition
            ) = abi.decode(params, (uint32[], uint32[], bytes[], bytes[], uint8));

            for (uint32 i; i < _creditPositions.length; ++i) {
                _addPosition(i, _creditPositions[i], _creditConfigurationData[i], false);
            }
            for (uint32 i; i < _debtPositions.length; ++i) {
                _addPosition(i, _debtPositions[i], _debtConfigurationData[i], true);
            }

            if (_creditPositions.length > 0) _setHoldingPosition(_holdingPosition);
        }

        feeData.lastAccrual = uint64(block.timestamp);

        (, , , , , address _strategistPayout, uint128 _assetRiskTolerance, uint128 _protocolRiskTolerance) = abi.decode(
            params,
            (uint32[], uint32[], bytes[], bytes[], uint8, address, uint128, uint128)
        );

        feeData.strategistPayoutAddress = _strategistPayout;

        assetRiskTolerance = _assetRiskTolerance;
        protocolRiskTolerance = _protocolRiskTolerance;
    }

    event ShareLockingPeriodChanged(uint256 oldPeriod, uint256 newPeriod);

    error Cellar__ZeroShares();

    error Cellar__ZeroAssets();

    error Cellar__IncompleteWithdraw(uint256 assetsOwed);

    error Cellar__IlliquidWithdraw(address illiquidPosition);

    error Cellar__InvalidShareLockPeriod();

    error Cellar__SharesAreLocked(uint256 timeSharesAreUnlocked, uint256 currentBlock);

    error Cellar__NotApprovedToDepositOnBehalf(address depositor);

    uint256 public constant MINIMUM_SHARE_LOCK_PERIOD = 5 * 60;

    uint256 public constant MAXIMUM_SHARE_LOCK_PERIOD = 2 days;

    uint256 public shareLockPeriod = MAXIMUM_SHARE_LOCK_PERIOD;

    mapping(address => uint256) public userShareLockStartTime;

    function setShareLockPeriod(uint256 newLock) external onlyOwner {
        if (newLock < MINIMUM_SHARE_LOCK_PERIOD || newLock > MAXIMUM_SHARE_LOCK_PERIOD)
            revert Cellar__InvalidShareLockPeriod();
        uint256 oldLockingPeriod = shareLockPeriod;
        shareLockPeriod = newLock;
        emit ShareLockingPeriodChanged(oldLockingPeriod, newLock);
    }

    function _checkIfSharesLocked(address owner) internal view {
        uint256 lockTime = userShareLockStartTime[owner];
        if (lockTime != 0) {
            uint256 timeSharesAreUnlocked = lockTime + shareLockPeriod;
            if (timeSharesAreUnlocked > block.timestamp)
                revert Cellar__SharesAreLocked(timeSharesAreUnlocked, block.timestamp);
        }
    }

    function transfer(address to, uint256 amount) public override returns (bool) {
        _checkIfSharesLocked(msg.sender);
        return super.transfer(to, amount);
    }

    function transferFrom(address from, address to, uint256 amount) public override returns (bool) {
        _checkIfSharesLocked(from);
        return super.transferFrom(from, to, amount);
    }

    error Cellar__DepositRestricted(uint256 assets, uint256 maxDeposit);

    function beforeDeposit(uint256 assets, uint256, address receiver) internal view override whenNotShutdown {
        if (msg.sender != receiver) {
            if (!registry.approvedForDepositOnBehalf(msg.sender))
                revert Cellar__NotApprovedToDepositOnBehalf(msg.sender);
        }
        uint256 maxAssets = maxDeposit(receiver);
        if (assets > maxAssets) revert Cellar__DepositRestricted(assets, maxAssets);
    }

    function afterDeposit(uint256 assets, uint256, address receiver) internal override {
        _depositTo(holdingPosition, assets);
        userShareLockStartTime[receiver] = block.timestamp;
    }

    function beforeWithdraw(uint256, uint256, address, address owner) internal view override {

        _checkIfSharesLocked(owner);
    }

    function _enter(uint256 assets, uint256 shares, address receiver) internal {
        beforeDeposit(assets, shares, receiver);

        asset.safeTransferFrom(msg.sender, address(this), assets);

        _mint(receiver, shares);

        emit Deposit(msg.sender, receiver, assets, shares);

        afterDeposit(assets, shares, receiver);
    }

    function deposit(uint256 assets, address receiver) public override nonReentrant returns (uint256 shares) {

        uint256 _totalAssets = _accounting(false);

        if ((shares = _convertToShares(assets, _totalAssets)) == 0) revert Cellar__ZeroShares();

        _enter(assets, shares, receiver);
    }

    function mint(uint256 shares, address receiver) public override nonReentrant returns (uint256 assets) {

        uint256 _totalAssets = _accounting(false);

        if ((assets = _previewMint(shares, _totalAssets)) == 0) revert Cellar__ZeroAssets();

        _enter(assets, shares, receiver);
    }

    function _exit(uint256 assets, uint256 shares, address receiver, address owner) internal {
        beforeWithdraw(assets, shares, receiver, owner);

        if (msg.sender != owner) {
            uint256 allowed = allowance[owner][msg.sender];

            if (allowed != type(uint256).max) allowance[owner][msg.sender] = allowed - shares;
        }

        _burn(owner, shares);

        emit Withdraw(msg.sender, receiver, owner, assets, shares);
        _withdrawInOrder(assets, receiver);

    }

    function withdraw(
        uint256 assets,
        address receiver,
        address owner
    ) public override nonReentrant returns (uint256 shares) {

        uint256 _totalAssets = _accounting(false);

        shares = _previewWithdraw(assets, _totalAssets);

        _exit(assets, shares, receiver, owner);
    }

    function redeem(
        uint256 shares,
        address receiver,
        address owner
    ) public override nonReentrant returns (uint256 assets) {

        uint256 _totalAssets = _accounting(false);

        if ((assets = _convertToAssets(shares, _totalAssets)) == 0) revert Cellar__ZeroAssets();

        _exit(assets, shares, receiver, owner);
    }

    struct WithdrawPricing {
        uint256 priceBaseUSD;
        uint256 oneBase;
        uint256 priceQuoteUSD;
        uint256 oneQuote;
    }

    uint256 private constant PRECISION_MULTIPLIER = 1e18;

    function _withdrawInOrder(uint256 assets, address receiver) internal {

        PriceRouter priceRouter = PriceRouter(registry.getAddress(PRICE_ROUTER_REGISTRY_SLOT));

        WithdrawPricing memory pricingInfo;
        pricingInfo.priceQuoteUSD = priceRouter.getPriceInUSD(asset);
        pricingInfo.oneQuote = 10 ** asset.decimals();
        uint256 creditLength = creditPositions.length;
        for (uint256 i; i < creditLength; ++i) {
            uint32 position = creditPositions[i];
            uint256 withdrawableBalance = _withdrawableFrom(position);

            if (withdrawableBalance == 0) continue;
            ERC20 positionAsset = _assetOf(position);

            pricingInfo.priceBaseUSD = priceRouter.getPriceInUSD(positionAsset);
            pricingInfo.oneBase = 10 ** positionAsset.decimals();
            uint256 totalWithdrawableBalanceInAssets;
            {
                uint256 withdrawableBalanceInUSD = (PRECISION_MULTIPLIER * withdrawableBalance).mulDivDown(
                    pricingInfo.priceBaseUSD,
                    pricingInfo.oneBase
                );
                totalWithdrawableBalanceInAssets = withdrawableBalanceInUSD.mulDivDown(
                    pricingInfo.oneQuote,
                    pricingInfo.priceQuoteUSD
                );
                totalWithdrawableBalanceInAssets = totalWithdrawableBalanceInAssets / PRECISION_MULTIPLIER;
            }

            uint256 amount;

            if (totalWithdrawableBalanceInAssets > assets) {

                uint256 assetsInUSD = (PRECISION_MULTIPLIER * assets).mulDivDown(
                    pricingInfo.priceQuoteUSD,
                    pricingInfo.oneQuote
                );
                amount = assetsInUSD.mulDivDown(pricingInfo.oneBase, pricingInfo.priceBaseUSD);
                amount = amount / PRECISION_MULTIPLIER;
                assets = 0;
            } else {
                amount = withdrawableBalance;
                assets = assets - totalWithdrawableBalanceInAssets;
            }

            _withdrawFrom(position, amount, receiver);

            if (assets == 0) break;
        }

        if (assets > 0) revert Cellar__IncompleteWithdraw(assets);
    }

    function _accounting(bool reportWithdrawable) internal view returns (uint256 assets) {
        uint256 numOfCreditPositions = creditPositions.length;
        ERC20[] memory creditAssets = new ERC20[](numOfCreditPositions);
        uint256[] memory creditBalances = new uint256[](numOfCreditPositions);
        PriceRouter priceRouter = PriceRouter(registry.getAddress(PRICE_ROUTER_REGISTRY_SLOT));

        if (reportWithdrawable) {
            for (uint256 i; i < numOfCreditPositions; ++i) {
                uint32 position = creditPositions[i];

                if ((creditBalances[i] = _withdrawableFrom(position)) == 0) continue;
                creditAssets[i] = _assetOf(position);
            }
            assets = priceRouter.getValues(creditAssets, creditBalances, asset);
        } else {
            uint256 numOfDebtPositions = debtPositions.length;
            ERC20[] memory debtAssets = new ERC20[](numOfDebtPositions);
            uint256[] memory debtBalances = new uint256[](numOfDebtPositions);
            for (uint256 i; i < numOfCreditPositions; ++i) {
                uint32 position = creditPositions[i];

                if ((creditBalances[i] = _balanceOf(position)) == 0) continue;
                creditAssets[i] = _assetOf(position);
            }
            for (uint256 i; i < numOfDebtPositions; ++i) {
                uint32 position = debtPositions[i];

                if ((debtBalances[i] = _balanceOf(position)) == 0) continue;
                debtAssets[i] = _assetOf(position);
            }
            assets = priceRouter.getValuesDelta(creditAssets, creditBalances, debtAssets, debtBalances, asset);
        }
    }

    function totalAssets() public view override returns (uint256 assets) {
        require(locked == 1, "REENTRANCY");
        assets = _accounting(false);
    }

    function totalAssetsWithdrawable() public view returns (uint256 assets) {
        require(locked == 1, "REENTRANCY");
        assets = _accounting(true);
    }

    function convertToAssets(uint256 shares) public view override returns (uint256 assets) {
        assets = _convertToAssets(shares, totalAssets());
    }

    function convertToShares(uint256 assets) public view override returns (uint256 shares) {
        shares = _convertToShares(assets, totalAssets());
    }

    function previewMint(uint256 shares) public view override returns (uint256 assets) {
        uint256 _totalAssets = totalAssets();
        assets = _previewMint(shares, _totalAssets);
    }

    function previewWithdraw(uint256 assets) public view override returns (uint256 shares) {
        uint256 _totalAssets = totalAssets();
        shares = _previewWithdraw(assets, _totalAssets);
    }

    function previewDeposit(uint256 assets) public view override returns (uint256 shares) {
        uint256 _totalAssets = totalAssets();
        shares = _convertToShares(assets, _totalAssets);
    }

    function previewRedeem(uint256 shares) public view override returns (uint256 assets) {
        uint256 _totalAssets = totalAssets();
        assets = _convertToAssets(shares, _totalAssets);
    }

    function _findMax(address owner, bool inShares) internal view returns (uint256 maxOut) {

        uint256 lockTime = userShareLockStartTime[owner];
        if (lockTime != 0) {
            uint256 timeSharesAreUnlocked = lockTime + shareLockPeriod;
            if (timeSharesAreUnlocked > block.timestamp) return 0;
        }

        uint256 _totalAssets = _accounting(false);
        uint256 assets = _convertToAssets(balanceOf[owner], _totalAssets);

        uint256 withdrawable = _accounting(true);
        maxOut = assets <= withdrawable ? assets : withdrawable;

        if (inShares) maxOut = _convertToShares(maxOut, _totalAssets);

    }

    function maxWithdraw(address owner) public view override returns (uint256) {
        require(locked == 1, "REENTRANCY");
        return _findMax(owner, false);
    }

    function maxRedeem(address owner) public view override returns (uint256) {
        require(locked == 1, "REENTRANCY");
        return _findMax(owner, true);
    }

    function _convertToAssets(uint256 shares, uint256 _totalAssets) internal view returns (uint256 assets) {
        uint256 totalShares = totalSupply;

        assets = totalShares == 0
            ? shares.changeDecimals(18, asset.decimals())
            : shares.mulDivDown(_totalAssets, totalShares);
    }

    function _convertToShares(uint256 assets, uint256 _totalAssets) internal view returns (uint256 shares) {
        uint256 totalShares = totalSupply;

        shares = totalShares == 0
            ? assets.changeDecimals(asset.decimals(), 18)
            : assets.mulDivDown(totalShares, _totalAssets);
    }

    function _previewMint(uint256 shares, uint256 _totalAssets) internal view returns (uint256 assets) {
        uint256 totalShares = totalSupply;

        assets = totalShares == 0
            ? shares.changeDecimals(18, asset.decimals())
            : shares.mulDivUp(_totalAssets, totalShares);
    }

    function _previewWithdraw(uint256 assets, uint256 _totalAssets) internal view returns (uint256 shares) {
        uint256 totalShares = totalSupply;

        shares = totalShares == 0
            ? assets.changeDecimals(asset.decimals(), 18)
            : assets.mulDivUp(totalShares, _totalAssets);
    }

    event RebalanceDeviationChanged(uint256 oldDeviation, uint256 newDeviation);

    error Cellar__TotalAssetDeviatedOutsideRange(uint256 assets, uint256 min, uint256 max);

    error Cellar__TotalSharesMustRemainConstant(uint256 current, uint256 expected);

    error Cellar__InvalidRebalanceDeviation(uint256 requested, uint256 max);

    error Cellar__AdaptorNotSetUp(address adaptor);

    mapping(address => bool) public isAdaptorSetup;

    function setupAdaptor(address _adaptor) external onlyOwner {

        registry.cellarSetupAdaptor(_adaptor, assetRiskTolerance, protocolRiskTolerance);
        isAdaptorSetup[_adaptor] = true;
    }

    uint64 public constant MAX_REBALANCE_DEVIATION = 0.1e18;

    uint256 public allowedRebalanceDeviation = 0.0003e18;

    function setRebalanceDeviation(uint256 newDeviation) external onlyOwner {
        if (newDeviation > MAX_REBALANCE_DEVIATION)
            revert Cellar__InvalidRebalanceDeviation(newDeviation, MAX_REBALANCE_DEVIATION);

        uint256 oldDeviation = allowedRebalanceDeviation;
        allowedRebalanceDeviation = newDeviation;

        emit RebalanceDeviationChanged(oldDeviation, newDeviation);
    }

    bool public blockExternalReceiver;

    struct AdaptorCall {
        address adaptor;
        bytes[] callData;
    }

    function callOnAdaptor(AdaptorCall[] memory data) external onlyOwner whenNotShutdown nonReentrant {
        blockExternalReceiver = true;

        uint256 minimumAllowedAssets;
        uint256 maximumAllowedAssets;
        uint256 totalShares;
        {
            uint256 assetsBeforeAdaptorCall = _accounting(false);
            minimumAllowedAssets = assetsBeforeAdaptorCall.mulDivUp((1e18 - allowedRebalanceDeviation), 1e18);
            maximumAllowedAssets = assetsBeforeAdaptorCall.mulDivUp((1e18 + allowedRebalanceDeviation), 1e18);
            totalShares = totalSupply;
        }

        for (uint8 i = 0; i < data.length; ++i) {
            address adaptor = data[i].adaptor;
            if (!isAdaptorSetup[adaptor]) revert Cellar__AdaptorNotSetUp(adaptor);
            for (uint8 j = 0; j < data[i].callData.length; j++) {
                adaptor.functionDelegateCall(data[i].callData[j]);
            }
        }

        uint256 assets = _accounting(false);
        if (assets < minimumAllowedAssets || assets > maximumAllowedAssets) {
            revert Cellar__TotalAssetDeviatedOutsideRange(assets, minimumAllowedAssets, maximumAllowedAssets);
        }
        if (totalShares != totalSupply) revert Cellar__TotalSharesMustRemainConstant(totalSupply, totalShares);

        blockExternalReceiver = false;
    }

    error Cellar__ExternalInitiator();

    error Cellar__CallerNotAavePool();

    address public aavePool = 0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9;

    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external returns (bool) {
        if (initiator != address(this)) revert Cellar__ExternalInitiator();
        if (msg.sender != aavePool) revert Cellar__CallerNotAavePool();

        AdaptorCall[] memory data = abi.decode(params, (AdaptorCall[]));

        for (uint8 i = 0; i < data.length; ++i) {
            address adaptor = data[i].adaptor;
            if (!isAdaptorSetup[adaptor]) revert Cellar__AdaptorNotSetUp(adaptor);
            for (uint8 j = 0; j < data[i].callData.length; j++) {
                adaptor.functionDelegateCall(data[i].callData[j]);
            }
        }

        for (uint256 i = 0; i < amounts.length; ++i) {
            ERC20(assets[i]).safeApprove(aavePool, (amounts[i] + premiums[i]));
        }

        return true;
    }

    function maxDeposit(address) public view override returns (uint256) {
        if (isShutdown) return 0;

        return type(uint256).max;
    }

    function maxMint(address) public view override returns (uint256) {
        if (isShutdown) return 0;

        return type(uint256).max;
    }

    error Cellar__PayoutNotSet();

    function _convertToFees(uint256 feesInShares) internal view returns (uint256 fees) {

        uint256 totalShares = totalSupply;

        if (totalShares > feesInShares) {

            uint256 denominator = totalShares - feesInShares;
            fees = feesInShares.mulDivUp(totalShares, denominator);
        }

    }

    event SendFees(uint256 feesInSharesRedeemed, uint256 feesInAssetsSent);

    function sendFees() external nonReentrant {
        address strategistPayoutAddress = feeData.strategistPayoutAddress;
        if (strategistPayoutAddress == address(0)) revert Cellar__PayoutNotSet();

        uint256 _totalAssets = _accounting(false);

        uint256 elapsedTime = block.timestamp - feeData.lastAccrual;
        uint256 platformFeeInAssets = (_totalAssets * elapsedTime * feeData.platformFee) / 1e18 / 365 days;
        uint256 platformFees = _convertToFees(_convertToShares(platformFeeInAssets, _totalAssets));
        _mint(address(this), platformFees);

        uint256 strategistFeeSharesDue = platformFees.mulWadDown(feeData.strategistPlatformCut);
        if (strategistFeeSharesDue > 0) {

            {
                balanceOf[address(this)] -= strategistFeeSharesDue;

                unchecked {
                    balanceOf[strategistPayoutAddress] += strategistFeeSharesDue;
                }

                emit Transfer(address(this), strategistPayoutAddress, strategistFeeSharesDue);
            }

            platformFees -= strategistFeeSharesDue;
        }

        feeData.lastAccrual = uint32(block.timestamp);

        uint256 assets = _convertToAssets(platformFees, _totalAssets);
        if (assets > 0) {
            _burn(address(this), platformFees);

            IGravity gravityBridge = IGravity(registry.getAddress(GRAVITY_BRIDGE_REGISTRY_SLOT));
            asset.safeApprove(address(gravityBridge), assets);
            gravityBridge.sendToCosmos(address(asset), registry.feesDistributor(), assets);
        }

        emit SendFees(platformFees, assets);
    }

    function _depositTo(uint32 position, uint256 assets) internal {
        address adaptor = getPositionData[position].adaptor;
        adaptor.functionDelegateCall(
            abi.encodeWithSelector(
                BaseAdaptor.deposit.selector,
                assets,
                getPositionData[position].adaptorData,
                getPositionData[position].configurationData
            )
        );
    }

    function _withdrawFrom(uint32 position, uint256 assets, address receiver) internal {
        address adaptor = getPositionData[position].adaptor;
        adaptor.functionDelegateCall(
            abi.encodeWithSelector(
                BaseAdaptor.withdraw.selector,
                assets,
                receiver,
                getPositionData[position].adaptorData,
                getPositionData[position].configurationData
            )
        );
    }

    function _withdrawableFrom(uint32 position) internal view returns (uint256) {

        if (getPositionData[position].isDebt) return 0;
        return
            BaseAdaptor(getPositionData[position].adaptor).withdrawableFrom(
                getPositionData[position].adaptorData,
                getPositionData[position].configurationData
            );
    }

    function _balanceOf(uint32 position) internal view returns (uint256) {
        address adaptor = getPositionData[position].adaptor;
        return BaseAdaptor(adaptor).balanceOf(getPositionData[position].adaptorData);
    }

    function _assetOf(uint32 position) internal view returns (ERC20) {
        address adaptor = getPositionData[position].adaptor;
        return BaseAdaptor(adaptor).assetOf(getPositionData[position].adaptorData);
    }

    function getPositionAssets() external view returns (ERC20[] memory assets) {
        assets = new ERC20[](creditPositions.length);
        for (uint256 i = 0; i < creditPositions.length; ++i) {
            assets[i] = _assetOf(creditPositions[i]);
        }
    }
}

contract Registry is Ownable {

    event AddressChanged(uint256 indexed id, address oldAddress, address newAddress);

    error Registry__ContractNotRegistered(uint256 id);

    event DepositorOnBehalfChanged(address depositor, bool state);

    uint256 public nextId;

    mapping(uint256 => address) public getAddress;

    mapping(address => bool) public approvedForDepositOnBehalf;

    function setApprovedForDepositOnBehalf(address depositor, bool state) external onlyOwner {
        approvedForDepositOnBehalf[depositor] = state;
        emit DepositorOnBehalfChanged(depositor, state);
    }

    function setAddress(uint256 id, address newAddress) external onlyOwner {
        if (id >= nextId) revert Registry__ContractNotRegistered(id);

        emit AddressChanged(id, getAddress[id], newAddress);

        getAddress[id] = newAddress;
    }

    constructor(address gravityBridge, address swapRouter, address priceRouter) Ownable() {
        _register(gravityBridge);
        _register(swapRouter);
        _register(priceRouter);
    }

    event Registered(uint256 indexed id, address indexed newContract);

    function register(address newContract) external onlyOwner {
        _register(newContract);
    }

    function _register(address newContract) internal {
        getAddress[nextId] = newContract;

        emit Registered(nextId, newContract);

        nextId++;
    }

    event FeesDistributorChanged(bytes32 oldFeesDistributor, bytes32 newFeesDistributor);

    error Registry__InvalidCosmosAddress();

    bytes32 public feesDistributor = hex"000000000000000000000000b813554b423266bbd4c16c32fa383394868c1f55";

    function setFeesDistributor(bytes32 newFeesDistributor) external onlyOwner {
        if (uint256(newFeesDistributor) > type(uint160).max) revert Registry__InvalidCosmosAddress();
        emit FeesDistributorChanged(feesDistributor, newFeesDistributor);

        feesDistributor = newFeesDistributor;
    }

    struct PositionData {
        address adaptor;
        bool isDebt;
        bytes adaptorData;
        bytes configurationData;
    }

    struct RiskData {
        uint128 assetRisk;
        uint128 protocolRisk;
    }

    event PositionAdded(uint32 id, address adaptor, bool isDebt, bytes adaptorData);

    error Registry__PositionPricingNotSetUp(address position);

    error Registry__InvalidPositionInput();

    error Registry__AssetTooRisky();

    error Registry__ProtocolTooRisky();

    error Registry__PositionDoesNotExist();

    uint256 public constant PRICE_ROUTER_REGISTRY_SLOT = 2;

    mapping(uint32 => RiskData) public getRiskData;

    mapping(address => RiskData) public getAdaptorRiskData;

    uint32 public positionCount;

    mapping(bytes32 => uint32) public getPositionHashToPositionId;

    mapping(uint32 => PositionData) public getPositionIdToPositionData;

    function trustPosition(
        address adaptor,
        bytes memory adaptorData,
        uint128 assetRisk,
        uint128 protocolRisk
    ) external onlyOwner returns (uint32 positionId) {
        bytes32 identifier = BaseAdaptor(adaptor).identifier();
        bool isDebt = BaseAdaptor(adaptor).isDebt();
        bytes32 positionHash = keccak256(abi.encode(identifier, isDebt, adaptorData));
        positionId = positionCount + 1;

        if (adaptor == address(0) || getPositionHashToPositionId[positionHash] != 0)
            revert Registry__InvalidPositionInput();

        if (!isAdaptorTrusted[adaptor]) revert Registry__AdaptorNotTrusted();

        getPositionIdToPositionData[positionId] = PositionData({
            adaptor: adaptor,
            isDebt: isDebt,
            adaptorData: adaptorData,
            configurationData: abi.encode(0)
        });

        getRiskData[positionId] = RiskData({ assetRisk: assetRisk, protocolRisk: protocolRisk });

        getPositionHashToPositionId[positionHash] = positionId;

        ERC20[] memory assets = BaseAdaptor(adaptor).assetsUsed(adaptorData);
        PriceRouter priceRouter = PriceRouter(getAddress[PRICE_ROUTER_REGISTRY_SLOT]);
        for (uint256 i; i < assets.length; i++) {
            if (!priceRouter.isSupported(assets[i])) revert Registry__PositionPricingNotSetUp(address(assets[i]));
        }

        positionCount = positionId;

        emit PositionAdded(positionId, adaptor, isDebt, adaptorData);
    }

    function cellarAddPosition(
        uint32 positionId,
        uint128 assetRiskTolerance,
        uint128 protocolRiskTolerance
    ) external view returns (address adaptor, bool isDebt, bytes memory adaptorData) {
        if (positionId > positionCount || positionId == 0) revert Registry__PositionDoesNotExist();
        RiskData memory data = getRiskData[positionId];
        if (assetRiskTolerance < data.assetRisk) revert Registry__AssetTooRisky();
        if (protocolRiskTolerance < data.protocolRisk) revert Registry__ProtocolTooRisky();
        PositionData memory positionData = getPositionIdToPositionData[positionId];
        return (positionData.adaptor, positionData.isDebt, positionData.adaptorData);
    }

    error Registry__IdentifierNotUnique();

    error Registry__AdaptorNotTrusted();

    mapping(address => bool) public isAdaptorTrusted;

    mapping(bytes32 => bool) public isIdentifierUsed;

    function trustAdaptor(address adaptor, uint128 assetRisk, uint128 protocolRisk) external onlyOwner {
        bytes32 identifier = BaseAdaptor(adaptor).identifier();
        if (isIdentifierUsed[identifier]) revert Registry__IdentifierNotUnique();
        isAdaptorTrusted[adaptor] = true;
        isIdentifierUsed[identifier] = true;
        getAdaptorRiskData[adaptor] = RiskData({ assetRisk: assetRisk, protocolRisk: protocolRisk });
    }

    function cellarSetupAdaptor(
        address adaptor,
        uint128 assetRiskTolerance,
        uint128 protocolRiskTolerance
    ) external view {
        RiskData memory data = getAdaptorRiskData[adaptor];
        if (assetRiskTolerance < data.assetRisk) revert Registry__AssetTooRisky();
        if (protocolRiskTolerance < data.protocolRisk) revert Registry__ProtocolTooRisky();
        if (!isAdaptorTrusted[adaptor]) revert Registry__AdaptorNotTrusted();
    }
}

interface IMulticall {

    function multicall(bytes[] calldata data) external payable returns (bytes[] memory results);
}

abstract contract Multicall is IMulticall {

    function multicall(bytes[] calldata data) public payable override returns (bytes[] memory results) {
        results = new bytes[](data.length);
        for (uint256 i = 0; i < data.length; i++) {
            (bool success, bytes memory result) = address(this).delegatecall(data[i]);

            if (!success) {

                if (result.length < 68) revert();
                assembly {
                    result := add(result, 0x04)
                }
                revert(abi.decode(result, (string)));
            }

            results[i] = result;
        }
    }
}

interface IUniswapV2Router01 {
    function factory() external pure returns (address);

    function WETH() external pure returns (address);

    function addLiquidity(
        address tokenA,
        address tokenB,
        uint256 amountADesired,
        uint256 amountBDesired,
        uint256 amountAMin,
        uint256 amountBMin,
        address to,
        uint256 deadline
    ) external returns (uint256 amountA, uint256 amountB, uint256 liquidity);

    function addLiquidityETH(
        address token,
        uint256 amountTokenDesired,
        uint256 amountTokenMin,
        uint256 amountETHMin,
        address to,
        uint256 deadline
    ) external payable returns (uint256 amountToken, uint256 amountETH, uint256 liquidity);

    function removeLiquidity(
        address tokenA,
        address tokenB,
        uint256 liquidity,
        uint256 amountAMin,
        uint256 amountBMin,
        address to,
        uint256 deadline
    ) external returns (uint256 amountA, uint256 amountB);

    function removeLiquidityETH(
        address token,
        uint256 liquidity,
        uint256 amountTokenMin,
        uint256 amountETHMin,
        address to,
        uint256 deadline
    ) external returns (uint256 amountToken, uint256 amountETH);

    function removeLiquidityWithPermit(
        address tokenA,
        address tokenB,
        uint256 liquidity,
        uint256 amountAMin,
        uint256 amountBMin,
        address to,
        uint256 deadline,
        bool approveMax,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external returns (uint256 amountA, uint256 amountB);

    function removeLiquidityETHWithPermit(
        address token,
        uint256 liquidity,
        uint256 amountTokenMin,
        uint256 amountETHMin,
        address to,
        uint256 deadline,
        bool approveMax,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external returns (uint256 amountToken, uint256 amountETH);

    function swapExactTokensForTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external returns (uint256[] memory amounts);

    function swapTokensForExactTokens(
        uint256 amountOut,
        uint256 amountInMax,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external returns (uint256[] memory amounts);

    function swapExactETHForTokens(
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external payable returns (uint256[] memory amounts);

    function swapTokensForExactETH(
        uint256 amountOut,
        uint256 amountInMax,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external returns (uint256[] memory amounts);

    function swapExactTokensForETH(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external returns (uint256[] memory amounts);

    function swapETHForExactTokens(
        uint256 amountOut,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external payable returns (uint256[] memory amounts);

    function quote(uint256 amountA, uint256 reserveA, uint256 reserveB) external pure returns (uint256 amountB);

    function getAmountOut(
        uint256 amountIn,
        uint256 reserveIn,
        uint256 reserveOut
    ) external pure returns (uint256 amountOut);

    function getAmountIn(
        uint256 amountOut,
        uint256 reserveIn,
        uint256 reserveOut
    ) external pure returns (uint256 amountIn);

    function getAmountsOut(uint256 amountIn, address[] calldata path) external view returns (uint256[] memory amounts);

    function getAmountsIn(uint256 amountOut, address[] calldata path) external view returns (uint256[] memory amounts);
}

interface IUniswapV2Router02 is IUniswapV2Router01 {
    function removeLiquidityETHSupportingFeeOnTransferTokens(
        address token,
        uint256 liquidity,
        uint256 amountTokenMin,
        uint256 amountETHMin,
        address to,
        uint256 deadline
    ) external returns (uint256 amountETH);

    function removeLiquidityETHWithPermitSupportingFeeOnTransferTokens(
        address token,
        uint256 liquidity,
        uint256 amountTokenMin,
        uint256 amountETHMin,
        address to,
        uint256 deadline,
        bool approveMax,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external returns (uint256 amountETH);

    function swapExactTokensForTokensSupportingFeeOnTransferTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external;

    function swapExactETHForTokensSupportingFeeOnTransferTokens(
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external payable;

    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint256 amountIn,
        uint256 amountOutMin,
        address[] calldata path,
        address to,
        uint256 deadline
    ) external;
}

interface IUniswapV3SwapCallback {

    function uniswapV3SwapCallback(int256 amount0Delta, int256 amount1Delta, bytes calldata data) external;
}

interface IUniswapV3Router is IUniswapV3SwapCallback {
    struct ExactInputSingleParams {
        address tokenIn;
        address tokenOut;
        uint24 fee;
        address recipient;
        uint256 deadline;
        uint256 amountIn;
        uint256 amountOutMinimum;
        uint160 sqrtPriceLimitX96;
    }

    function exactInputSingle(ExactInputSingleParams calldata params) external payable returns (uint256 amountOut);

    struct ExactInputParams {
        bytes path;
        address recipient;
        uint256 deadline;
        uint256 amountIn;
        uint256 amountOutMinimum;
    }

    function exactInput(ExactInputParams calldata params) external payable returns (uint256 amountOut);

    struct ExactOutputSingleParams {
        address tokenIn;
        address tokenOut;
        uint24 fee;
        address recipient;
        uint256 deadline;
        uint256 amountOut;
        uint256 amountInMaximum;
        uint160 sqrtPriceLimitX96;
    }

    function exactOutputSingle(ExactOutputSingleParams calldata params) external payable returns (uint256 amountIn);

    struct ExactOutputParams {
        bytes path;
        address recipient;
        uint256 deadline;
        uint256 amountOut;
        uint256 amountInMaximum;
    }

    function exactOutput(ExactOutputParams calldata params) external payable returns (uint256 amountIn);
}

contract SwapRouter is Multicall {
    using SafeTransferLib for ERC20;

    enum Exchange {
        UNIV2,
        UNIV3
    }

    mapping(Exchange => bytes4) public getExchangeSelector;

    IUniswapV2Router02 public immutable uniswapV2Router;

    IUniswapV3Router public immutable uniswapV3Router;

    constructor(IUniswapV2Router02 _uniswapV2Router, IUniswapV3Router _uniswapV3Router) {

        uniswapV2Router = _uniswapV2Router;
        uniswapV3Router = _uniswapV3Router;

        getExchangeSelector[Exchange.UNIV2] = SwapRouter(this).swapWithUniV2.selector;
        getExchangeSelector[Exchange.UNIV3] = SwapRouter(this).swapWithUniV3.selector;
    }

    error SwapRouter__SwapReverted();

    error SwapRouter__AssetInMisMatch(address actual, address expected);

    error SwapRouter__AssetOutMisMatch(address actual, address expected);

    function swap(
        Exchange exchange,
        bytes memory swapData,
        address receiver,
        ERC20 assetIn,
        ERC20 assetOut
    ) external returns (uint256 amountOut) {

        (bool success, bytes memory result) = address(this).delegatecall(
            abi.encodeWithSelector(getExchangeSelector[exchange], swapData, receiver, assetIn, assetOut)
        );

        if (!success) {

            if (result.length > 0) {
                assembly {
                    let returndata_size := mload(result)
                    revert(add(32, result), returndata_size)
                }
            } else {
                revert SwapRouter__SwapReverted();
            }
        }

        amountOut = abi.decode(result, (uint256));
    }

    function swapWithUniV2(
        bytes memory swapData,
        address receiver,
        ERC20 assetIn,
        ERC20 assetOut
    ) public returns (uint256 amountOut) {
        (address[] memory path, uint256 amount, uint256 amountOutMin) = abi.decode(
            swapData,
            (address[], uint256, uint256)
        );

        if (assetIn != ERC20(path[0])) revert SwapRouter__AssetInMisMatch(path[0], address(assetIn));
        if (assetOut != ERC20(path[path.length - 1]))
            revert SwapRouter__AssetOutMisMatch(path[path.length - 1], address(assetOut));

        assetIn.safeTransferFrom(msg.sender, address(this), amount);

        assetIn.safeApprove(address(uniswapV2Router), amount);

        uint256[] memory amountsOut = uniswapV2Router.swapExactTokensForTokens(
            amount,
            amountOutMin,
            path,
            receiver,
            block.timestamp + 60
        );

        amountOut = amountsOut[amountsOut.length - 1];

        _checkApprovalIsZero(assetIn, address(uniswapV2Router));
    }

    function swapWithUniV3(
        bytes memory swapData,
        address receiver,
        ERC20 assetIn,
        ERC20 assetOut
    ) public returns (uint256 amountOut) {
        (address[] memory path, uint24[] memory poolFees, uint256 amount, uint256 amountOutMin) = abi.decode(
            swapData,
            (address[], uint24[], uint256, uint256)
        );

        if (assetIn != ERC20(path[0])) revert SwapRouter__AssetInMisMatch(path[0], address(assetIn));
        if (assetOut != ERC20(path[path.length - 1]))
            revert SwapRouter__AssetOutMisMatch(path[path.length - 1], address(assetOut));

        assetIn.safeTransferFrom(msg.sender, address(this), amount);

        assetIn.safeApprove(address(uniswapV3Router), amount);

        bytes memory encodePackedPath = abi.encodePacked(address(assetIn));
        for (uint256 i = 1; i < path.length; i++)
            encodePackedPath = abi.encodePacked(encodePackedPath, poolFees[i - 1], path[i]);

        amountOut = uniswapV3Router.exactInput(
            IUniswapV3Router.ExactInputParams({
                path: encodePackedPath,
                recipient: receiver,
                deadline: block.timestamp + 60,
                amountIn: amount,
                amountOutMinimum: amountOutMin
            })
        );

        _checkApprovalIsZero(assetIn, address(uniswapV3Router));
    }

    error SwapRouter__UnusedApproval();

    function _checkApprovalIsZero(ERC20 asset, address spender) internal view {
        if (asset.allowance(address(this), spender) != 0) revert SwapRouter__UnusedApproval();
    }
}

abstract contract BaseAdaptor {
    using SafeTransferLib for ERC20;
    using Math for uint256;

    error BaseAdaptor__ExternalReceiverBlocked();

    error BaseAdaptor__UserDepositsNotAllowed();

    error BaseAdaptor__UserWithdrawsNotAllowed();

    function identifier() public pure virtual returns (bytes32) {
        return keccak256(abi.encode("Base Adaptor V 0.0"));
    }

    function SWAP_ROUTER_REGISTRY_SLOT() internal pure returns (uint256) {
        return 1;
    }

    function PRICE_ROUTER_REGISTRY_SLOT() internal pure returns (uint256) {
        return 2;
    }

    function deposit(uint256 assets, bytes memory adaptorData, bytes memory configurationData) public virtual;

    function withdraw(
        uint256 assets,
        address receiver,
        bytes memory adaptorData,
        bytes memory configurationData
    ) public virtual;

    function balanceOf(bytes memory adaptorData) public view virtual returns (uint256);

    function withdrawableFrom(bytes memory, bytes memory) public view virtual returns (uint256);

    function assetOf(bytes memory adaptorData) public view virtual returns (ERC20);

    function assetsUsed(bytes memory adaptorData) public view virtual returns (ERC20[] memory assets) {
        assets = new ERC20[](1);
        assets[0] = assetOf(adaptorData);
    }

    function isDebt() public view virtual returns (bool);

    function _maxAvailable(ERC20 token, uint256 amount) internal view virtual returns (uint256) {
        if (amount == type(uint256).max) return token.balanceOf(address(this));
        else return amount;
    }

    function _revokeExternalApproval(ERC20 asset, address spender) internal {
        if (asset.allowance(address(this), spender) > 0) asset.safeApprove(spender, 0);
    }

    function swap(
        ERC20 assetIn,
        ERC20 assetOut,
        uint256 amountIn,
        SwapRouter.Exchange exchange,
        bytes memory params
    ) public returns (uint256 amountOut) {

        SwapRouter swapRouter = SwapRouter(Cellar(address(this)).registry().getAddress(SWAP_ROUTER_REGISTRY_SLOT()));

        assetIn.safeApprove(address(swapRouter), amountIn);

        amountOut = swapRouter.swap(exchange, params, address(this), assetIn, assetOut);
    }

    function _externalReceiverCheck(address receiver) internal view {
        if (receiver != address(this) && Cellar(address(this)).blockExternalReceiver())
            revert BaseAdaptor__ExternalReceiverBlocked();
    }

    error BaseAdaptor__BadSlippage();

    error BaseAdaptor__ExchangeNotSupported();

    function oracleSwap(
        ERC20 assetIn,
        ERC20 assetOut,
        uint256 amountIn,
        SwapRouter.Exchange exchange,
        bytes memory params,
        uint64 slippage
    ) public returns (uint256 amountOut) {
        amountIn = _maxAvailable(assetIn, amountIn);

        if (exchange == SwapRouter.Exchange.UNIV2) {
            address[] memory path = abi.decode(params, (address[]));
            params = abi.encode(path, amountIn, 0);
        } else if (exchange == SwapRouter.Exchange.UNIV3) {
            (address[] memory path, uint24[] memory poolFees) = abi.decode(params, (address[], uint24[]));
            params = abi.encode(path, poolFees, amountIn, 0);
        } else revert BaseAdaptor__ExchangeNotSupported();

        SwapRouter swapRouter = SwapRouter(Cellar(address(this)).registry().getAddress(SWAP_ROUTER_REGISTRY_SLOT()));

        PriceRouter priceRouter = PriceRouter(
            Cellar(address(this)).registry().getAddress(PRICE_ROUTER_REGISTRY_SLOT())
        );

        assetIn.safeApprove(address(swapRouter), amountIn);

        amountOut = swapRouter.swap(exchange, params, address(this), assetIn, assetOut);

        amountIn = priceRouter.getValue(assetIn, amountIn, assetOut);
        uint256 amountInWithSlippage = amountIn.mulDivDown(slippage, 1e18);
        if (amountOut < amountInWithSlippage) revert BaseAdaptor__BadSlippage();
    }

    function revokeApproval(ERC20 asset, address spender) public {
        asset.safeApprove(spender, 0);
    }
}

pragma abicoder v2;

interface IERC165 {

    function supportsInterface(bytes4 interfaceId) external view returns (bool);
}

interface IERC721 is IERC165 {

    event Transfer(address indexed from, address indexed to, uint256 indexed tokenId);

    event Approval(address indexed owner, address indexed approved, uint256 indexed tokenId);

    event ApprovalForAll(address indexed owner, address indexed operator, bool approved);

    function balanceOf(address owner) external view returns (uint256 balance);

    function ownerOf(uint256 tokenId) external view returns (address owner);

    function safeTransferFrom(address from, address to, uint256 tokenId, bytes calldata data) external;

    function safeTransferFrom(address from, address to, uint256 tokenId) external;

    function transferFrom(address from, address to, uint256 tokenId) external;

    function approve(address to, uint256 tokenId) external;

    function setApprovalForAll(address operator, bool _approved) external;

    function getApproved(uint256 tokenId) external view returns (address operator);

    function isApprovedForAll(address owner, address operator) external view returns (bool);
}

interface IERC721Metadata is IERC721 {

    function name() external view returns (string memory);

    function symbol() external view returns (string memory);

    function tokenURI(uint256 tokenId) external view returns (string memory);
}

interface IERC721Enumerable is IERC721 {

    function totalSupply() external view returns (uint256);

    function tokenOfOwnerByIndex(address owner, uint256 index) external view returns (uint256);

    function tokenByIndex(uint256 index) external view returns (uint256);
}

interface IPoolInitializer {

    function createAndInitializePoolIfNecessary(
        address token0,
        address token1,
        uint24 fee,
        uint160 sqrtPriceX96
    ) external payable returns (address pool);
}

interface IERC721Permit is IERC721 {

    function PERMIT_TYPEHASH() external pure returns (bytes32);

    function DOMAIN_SEPARATOR() external view returns (bytes32);

    function permit(address spender, uint256 tokenId, uint256 deadline, uint8 v, bytes32 r, bytes32 s) external payable;
}

interface IPeripheryPayments {

    function unwrapWETH9(uint256 amountMinimum, address recipient) external payable;

    function refundETH() external payable;

    function sweepToken(address token, uint256 amountMinimum, address recipient) external payable;
}

interface IPeripheryImmutableState {

    function factory() external view returns (address);

    function WETH9() external view returns (address);
}

library PoolAddress {
    bytes32 internal constant POOL_INIT_CODE_HASH = 0xa598dd2fba360510c5a8f02f44423a4468e902df5857dbce3ca162a43a3a31ff;

    struct PoolKey {
        address token0;
        address token1;
        uint24 fee;
    }

    function getPoolKey(address tokenA, address tokenB, uint24 fee) internal pure returns (PoolKey memory) {
        if (tokenA > tokenB) (tokenA, tokenB) = (tokenB, tokenA);
        return PoolKey({ token0: tokenA, token1: tokenB, fee: fee });
    }

    function computeAddress(address factory, PoolKey memory key) internal pure returns (address pool) {
        require(key.token0 < key.token1);
        pool = address(
            uint160(
                uint256(
                    keccak256(
                        abi.encodePacked(
                            hex"ff",
                            factory,
                            keccak256(abi.encode(key.token0, key.token1, key.fee)),
                            POOL_INIT_CODE_HASH
                        )
                    )
                )
            )
        );
    }
}

interface INonfungiblePositionManager is
    IPoolInitializer,
    IPeripheryPayments,
    IPeripheryImmutableState,
    IERC721Metadata,
    IERC721Enumerable,
    IERC721Permit
{

    event IncreaseLiquidity(uint256 indexed tokenId, uint128 liquidity, uint256 amount0, uint256 amount1);

    event DecreaseLiquidity(uint256 indexed tokenId, uint128 liquidity, uint256 amount0, uint256 amount1);

    event Collect(uint256 indexed tokenId, address recipient, uint256 amount0, uint256 amount1);

    function positions(
        uint256 tokenId
    )
        external
        view
        returns (
            uint96 nonce,
            address operator,
            address token0,
            address token1,
            uint24 fee,
            int24 tickLower,
            int24 tickUpper,
            uint128 liquidity,
            uint256 feeGrowthInside0LastX128,
            uint256 feeGrowthInside1LastX128,
            uint128 tokensOwed0,
            uint128 tokensOwed1
        );

    struct MintParams {
        address token0;
        address token1;
        uint24 fee;
        int24 tickLower;
        int24 tickUpper;
        uint256 amount0Desired;
        uint256 amount1Desired;
        uint256 amount0Min;
        uint256 amount1Min;
        address recipient;
        uint256 deadline;
    }

    function mint(
        MintParams calldata params
    ) external payable returns (uint256 tokenId, uint128 liquidity, uint256 amount0, uint256 amount1);

    struct IncreaseLiquidityParams {
        uint256 tokenId;
        uint256 amount0Desired;
        uint256 amount1Desired;
        uint256 amount0Min;
        uint256 amount1Min;
        uint256 deadline;
    }

    function increaseLiquidity(
        IncreaseLiquidityParams calldata params
    ) external payable returns (uint128 liquidity, uint256 amount0, uint256 amount1);

    struct DecreaseLiquidityParams {
        uint256 tokenId;
        uint128 liquidity;
        uint256 amount0Min;
        uint256 amount1Min;
        uint256 deadline;
    }

    function decreaseLiquidity(
        DecreaseLiquidityParams calldata params
    ) external payable returns (uint256 amount0, uint256 amount1);

    struct CollectParams {
        uint256 tokenId;
        address recipient;
        uint128 amount0Max;
        uint128 amount1Max;
    }

    function collect(CollectParams calldata params) external payable returns (uint256 amount0, uint256 amount1);

    function burn(uint256 tokenId) external payable;
}

contract UniswapV3PositionTracker {

    uint256 public constant MAX_HOLDINGS = 20;

    mapping(address => mapping(ERC20 => mapping(ERC20 => uint256[]))) private callerToToken0ToToken1ToHoldings;

    error UniswapV3PositionTracker__MaxHoldingsExceeded();
    error UniswapV3PositionTracker__CallerDoesNotOwnTokenId();
    error UniswapV3PositionTracker__CallerOwnsTokenId();
    error UniswapV3PositionTracker__TokenIdAlreadyTracked();
    error UniswapV3PositionTracker__TokenIdMustBeOwnedByDeadAddress();
    error UniswapV3PositionTracker__TokenIdNotFound();

    INonfungiblePositionManager public immutable positionManager;

    constructor(INonfungiblePositionManager _positionManager) {
        positionManager = _positionManager;
    }

    function addPositionToArray(uint256 tokenId, ERC20 token0, ERC20 token1) external {
        if (positionManager.ownerOf(tokenId) != msg.sender) revert UniswapV3PositionTracker__CallerDoesNotOwnTokenId();
        uint256 holdingLength = callerToToken0ToToken1ToHoldings[msg.sender][token0][token1].length;

        if (holdingLength >= MAX_HOLDINGS) revert UniswapV3PositionTracker__MaxHoldingsExceeded();

        if (checkIfPositionIsInTracker(msg.sender, tokenId, token0, token1))
            revert UniswapV3PositionTracker__TokenIdAlreadyTracked();

        callerToToken0ToToken1ToHoldings[msg.sender][token0][token1].push(tokenId);
    }

    function removePositionFromArray(uint256 tokenId, ERC20 token0, ERC20 token1) external {
        _iterateArrayAndPopTarget(tokenId, token0, token1);

        positionManager.transferFrom(msg.sender, address(this), tokenId);

        positionManager.burn(tokenId);
    }

    function removePositionFromArrayThatIsNotOwnedByCaller(uint256 tokenId, ERC20 token0, ERC20 token1) external {
        if (positionManager.ownerOf(tokenId) == msg.sender) revert UniswapV3PositionTracker__CallerOwnsTokenId();

        _iterateArrayAndPopTarget(tokenId, token0, token1);
    }

    function checkIfPositionIsInTracker(
        address caller,
        uint256 tokenId,
        ERC20 token0,
        ERC20 token1
    ) public view returns (bool tokenFound) {

        uint256[] storage holdings = callerToToken0ToToken1ToHoldings[caller][token0][token1];
        uint256 holdingLength = holdings.length;

        for (uint256 i; i < holdingLength; ++i) {
            uint256 currentTokenId = holdings[i];
            if (currentTokenId == tokenId) {
                return true;
            }
        }

        return false;
    }

    function getTokens(address caller, ERC20 token0, ERC20 token1) external view returns (uint256[] memory tokens) {
        return callerToToken0ToToken1ToHoldings[caller][token0][token1];
    }

    function _iterateArrayAndPopTarget(uint256 targetId, ERC20 token0, ERC20 token1) internal {
        uint256[] storage holdings = callerToToken0ToToken1ToHoldings[msg.sender][token0][token1];
        uint256 holdingLength = holdings.length;

        for (uint256 i; i < holdingLength; ++i) {
            uint256 currentTokenId = holdings[i];
            if (currentTokenId == targetId) {

                holdings[i] = holdings[holdingLength - 1];
                holdings.pop();
                return;
            }
        }

        revert UniswapV3PositionTracker__TokenIdNotFound();
    }
}
