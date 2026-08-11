pragma solidity >=0.8.0;

abstract contract Context {
    function _msgSender() internal view virtual returns (address payable) {
        return payable(msg.sender);
    }

    function _msgData() internal view virtual returns (bytes memory) {
        this;
        return msg.data;
    }
}

library SafeMath {

    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");

        return c;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        return sub(a, b, "SafeMath: subtraction overflow");
    }

    function sub(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        require(b <= a, errorMessage);
        uint256 c = a - b;

        return c;
    }

    function mul(uint256 a, uint256 b) internal pure returns (uint256) {

        if (a == 0) {
            return 0;
        }

        uint256 c = a * b;
        require(c / a == b, "SafeMath: multiplication overflow");

        return c;
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        return div(a, b, "SafeMath: division by zero");
    }

    function div(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {

        require(b > 0, errorMessage);
        uint256 c = a / b;

        return c;
    }

    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        return mod(a, b, "SafeMath: modulo by zero");
    }

    function mod(uint256 a, uint256 b, string memory errorMessage) internal pure returns (uint256) {
        require(b != 0, errorMessage);
        return a % b;
    }
}

interface IERC20 {

    function totalSupply() external view returns (uint256);

    function balanceOf(address account) external view returns (uint256);

    function transfer(address recipient, uint256 amount) external returns (bool);

    function allowance(address owner, address spender) external view returns (uint256);

    function approve(address spender, uint256 amount) external returns (bool);

    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);

    event Transfer(address indexed from, address indexed to, uint256 value);

    event Approval(address indexed owner, address indexed spender, uint256 value);
}

interface IUniswapV3PoolActions {

    function initialize(uint160 sqrtPriceX96) external;

    function mint(
        address recipient,
        int24 tickLower,
        int24 tickUpper,
        uint128 amount,
        bytes calldata data
    ) external returns (uint256 amount0, uint256 amount1);

    function collect(
        address recipient,
        int24 tickLower,
        int24 tickUpper,
        uint128 amount0Requested,
        uint128 amount1Requested
    ) external returns (uint128 amount0, uint128 amount1);

    function burn(
        int24 tickLower,
        int24 tickUpper,
        uint128 amount
    ) external returns (uint256 amount0, uint256 amount1);

    function swap(
        address recipient,
        bool zeroForOne,
        int256 amountSpecified,
        uint160 sqrtPriceLimitX96,
        bytes calldata data
    ) external returns (int256 amount0, int256 amount1);

    function flash(
        address recipient,
        uint256 amount0,
        uint256 amount1,
        bytes calldata data
    ) external;

    function increaseObservationCardinalityNext(uint16 observationCardinalityNext) external;
}

interface IUniswapV3PoolDerivedState {

    function observe(uint32[] calldata secondsAgos)
        external
        view
        returns (int56[] memory tickCumulatives, uint160[] memory secondsPerLiquidityCumulativeX128s);

    function snapshotCumulativesInside(int24 tickLower, int24 tickUpper)
        external
        view
        returns (
            int56 tickCumulativeInside,
            uint160 secondsPerLiquidityInsideX128,
            uint32 secondsInside
        );
}

interface IUniswapV3PoolEvents {

    event Initialize(uint160 sqrtPriceX96, int24 tick);

    event Mint(
        address sender,
        address indexed owner,
        int24 indexed tickLower,
        int24 indexed tickUpper,
        uint128 amount,
        uint256 amount0,
        uint256 amount1
    );

    event Collect(
        address indexed owner,
        address recipient,
        int24 indexed tickLower,
        int24 indexed tickUpper,
        uint128 amount0,
        uint128 amount1
    );

    event Burn(
        address indexed owner,
        int24 indexed tickLower,
        int24 indexed tickUpper,
        uint128 amount,
        uint256 amount0,
        uint256 amount1
    );

    event Swap(
        address indexed sender,
        address indexed recipient,
        int256 amount0,
        int256 amount1,
        uint160 sqrtPriceX96,
        uint128 liquidity,
        int24 tick
    );

    event Flash(
        address indexed sender,
        address indexed recipient,
        uint256 amount0,
        uint256 amount1,
        uint256 paid0,
        uint256 paid1
    );

    event IncreaseObservationCardinalityNext(
        uint16 observationCardinalityNextOld,
        uint16 observationCardinalityNextNew
    );

    event SetFeeProtocol(uint8 feeProtocol0Old, uint8 feeProtocol1Old, uint8 feeProtocol0New, uint8 feeProtocol1New);

    event CollectProtocol(address indexed sender, address indexed recipient, uint128 amount0, uint128 amount1);
}

interface IUniswapV3PoolImmutables {

    function factory() external view returns (address);

    function token0() external view returns (address);

    function token1() external view returns (address);

    function fee() external view returns (uint24);

    function tickSpacing() external view returns (int24);

    function maxLiquidityPerTick() external view returns (uint128);
}

interface IUniswapV3PoolOwnerActions {

    function setFeeProtocol(uint8 feeProtocol0, uint8 feeProtocol1) external;

    function collectProtocol(
        address recipient,
        uint128 amount0Requested,
        uint128 amount1Requested
    ) external returns (uint128 amount0, uint128 amount1);
}

interface IUniswapV3PoolState {

    function slot0()
        external
        view
        returns (
            uint160 sqrtPriceX96,
            int24 tick,
            uint16 observationIndex,
            uint16 observationCardinality,
            uint16 observationCardinalityNext,
            uint8 feeProtocol,
            bool unlocked
        );

    function feeGrowthGlobal0X128() external view returns (uint256);

    function feeGrowthGlobal1X128() external view returns (uint256);

    function protocolFees() external view returns (uint128 token0, uint128 token1);

    function liquidity() external view returns (uint128);

    function ticks(int24 tick)
        external
        view
        returns (
            uint128 liquidityGross,
            int128 liquidityNet,
            uint256 feeGrowthOutside0X128,
            uint256 feeGrowthOutside1X128,
            int56 tickCumulativeOutside,
            uint160 secondsPerLiquidityOutsideX128,
            uint32 secondsOutside,
            bool initialized
        );

    function tickBitmap(int16 wordPosition) external view returns (uint256);

    function positions(bytes32 key)
        external
        view
        returns (
            uint128 _liquidity,
            uint256 feeGrowthInside0LastX128,
            uint256 feeGrowthInside1LastX128,
            uint128 tokensOwed0,
            uint128 tokensOwed1
        );

    function observations(uint256 index)
        external
        view
        returns (
            uint32 blockTimestamp,
            int56 tickCumulative,
            uint160 secondsPerLiquidityCumulativeX128,
            bool initialized
        );
}

interface IUniswapV3Pool is
    IUniswapV3PoolImmutables,
    IUniswapV3PoolState,
    IUniswapV3PoolDerivedState,
    IUniswapV3PoolActions,
    IUniswapV3PoolOwnerActions,
    IUniswapV3PoolEvents
{

}

interface IBunniLens {
    struct BunniKey {
        IUniswapV3Pool pool;
        int24 tickLower;
        int24 tickUpper;
    }

    function getReserves (BunniKey calldata key) external view returns (uint112 reserve0, uint112 reserve1);
    function hub () external view returns (address);
    function pricePerFullShare (BunniKey calldata key) external view returns (uint128 liquidity, uint256 amount0, uint256 amount1);
}

interface IBunniGauge {
  function deposit ( uint256 _value ) external;
  function deposit ( uint256 _value, address _addr ) external;
  function deposit ( uint256 _value, address _addr, bool _claim_rewards ) external;
  function withdraw ( uint256 _value ) external;
  function withdraw ( uint256 _value, bool _claim_rewards ) external;
  function claim_rewards (  ) external;
  function claim_rewards ( address _addr ) external;
  function claim_rewards ( address _addr, address _receiver ) external;
  function transferFrom ( address _from, address _to, uint256 _value ) external returns ( bool );
  function transfer ( address _to, uint256 _value ) external returns ( bool );
  function approve ( address _spender, uint256 _value ) external returns ( bool );
  function permit ( address _owner, address _spender, uint256 _value, uint256 _deadline, uint8 _v, bytes32 _r, bytes32 _s ) external returns ( bool );
  function increaseAllowance ( address _spender, uint256 _added_value ) external returns ( bool );
  function decreaseAllowance ( address _spender, uint256 _subtracted_value ) external returns ( bool );
  function user_checkpoint ( address addr ) external returns ( bool );
  function set_rewards_receiver ( address _receiver ) external;
  function kick ( address addr ) external;
  function deposit_reward_token ( address _reward_token, uint256 _amount ) external;
  function add_reward ( address _reward_token, address _distributor ) external;
  function set_reward_distributor ( address _reward_token, address _distributor ) external;
  function makeGaugePermissionless (  ) external;
  function killGauge (  ) external;
  function unkillGauge (  ) external;
  function change_pending_admin ( address new_pending_admin ) external;
  function claim_admin (  ) external;
  function set_tokenless_production ( uint8 new_tokenless_production ) external;
  function claimed_reward ( address _addr, address _token ) external view returns ( uint256 );
  function claimable_reward ( address _user, address _reward_token ) external view returns ( uint256 );
  function claimable_tokens ( address addr ) external returns ( uint256 );
  function integrate_checkpoint (  ) external view returns ( uint256 );
  function future_epoch_time (  ) external view returns ( uint256 );
  function inflation_rate (  ) external view returns ( uint256 );
  function decimals (  ) external view returns ( uint256 );
  function version (  ) external view returns ( string memory );
  function allowance ( address owner, address spender ) external view returns ( uint256 );
  function is_killed (  ) external view returns ( bool );
  function initialize ( address _lp_token, uint256 relative_weight_cap, address _voting_escrow_delegation, address _admin, bytes32 _position_key ) external;
  function setRelativeWeightCap ( uint256 relative_weight_cap ) external;
  function getRelativeWeightCap (  ) external view returns ( uint256 );
  function getCappedRelativeWeight ( uint256 time ) external view returns ( uint256 );
  function getMaxRelativeWeightCap (  ) external pure returns ( uint256 );
  function tokenless_production (  ) external view returns ( uint8 );
  function pending_admin (  ) external view returns ( address );
  function admin (  ) external view returns ( address );
  function voting_escrow_delegation (  ) external view returns ( address );
  function balanceOf ( address arg0 ) external view returns ( uint256 );
  function totalSupply (  ) external view returns ( uint256 );
  function name (  ) external view returns ( string memory );
  function symbol (  ) external view returns ( string memory );
  function DOMAIN_SEPARATOR (  ) external view returns ( bytes32 );
  function nonces ( address arg0 ) external view returns ( uint256 );
  function lp_token (  ) external view returns ( address );
  function gauge_state (  ) external view returns ( uint8 );
  function position_key (  ) external view returns ( bytes32 );
  function reward_count (  ) external view returns ( uint256 );

  function rewards_receiver ( address arg0 ) external view returns ( address );
  function reward_integral_for ( address arg0, address arg1 ) external view returns ( uint256 );
  function working_balances ( address arg0 ) external view returns ( uint256 );
  function working_supply (  ) external view returns ( uint256 );
  function integrate_inv_supply_of ( address arg0 ) external view returns ( uint256 );
  function integrate_checkpoint_of ( address arg0 ) external view returns ( uint256 );
  function integrate_fraction ( address arg0 ) external view returns ( uint256 );
  function period (  ) external view returns ( int128 );
  function reward_tokens ( uint256 arg0 ) external view returns ( address );
  function period_timestamp ( uint256 arg0 ) external view returns ( uint256 );
  function integrate_inv_supply ( uint256 arg0 ) external view returns ( uint256 );
}

interface IFraxGaugeController {
    struct Point {
        uint256 bias;
        uint256 slope;
    }

    struct VotedSlope {
        uint256 slope;
        uint256 power;
        uint256 end;
    }

    function admin() external view returns (address);
    function future_admin() external view returns (address);
    function token() external view returns (address);
    function voting_escrow() external view returns (address);
    function n_gauge_types() external view returns (int128);
    function n_gauges() external view returns (int128);
    function gauge_type_names(int128) external view returns (string memory);
    function gauges(uint256) external view returns (address);
    function vote_user_slopes(address, address)
        external
        view
        returns (VotedSlope memory);
    function vote_user_power(address) external view returns (uint256);
    function last_user_vote(address, address) external view returns (uint256);
    function points_weight(address, uint256)
        external
        view
        returns (Point memory);
    function time_weight(address) external view returns (uint256);
    function points_sum(int128, uint256) external view returns (Point memory);
    function time_sum(uint256) external view returns (uint256);
    function points_total(uint256) external view returns (uint256);
    function time_total() external view returns (uint256);
    function points_type_weight(int128, uint256)
        external
        view
        returns (uint256);
    function time_type_weight(uint256) external view returns (uint256);

    function gauge_types(address) external view returns (int128);
    function gauge_relative_weight(address) external view returns (uint256);
    function gauge_relative_weight(address, uint256) external view returns (uint256);
    function get_gauge_weight(address) external view returns (uint256);
    function get_type_weight(int128) external view returns (uint256);
    function get_total_weight() external view returns (uint256);
    function get_weights_sum_per_type(int128) external view returns (uint256);

    function commit_transfer_ownership(address) external;
    function apply_transfer_ownership() external;
    function add_gauge(
        address,
        int128,
        uint256
    ) external;
    function checkpoint() external;
    function checkpoint_gauge(address) external;
    function global_emission_rate() external view returns (uint256);
    function gauge_relative_weight_write(address)
        external
        returns (uint256);
    function gauge_relative_weight_write(address, uint256)
        external
        returns (uint256);
    function add_type(string memory, uint256) external;
    function change_type_weight(int128, uint256) external;
    function change_gauge_weight(address, uint256) external;
    function change_global_emission_rate(uint256) external;
    function vote_for_gauge_weights(address, uint256) external;
}

interface IFraxGaugeFXSRewardsDistributor {
  function acceptOwnership() external;
  function curator_address() external view returns(address);
  function currentReward(address gauge_address) external view returns(uint256 reward_amount);
  function distributeReward(address gauge_address) external returns(uint256 weeks_elapsed, uint256 reward_tally);
  function distributionsOn() external view returns(bool);
  function gauge_whitelist(address) external view returns(bool);
  function is_middleman(address) external view returns(bool);
  function last_time_gauge_paid(address) external view returns(uint256);
  function nominateNewOwner(address _owner) external;
  function nominatedOwner() external view returns(address);
  function owner() external view returns(address);
  function recoverERC20(address tokenAddress, uint256 tokenAmount) external;
  function setCurator(address _new_curator_address) external;
  function setGaugeController(address _gauge_controller_address) external;
  function setGaugeState(address _gauge_address, bool _is_middleman, bool _is_active) external;
  function setTimelock(address _new_timelock) external;
  function timelock_address() external view returns(address);
  function toggleDistributions() external;
}

interface IveFXS {

    struct LockedBalance {
        int128 amount;
        uint256 end;
    }

    function commit_transfer_ownership(address addr) external;
    function apply_transfer_ownership() external;
    function commit_smart_wallet_checker(address addr) external;
    function apply_smart_wallet_checker() external;
    function toggleEmergencyUnlock() external;
    function recoverERC20(address token_addr, uint256 amount) external;
    function get_last_user_slope(address addr) external view returns (int128);
    function user_point_history__ts(address _addr, uint256 _idx) external view returns (uint256);
    function locked__end(address _addr) external view returns (uint256);
    function checkpoint() external;
    function deposit_for(address _addr, uint256 _value) external;
    function create_lock(uint256 _value, uint256 _unlock_time) external;
    function increase_amount(uint256 _value) external;
    function increase_unlock_time(uint256 _unlock_time) external;
    function withdraw() external;
    function balanceOf(address addr) external view returns (uint256);
    function balanceOf(address addr, uint256 _t) external view returns (uint256);
    function balanceOfAt(address addr, uint256 _block) external view returns (uint256);
    function totalSupply() external view returns (uint256);
    function totalSupply(uint256 t) external view returns (uint256);
    function totalSupplyAt(uint256 _block) external view returns (uint256);
    function totalFXSSupply() external view returns (uint256);
    function totalFXSSupplyAt(uint256 _block) external view returns (uint256);
    function changeController(address _newController) external;
    function token() external view returns (address);
    function supply() external view returns (uint256);
    function locked(address addr) external view returns (LockedBalance memory);
    function epoch() external view returns (uint256);
    function point_history(uint256 arg0) external view returns (int128 bias, int128 slope, uint256 ts, uint256 blk, uint256 fxs_amt);
    function user_point_history(address arg0, uint256 arg1) external view returns (int128 bias, int128 slope, uint256 ts, uint256 blk, uint256 fxs_amt);
    function user_point_epoch(address arg0) external view returns (uint256);
    function slope_changes(uint256 arg0) external view returns (int128);
    function controller() external view returns (address);
    function transfersEnabled() external view returns (bool);
    function emergencyUnlockActive() external view returns (bool);
    function name() external view returns (string memory);
    function symbol() external view returns (string memory);
    function version() external view returns (string memory);
    function decimals() external view returns (uint256);
    function future_smart_wallet_checker() external view returns (address);
    function smart_wallet_checker() external view returns (address);
    function admin() external view returns (address);
    function future_admin() external view returns (address);
}

library Math {

    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return a >= b ? a : b;
    }

    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }

    function average(uint256 a, uint256 b) internal pure returns (uint256) {

        return (a / 2) + (b / 2) + ((a % 2 + b % 2) / 2);
    }

    function sqrt(uint y) internal pure returns (uint z) {
        if (y > 3) {
            z = y;
            uint x = y / 2 + 1;
            while (x < z) {
                z = x;
                x = (y / x + x) / 2;
            }
        } else if (y != 0) {
            z = 1;
        }
    }
}

interface IBunniMinter {
  function allowed_to_mint_for ( address minter, address user ) external view returns ( bool );
  function getGaugeController (  ) external view returns ( address );
  function getMinterApproval ( address minter, address user ) external view returns ( bool );
  function getToken (  ) external view returns ( address );
  function getTokenAdmin (  ) external view returns ( address );
  function mint ( address gauge ) external returns ( uint256 );
  function mintFor ( address gauge, address user ) external returns ( uint256 );
  function mintMany ( address[] memory gauges ) external returns ( uint256 );
  function mintManyFor ( address[] memory gauges, address user ) external returns ( uint256 );
  function mint_for ( address gauge, address user ) external;
  function mint_many ( address[8] memory gauges ) external;
  function minted ( address user, address gauge ) external view returns ( uint256 );
  function setMinterApproval ( address minter, bool approval ) external;
  function toggle_approve_mint ( address minter ) external;
}

contract Owned {
    address public owner;
    address public nominatedOwner;

    constructor (address _owner) public {
        require(_owner != address(0), "Owner address cannot be 0");
        owner = _owner;
        emit OwnerChanged(address(0), _owner);
    }

    function nominateNewOwner(address _owner) external onlyOwner {
        nominatedOwner = _owner;
        emit OwnerNominated(_owner);
    }

    function acceptOwnership() external {
        require(msg.sender == nominatedOwner, "You must be nominated before you can accept ownership");
        emit OwnerChanged(owner, nominatedOwner);
        owner = nominatedOwner;
        nominatedOwner = address(0);
    }

    modifier onlyOwner {
        require(msg.sender == owner, "Only the contract owner may perform this action");
        _;
    }

    event OwnerNominated(address newOwner);
    event OwnerChanged(address oldOwner, address newOwner);
}

library TransferHelper {
    function safeApprove(address token, address to, uint value) internal {

        (bool success, bytes memory data) = token.call(abi.encodeWithSelector(0x095ea7b3, to, value));
        require(success && (data.length == 0 || abi.decode(data, (bool))), 'TransferHelper: APPROVE_FAILED');
    }

    function safeTransfer(address token, address to, uint value) internal {

        (bool success, bytes memory data) = token.call(abi.encodeWithSelector(0xa9059cbb, to, value));
        require(success && (data.length == 0 || abi.decode(data, (bool))), 'TransferHelper: TRANSFER_FAILED');
    }

    function safeTransferFrom(address token, address from, address to, uint value) internal {

        (bool success, bytes memory data) = token.call(abi.encodeWithSelector(0x23b872dd, from, to, value));
        require(success && (data.length == 0 || abi.decode(data, (bool))), 'TransferHelper: TRANSFER_FROM_FAILED');
    }

    function safeTransferETH(address to, uint value) internal {
        (bool success,) = to.call{value:value}(new bytes(0));
        require(success, 'TransferHelper: ETH_TRANSFER_FAILED');
    }
}

abstract contract ReentrancyGuard {

    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;

    uint256 private _status;

    constructor () internal {
        _status = _NOT_ENTERED;
    }

    modifier nonReentrant() {

        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");

        _status = _ENTERED;

        _;

        _status = _NOT_ENTERED;
    }
}

contract FraxUnifiedFarmTemplate is Owned, ReentrancyGuard {

    IBunniGauge public stakingToken;
    IBunniLens public lens = IBunniLens(0xb73F303472C4fD4FF3B9f59ce0F9b13E47fbfD19);
    IBunniMinter public minter = IBunniMinter(0xF087521Ffca0Fa8A43F5C445773aB37C5f574DA0);

    IveFXS private constant veFXS = IveFXS(0xc8418aF6358FFddA74e09Ca9CC3Fe03Ca6aDC5b0);

    address internal constant frax_address = 0x853d955aCEf822Db058eb8505911ED77F175b99e;

    uint256 public fraxPerLPStored;

    uint256 internal constant MULTIPLIER_PRECISION = 1e18;

    uint256 public periodFinish;

    uint256 public lastUpdateTime;

    uint256 public lock_max_multiplier = uint256(2e18);
    uint256 public lock_time_for_max_multiplier = 1 * 1095 * 86400;

    uint256 public lock_time_min = 594000;

    uint256 public vefxs_boost_scale_factor = uint256(4e18);
    uint256 public vefxs_max_multiplier = uint256(2e18);
    uint256 public vefxs_per_frax_for_max_boost = uint256(4e18);
    mapping(address => uint256) internal _vefxsMultiplierStored;
    mapping(address => bool) internal valid_vefxs_proxies;
    mapping(address => mapping(address => bool)) internal proxy_allowed_stakers;

    mapping(address => address) public rewardManagers;
    address[] internal rewardTokens;
    address[] internal gaugeControllers;
    address[] internal rewardDistributors;
    uint256[] internal rewardRatesManual;
    mapping(address => bool) internal isRewardToken;

    mapping(address => uint256) public rewardTokenAddrToIdx;

    uint256 public constant rewardsDuration = 604800;

    uint256[] private rewardsPerTokenStored;
    mapping(address => mapping(uint256 => uint256)) private userRewardsPerTokenPaid;
    mapping(address => mapping(uint256 => uint256)) private rewards;
    mapping(address => uint256) public lastRewardClaimTime;

    uint256[] private last_gauge_relative_weights;
    uint256[] private last_gauge_time_totals;

    uint256 internal _total_liquidity_locked;
    uint256 internal _total_combined_weight;
    mapping(address => uint256) internal _locked_liquidity;
    mapping(address => uint256) internal _combined_weights;

    mapping(address => uint256) public proxy_lp_balances;

    mapping(address => address) public staker_designated_proxies;

    bool public stakesUnlocked;
    bool internal withdrawalsPaused;
    bool internal rewardsCollectionPaused;
    bool internal stakingPaused;
    bool internal collectRewardsOnWithdrawalPaused;

    modifier onlyByOwnGov() {
        require(msg.sender == owner || msg.sender == 0x8412ebf45bAC1B340BbE8F318b928C466c4E39CA, "Not owner or timelock");
        _;
    }

    modifier onlyTknMgrs(address reward_token_address) {
        require(msg.sender == owner || isTokenManagerFor(msg.sender, reward_token_address), "Not owner or tkn mgr");
        _;
    }

    modifier updateRewardAndBalanceMdf(address account, bool sync_too) {
        _updateRewardAndBalance(account, sync_too, false);
        _;
    }

    constructor (
        address _owner,
        address[] memory _rewardTokens,
        address[] memory _rewardManagers,
        uint256[] memory _rewardRatesManual,
        address[] memory _gaugeControllers,
        address[] memory _rewardDistributors
    ) Owned(_owner) {

        rewardTokens = _rewardTokens;
        gaugeControllers = _gaugeControllers;
        rewardDistributors = _rewardDistributors;
        rewardRatesManual = _rewardRatesManual;

        for (uint256 i = 0; i < _rewardTokens.length; i++){

            rewardTokenAddrToIdx[_rewardTokens[i]] = i;

            isRewardToken[_rewardTokens[i]] = true;

            rewardsPerTokenStored.push(0);

            rewardManagers[_rewardTokens[i]] = _rewardManagers[i];

            last_gauge_relative_weights.push(0);

            last_gauge_time_totals.push(0);
        }

        stakesUnlocked = false;

        lastUpdateTime = block.timestamp;

        periodFinish = IFraxGaugeController(0x3669C421b77340B2979d1A00a792CC2ee0FcE737).time_total();

    }

    function isTokenManagerFor(address caller_addr, address reward_token_addr) public view returns (bool){
        if (!isRewardToken[reward_token_addr]) return false;
        else if (caller_addr == address(0) || reward_token_addr == address(0)) return false;
        else if (caller_addr == owner) return true;
        else if (rewardManagers[reward_token_addr] == caller_addr) return true;
        return false;
    }

    function getAllRewardTokens() external view returns (address[] memory) {
        return rewardTokens;
    }

    function lastTimeRewardApplicable() internal view returns (uint256) {
        return Math.min(block.timestamp, periodFinish);
    }

    function rewardRates(uint256 token_idx) public view returns (uint256 rwd_rate) {
        address gauge_controller_address = gaugeControllers[token_idx];
        if (gauge_controller_address != address(0)) {
            rwd_rate = (IFraxGaugeController(gauge_controller_address).global_emission_rate() * last_gauge_relative_weights[token_idx]) / 1e18;
        }
        else {
            rwd_rate = rewardRatesManual[token_idx];
        }
    }

    function rewardsPerToken() public view returns (uint256[] memory newRewardsPerTokenStored) {
        if (_total_liquidity_locked == 0 || _total_combined_weight == 0) {
            return rewardsPerTokenStored;
        }
        else {
            newRewardsPerTokenStored = new uint256[](rewardTokens.length);
            for (uint256 i = 0; i < rewardsPerTokenStored.length; i++){
                newRewardsPerTokenStored[i] = rewardsPerTokenStored[i] + (
                    ((lastTimeRewardApplicable() - lastUpdateTime) * rewardRates(i) * 1e18) / _total_combined_weight
                );
            }
            return newRewardsPerTokenStored;
        }
    }

    function earned(address account) public view returns (uint256[] memory new_earned) {
        uint256[] memory reward_arr = rewardsPerToken();
        new_earned = new uint256[](rewardTokens.length);

        if (_combined_weights[account] > 0){
            for (uint256 i = 0; i < rewardTokens.length; i++){
                new_earned[i] = ((_combined_weights[account] * (reward_arr[i] - userRewardsPerTokenPaid[account][i])) / 1e18)
                                + rewards[account][i];
            }
        }
    }

    function getRewardForDuration() external view returns (uint256[] memory rewards_per_duration_arr) {
        rewards_per_duration_arr = new uint256[](rewardRatesManual.length);

        for (uint256 i = 0; i < rewardRatesManual.length; i++){
            rewards_per_duration_arr[i] = rewardRates(i) * rewardsDuration;
        }
    }

    function totalLiquidityLocked() external view returns (uint256) {
        return _total_liquidity_locked;
    }

    function lockedLiquidityOf(address account) external view returns (uint256) {
        return _locked_liquidity[account];
    }

    function totalCombinedWeight() external view returns (uint256) {
        return _total_combined_weight;
    }

    function combinedWeightOf(address account) external view returns (uint256) {
        return _combined_weights[account];
    }

    function calcCurCombinedWeight(address account) public virtual view
        returns (
            uint256 old_combined_weight,
            uint256 new_vefxs_multiplier,
            uint256 new_combined_weight
        )
    {
        revert("Need cCCW logic");
    }

    function lockMultiplier(uint256 secs) public view returns (uint256) {
        return Math.min(
            lock_max_multiplier,
            (secs * lock_max_multiplier) / lock_time_for_max_multiplier
        ) ;
    }

    function userStakedFrax(address account) public view returns (uint256) {
        return (fraxPerLPStored * _locked_liquidity[account]) / MULTIPLIER_PRECISION;
    }

    function proxyStakedFrax(address proxy_address) public view returns (uint256) {
        return (fraxPerLPStored * proxy_lp_balances[proxy_address]) / MULTIPLIER_PRECISION;
    }

    function maxLPForMaxBoost(address account) external view returns (uint256) {
        return (veFXS.balanceOf(account) * MULTIPLIER_PRECISION * MULTIPLIER_PRECISION) / (vefxs_per_frax_for_max_boost * fraxPerLPStored);
    }

    function fraxPerLPToken() public virtual view returns (uint256) {
        revert("Need fPLPT logic");
    }

    function minVeFXSForMaxBoost(address account) public view returns (uint256) {
        return (userStakedFrax(account) * vefxs_per_frax_for_max_boost) / MULTIPLIER_PRECISION;
    }

    function minVeFXSForMaxBoostProxy(address proxy_address) public view returns (uint256) {
        return (proxyStakedFrax(proxy_address) * vefxs_per_frax_for_max_boost) / MULTIPLIER_PRECISION;
    }

    function getProxyFor(address addr) public view returns (address){
        if (valid_vefxs_proxies[addr]) {

            return addr;
        }
        else {

            return staker_designated_proxies[addr];
        }
    }

    function veFXSMultiplier(address account) public view returns (uint256 vefxs_multiplier) {

        uint256 vefxs_bal_to_use = 0;
        address the_proxy = getProxyFor(account);
        vefxs_bal_to_use = (the_proxy == address(0)) ? veFXS.balanceOf(account) : veFXS.balanceOf(the_proxy);

        uint256 mult_optn_1 = (vefxs_bal_to_use * vefxs_max_multiplier * vefxs_boost_scale_factor)
                            / (veFXS.totalSupply() * MULTIPLIER_PRECISION);

        uint256 mult_optn_2;
        {
            uint256 veFXS_needed_for_max_boost;

            veFXS_needed_for_max_boost = (the_proxy == address(0)) ? minVeFXSForMaxBoost(account) : minVeFXSForMaxBoostProxy(the_proxy);

            if (veFXS_needed_for_max_boost > 0){
                uint256 user_vefxs_fraction = (vefxs_bal_to_use * MULTIPLIER_PRECISION) / veFXS_needed_for_max_boost;

                mult_optn_2 = (user_vefxs_fraction * vefxs_max_multiplier) / MULTIPLIER_PRECISION;
            }
            else mult_optn_2 = 0;
        }

        vefxs_multiplier = (mult_optn_1 > mult_optn_2 ? mult_optn_1 : mult_optn_2);

        if (vefxs_multiplier > vefxs_max_multiplier) vefxs_multiplier = vefxs_max_multiplier;
    }

    function proxyToggleStaker(address staker_address) external {
        require(valid_vefxs_proxies[msg.sender], "Invalid proxy");
        proxy_allowed_stakers[msg.sender][staker_address] = !proxy_allowed_stakers[msg.sender][staker_address];

        if (staker_designated_proxies[staker_address] == msg.sender){
            staker_designated_proxies[staker_address] = address(0);

            proxy_lp_balances[msg.sender] -= _locked_liquidity[staker_address];
        }
    }

    function stakerSetVeFXSProxy(address proxy_address) external {
        require(valid_vefxs_proxies[proxy_address], "Invalid proxy");
        require(proxy_allowed_stakers[proxy_address][msg.sender], "Proxy has not allowed you yet");

        address old_proxy_addr = staker_designated_proxies[msg.sender];
        if (old_proxy_addr != address(0)) {

            proxy_lp_balances[old_proxy_addr] -= _locked_liquidity[msg.sender];
        }

        staker_designated_proxies[msg.sender] = proxy_address;

        proxy_lp_balances[proxy_address] += _locked_liquidity[msg.sender];
    }

    function _updateRewardAndBalance(address account, bool sync_too) internal {
        _updateRewardAndBalance(account, sync_too, false);
    }

    function _updateRewardAndBalance(address account, bool sync_too, bool pre_sync_vemxstored) internal {

        if (sync_too){
            sync();
        }

        if (pre_sync_vemxstored){
            _vefxsMultiplierStored[account] = veFXSMultiplier(account);
        }

        if (account != address(0)) {

            (
                uint256 old_combined_weight,
                uint256 new_vefxs_multiplier,
                uint256 new_combined_weight
            ) = calcCurCombinedWeight(account);

            _syncEarned(account);

            _vefxsMultiplierStored[account] = new_vefxs_multiplier;

            if (new_combined_weight >= old_combined_weight) {
                uint256 weight_diff = new_combined_weight - old_combined_weight;
                _total_combined_weight = _total_combined_weight + weight_diff;
                _combined_weights[account] = old_combined_weight + weight_diff;
            } else {
                uint256 weight_diff = old_combined_weight - new_combined_weight;
                _total_combined_weight = _total_combined_weight - weight_diff;
                _combined_weights[account] = old_combined_weight - weight_diff;
            }

        }
    }

    function _syncEarned(address account) internal {
        if (account != address(0)) {

            uint256[] memory earned_arr = earned(account);

            for (uint256 i = 0; i < earned_arr.length; i++){
                rewards[account][i] = earned_arr[i];
            }

            for (uint256 i = 0; i < earned_arr.length; i++){
                userRewardsPerTokenPaid[account][i] = rewardsPerTokenStored[i];
            }
        }
    }

    function getRewardExtraLogic(address destination_address) public nonReentrant {
        require(rewardsCollectionPaused == false, "Rewards collection paused");
        return _getRewardExtraLogic(msg.sender, destination_address);
    }

    function _getRewardExtraLogic(address rewardee, address destination_address) internal virtual {
        revert("Need gREL logic");
    }

    function getReward(address destination_address) external nonReentrant returns (uint256[] memory) {
        return _getReward(msg.sender, destination_address, true);
    }

    function getReward2(address destination_address, bool claim_extra_too) external nonReentrant returns (uint256[] memory) {
        return _getReward(msg.sender, destination_address, claim_extra_too);
    }

    function _getReward(address rewardee, address destination_address, bool do_extra_logic) internal updateRewardAndBalanceMdf(rewardee, true) returns (uint256[] memory rewards_before) {

        lastRewardClaimTime[rewardee] = block.timestamp;

        require(rewardsCollectionPaused == false, "Rewards collection paused");

        rewards_before = new uint256[](rewardTokens.length);

        for (uint256 i = 0; i < rewardTokens.length; i++){
            rewards_before[i] = rewards[rewardee][i];
            rewards[rewardee][i] = 0;
            if (rewards_before[i] > 0) {
                TransferHelper.safeTransfer(rewardTokens[i], destination_address, rewards_before[i]);

                emit RewardPaid(rewardee, rewards_before[i], rewardTokens[i], destination_address);
            }
        }

        if (do_extra_logic) {
            _getRewardExtraLogic(rewardee, destination_address);
        }
    }

    function retroCatchUp() internal {

        _updateStoredRewardsAndTime();

        for (uint256 i = 0; i < rewardDistributors.length; i++){
            address reward_distributor_address = rewardDistributors[i];
            if (reward_distributor_address != address(0)) {
                IFraxGaugeFXSRewardsDistributor(reward_distributor_address).distributeReward(address(this));
            }
        }

        uint256 num_periods_elapsed = uint256(block.timestamp - periodFinish) / rewardsDuration;

        for (uint256 i = 0; i < rewardTokens.length; i++){
            require((rewardRates(i) * rewardsDuration * (num_periods_elapsed + 1)) <= IERC20(rewardTokens[i]).balanceOf(address(this)), string(abi.encodePacked("Not enough reward tokens available: ", rewardTokens[i])) );
        }

        periodFinish = periodFinish + ((num_periods_elapsed + 1) * rewardsDuration);

        if (rewardRatesManual[1] != 0) {

            uint256 olit_before = IERC20(rewardTokens[1]).balanceOf(address(this));
            minter.mint(address(stakingToken));
            uint256 olit_after = IERC20(rewardTokens[1]).balanceOf(address(this));

            rewardRatesManual[1] = (olit_after - olit_before) / rewardsDuration;
        }

        _updateStoredRewardsAndTime();
    }

    function _updateStoredRewardsAndTime() internal {

        uint256[] memory rewards_per_token = rewardsPerToken();

        for (uint256 i = 0; i < rewardsPerTokenStored.length; i++){
            rewardsPerTokenStored[i] = rewards_per_token[i];
        }

        lastUpdateTime = lastTimeRewardApplicable();
    }

    function sync_gauge_weights(bool force_update) public {

        for (uint256 i = 0; i < gaugeControllers.length; i++){
            address gauge_controller_address = gaugeControllers[i];
            if (gauge_controller_address != address(0)) {
                if (force_update || (block.timestamp > last_gauge_time_totals[i])){

                    last_gauge_relative_weights[i] = IFraxGaugeController(gauge_controller_address).gauge_relative_weight_write(address(this), block.timestamp);
                    last_gauge_time_totals[i] = IFraxGaugeController(gauge_controller_address).time_total();
                }
            }
        }
    }

    function sync() public {

        sync_gauge_weights(false);

        fraxPerLPStored = fraxPerLPToken();

        if (block.timestamp >= periodFinish) {
            retroCatchUp();
        }
        else {
            _updateStoredRewardsAndTime();
        }
    }

    function setPauses(
        bool _stakingPaused,
        bool _withdrawalsPaused,
        bool _rewardsCollectionPaused,
        bool _collectRewardsOnWithdrawalPaused
    ) external onlyByOwnGov {
        stakingPaused = _stakingPaused;
        withdrawalsPaused = _withdrawalsPaused;
        rewardsCollectionPaused = _rewardsCollectionPaused;
        collectRewardsOnWithdrawalPaused = _collectRewardsOnWithdrawalPaused;
    }

    function unlockStakes() external onlyByOwnGov {
        stakesUnlocked = !stakesUnlocked;
    }

    function toggleValidVeFXSProxy(address _proxy_addr) external onlyByOwnGov {
        valid_vefxs_proxies[_proxy_addr] = !valid_vefxs_proxies[_proxy_addr];
    }

    function recoverERC20(address tokenAddress, uint256 tokenAmount) external onlyTknMgrs(tokenAddress) {

        bool isRewTkn = isRewardToken[tokenAddress];

        if (
                (isRewTkn && rewardManagers[tokenAddress] == msg.sender)
                || (!isRewTkn && (msg.sender == owner))
            ) {
            TransferHelper.safeTransfer(tokenAddress, msg.sender, tokenAmount);
            return;
        }

        else {
            revert("No valid tokens to recover");
        }
    }

    function setMiscVariables(
        uint256[6] memory _misc_vars

    ) external onlyByOwnGov {
        require(_misc_vars[0] >= MULTIPLIER_PRECISION, "Must be >= MUL PREC");
        require((_misc_vars[1] >= 0) && (_misc_vars[2] >= 0) && (_misc_vars[3] >= 0), "Must be >= 0");
        require((_misc_vars[4] >= 1) && (_misc_vars[5] >= 1), "Must be >= 1");

        lock_max_multiplier = _misc_vars[0];
        vefxs_max_multiplier = _misc_vars[1];
        vefxs_per_frax_for_max_boost = _misc_vars[2];
        vefxs_boost_scale_factor = _misc_vars[3];
        lock_time_for_max_multiplier = _misc_vars[4];
        lock_time_min = _misc_vars[5];
    }

    function setRewardVars(address reward_token_address, uint256 _new_rate, address _gauge_controller_address, address _rewards_distributor_address) external onlyTknMgrs(reward_token_address) {
        rewardRatesManual[rewardTokenAddrToIdx[reward_token_address]] = _new_rate;
        gaugeControllers[rewardTokenAddrToIdx[reward_token_address]] = _gauge_controller_address;
        rewardDistributors[rewardTokenAddrToIdx[reward_token_address]] = _rewards_distributor_address;
    }

    function changeTokenManager(address reward_token_address, address new_manager_address) external onlyTknMgrs(reward_token_address) {
        rewardManagers[reward_token_address] = new_manager_address;
    }

    event RewardPaid(address indexed user, uint256 amount, address token_address, address destination_address);

}

contract FraxUnifiedFarm_ERC20 is FraxUnifiedFarmTemplate {

    bool internal frax_is_token0;

    mapping(address => LockedStake[]) public lockedStakes;

    struct LockedStake {
        bytes32 kek_id;
        uint256 start_timestamp;
        uint256 liquidity;
        uint256 ending_timestamp;
        uint256 lock_multiplier;
    }

    constructor (
        address _owner,
        address[] memory _rewardTokens,
        address[] memory _rewardManagers,
        uint256[] memory _rewardRatesManual,
        address[] memory _gaugeControllers,
        address[] memory _rewardDistributors,
        address _stakingToken
    )
    FraxUnifiedFarmTemplate(_owner, _rewardTokens, _rewardManagers, _rewardRatesManual, _gaugeControllers, _rewardDistributors)
    {

    }

    function fraxPerLPToken() public virtual view override returns (uint256) {

        uint256 frax_per_lp_token;

        return frax_per_lp_token;
    }

    function calcCurrLockMultiplier(address account, uint256 stake_idx) public view returns (uint256 midpoint_lock_multiplier) {

        LockedStake memory thisStake = lockedStakes[account][stake_idx];

        uint256 accrue_start_time;
        if (lastRewardClaimTime[account] < thisStake.start_timestamp) {
            accrue_start_time = thisStake.start_timestamp;
        }
        else {
            accrue_start_time = lastRewardClaimTime[account];
        }

        if (thisStake.ending_timestamp <= block.timestamp) {

            if (lastRewardClaimTime[account] < thisStake.ending_timestamp){
                uint256 time_before_expiry = thisStake.ending_timestamp - accrue_start_time;
                uint256 time_after_expiry = block.timestamp - thisStake.ending_timestamp;

                uint256 pre_expiry_avg_multiplier = lockMultiplier(time_before_expiry / 2);

                uint256 numerator = (pre_expiry_avg_multiplier * time_before_expiry) + (0 * time_after_expiry);
                midpoint_lock_multiplier = numerator / (time_before_expiry + time_after_expiry);
            }
            else {

                midpoint_lock_multiplier = 0;
            }
        }

        else {

            uint256 avg_time_left;
            {
                uint256 time_left_p1 = thisStake.ending_timestamp - accrue_start_time;
                uint256 time_left_p2 = thisStake.ending_timestamp - block.timestamp;
                avg_time_left = (time_left_p1 + time_left_p2) / 2;
            }
            midpoint_lock_multiplier = lockMultiplier(avg_time_left);
        }

        if (midpoint_lock_multiplier > thisStake.lock_multiplier) midpoint_lock_multiplier = thisStake.lock_multiplier;
    }

    function calcCurCombinedWeight(address account) public override view
        returns (
            uint256 old_combined_weight,
            uint256 new_vefxs_multiplier,
            uint256 new_combined_weight
        )
    {

        old_combined_weight = _combined_weights[account];

        new_vefxs_multiplier = veFXSMultiplier(account);

        uint256 midpoint_vefxs_multiplier;
        if (
            (_locked_liquidity[account] == 0 && _combined_weights[account] == 0) ||
            (new_vefxs_multiplier >= _vefxsMultiplierStored[account])
        ) {

            midpoint_vefxs_multiplier = new_vefxs_multiplier;
        }
        else {

            midpoint_vefxs_multiplier = (new_vefxs_multiplier + _vefxsMultiplierStored[account]) / 2;
        }

        new_combined_weight = 0;
        for (uint256 i = 0; i < lockedStakes[account].length; i++) {
            LockedStake memory thisStake = lockedStakes[account][i];

            uint256 midpoint_lock_multiplier = calcCurrLockMultiplier(account, i);

            uint256 liquidity = thisStake.liquidity;
            uint256 combined_boosted_amount = liquidity + ((liquidity * (midpoint_lock_multiplier + midpoint_vefxs_multiplier)) / MULTIPLIER_PRECISION);
            new_combined_weight += combined_boosted_amount;
        }
    }

    function lockedStakesOf(address account) external view returns (LockedStake[] memory) {
        return lockedStakes[account];
    }

    function lockedStakesOfLength(address account) external view returns (uint256) {
        return lockedStakes[account].length;
    }

    function _updateLiqAmts(address staker_address, uint256 amt, bool is_add) internal {

        address the_proxy = getProxyFor(staker_address);

        if (is_add) {

            _total_liquidity_locked += amt;
            _locked_liquidity[staker_address] += amt;

            if (the_proxy != address(0)) proxy_lp_balances[the_proxy] += amt;
        }
        else {

            _total_liquidity_locked -= amt;
            _locked_liquidity[staker_address] -= amt;

            if (the_proxy != address(0)) proxy_lp_balances[the_proxy] -= amt;
        }

        _updateRewardAndBalance(staker_address, false, true);
    }

    function _getStake(address staker_address, bytes32 kek_id) internal view returns (LockedStake memory locked_stake, uint256 arr_idx) {
        for (uint256 i = 0; i < lockedStakes[staker_address].length; i++){
            if (kek_id == lockedStakes[staker_address][i].kek_id){
                locked_stake = lockedStakes[staker_address][i];
                arr_idx = i;
                break;
            }
        }
        require(locked_stake.kek_id == kek_id, "Stake not found");

    }

    function lockAdditional(bytes32 kek_id, uint256 addl_liq) nonReentrant updateRewardAndBalanceMdf(msg.sender, true) public {

        (LockedStake memory thisStake, uint256 theArrayIndex) = _getStake(msg.sender, kek_id);

        uint256 new_amt = thisStake.liquidity + addl_liq;

        require(addl_liq >= 0, "Must be positive");

        TransferHelper.safeTransferFrom(address(stakingToken), msg.sender, address(this), addl_liq);

        lockedStakes[msg.sender][theArrayIndex] = LockedStake(
            kek_id,
            thisStake.start_timestamp,
            new_amt,
            thisStake.ending_timestamp,
            thisStake.lock_multiplier
        );

        _updateLiqAmts(msg.sender, addl_liq, true);

        emit LockedAdditional(msg.sender, kek_id, addl_liq);
    }

    function lockLonger(bytes32 kek_id, uint256 new_ending_ts) nonReentrant updateRewardAndBalanceMdf(msg.sender, true) public {

        (LockedStake memory thisStake, uint256 theArrayIndex) = _getStake(msg.sender, kek_id);

        require(new_ending_ts > block.timestamp, "Must be in the future");

        uint256 time_left = (thisStake.ending_timestamp > block.timestamp) ? thisStake.ending_timestamp - block.timestamp : 0;
        uint256 new_secs = new_ending_ts - block.timestamp;

        require(new_secs > time_left, "Cannot shorten lock time");
        require(new_secs >= lock_time_min, "Minimum stake time not met");
        require(new_secs <= lock_time_for_max_multiplier, "Trying to lock for too long");

        lockedStakes[msg.sender][theArrayIndex] = LockedStake(
            kek_id,
            block.timestamp,
            thisStake.liquidity,
            new_ending_ts,
            lockMultiplier(new_secs)
        );

        _updateRewardAndBalance(msg.sender, false, true);

        emit LockedLonger(msg.sender, kek_id, new_secs, block.timestamp, new_ending_ts);
    }

    function stakeLocked(uint256 liquidity, uint256 secs) nonReentrant external returns (bytes32) {
        return _stakeLocked(msg.sender, msg.sender, liquidity, secs, block.timestamp);
    }

    function _stakeLocked(
        address staker_address,
        address source_address,
        uint256 liquidity,
        uint256 secs,
        uint256 start_timestamp
    ) internal updateRewardAndBalanceMdf(staker_address, true) returns (bytes32) {
        require(stakingPaused == false, "Staking paused");
        require(secs >= lock_time_min, "Minimum stake time not met");
        require(secs <= lock_time_for_max_multiplier,"Trying to lock for too long");

        TransferHelper.safeTransferFrom(address(stakingToken), source_address, address(this), liquidity);

        uint256 lock_multiplier = lockMultiplier(secs);
        bytes32 kek_id = keccak256(abi.encodePacked(staker_address, start_timestamp, liquidity, _locked_liquidity[staker_address]));

        lockedStakes[staker_address].push(LockedStake(
            kek_id,
            start_timestamp,
            liquidity,
            start_timestamp + secs,
            lock_multiplier
        ));

        _updateLiqAmts(staker_address, liquidity, true);

        emit StakeLocked(staker_address, liquidity, secs, kek_id, source_address);

        return kek_id;
    }

    function withdrawLocked(bytes32 kek_id, address destination_address, bool claim_rewards) nonReentrant external returns (uint256) {
        require(withdrawalsPaused == false, "Withdrawals paused");
        return _withdrawLocked(msg.sender, destination_address, kek_id, claim_rewards);
    }

    function _withdrawLocked(
        address staker_address,
        address destination_address,
        bytes32 kek_id,
        bool claim_rewards
    ) internal returns (uint256) {

        if (claim_rewards || !collectRewardsOnWithdrawalPaused) _getReward(staker_address, destination_address, true);
        else {

            _updateRewardAndBalance(staker_address, true, false);
        }

        (LockedStake memory thisStake, uint256 theArrayIndex) = _getStake(staker_address, kek_id);
        require(block.timestamp >= thisStake.ending_timestamp || stakesUnlocked == true, "Stake is still locked!");
        uint256 liquidity = thisStake.liquidity;

        if (liquidity > 0) {

            TransferHelper.safeTransfer(address(stakingToken), destination_address, liquidity);

            delete lockedStakes[staker_address][theArrayIndex];

            _updateLiqAmts(staker_address, liquidity, false);

            emit WithdrawLocked(staker_address, liquidity, kek_id, destination_address);
        }

        return liquidity;
    }

    function _getRewardExtraLogic(address rewardee, address destination_address) internal override {

    }

    event LockedAdditional(address indexed user, bytes32 kek_id, uint256 amount);
    event LockedLonger(address indexed user, bytes32 kek_id, uint256 new_secs, uint256 new_start_ts, uint256 new_end_ts);
    event StakeLocked(address indexed user, uint256 amount, uint256 secs, bytes32 kek_id, address source_address);
    event WithdrawLocked(address indexed user, uint256 liquidity, bytes32 kek_id, address destination_address);
}

interface IBunniTokenLP {
  function DOMAIN_SEPARATOR () external view returns (bytes32);
  function allowance (address, address) external view returns (uint256);
  function approve (address spender, uint256 amount) external returns (bool);
  function balanceOf (address) external view returns (uint256);
  function burn (address from, uint256 amount) external;
  function decimals () external view returns (uint8);
  function hub () external view returns (address);
  function mint (address to, uint256 amount) external;
  function name () external view returns (string memory);
  function nonces (address) external view returns (uint256);
  function permit (address owner, address spender, uint256 value, uint256 deadline, uint8 v, bytes32 r, bytes32 s) external;
  function pool () external view returns (address);
  function symbol () external view returns (string memory);
  function tickLower () external view returns (int24);
  function tickUpper () external view returns (int24);
  function totalSupply () external view returns (uint256);
  function transfer (address to, uint256 amount) external returns (bool);
  function transferFrom (address from, address to, uint256 amount) external returns (bool);
}

interface AggregatorV3Interface {

  function decimals() external view returns (uint8);
  function description() external view returns (string memory);
  function version() external view returns (uint256);

  function getRoundData(uint80 _roundId)
    external
    view
    returns (
      uint80 roundId,
      int256 answer,
      uint256 startedAt,
      uint256 updatedAt,
      uint80 answeredInRound
    );
  function latestRoundData()
    external
    view
    returns (
      uint80 roundId,
      int256 answer,
      uint256 startedAt,
      uint256 updatedAt,
      uint80 answeredInRound
    );

}

contract FraxUnifiedFarm_ERC20_Other is FraxUnifiedFarm_ERC20 {

    IBunniTokenLP public lp_tkn;
    IUniswapV3Pool public univ3_pool;

    constructor (
        address _owner,
        address[] memory _rewardTokens,
        address[] memory _rewardManagers,
        uint256[] memory _rewardRates,
        address[] memory _gaugeControllers,
        address[] memory _rewardDistributors,
        address _stakingToken
    )
    FraxUnifiedFarm_ERC20(_owner , _rewardTokens, _rewardManagers, _rewardRates, _gaugeControllers, _rewardDistributors, _stakingToken)
    {

        stakingToken = IBunniGauge(_stakingToken);
        lp_tkn = IBunniTokenLP(stakingToken.lp_token());
        univ3_pool = IUniswapV3Pool(lp_tkn.pool());
        address token0 = univ3_pool.token0();
        frax_is_token0 = (token0 == frax_address);
    }

    function setBunniAddrs(address _lens, address _minter) public onlyByOwnGov {
        lens = IBunniLens(_lens);
        minter = IBunniMinter(_minter);
    }

    function toggleBunni3rdPartyOLITClaimer(address _claimer) public onlyByOwnGov {
        minter.toggle_approve_mint(_claimer);
    }

    function fraxPerLPToken() public view override returns (uint256) {

        uint256 frax_per_lp_token;

        {

            IBunniLens.BunniKey memory bkey = IBunniLens.BunniKey({
                pool: univ3_pool,
                tickLower: lp_tkn.tickLower(),
                tickUpper: lp_tkn.tickUpper()
            });
            (, uint256 amt0, uint256 amt1) = lens.pricePerFullShare(bkey);

            if (frax_is_token0) frax_per_lp_token = amt0;
            else frax_per_lp_token = amt1;
        }

        return frax_per_lp_token;
    }
}
