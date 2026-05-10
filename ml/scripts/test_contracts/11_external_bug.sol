// expect: ExternalBug
// Trust assumption on external contract: price comes from an oracle that can
// be manipulated. Also: callback to untrusted contract during state change.
pragma solidity ^0.8.0;

interface IPriceOracle {
    function getPrice(address token) external view returns (uint256);
}

interface ICallback {
    function onBorrow(uint256 amount) external;
}

contract LendingPool {
    IPriceOracle public oracle;
    mapping(address => uint256) public collateral;
    mapping(address => uint256) public debt;
    address public baseToken;

    constructor(address _oracle, address _token) {
        oracle = IPriceOracle(_oracle);
        baseToken = _token;
    }

    function depositCollateral() external payable {
        collateral[msg.sender] += msg.value;
    }

    function borrow(uint256 amount, address callbackTarget) external {
        // VULNERABILITY: price from external oracle — single-block flash loan manipulation
        uint256 price = oracle.getPrice(baseToken);
        uint256 collateralValue = collateral[msg.sender] * price;
        require(collateralValue >= amount * 2, "undercollateralised");

        debt[msg.sender] += amount;

        // VULNERABILITY: external call to untrusted callbackTarget before state is settled
        // callbackTarget can re-enter borrow() with same collateral still credited
        ICallback(callbackTarget).onBorrow(amount);
    }
}
