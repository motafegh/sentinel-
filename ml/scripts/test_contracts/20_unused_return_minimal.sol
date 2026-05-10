// expect: UnusedReturn
// Minimal: ignore bool return from ERC20 transfer (USDT-style non-reverting token).
pragma solidity ^0.8.0;
interface IToken { function transfer(address to, uint256 amt) external returns (bool); }
contract IgnoreReturn {
    IToken public token;
    constructor(address t) { token = IToken(t); }
    function pay(address to, uint256 amt) external {
        token.transfer(to, amt); // VULNERABILITY: return value discarded
    }
}
