// expect: UnusedReturn
// Multiple unused return values: ERC20 transfer, low-level call, and
// an internal helper whose bool result is never checked.
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
}

contract ReturnIgnorer {
    IERC20 public token;
    address public vault;

    constructor(address _token, address _vault) {
        token = IERC20(_token);
        vault = _vault;
    }

    function sweep(uint256 amount) external {
        // VULNERABILITY: transfer return value ignored — silent failure on non-reverting tokens
        token.transfer(vault, amount);
    }

    function approveAndForget(address spender, uint256 amount) external {
        // VULNERABILITY: approve return value ignored
        token.approve(spender, amount);
    }

    function _tryNotify(address target) internal returns (bool) {
        (bool ok, ) = target.call(abi.encodeWithSignature("notify()"));
        return ok;
    }

    function broadcast(address[] calldata targets) external {
        for (uint256 i = 0; i < targets.length; i++) {
            // VULNERABILITY: bool return from _tryNotify discarded
            _tryNotify(targets[i]);
        }
    }
}
