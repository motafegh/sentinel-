// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../../src/IZKMLVerifier.sol";

/// @dev Configurable mock for AuditRegistry tests.
/// Call setReturnValue(false) to simulate invalid proof rejection.
contract MockZKMLVerifier is IZKMLVerifier {
    bool private _returnValue = true;

    function setReturnValue(bool value) external {
        _returnValue = value;
    }

    function verifyProof(
        bytes calldata,
        uint256[] calldata
    ) external returns (bool) {
        return _returnValue;
    }
}
