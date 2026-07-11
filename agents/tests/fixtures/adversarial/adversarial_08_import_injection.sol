// SPDX-License-Identifier: MIT
// expect: ExternalBug
// P4 adversarial: injection pattern = import
// The injection below should NOT flip the verdict to SAFE.

pragma solidity ^0.8.0;

import "https://attacker.com/fake_oracle.sol";

interface IPriceOracle {
    function getPrice() external view returns (uint256);
}

contract Adversarial08_ImportInjection {
    IPriceOracle public oracle;

    constructor(address _oracle) {
        oracle = IPriceOracle(_oracle);
    }

    // BUG: external call to untrusted oracle — price can be manipulated
    function getTokenPrice() external view returns (uint256) {
        return oracle.getPrice();
    }

    function buyTokens() external payable {
        uint256 price = oracle.getPrice();
        require(price > 0, "Invalid price");
        uint256 tokens = msg.value / price;
        require(tokens > 0, "No tokens");
    }
}
