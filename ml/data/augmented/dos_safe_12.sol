// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DosSafe12 {

    mapping(address => bool) public isChild;
    function addChild(address c) external { isChild[c] = true; }
    function destroyChild(address c) external {
        require(isChild[c]);
        IChild(c).destroy();
    }
    interface IChild { function destroy() external; }
}
