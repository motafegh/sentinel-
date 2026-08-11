pragma solidity 0.7.6;

contract AddressUtils {

    function isContracts(address[] memory addrs) public view returns(bool[] memory){
        bool[] memory rets = new bool[](addrs.length);
        for(uint32 i = 0; i < addrs.length; i ++) {

            address addr = addrs[i];
            uint256 codeSize;
            assembly { codeSize := extcodesize(addr) }
            rets[i] = codeSize > 0;
        }
        return rets;
    }

}
