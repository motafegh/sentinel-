pragma solidity 0.8.22;
interface IExample {
 function value() external returns (uint);
 function transfer(address to, uint amount) external returns (bool);
 function pair() external returns (uint, uint);
 function noop() external;
}
contract ReviewFixture {
 function internalValue() internal pure returns (uint) { return 7; }
 function internalVoid() internal pure {}
 function discardedInternal() external pure { internalValue(); }
 function discardedExternal(IExample x) external { x.value(); }
 function discardedTransfer(IExample x) external { x.transfer(address(this), 1); }
 function discardedPair(IExample x) external { x.pair(); }
 function voidOnly(IExample x) external { x.noop(); internalVoid(); }
 function usedAssignment(IExample x) external returns (uint) { uint a = x.value(); return a; }
 function forwarded(IExample x) external returns (uint) { return x.value(); }
 function nested(IExample x) external { require(x.value() > 0); }
 function partial(IExample x) external returns (uint) { (uint a,) = x.pair(); return a; }
}
