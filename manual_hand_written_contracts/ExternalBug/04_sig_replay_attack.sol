// expect: ExternalBug,MishandledException
// Meta-transaction relayer that does not track nonces or used signatures.
// A valid signed message can be replayed multiple times across different
// relayers or even on the same contract — the signature check passes
// every time because there is no storage of used hashes.
// Also: the permit-style approve pattern uses an unchecked external call
// and the return value of the token transfer is silently ignored.
pragma solidity ^0.8.0;

interface IERC20Permit {
    function permit(address owner, address spender, uint256 value, uint256 deadline, uint8 v, bytes32 r, bytes32 s) external;
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function transfer(address to, uint256 amount) external returns (bool);
}

contract MetaRelayer {
    struct Signature {
        uint8 v;
        bytes32 r;
        bytes32 s;
    }

    address public owner;
    IERC20Permit public token;
    uint256 public relayerFee;

    event Relayed(address indexed target, bytes data, address indexed sender);
    event FeeCollected(uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _token) {
        owner = msg.sender;
        token = IERC20Permit(_token);
        relayerFee = 0.001 ether;
    }

    function relayPermitAndTransfer(
        address from,
        address to,
        uint256 amount,
        uint256 deadline,
        Signature calldata permitSig,
        uint256 transferAmount
    ) external payable {
        require(msg.value >= relayerFee, "insufficient fee");
        token.permit(from, address(this), amount, deadline, permitSig.v, permitSig.r, permitSig.s);
        token.transferFrom(from, to, transferAmount);
        (bool feeOk, ) = owner.call{value: msg.value}("");
        require(feeOk, "fee transfer failed");
        emit Relayed(address(token), abi.encodeWithSignature("transferFrom(address,address,uint256)", from, to, transferAmount), msg.sender);
    }

    function relayBatchPermit(
        address[] calldata fromList,
        address[] calldata toList,
        uint256[] calldata amounts,
        uint256 deadline,
        Signature[] calldata sigs
    ) external payable {
        require(fromList.length == toList.length, "length mismatch");
        require(fromList.length == sigs.length, "sig mismatch");
        require(msg.value >= relayerFee * fromList.length, "insufficient fees");
        for (uint256 i = 0; i < fromList.length; i++) {
            Signature calldata sig = sigs[i];
            token.permit(fromList[i], address(this), amounts[i], deadline, sig.v, sig.r, sig.s);
            token.transferFrom(fromList[i], toList[i], amounts[i]);
        }
        (bool feeOk, ) = owner.call{value: msg.value}("");
        require(feeOk, "fee transfer failed");
    }

    function relayMultiCall(
        address[] calldata targets,
        bytes[] calldata calldatas
    ) external payable returns (bytes[] memory) {
        require(targets.length == calldatas.length, "length mismatch");
        bytes[] memory results = new bytes[](targets.length);
        for (uint256 i = 0; i < targets.length; i++) {
            (bool ok, bytes memory ret) = targets[i].call(calldatas[i]);
            require(ok, "multicall failed");
            results[i] = ret;
        }
        if (msg.value > 0) {
            (bool feeOk, ) = owner.call{value: msg.value}("");
            require(feeOk, "fee transfer failed");
        }
        return results;
    }

    function setFee(uint256 newFee) external onlyOwner {
        relayerFee = newFee;
    }

    function drain() external onlyOwner {
        (bool ok, ) = owner.call{value: address(this).balance}("");
        require(ok, "drain failed");
    }

    receive() external payable {}
}