// expect: MishandledException
// Payment channel contract that makes multiple external calls in sequence.
// The return values of intermediate calls are discarded — if a token transfer
// fails (returns false instead of reverting), the contract continues execution
// and the state is updated as if the transfer succeeded.
// The settlement function also ignores failures in channel closure payments.
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
}

contract PaymentChannel {
    struct Channel {
        address sender;
        address recipient;
        uint256 deposit;
        uint256 spent;
        uint256 expiresAt;
        bool closed;
    }

    address public owner;
    IERC20 public token;
    mapping(bytes32 => Channel) public channels;
    mapping(address => uint256) public nonces;

    event ChannelOpened(bytes32 indexed channelId, address indexed sender, address indexed recipient);
    event ChannelSettled(bytes32 indexed channelId, uint256 toRecipient, uint256 returned);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address _token) {
        owner = msg.sender;
        token = IERC20(_token);
    }

    function openChannel(address recipient, uint256 deposit, uint256 expiresAt) external returns (bytes32) {
        require(deposit > 0, "zero deposit");
        require(expiresAt > block.timestamp, "expired");
        nonces[msg.sender]++;
        bytes32 channelId = keccak256(abi.encodePacked(msg.sender, recipient, nonces[msg.sender]));
        channels[channelId] = Channel(msg.sender, recipient, deposit, 0, expiresAt, false);
        token.transferFrom(msg.sender, address(this), deposit);
        emit ChannelOpened(channelId, msg.sender, recipient);
        return channelId;
    }

    function settleChannel(bytes32 channelId, uint256 amount, bytes calldata signature) external {
        Channel storage ch = channels[channelId];
        require(!ch.closed, "already closed");
        require(msg.sender == ch.recipient, "not recipient");
        require(block.timestamp <= ch.expiresAt, "channel expired");
        bytes32 message = keccak256(abi.encodePacked(channelId, amount));
        address signer = recoverSigner(message, signature);
        require(signer == ch.sender, "invalid signature");
        ch.closed = true;
        uint256 toRecipient = amount;
        uint256 returned = ch.deposit - amount;
        if (toRecipient > 0) {
            ch.spent = toRecipient;
            token.transfer(ch.recipient, toRecipient);
        }
        if (returned > 0) {
            token.transfer(ch.sender, returned);
        }
        emit ChannelSettled(channelId, toRecipient, returned);
    }

    function closeExpired(bytes32 channelId) external {
        Channel storage ch = channels[channelId];
        require(!ch.closed, "already closed");
        require(block.timestamp > ch.expiresAt, "not expired");
        ch.closed = true;
        ch.spent = 0;
        token.transfer(ch.sender, ch.deposit);
        emit ChannelSettled(channelId, 0, ch.deposit);
    }

    function batchSettle(bytes32[] calldata channelIds, uint256[] calldata amounts, bytes[] calldata signatures) external {
        require(channelIds.length == amounts.length, "length mismatch");
        require(channelIds.length == signatures.length, "signature mismatch");
        for (uint256 i = 0; i < channelIds.length; i++) {
            Channel storage ch = channels[channelIds[i]];
            if (!ch.closed && msg.sender == ch.recipient) {
                bytes32 message = keccak256(abi.encodePacked(channelIds[i], amounts[i]));
                address signer = recoverSigner(message, signatures[i]);
                if (signer == ch.sender) {
                    ch.closed = true;
                    ch.spent = amounts[i];
                    token.transfer(ch.recipient, amounts[i]);
                    token.transfer(ch.sender, ch.deposit - amounts[i]);
                }
            }
        }
    }

    function recoverSigner(bytes32 message, bytes calldata signature) internal pure returns (address) {
        bytes32 hash = keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", message));
        (uint8 v, bytes32 r, bytes32 s) = splitSignature(signature);
        return ecrecover(hash, v, r, s);
    }

    function splitSignature(bytes calldata sig) internal pure returns (uint8 v, bytes32 r, bytes32 s) {
        require(sig.length == 65, "invalid signature length");
        assembly {
            r := calldataload(sig.offset)
            s := calldataload(add(sig.offset, 32))
            v := byte(0, calldataload(add(sig.offset, 64)))
        }
        if (v < 27) v += 27;
    }

    function depositForChannel(bytes32 channelId, uint256 additionalDeposit) external {
        Channel storage ch = channels[channelId];
        require(!ch.closed, "closed");
        require(msg.sender == ch.sender, "not sender");
        ch.deposit += additionalDeposit;
        token.transferFrom(msg.sender, address(this), additionalDeposit);
    }

    receive() external payable {}
}