pragma solidity ^0.8.0;

library Math {
    enum Rounding {
        Down,
        Up,
        Zero
    }

    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return a > b ? a : b;
    }

    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }

    function average(uint256 a, uint256 b) internal pure returns (uint256) {

        return (a & b) + (a ^ b) / 2;
    }

    function ceilDiv(uint256 a, uint256 b) internal pure returns (uint256) {

        return a == 0 ? 0 : (a - 1) / b + 1;
    }

    function mulDiv(
        uint256 x,
        uint256 y,
        uint256 denominator
    ) internal pure returns (uint256 result) {
        unchecked {

            uint256 prod0;
            uint256 prod1;
            assembly {
                let mm := mulmod(x, y, not(0))
                prod0 := mul(x, y)
                prod1 := sub(sub(mm, prod0), lt(mm, prod0))
            }

            if (prod1 == 0) {
                return prod0 / denominator;
            }

            require(denominator > prod1);

            uint256 remainder;
            assembly {

                remainder := mulmod(x, y, denominator)

                prod1 := sub(prod1, gt(remainder, prod0))
                prod0 := sub(prod0, remainder)
            }

            uint256 twos = denominator & (~denominator + 1);
            assembly {

                denominator := div(denominator, twos)

                prod0 := div(prod0, twos)

                twos := add(div(sub(0, twos), twos), 1)
            }

            prod0 |= prod1 * twos;

            uint256 inverse = (3 * denominator) ^ 2;

            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;

            result = prod0 * inverse;
            return result;
        }
    }

    function mulDiv(
        uint256 x,
        uint256 y,
        uint256 denominator,
        Rounding rounding
    ) internal pure returns (uint256) {
        uint256 result = mulDiv(x, y, denominator);
        if (rounding == Rounding.Up && mulmod(x, y, denominator) > 0) {
            result += 1;
        }
        return result;
    }

    function sqrt(uint256 a) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }

        uint256 result = 1 << (log2(a) >> 1);

        unchecked {
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            result = (result + a / result) >> 1;
            return min(result, a / result);
        }
    }

    function sqrt(uint256 a, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = sqrt(a);
            return result + (rounding == Rounding.Up && result * result < a ? 1 : 0);
        }
    }

    function log2(uint256 value) internal pure returns (uint256) {
        uint256 result = 0;
        unchecked {
            if (value >> 128 > 0) {
                value >>= 128;
                result += 128;
            }
            if (value >> 64 > 0) {
                value >>= 64;
                result += 64;
            }
            if (value >> 32 > 0) {
                value >>= 32;
                result += 32;
            }
            if (value >> 16 > 0) {
                value >>= 16;
                result += 16;
            }
            if (value >> 8 > 0) {
                value >>= 8;
                result += 8;
            }
            if (value >> 4 > 0) {
                value >>= 4;
                result += 4;
            }
            if (value >> 2 > 0) {
                value >>= 2;
                result += 2;
            }
            if (value >> 1 > 0) {
                result += 1;
            }
        }
        return result;
    }

    function log2(uint256 value, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = log2(value);
            return result + (rounding == Rounding.Up && 1 << result < value ? 1 : 0);
        }
    }

    function log10(uint256 value) internal pure returns (uint256) {
        uint256 result = 0;
        unchecked {
            if (value >= 10**64) {
                value /= 10**64;
                result += 64;
            }
            if (value >= 10**32) {
                value /= 10**32;
                result += 32;
            }
            if (value >= 10**16) {
                value /= 10**16;
                result += 16;
            }
            if (value >= 10**8) {
                value /= 10**8;
                result += 8;
            }
            if (value >= 10**4) {
                value /= 10**4;
                result += 4;
            }
            if (value >= 10**2) {
                value /= 10**2;
                result += 2;
            }
            if (value >= 10**1) {
                result += 1;
            }
        }
        return result;
    }

    function log10(uint256 value, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = log10(value);
            return result + (rounding == Rounding.Up && 10**result < value ? 1 : 0);
        }
    }

    function log256(uint256 value) internal pure returns (uint256) {
        uint256 result = 0;
        unchecked {
            if (value >> 128 > 0) {
                value >>= 128;
                result += 16;
            }
            if (value >> 64 > 0) {
                value >>= 64;
                result += 8;
            }
            if (value >> 32 > 0) {
                value >>= 32;
                result += 4;
            }
            if (value >> 16 > 0) {
                value >>= 16;
                result += 2;
            }
            if (value >> 8 > 0) {
                result += 1;
            }
        }
        return result;
    }

    function log256(uint256 value, Rounding rounding) internal pure returns (uint256) {
        unchecked {
            uint256 result = log256(value);
            return result + (rounding == Rounding.Up && 1 << (result * 8) < value ? 1 : 0);
        }
    }
}

pragma solidity ^0.8.0;

library Strings {
    bytes16 private constant _SYMBOLS = "0123456789abcdef";
    uint8 private constant _ADDRESS_LENGTH = 20;

    function toString(uint256 value) internal pure returns (string memory) {
        unchecked {
            uint256 length = Math.log10(value) + 1;
            string memory buffer = new string(length);
            uint256 ptr;

            assembly {
                ptr := add(buffer, add(32, length))
            }
            while (true) {
                ptr--;

                assembly {
                    mstore8(ptr, byte(mod(value, 10), _SYMBOLS))
                }
                value /= 10;
                if (value == 0) break;
            }
            return buffer;
        }
    }

    function toHexString(uint256 value) internal pure returns (string memory) {
        unchecked {
            return toHexString(value, Math.log256(value) + 1);
        }
    }

    function toHexString(uint256 value, uint256 length) internal pure returns (string memory) {
        bytes memory buffer = new bytes(2 * length + 2);
        buffer[0] = "0";
        buffer[1] = "x";
        for (uint256 i = 2 * length + 1; i > 1; --i) {
            buffer[i] = _SYMBOLS[value & 0xf];
            value >>= 4;
        }
        require(value == 0, "Strings: hex length insufficient");
        return string(buffer);
    }

    function toHexString(address addr) internal pure returns (string memory) {
        return toHexString(uint256(uint160(addr)), _ADDRESS_LENGTH);
    }
}

pragma solidity ^0.8.0;

abstract contract ReentrancyGuard {

    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;

    uint256 private _status;

    constructor() {
        _status = _NOT_ENTERED;
    }

    modifier nonReentrant() {
        _nonReentrantBefore();
        _;
        _nonReentrantAfter();
    }

    function _nonReentrantBefore() private {

        require(_status != _ENTERED, "ReentrancyGuard: reentrant call");

        _status = _ENTERED;
    }

    function _nonReentrantAfter() private {

        _status = _NOT_ENTERED;
    }
}

pragma solidity ^0.8.0;

library MerkleProof {

    function verify(
        bytes32[] memory proof,
        bytes32 root,
        bytes32 leaf
    ) internal pure returns (bool) {
        return processProof(proof, leaf) == root;
    }

    function verifyCalldata(
        bytes32[] calldata proof,
        bytes32 root,
        bytes32 leaf
    ) internal pure returns (bool) {
        return processProofCalldata(proof, leaf) == root;
    }

    function processProof(bytes32[] memory proof, bytes32 leaf) internal pure returns (bytes32) {
        bytes32 computedHash = leaf;
        for (uint256 i = 0; i < proof.length; i++) {
            computedHash = _hashPair(computedHash, proof[i]);
        }
        return computedHash;
    }

    function processProofCalldata(bytes32[] calldata proof, bytes32 leaf) internal pure returns (bytes32) {
        bytes32 computedHash = leaf;
        for (uint256 i = 0; i < proof.length; i++) {
            computedHash = _hashPair(computedHash, proof[i]);
        }
        return computedHash;
    }

    function multiProofVerify(
        bytes32[] memory proof,
        bool[] memory proofFlags,
        bytes32 root,
        bytes32[] memory leaves
    ) internal pure returns (bool) {
        return processMultiProof(proof, proofFlags, leaves) == root;
    }

    function multiProofVerifyCalldata(
        bytes32[] calldata proof,
        bool[] calldata proofFlags,
        bytes32 root,
        bytes32[] memory leaves
    ) internal pure returns (bool) {
        return processMultiProofCalldata(proof, proofFlags, leaves) == root;
    }

    function processMultiProof(
        bytes32[] memory proof,
        bool[] memory proofFlags,
        bytes32[] memory leaves
    ) internal pure returns (bytes32 merkleRoot) {

        uint256 leavesLen = leaves.length;
        uint256 totalHashes = proofFlags.length;

        require(leavesLen + proof.length - 1 == totalHashes, "MerkleProof: invalid multiproof");

        bytes32[] memory hashes = new bytes32[](totalHashes);
        uint256 leafPos = 0;
        uint256 hashPos = 0;
        uint256 proofPos = 0;

        for (uint256 i = 0; i < totalHashes; i++) {
            bytes32 a = leafPos < leavesLen ? leaves[leafPos++] : hashes[hashPos++];
            bytes32 b = proofFlags[i] ? leafPos < leavesLen ? leaves[leafPos++] : hashes[hashPos++] : proof[proofPos++];
            hashes[i] = _hashPair(a, b);
        }

        if (totalHashes > 0) {
            return hashes[totalHashes - 1];
        } else if (leavesLen > 0) {
            return leaves[0];
        } else {
            return proof[0];
        }
    }

    function processMultiProofCalldata(
        bytes32[] calldata proof,
        bool[] calldata proofFlags,
        bytes32[] memory leaves
    ) internal pure returns (bytes32 merkleRoot) {

        uint256 leavesLen = leaves.length;
        uint256 totalHashes = proofFlags.length;

        require(leavesLen + proof.length - 1 == totalHashes, "MerkleProof: invalid multiproof");

        bytes32[] memory hashes = new bytes32[](totalHashes);
        uint256 leafPos = 0;
        uint256 hashPos = 0;
        uint256 proofPos = 0;

        for (uint256 i = 0; i < totalHashes; i++) {
            bytes32 a = leafPos < leavesLen ? leaves[leafPos++] : hashes[hashPos++];
            bytes32 b = proofFlags[i] ? leafPos < leavesLen ? leaves[leafPos++] : hashes[hashPos++] : proof[proofPos++];
            hashes[i] = _hashPair(a, b);
        }

        if (totalHashes > 0) {
            return hashes[totalHashes - 1];
        } else if (leavesLen > 0) {
            return leaves[0];
        } else {
            return proof[0];
        }
    }

    function _hashPair(bytes32 a, bytes32 b) private pure returns (bytes32) {
        return a < b ? _efficientHash(a, b) : _efficientHash(b, a);
    }

    function _efficientHash(bytes32 a, bytes32 b) private pure returns (bytes32 value) {

        assembly {
            mstore(0x00, a)
            mstore(0x20, b)
            value := keccak256(0x00, 0x40)
        }
    }
}

pragma solidity ^0.8.0;

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}

pragma solidity ^0.8.0;

abstract contract Ownable is Context {
    address private _owner;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        _transferOwnership(_msgSender());
    }

    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    function _checkOwner() internal view virtual {
        require(owner() == _msgSender(), "Ownable: caller is not the owner");
    }

    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        _transferOwnership(newOwner);
    }

    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

pragma solidity ^0.8.4;

interface IERC721A {

    error ApprovalCallerNotOwnerNorApproved();

    error ApprovalQueryForNonexistentToken();

    error BalanceQueryForZeroAddress();

    error MintToZeroAddress();

    error MintZeroQuantity();

    error OwnerQueryForNonexistentToken();

    error TransferCallerNotOwnerNorApproved();

    error TransferFromIncorrectOwner();

    error TransferToNonERC721ReceiverImplementer();

    error TransferToZeroAddress();

    error URIQueryForNonexistentToken();

    error MintERC2309QuantityExceedsLimit();

    error OwnershipNotInitializedForExtraData();

    struct TokenOwnership {

        address addr;

        uint64 startTimestamp;

        bool burned;

        uint24 extraData;
    }

    function totalSupply() external view returns (uint256);

    function supportsInterface(bytes4 interfaceId) external view returns (bool);

    event Transfer(address indexed from, address indexed to, uint256 indexed tokenId);

    event Approval(address indexed owner, address indexed approved, uint256 indexed tokenId);

    event ApprovalForAll(address indexed owner, address indexed operator, bool approved);

    function balanceOf(address owner) external view returns (uint256 balance);

    function ownerOf(uint256 tokenId) external view returns (address owner);

    function safeTransferFrom(
        address from,
        address to,
        uint256 tokenId,
        bytes calldata data
    ) external payable;

    function safeTransferFrom(
        address from,
        address to,
        uint256 tokenId
    ) external payable;

    function transferFrom(
        address from,
        address to,
        uint256 tokenId
    ) external payable;

    function approve(address to, uint256 tokenId) external payable;

    function setApprovalForAll(address operator, bool _approved) external;

    function getApproved(uint256 tokenId) external view returns (address operator);

    function isApprovedForAll(address owner, address operator) external view returns (bool);

    function name() external view returns (string memory);

    function symbol() external view returns (string memory);

    function tokenURI(uint256 tokenId) external view returns (string memory);

    event ConsecutiveTransfer(uint256 indexed fromTokenId, uint256 toTokenId, address indexed from, address indexed to);
}

pragma solidity ^0.8.4;

interface ERC721A__IERC721Receiver {
    function onERC721Received(
        address operator,
        address from,
        uint256 tokenId,
        bytes calldata data
    ) external returns (bytes4);
}

contract ERC721A is IERC721A {

    struct TokenApprovalRef {
        address value;
    }

    uint256 private constant _BITMASK_ADDRESS_DATA_ENTRY = (1 << 64) - 1;

    uint256 private constant _BITPOS_NUMBER_MINTED = 64;

    uint256 private constant _BITPOS_NUMBER_BURNED = 128;

    uint256 private constant _BITPOS_AUX = 192;

    uint256 private constant _BITMASK_AUX_COMPLEMENT = (1 << 192) - 1;

    uint256 private constant _BITPOS_START_TIMESTAMP = 160;

    uint256 private constant _BITMASK_BURNED = 1 << 224;

    uint256 private constant _BITPOS_NEXT_INITIALIZED = 225;

    uint256 private constant _BITMASK_NEXT_INITIALIZED = 1 << 225;

    uint256 private constant _BITPOS_EXTRA_DATA = 232;

    uint256 private constant _BITMASK_EXTRA_DATA_COMPLEMENT = (1 << 232) - 1;

    uint256 private constant _BITMASK_ADDRESS = (1 << 160) - 1;

    uint256 private constant _MAX_MINT_ERC2309_QUANTITY_LIMIT = 5000;

    bytes32 private constant _TRANSFER_EVENT_SIGNATURE =
        0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef;

    uint256 private _currentIndex;

    uint256 private _burnCounter;

    string private _name;

    string private _symbol;

    mapping(uint256 => uint256) private _packedOwnerships;

    mapping(address => uint256) private _packedAddressData;

    mapping(uint256 => TokenApprovalRef) private _tokenApprovals;

    mapping(address => mapping(address => bool)) private _operatorApprovals;

    constructor(string memory name_, string memory symbol_) {
        _name = name_;
        _symbol = symbol_;
        _currentIndex = _startTokenId();
    }

    function _startTokenId() internal view virtual returns (uint256) {
        return 0;
    }

    function _nextTokenId() internal view virtual returns (uint256) {
        return _currentIndex;
    }

    function totalSupply() public view virtual override returns (uint256) {

        unchecked {
            return _currentIndex - _burnCounter - _startTokenId();
        }
    }

    function _totalMinted() internal view virtual returns (uint256) {

        unchecked {
            return _currentIndex - _startTokenId();
        }
    }

    function _totalBurned() internal view virtual returns (uint256) {
        return _burnCounter;
    }

    function balanceOf(address owner) public view virtual override returns (uint256) {
        if (owner == address(0)) revert BalanceQueryForZeroAddress();
        return _packedAddressData[owner] & _BITMASK_ADDRESS_DATA_ENTRY;
    }

    function _numberMinted(address owner) internal view returns (uint256) {
        return (_packedAddressData[owner] >> _BITPOS_NUMBER_MINTED) & _BITMASK_ADDRESS_DATA_ENTRY;
    }

    function _numberBurned(address owner) internal view returns (uint256) {
        return (_packedAddressData[owner] >> _BITPOS_NUMBER_BURNED) & _BITMASK_ADDRESS_DATA_ENTRY;
    }

    function _getAux(address owner) internal view returns (uint64) {
        return uint64(_packedAddressData[owner] >> _BITPOS_AUX);
    }

    function _setAux(address owner, uint64 aux) internal virtual {
        uint256 packed = _packedAddressData[owner];
        uint256 auxCasted;

        assembly {
            auxCasted := aux
        }
        packed = (packed & _BITMASK_AUX_COMPLEMENT) | (auxCasted << _BITPOS_AUX);
        _packedAddressData[owner] = packed;
    }

    function supportsInterface(bytes4 interfaceId) public view virtual override returns (bool) {

        return
            interfaceId == 0x01ffc9a7 ||
            interfaceId == 0x80ac58cd ||
            interfaceId == 0x5b5e139f;
    }

    function name() public view virtual override returns (string memory) {
        return _name;
    }

    function symbol() public view virtual override returns (string memory) {
        return _symbol;
    }

    function tokenURI(uint256 tokenId) public view virtual override returns (string memory) {
        if (!_exists(tokenId)) revert URIQueryForNonexistentToken();

        string memory baseURI = _baseURI();
        return bytes(baseURI).length != 0 ? string(abi.encodePacked(baseURI, _toString(tokenId))) : '';
    }

    function _baseURI() internal view virtual returns (string memory) {
        return '';
    }

    function ownerOf(uint256 tokenId) public view virtual override returns (address) {
        return address(uint160(_packedOwnershipOf(tokenId)));
    }

    function _ownershipOf(uint256 tokenId) internal view virtual returns (TokenOwnership memory) {
        return _unpackedOwnership(_packedOwnershipOf(tokenId));
    }

    function _ownershipAt(uint256 index) internal view virtual returns (TokenOwnership memory) {
        return _unpackedOwnership(_packedOwnerships[index]);
    }

    function _initializeOwnershipAt(uint256 index) internal virtual {
        if (_packedOwnerships[index] == 0) {
            _packedOwnerships[index] = _packedOwnershipOf(index);
        }
    }

    function _packedOwnershipOf(uint256 tokenId) private view returns (uint256) {
        uint256 curr = tokenId;

        unchecked {
            if (_startTokenId() <= curr)
                if (curr < _currentIndex) {
                    uint256 packed = _packedOwnerships[curr];

                    if (packed & _BITMASK_BURNED == 0) {

                        while (packed == 0) {
                            packed = _packedOwnerships[--curr];
                        }
                        return packed;
                    }
                }
        }
        revert OwnerQueryForNonexistentToken();
    }

    function _unpackedOwnership(uint256 packed) private pure returns (TokenOwnership memory ownership) {
        ownership.addr = address(uint160(packed));
        ownership.startTimestamp = uint64(packed >> _BITPOS_START_TIMESTAMP);
        ownership.burned = packed & _BITMASK_BURNED != 0;
        ownership.extraData = uint24(packed >> _BITPOS_EXTRA_DATA);
    }

    function _packOwnershipData(address owner, uint256 flags) private view returns (uint256 result) {
        assembly {

            owner := and(owner, _BITMASK_ADDRESS)

            result := or(owner, or(shl(_BITPOS_START_TIMESTAMP, timestamp()), flags))
        }
    }

    function _nextInitializedFlag(uint256 quantity) private pure returns (uint256 result) {

        assembly {

            result := shl(_BITPOS_NEXT_INITIALIZED, eq(quantity, 1))
        }
    }

    function approve(address to, uint256 tokenId) public payable virtual override {
        address owner = ownerOf(tokenId);

        if (_msgSenderERC721A() != owner)
            if (!isApprovedForAll(owner, _msgSenderERC721A())) {
                revert ApprovalCallerNotOwnerNorApproved();
            }

        _tokenApprovals[tokenId].value = to;
        emit Approval(owner, to, tokenId);
    }

    function getApproved(uint256 tokenId) public view virtual override returns (address) {
        if (!_exists(tokenId)) revert ApprovalQueryForNonexistentToken();

        return _tokenApprovals[tokenId].value;
    }

    function setApprovalForAll(address operator, bool approved) public virtual override {
        _operatorApprovals[_msgSenderERC721A()][operator] = approved;
        emit ApprovalForAll(_msgSenderERC721A(), operator, approved);
    }

    function isApprovedForAll(address owner, address operator) public view virtual override returns (bool) {
        return _operatorApprovals[owner][operator];
    }

    function _exists(uint256 tokenId) internal view virtual returns (bool) {
        return
            _startTokenId() <= tokenId &&
            tokenId < _currentIndex &&
            _packedOwnerships[tokenId] & _BITMASK_BURNED == 0;
    }

    function _isSenderApprovedOrOwner(
        address approvedAddress,
        address owner,
        address msgSender
    ) private pure returns (bool result) {
        assembly {

            owner := and(owner, _BITMASK_ADDRESS)

            msgSender := and(msgSender, _BITMASK_ADDRESS)

            result := or(eq(msgSender, owner), eq(msgSender, approvedAddress))
        }
    }

    function _getApprovedSlotAndAddress(uint256 tokenId)
        private
        view
        returns (uint256 approvedAddressSlot, address approvedAddress)
    {
        TokenApprovalRef storage tokenApproval = _tokenApprovals[tokenId];

        assembly {
            approvedAddressSlot := tokenApproval.slot
            approvedAddress := sload(approvedAddressSlot)
        }
    }

    function transferFrom(
        address from,
        address to,
        uint256 tokenId
    ) public payable virtual override {
        uint256 prevOwnershipPacked = _packedOwnershipOf(tokenId);

        if (address(uint160(prevOwnershipPacked)) != from) revert TransferFromIncorrectOwner();

        (uint256 approvedAddressSlot, address approvedAddress) = _getApprovedSlotAndAddress(tokenId);

        if (!_isSenderApprovedOrOwner(approvedAddress, from, _msgSenderERC721A()))
            if (!isApprovedForAll(from, _msgSenderERC721A())) revert TransferCallerNotOwnerNorApproved();

        if (to == address(0)) revert TransferToZeroAddress();

        _beforeTokenTransfers(from, to, tokenId, 1);

        assembly {
            if approvedAddress {

                sstore(approvedAddressSlot, 0)
            }
        }

        unchecked {

            --_packedAddressData[from];
            ++_packedAddressData[to];

            _packedOwnerships[tokenId] = _packOwnershipData(
                to,
                _BITMASK_NEXT_INITIALIZED | _nextExtraData(from, to, prevOwnershipPacked)
            );

            if (prevOwnershipPacked & _BITMASK_NEXT_INITIALIZED == 0) {
                uint256 nextTokenId = tokenId + 1;

                if (_packedOwnerships[nextTokenId] == 0) {

                    if (nextTokenId != _currentIndex) {

                        _packedOwnerships[nextTokenId] = prevOwnershipPacked;
                    }
                }
            }
        }

        emit Transfer(from, to, tokenId);
        _afterTokenTransfers(from, to, tokenId, 1);
    }

    function safeTransferFrom(
        address from,
        address to,
        uint256 tokenId
    ) public payable virtual override {
        safeTransferFrom(from, to, tokenId, '');
    }

    function safeTransferFrom(
        address from,
        address to,
        uint256 tokenId,
        bytes memory _data
    ) public payable virtual override {
        transferFrom(from, to, tokenId);
        if (to.code.length != 0)
            if (!_checkContractOnERC721Received(from, to, tokenId, _data)) {
                revert TransferToNonERC721ReceiverImplementer();
            }
    }

    function _beforeTokenTransfers(
        address from,
        address to,
        uint256 startTokenId,
        uint256 quantity
    ) internal virtual {}

    function _afterTokenTransfers(
        address from,
        address to,
        uint256 startTokenId,
        uint256 quantity
    ) internal virtual {}

    function _checkContractOnERC721Received(
        address from,
        address to,
        uint256 tokenId,
        bytes memory _data
    ) private returns (bool) {
        try ERC721A__IERC721Receiver(to).onERC721Received(_msgSenderERC721A(), from, tokenId, _data) returns (
            bytes4 retval
        ) {
            return retval == ERC721A__IERC721Receiver(to).onERC721Received.selector;
        } catch (bytes memory reason) {
            if (reason.length == 0) {
                revert TransferToNonERC721ReceiverImplementer();
            } else {
                assembly {
                    revert(add(32, reason), mload(reason))
                }
            }
        }
    }

    function _mint(address to, uint256 quantity) internal virtual {
        uint256 startTokenId = _currentIndex;
        if (quantity == 0) revert MintZeroQuantity();

        _beforeTokenTransfers(address(0), to, startTokenId, quantity);

        unchecked {

            _packedAddressData[to] += quantity * ((1 << _BITPOS_NUMBER_MINTED) | 1);

            _packedOwnerships[startTokenId] = _packOwnershipData(
                to,
                _nextInitializedFlag(quantity) | _nextExtraData(address(0), to, 0)
            );

            uint256 toMasked;
            uint256 end = startTokenId + quantity;

            assembly {

                toMasked := and(to, _BITMASK_ADDRESS)

                log4(
                    0,
                    0,
                    _TRANSFER_EVENT_SIGNATURE,
                    0,
                    toMasked,
                    startTokenId
                )

                for {
                    let tokenId := add(startTokenId, 1)
                } iszero(eq(tokenId, end)) {
                    tokenId := add(tokenId, 1)
                } {

                    log4(0, 0, _TRANSFER_EVENT_SIGNATURE, 0, toMasked, tokenId)
                }
            }
            if (toMasked == 0) revert MintToZeroAddress();

            _currentIndex = end;
        }
        _afterTokenTransfers(address(0), to, startTokenId, quantity);
    }

    function _mintERC2309(address to, uint256 quantity) internal virtual {
        uint256 startTokenId = _currentIndex;
        if (to == address(0)) revert MintToZeroAddress();
        if (quantity == 0) revert MintZeroQuantity();
        if (quantity > _MAX_MINT_ERC2309_QUANTITY_LIMIT) revert MintERC2309QuantityExceedsLimit();

        _beforeTokenTransfers(address(0), to, startTokenId, quantity);

        unchecked {

            _packedAddressData[to] += quantity * ((1 << _BITPOS_NUMBER_MINTED) | 1);

            _packedOwnerships[startTokenId] = _packOwnershipData(
                to,
                _nextInitializedFlag(quantity) | _nextExtraData(address(0), to, 0)
            );

            emit ConsecutiveTransfer(startTokenId, startTokenId + quantity - 1, address(0), to);

            _currentIndex = startTokenId + quantity;
        }
        _afterTokenTransfers(address(0), to, startTokenId, quantity);
    }

    function _safeMint(
        address to,
        uint256 quantity,
        bytes memory _data
    ) internal virtual {
        _mint(to, quantity);

        unchecked {
            if (to.code.length != 0) {
                uint256 end = _currentIndex;
                uint256 index = end - quantity;
                do {
                    if (!_checkContractOnERC721Received(address(0), to, index++, _data)) {
                        revert TransferToNonERC721ReceiverImplementer();
                    }
                } while (index < end);

                if (_currentIndex != end) revert();
            }
        }
    }

    function _safeMint(address to, uint256 quantity) internal virtual {
        _safeMint(to, quantity, '');
    }

    function _burn(uint256 tokenId) internal virtual {
        _burn(tokenId, false);
    }

    function _burn(uint256 tokenId, bool approvalCheck) internal virtual {
        uint256 prevOwnershipPacked = _packedOwnershipOf(tokenId);

        address from = address(uint160(prevOwnershipPacked));

        (uint256 approvedAddressSlot, address approvedAddress) = _getApprovedSlotAndAddress(tokenId);

        if (approvalCheck) {

            if (!_isSenderApprovedOrOwner(approvedAddress, from, _msgSenderERC721A()))
                if (!isApprovedForAll(from, _msgSenderERC721A())) revert TransferCallerNotOwnerNorApproved();
        }

        _beforeTokenTransfers(from, address(0), tokenId, 1);

        assembly {
            if approvedAddress {

                sstore(approvedAddressSlot, 0)
            }
        }

        unchecked {

            _packedAddressData[from] += (1 << _BITPOS_NUMBER_BURNED) - 1;

            _packedOwnerships[tokenId] = _packOwnershipData(
                from,
                (_BITMASK_BURNED | _BITMASK_NEXT_INITIALIZED) | _nextExtraData(from, address(0), prevOwnershipPacked)
            );

            if (prevOwnershipPacked & _BITMASK_NEXT_INITIALIZED == 0) {
                uint256 nextTokenId = tokenId + 1;

                if (_packedOwnerships[nextTokenId] == 0) {

                    if (nextTokenId != _currentIndex) {

                        _packedOwnerships[nextTokenId] = prevOwnershipPacked;
                    }
                }
            }
        }

        emit Transfer(from, address(0), tokenId);
        _afterTokenTransfers(from, address(0), tokenId, 1);

        unchecked {
            _burnCounter++;
        }
    }

    function _setExtraDataAt(uint256 index, uint24 extraData) internal virtual {
        uint256 packed = _packedOwnerships[index];
        if (packed == 0) revert OwnershipNotInitializedForExtraData();
        uint256 extraDataCasted;

        assembly {
            extraDataCasted := extraData
        }
        packed = (packed & _BITMASK_EXTRA_DATA_COMPLEMENT) | (extraDataCasted << _BITPOS_EXTRA_DATA);
        _packedOwnerships[index] = packed;
    }

    function _extraData(
        address from,
        address to,
        uint24 previousExtraData
    ) internal view virtual returns (uint24) {}

    function _nextExtraData(
        address from,
        address to,
        uint256 prevOwnershipPacked
    ) private view returns (uint256) {
        uint24 extraData = uint24(prevOwnershipPacked >> _BITPOS_EXTRA_DATA);
        return uint256(_extraData(from, to, extraData)) << _BITPOS_EXTRA_DATA;
    }

    function _msgSenderERC721A() internal view virtual returns (address) {
        return msg.sender;
    }

    function _toString(uint256 value) internal pure virtual returns (string memory str) {
        assembly {

            let m := add(mload(0x40), 0xa0)

            mstore(0x40, m)

            str := sub(m, 0x20)

            mstore(str, 0)

            let end := str

            for { let temp := value } 1 {} {
                str := sub(str, 1)

                mstore8(str, add(48, mod(temp, 10)))

                temp := div(temp, 10)

                if iszero(temp) { break }
            }

            let length := sub(end, str)

            str := sub(str, 0x20)

            mstore(str, length)
        }
    }
}

pragma solidity ^0.8.17;

library Errors{
  error InvalidMintAmt();
  error MaxSupplyExceeded();
  error MaxWLSupplyExceeded();
  error mageLimitExceeeded();
  error limitExceeeded();
  error InsufficientFund();
  error TokenDoesnotExists();
  error NotOwner();
  error NotAuthorized();
  error wlClaimed();
  error StakingNotSet();
  error WLSaleNotActive();
  error ContractPaused();
  error InsuffWithdrawAmount();
  error AlreadyRevealed();
  error NotOwnerOrAdmin();
  error RevealClosed();
}

contract graph is ERC721A, Ownable, ReentrancyGuard {
    using Strings for uint256;
    bytes32 merkleRoot;
    bytes32 mageMerkleRoot;

    uint256 public cost;
    uint256 maxSupply;

    bool paused = true;
    bool whitelistMintEnabled = false;

    uint256 mageMint;
    uint256 wlMint;
    uint256 publicMint;

    string uriPrefix = '';
    string uriSuffix = '.json';

    mapping(uint256=>bool) revealed;
    mapping(uint256=>uint256) rarityStake;
    mapping(uint256=>uint256) rarityTokenId;

    uint256 private _poolA = 0;
    uint256 private _poolB = 4000;

    address payable private payments;
    bool paymentStatus = false;
    bool reveal = false;

    mapping(address=>uint256) mageList;
    uint256 public mageListAllowedNFTs;

    mapping(address=>bool) allowedList;

    uint256 totalBalance;

    address stakingAddress;

    bool mintStatus = false;

    address admin;

    event WhitelistMinted(address minter,uint256 amount);
    event minted(address minter,uint256 amount);
    event amountReleased();

    constructor(
    string memory _tokenName,
    string memory _tokenSymbol,
    uint256 _maxSupply,
    uint256 _mageListAllowedNFTs,

    address _payments
  ) ERC721A(_tokenName, _tokenSymbol) {
    maxSupply = _maxSupply;
    mageListAllowedNFTs = _mageListAllowedNFTs;

    payments = payable(_payments);
    cost = 0.0369 ether;
  }

  modifier mintCompliance(uint256 _mintQuantity) {
    require(_mintQuantity > 0, 'Invalid mint amount!');
    require(totalSupply() + _mintQuantity <= maxSupply, 'Max supply exceeded!');
    require(msg.value >= cost * _mintQuantity, 'Insufficient funds!');
    _;
  }

  modifier onlyAdminOrOwner() {
      if (owner() != _msgSender() && admin != _msgSender())
          revert Errors.NotOwnerOrAdmin();
      _;
    }

  modifier checkWalletAddress(){
    require(tx.origin == msg.sender,"Invalid Wallet Address");
    _;
  }

  function setMintDate(uint256 _mageMint,uint256 _wlMint, uint256 _publicMint) public onlyAdminOrOwner {
    mageMint = _mageMint;
    wlMint = _wlMint;
    publicMint = _publicMint;
  }
  function getMintDate() public view returns(uint256 _mageMint,uint256 _wlMint, uint256 _publicMint) {
    _mageMint = mageMint;
    _wlMint = wlMint;
    _publicMint = publicMint;
  }
  function whitelistVerify(bytes32[] calldata _merkleProof, bool _mage) checkWalletAddress() external view returns(bool){
    bytes32 leaf = keccak256(abi.encodePacked(_msgSender()));
    if(_mage){
      return MerkleProof.verify(_merkleProof, mageMerkleRoot, leaf);
    }else{
      return MerkleProof.verify(_merkleProof, merkleRoot, leaf);
    }
  }

  function revealNFT(uint256 _tokenId) public {
    if(ownerOf(_tokenId)!=_msgSender()) revert Errors.NotOwner();
    if(revealed[_tokenId]) revert Errors.AlreadyRevealed();
    if(!_exists(_tokenId)) revert Errors.TokenDoesnotExists();
    if(!reveal) revert Errors.RevealClosed();
    bool poolA = false;
    bool poolB = false;
    if(rarityStake[_tokenId] < 2592000) {
      if(_poolA + 1 <= 4000) poolA = true; else poolB = false;
    }

    if(rarityStake[_tokenId]  >= 2592000 ) {
      if(_poolB + 1 <= 8000) poolB = true; else poolA = false;
    }
    if(poolA){
      _poolA += 1;
      rarityTokenId[_tokenId] = _poolA;
      revealed[_tokenId] = true;
    }
    if(poolB){
      _poolB += 1;
      rarityTokenId[_tokenId] = _poolB;
      revealed[_tokenId] = true;
    }
  }

  function whitelistMint(uint256 _mintAmount, bytes32[] calldata _merkleProof) external payable mintCompliance(_mintAmount) checkWalletAddress() {
    if(!whitelistMintEnabled) revert Errors.WLSaleNotActive();
    if(block.timestamp >= mageMint && block.timestamp < wlMint){
      if(mageList[_msgSender()] + _mintAmount <= mageListAllowedNFTs ){
        mageList[_msgSender()] += _mintAmount;
      } else revert Errors.mageLimitExceeeded();
      bytes32 leaf = keccak256(abi.encodePacked(_msgSender()));
      require(MerkleProof.verify(_merkleProof, mageMerkleRoot, leaf), 'Invalid proof!');
    }else if(block.timestamp>=wlMint && block.timestamp < publicMint){
      if(allowedList[_msgSender()]){
        revert Errors.wlClaimed();
      }else{
        allowedList[_msgSender()] = true;
      }
      bytes32 leaf = keccak256(abi.encodePacked(_msgSender()));
      require(MerkleProof.verify(_merkleProof, merkleRoot, leaf), 'Invalid proof!');
    }
    totalBalance += msg.value;
    _safeMint(_msgSender(), _mintAmount);
    emit WhitelistMinted(_msgSender(), _mintAmount);
  }

  function mint(uint256 _mintAmount) public payable mintCompliance(_mintAmount) checkWalletAddress(){
    require(!paused, 'The contract is paused!');
    if(_mintAmount > mageListAllowedNFTs) revert Errors.limitExceeeded();
    totalBalance += msg.value;
    if(totalSupply() + _mintAmount >=maxSupply){mintStatus = true;}
    _safeMint(_msgSender(), _mintAmount);
    emit minted(_msgSender(), _mintAmount);
  }

  function releaseStake(uint256 _tokenId, uint256 _stake) external{
    if(stakingAddress!=_msgSender()) revert Errors.NotAuthorized();
    if(ownerOf(_tokenId) != _msgSender()) revert Errors.NotOwner();
    if(!_exists(_tokenId)) revert Errors.TokenDoesnotExists();
    if(_stake > 0 && !revealed[_tokenId]){
      rarityStake[_tokenId] += _stake;
    }
  }

  function tokenURI(uint256 _tokenId) public view virtual override returns (string memory) {
    if(!_exists(_tokenId)) revert Errors.TokenDoesnotExists();
    if(revealed[_tokenId]){
      uint256 tokenId = rarityTokenId[_tokenId];
      return bytes(_baseURI()).length > 0
          ? string(abi.encodePacked(_baseURI(), tokenId.toString(), uriSuffix))
          : '';
    }else{
      return bytes(_baseURI()).length > 0
          ? string(abi.encodePacked(_baseURI(), "hidden", uriSuffix))
          : '';
    }
  }

  function _baseURI() internal view virtual override returns (string memory) {
    return uriPrefix;
  }

  function mintForAddress(uint256 _mintAmount, address _receiver) public onlyAdminOrOwner {
    require(_mintAmount > 0, 'Invalid mint amount!');
    require(totalSupply() + _mintAmount <= maxSupply, 'Max supply exceeded!');
    if(totalSupply() + _mintAmount >=maxSupply){mintStatus = true;}
    _safeMint(_receiver, _mintAmount);
  }

  function mintForAddressBatch(address[] calldata _addresses,uint256[] calldata _mintAmount,uint256 _totalMintAmount) external onlyAdminOrOwner{
    require(_totalMintAmount > 0, 'Invalid mint amount!');
    require(totalSupply() + _totalMintAmount <= maxSupply, 'Max supply exceeded!');
      uint256 mintAmount = 0;
      for (uint256 i = 0; i < _addresses.length; i++) {
          _safeMint(_addresses[i], _mintAmount[i]);
          mintAmount += _mintAmount[i];
      }
      if(totalSupply() + mintAmount >=maxSupply){mintStatus = true;}
  }

  function setStakingAddress(address _stakingAddress) public onlyOwner{
    stakingAddress = _stakingAddress;
  }

  function setApproval(uint256 _tokenId) public{
    if(!_exists(_tokenId)) revert Errors.TokenDoesnotExists();
    if(ownerOf(_tokenId) != _msgSender()) revert Errors.NotOwner();
    approve(stakingAddress, _tokenId);
  }

  function setApprovalAll() public{
    if(stakingAddress==address(0)) revert Errors.StakingNotSet();
    setApprovalForAll(stakingAddress,true);
  }

  function setCost(uint256 _cost) public onlyAdminOrOwner() {
    cost = _cost;
  }

  function transferOwner(address newOwner) public onlyOwner{
    super.transferOwnership(newOwner);
  }

   function _startTokenId() internal view virtual override returns (uint256) {
    return 1;
  }

  function setUriPrefix(string memory _uriPrefix) public onlyOwner {
    uriPrefix = _uriPrefix;
  }

  function setUriSuffix(string memory _uriSuffix) public onlyOwner {
    uriSuffix = _uriSuffix;
  }

  function setPaused(bool _state) public onlyAdminOrOwner {
    paused = _state;
  }
  function getPaused() public view returns(bool _state){
    _state = paused;
  }
  function setMerkleRoot(bytes32 _merkleRoot, bytes32 _mageMerkleRoot) public onlyOwner {
    merkleRoot = _merkleRoot;
    mageMerkleRoot = _mageMerkleRoot;
  }
  function getMerkleRoot() public view returns(bytes32 _merkleRoot, bytes32 _mageMerkleRoot) {
    _merkleRoot = merkleRoot;
    _mageMerkleRoot = mageMerkleRoot;
  }
  function setMageLimit(uint256 _mageListAllowedNFTs) public onlyAdminOrOwner{
    mageListAllowedNFTs = _mageListAllowedNFTs;
  }
  function getMageList(address _address) public view returns(uint256 _tokens){
    _tokens = mageList[_address];
  }
    function getAllowList(address _address) public view returns(bool _state){
    _state = allowedList[_address];
  }
  function setWhitelistMintEnabled(bool _state) public onlyAdminOrOwner {
    whitelistMintEnabled = _state;
  }
  function getWhitelistMintEnabled() public view returns(bool _state) {
    _state = whitelistMintEnabled ;
  }
  function setMaxSupply(uint256 _maxSupply)public onlyAdminOrOwner{
    maxSupply = _maxSupply;
  }

  function currentBalance() public view returns(uint256){
    return address(this).balance;
  }

  function getTotalBalance() public view returns(uint256)
  {
    return totalBalance;
  }
  function getDetailsTokenId(uint256 _tokenId) public view returns(bool,uint256,uint256){
    return (revealed[_tokenId],rarityStake[_tokenId],rarityTokenId[_tokenId]);
  }
  function safeTransferFrom(
        address from,
        address to,
        uint256 tokenId
    ) public payable virtual override {
      if(to != stakingAddress){
        require(mintStatus,"STransfer: Can't Sell Now");
      }
      super.safeTransferFrom(from, to, tokenId, '');
  }

  function transferFrom(
        address from,
        address to,
        uint256 tokenId
    ) public payable virtual override(ERC721A) {
      if(to != stakingAddress){
        require(mintStatus,"Transfer: Can't Sell Now");
      }
      super.transferFrom(from, to, tokenId);
  }

  function safeTransferFrom(
        address from,
        address to,
        uint256 tokenId,
        bytes memory _data
    ) public payable virtual override{
      if(to != stakingAddress){
        require(mintStatus,"STransferD: Can't Sell Now");
      }
      super.safeTransferFrom(from,to,tokenId,_data);
    }

  function setApprovalForAll(address to, bool approved)
        public virtual override
    {
        if(to != stakingAddress){
          require(mintStatus,"Approve All: Can't Sell Now");
        }
        super.setApprovalForAll(to, approved);

    }

  function approve(address to, uint256 tokenId)
        public virtual payable override
    {
        if(to != stakingAddress){
          require(mintStatus,"Approve: Can't Sell Now");
        }
        super.approve(to, tokenId);
    }
  function setMintStatus(bool _mintStatus) public onlyAdminOrOwner{
    mintStatus = _mintStatus;
  }
  function setPayment(bool _status) public onlyOwner{
      paymentStatus = _status;
  }
  function setReveal(bool _status) public onlyOwner{
      reveal = _status;
  }
  function setAdmin(address _admin) public onlyOwner{
    admin = _admin;
  }
  function setPayments(address payable _payments) public onlyOwner{
    payments = payable(_payments);
  }
  function withdraw() external nonReentrant onlyAdminOrOwner{
    require(address(this).balance>0,"Release amount should be greater than zero");
    if(paymentStatus){
      (bool success, ) = payable(payments).call{value: address(this).balance}("");
      require(success);
      emit amountReleased();
    }
  }
}
