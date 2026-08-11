// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "../src/AuditRegistry.sol";

/// @notice Cross-language EIP-712 compatibility check.
/// @dev CI computes SENTINEL_GOLDEN_V3_DIGEST with the Python policy-signer
/// implementation and this test independently computes the digest in Solidity.
contract AuditRegistryV3GoldenDigestTest is Test {
    address internal constant AGENT = 0x1111111111111111111111111111111111111111;
    address internal constant TARGET = 0x2222222222222222222222222222222222222222;
    address internal constant REGISTRY = 0x3333333333333333333333333333333333333333;

    uint256 internal constant CHAIN_ID = 31337;
    uint256 internal constant ROUND_ID = 77;
    uint256 internal constant DEADLINE = 2_000_000_000;
    uint256 internal constant INPUT_OFFSET = 128;
    uint256 internal constant NUM_CLASSES = 10;
    uint256 internal constant TOTAL_SIGNALS = 138;

    bytes internal constant TARGET_RUNTIME_CODE = hex"6001600055";
    bytes internal constant PROOF = hex"deadbeef";

    bytes32 internal constant TEACHER_HASH = bytes32(uint256(type(uint256).max / 3));

    function setUp() public {
        vm.chainId(CHAIN_ID);

        // computeAuditDigestV3 has no storage dependency. Etching the deployed
        // implementation runtime at a fixed address gives the EIP-712 domain a
        // deterministic verifyingContract without needing proxy initialization.
        AuditRegistry implementation = new AuditRegistry();
        vm.etch(REGISTRY, address(implementation).code);
        vm.etch(TARGET, TARGET_RUNTIME_CODE);
    }

    function _scores() internal pure returns (uint256[10] memory scores) {
        for (uint256 i = 0; i < NUM_CLASSES; i++) {
            scores[i] = 1200 + i * 41;
        }
    }

    function _signals(uint256[10] memory scores)
        internal
        pure
        returns (uint256[] memory signals)
    {
        signals = new uint256[](TOTAL_SIGNALS);
        for (uint256 i = 0; i < INPUT_OFFSET; i++) {
            signals[i] = i + 17;
        }
        for (uint256 i = 0; i < NUM_CLASSES; i++) {
            signals[INPUT_OFFSET + i] = scores[i];
        }
    }

    function test_v3_digest_matches_python_golden() public {
        // Exact bytes32 constants from agents/tests/test_policy_signer_v3.py.
        bytes32 teacherHash = 0x5555555555555555555555555555555555555555555555555555555555555555;
        bytes32 bundleHash = 0x6666666666666666666666666666666666666666666666666666666666666666;
        bytes32 dataHash = 0x7777777777777777777777777777777777777777777777777777777777777777;
        bytes32 schemaHash = 0x8888888888888888888888888888888888888888888888888888888888888888;

        AuditRegistry.AuditContextV3 memory context = AuditRegistry.AuditContextV3({
            contractAddress: TARGET,
            roundId: ROUND_ID,
            teacherModelHash: teacherHash,
            proxyBundleHash: bundleHash,
            dataVersionHash: dataHash,
            classSchemaHash: schemaHash,
            deadline: DEADLINE
        });

        uint256[10] memory scores = _scores();
        uint256[] memory signals = _signals(scores);
        bytes32 actual = AuditRegistry(REGISTRY).computeAuditDigestV3(
            AGENT,
            context,
            keccak256(PROOF),
            keccak256(abi.encode(signals)),
            keccak256(abi.encode(scores))
        );

        bytes32 expected = vm.envBytes32("SENTINEL_GOLDEN_V3_DIGEST");
        assertEq(actual, expected, "Python/Solidity V3 EIP-712 digest drift");
        assertEq(TARGET.codehash, keccak256(TARGET_RUNTIME_CODE));
    }
}
