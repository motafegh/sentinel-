// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "../src/ZKMLVerifier.sol";
import "./generated/ZkmlV2ProofFixture.sol";

/// @notice Exercises the actual generated Halo2 verifier against the tracked
/// historical EZKL proof/signals. The fixture is generated in CI directly from
/// zkml/ezkl/proof.json; it is not a mock proof path.
contract ZKMLVerifierRealProofTest is Test {
    Halo2Verifier internal verifier;

    function setUp() public {
        verifier = new Halo2Verifier();
    }

    function test_tracked_v2_proof_verifies_with_canonical_generated_verifier() public {
        bytes memory proof = ZkmlV2ProofFixture.proof();
        uint256[] memory signals = ZkmlV2ProofFixture.signals();
        assertEq(signals.length, 138);
        assertTrue(verifier.verifyProof(proof, signals));
    }

    function test_mutated_public_output_is_rejected_fail_closed() public {
        bytes memory proof = ZkmlV2ProofFixture.proof();
        uint256[] memory signals = ZkmlV2ProofFixture.signals();
        signals[137] += 1;

        // EZKL-generated Halo2 verifiers are allowed to reject an invalid
        // proof either by returning false or by reverting when the pairing
        // check fails. Both outcomes are fail-closed. The only forbidden
        // outcome is a successful call returning true.
        (bool callSucceeded, bytes memory returndata) = address(verifier).call(
            abi.encodeCall(Halo2Verifier.verifyProof, (proof, signals))
        );

        if (callSucceeded) {
            assertEq(returndata.length, 32, "verifier returned malformed success data");
            assertFalse(abi.decode(returndata, (bool)), "mutated public output verified");
        }
    }
}
