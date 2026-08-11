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

    function test_mutated_public_output_does_not_verify() public {
        bytes memory proof = ZkmlV2ProofFixture.proof();
        uint256[] memory signals = ZkmlV2ProofFixture.signals();
        signals[137] += 1;
        assertFalse(verifier.verifyProof(proof, signals));
    }
}
