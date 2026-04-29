// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// RECALL — Interface matches Halo2Verifier's actual generated ABI exactly.
// Function name: verifyProof (EZKL's generated name, not verify)
// nonpayable not view — EZKL's verifier modifies memory internally
// even though it doesn't change contract storage.
// Lesson: always check the generated ABI before writing the interface.
interface IZKMLVerifier {
    function verifyProof(
        bytes calldata proof,
        uint256[] calldata instances
    ) external returns (bool);
}