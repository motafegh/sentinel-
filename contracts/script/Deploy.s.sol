// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import "../src/SentinelToken.sol";
import "../src/AuditRegistry.sol";

/// @notice Fresh SENTINEL deployment with V3 activated before the script exits.
/// @dev Privileged deployment is intentionally separate from runtime audit submission.
/// The generated verifier is deployed separately and provided by address. A fresh
/// registry is not considered successfully deployed unless V3 verifier/signer roots are
/// configured and the historical V1/V2 write paths are disabled.
contract Deploy is Script {
    function run() external {
        uint256 deployerKey = vm.envUint("DEPLOYER_PRIVATE_KEY");
        address verifierV3 = vm.envAddress("ZKML_VERIFIER_V3");
        address policySignerV3 = vm.envAddress("AUDIT_POLICY_SIGNER_V3");

        require(verifierV3 != address(0), "Deploy: zero V3 verifier");
        require(policySignerV3 != address(0), "Deploy: zero V3 policy signer");

        address deployer = vm.addr(deployerKey);
        console2.log("Deployer:", deployer);
        console2.log("V3 verifier:", verifierV3);
        console2.log("V3 policy signer:", policySignerV3);

        vm.startBroadcast(deployerKey);

        SentinelToken sentinelToken = new SentinelToken();

        AuditRegistry implementation = new AuditRegistry();
        bytes memory initData = abi.encodeCall(
            AuditRegistry.initialize,
            (verifierV3, address(sentinelToken))
        );
        ERC1967Proxy proxy = new ERC1967Proxy(address(implementation), initData);
        AuditRegistry registry = AuditRegistry(address(proxy));

        // Fresh deployments must enter the V3 fail-closed state immediately.
        // The base verifier is set to the same verifier for historical storage
        // compatibility, but V1/V2 writes are permanently disabled by initializeV3.
        registry.initializeV3(verifierV3, policySignerV3);

        vm.stopBroadcast();

        require(
            address(registry.sentinelToken()) == address(sentinelToken),
            "Deploy: token address mismatch"
        );
        require(
            address(registry.zkmlVerifierV3()) == verifierV3,
            "Deploy: V3 verifier address mismatch"
        );
        require(
            registry.auditPolicySignerV3() == policySignerV3,
            "Deploy: V3 policy signer mismatch"
        );
        require(
            registry.legacySubmissionsDisabled(),
            "Deploy: legacy submissions still enabled"
        );
        require(
            registry.owner() == deployer,
            "Deploy: unexpected registry owner"
        );
        require(
            sentinelToken.totalSupply() == 1_000_000 * 10 ** 18,
            "Deploy: unexpected total supply"
        );

        console2.log("SentinelToken:", address(sentinelToken));
        console2.log("AuditRegistry proxy:", address(registry));
        console2.log("AuditRegistry implementation:", address(implementation));
        console2.log("Legacy writes disabled:", registry.legacySubmissionsDisabled());
    }
}
