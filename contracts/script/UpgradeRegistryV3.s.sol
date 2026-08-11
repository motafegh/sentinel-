// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "../src/AuditRegistry.sol";

/// @notice Privileged deployment/upgrade workflow for activating AuditRegistry V3.
/// @dev This is intentionally separate from runtime audit submission. The script
/// requires the current proxy owner to be the broadcaster and performs a
/// UUPS upgradeToAndCall that initializes the dedicated V3 verifier + policy
/// signer and permanently disables new V1/V2 writes.
contract UpgradeRegistryV3 is Script {
    function run() external returns (address implementationAddress) {
        uint256 deployerKey = vm.envUint("DEPLOYER_PRIVATE_KEY");
        address deployer = vm.addr(deployerKey);
        address proxyAddress = vm.envAddress("AUDIT_REGISTRY_PROXY");
        address verifierV3 = vm.envAddress("ZKML_VERIFIER_V3");
        address policySignerV3 = vm.envAddress("AUDIT_POLICY_SIGNER_V3");

        require(proxyAddress != address(0), "UpgradeRegistryV3: zero proxy");
        require(verifierV3 != address(0), "UpgradeRegistryV3: zero verifier");
        require(policySignerV3 != address(0), "UpgradeRegistryV3: zero policy signer");

        AuditRegistry current = AuditRegistry(proxyAddress);
        require(current.owner() == deployer, "UpgradeRegistryV3: broadcaster is not proxy owner");
        require(!current.legacySubmissionsDisabled(), "UpgradeRegistryV3: V3 already active");

        vm.startBroadcast(deployerKey);
        AuditRegistry implementation = new AuditRegistry();
        current.upgradeToAndCall(
            address(implementation),
            abi.encodeCall(AuditRegistry.initializeV3, (verifierV3, policySignerV3))
        );
        vm.stopBroadcast();

        implementationAddress = address(implementation);

        // Read-back assertions are part of the deployment contract. A script
        // must fail rather than merely print a partially configured upgrade.
        AuditRegistry upgraded = AuditRegistry(proxyAddress);
        require(upgraded.legacySubmissionsDisabled(), "UpgradeRegistryV3: legacy writes still enabled");
        require(address(upgraded.zkmlVerifierV3()) == verifierV3, "UpgradeRegistryV3: verifier mismatch");
        require(upgraded.auditPolicySignerV3() == policySignerV3, "UpgradeRegistryV3: signer mismatch");
        require(upgraded.owner() == deployer, "UpgradeRegistryV3: owner changed unexpectedly");

        console2.log("AuditRegistry proxy:", proxyAddress);
        console2.log("V3 implementation:", implementationAddress);
        console2.log("V3 verifier:", verifierV3);
        console2.log("V3 policy signer:", policySignerV3);
        console2.log("Legacy writes disabled:", upgraded.legacySubmissionsDisabled());
    }
}
