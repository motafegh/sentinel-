// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import "../src/SentinelToken.sol";
import "../src/AuditRegistry.sol";

// RECALL — Deploy order matters:
//   1. SentinelToken  — no deps
//   2. ZKMLVerifier   — no deps (deployed separately with solc 0.8.17, address provided via env)
//   3. AuditRegistry  — needs verifier + token addresses
//
// ZKMLVerifier is NOT deployed here because it requires a different solc version (0.8.17).
// It must be deployed separately:
//   solc-select use 0.8.17
//   forge create src/ZKMLVerifier.sol:Halo2Verifier --rpc-url $SEPOLIA_RPC_URL --private-key $DEPLOYER_PRIVATE_KEY
//   solc-select use 0.8.20
// Then set ZKML_VERIFIER_ADDRESS in the environment before running this script.
contract Deploy is Script {
    function run() external {
        uint256 deployerKey       = vm.envUint("DEPLOYER_PRIVATE_KEY");
        address zkmlVerifierAddr  = vm.envAddress("ZKML_VERIFIER_ADDRESS");

        address deployer = vm.addr(deployerKey);
        console2.log("Deployer:       ", deployer);
        console2.log("ZKMLVerifier:   ", zkmlVerifierAddr);

        vm.startBroadcast(deployerKey);

        // 1 — SentinelToken
        SentinelToken sentinelToken = new SentinelToken();
        console2.log("SentinelToken:  ", address(sentinelToken));

        // 2 — AuditRegistry UUPS proxy
        AuditRegistry impl = new AuditRegistry();
        bytes memory initData = abi.encodeCall(
            AuditRegistry.initialize,
            (zkmlVerifierAddr, address(sentinelToken))
        );
        ERC1967Proxy proxy = new ERC1967Proxy(address(impl), initData);
        AuditRegistry registry = AuditRegistry(address(proxy));
        console2.log("AuditRegistry:  ", address(registry));
        console2.log("  impl:         ", address(impl));

        vm.stopBroadcast();

        // Sanity checks — revert here = deployment misconfigured
        require(
            address(registry.sentinelToken()) == address(sentinelToken),
            "Deploy: token address mismatch"
        );
        require(
            address(registry.zkmlVerifier()) == zkmlVerifierAddr,
            "Deploy: verifier address mismatch"
        );
        require(
            sentinelToken.totalSupply() == 1_000_000 * 10 ** 18,
            "Deploy: unexpected total supply"
        );

        console2.log("Deploy complete. Add to .env:");
        console2.log("SENTINEL_TOKEN_ADDRESS=",   address(sentinelToken));
        console2.log("AUDIT_REGISTRY_ADDRESS=",   address(registry));
    }
}
