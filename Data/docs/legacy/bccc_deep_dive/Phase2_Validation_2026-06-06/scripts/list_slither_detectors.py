import pkgutil
import slither.detectors as det

print("=" * 80)
print("SLITHER DETECTORS — Complete list with descriptions")
print("=" * 80)

total = 0
all_dets = []
for finder, name, ispkg in pkgutil.iter_modules(det.__path__):
    if name in ("abstract_detector", "all_detectors", "examples"):
        continue
    try:
        mod = __import__(f"slither.detectors.{name}", fromlist=["*"])
        dets = []
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type) and hasattr(obj, "ARGUMENT") and hasattr(obj, "WIKI"):
                wiki = obj.WIKI
                desc = wiki.split("\\n")[0][:100] if isinstance(wiki, str) else "?"
                dets.append((attr, desc))
        if dets:
            print(f"\n--- {name} ({len(dets)} detectors) ---")
            for d, desc in sorted(dets):
                print(f"  {d:45s} | {desc}")
            total += len(dets)
            all_dets.extend([(name, d, desc) for d, desc in dets])
    except Exception as e:
        print(f"  ERR {name}: {e}")

print(f"\nTotal detector classes: {total}")

print("\n" + "=" * 80)
print("BCCC CLASS MAPPING (preliminary — to be refined in Phase 3)")
print("=" * 80)
mapping = {
    "Class01:ExternalBug": ["arbitrary_send_erc20", "arbitrary_send_erc20_permit",
                            "controlled_delegatecall", "delegatecall_loop",
                            "low_level_calls", "unchecked_lowlevel", "unchecked_send",
                            "arbitrary_send", "msg_value_loop"],
    "Class02:GasException": ["void_constructor", "constant_function_asm",
                             "constant_function_state", "events_maths",
                             "locked_ether"],
    "Class03:MishandledException": ["incorrect_return", "locked_ether",
                                    "uninitialized_state", "uninitialized_storage",
                                    "uninitialized_local", "mapping_deletion",
                                    "modifying_storage_array_by_value"],
    "Class04:Timestamp": ["timestamp", "weak_prng", "block_timestamp",
                          "eth_balance_dependent", "encode_decode_assembly"],
    "Class05:TransactionOrderDependence": [],  # DROPPED per D-F1
    "Class06:UnusedReturn": ["unchecked_transfer", "low_level_calls",
                             "unchecked_lowlevel", "unchecked_send", "unused_return"],
    "Class07:WeakAccessMod": ["suicidal", "arbitrary_send_erc20",  # proxy
                              "tx_origin", "unprotected_upgrade"],
    "Class08:CallToUnknown": ["missing_zero_address_validation", "missing_zero_address",
                              "multiple_zero_address", "uninitialized_fptr"],
    "Class09:DenialOfService": ["calls_loop", "msg_value_loop", "locked_ether"],
    "Class10:IntegerUO": ["divide_before_multiply", "incorrect_exp",
                          "shift_parameter", "tautology", "tautological_compare",
                          "out_of_bounds_array", "incorrect_equality",
                          "strict_equality"],
    "Class11:Reentrancy": ["reentrancy_eth", "reentrancy_no_eth",
                           "reentrancy_events", "reentrancy_benign",
                           "reentrancy_unlimited_gas"],
    "Class12:NonVulnerable": ["naming_convention", "redundant_zero_check",
                              "similar_names", "events_access", "deprecated_standards",
                              "domain_separator_collision", "erc20_indexed",
                              "erc20_interface", "erc721_interface",
                              "unimplemented_functions", "unused_import",
                              "unused_state", "var_read_using_this"],
}

# Validate mapping
all_mapped = set()
for cls, det_list in mapping.items():
    all_mapped.update(det_list)
all_dets_set = set(d[1] for d in all_dets)
unmapped = all_dets_set - all_mapped
print(f"\nDetectors NOT mapped to BCCC class (likely benign or low-priority):")
for d in sorted(unmapped):
    if d not in ("ABSTRACT_BASE_CLASS_NAME", "AbstractDetector"):
        print(f"  {d}")
print(f"\nTotal slither detectors: {len(all_dets)}")
print(f"Detectors mapped to BCCC classes: {len(all_mapped)}")
print(f"Unmapped: {len(unmapped)}")
