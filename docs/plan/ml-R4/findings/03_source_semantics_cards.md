# 03 — Source Semantics Cards

These cards describe what each source-native record can and cannot mean before SENTINEL converts it into a fixed ten-cell target.

## SolidiFI

**Native assertion unit:** injection folder for one contract.  
**Positive meaning:** the named vulnerability was programmatically injected; T0 positive authority.  
**Negative meaning:** none for the other classes. The corpus is an injection benchmark, not a ten-class clean-control study.  
**Parser behavior:** maps one folder to one canonical class and writes that cell as `1`; every other canonical cell becomes `0`.  
**Loss introduced:** nine non-target cells become binary negatives without source-native negative evidence.  
**Canonical mappings:** Re-entrancy→Reentrancy; Timestamp-Dependency→Timestamp; Unhandled-Exceptions→MishandledException; TOD→TransactionOrderDependence; Overflow-Underflow→IntegerUO; Unchecked-Send→CallToUnknown; tx.origin→ExternalBug.  
**Unsupported canonical classes:** DenialOfService, GasException, UnusedReturn.

## DIVE

**Native assertion unit:** one CSV row with eight vulnerability columns plus the corresponding Solidity source.  
**Native cell semantics:** positive values are recognized as positive; empty cells are explicitly documented by folderization as `unknown`.  
**Folderization behavior:** creates symlinks only for positive cells. `0` and empty/unknown both create no symlink. Missing source files are skipped.  
**Parser behavior:** reconstructs labels from folder membership; membership→`1`, no membership→`0`; if no mapped memberships exist, the record is counted as NonVulnerable.  
**Crosswalk:** Reentrancy→Reentrancy; DoS→DenialOfService; Arithmetic→IntegerUO; Time manipulation→Timestamp; Front Running→TransactionOrderDependence; Access Control→ExternalBug; Unchecked Return Values→UnusedReturn.  
**Dropped native category:** Bad Randomness. 634 files are reported in that folder; exclusive Bad-Randomness rows become all-zero after the drop, but the exact exclusive count is not retained remotely.  
**Unsupported canonical classes:** CallToUnknown, GasException, MishandledException.  
**Recovered quality evidence:** ExternalBug and Reentrancy folder labels have very low manual TP rates; DIVE+tool agreement did not materially improve precision.  
**Primary corruption:** `unknown` and `explicit 0` become observationally identical once folderized, and dropped/unsupported categories also become zeros.

## SmartBugs Curated

**Native assertion unit:** category-folder membership in a hand-labeled benchmark.  
**Positive meaning:** the folder category is the source assertion.  
**Parser behavior:** one mapped category becomes `1`; all other canonical classes become `0`.  
**Lossy mappings:** `bad_randomness`→Timestamp.  
**Mapped-to-NonVulnerable categories:** `short_addresses`, `other`.  
**Recovered source evidence:** 143 contracts; four explicit NonVulnerable examples reported in Phase 1; aggregate semantic recall 94.4%.  
**Primary corruption:** a single-category source assertion is expanded into nine negative cells even though those classes were not independently reviewed; categories outside the ten-class taxonomy can become an all-zero target.

## Web3Bugs

**Configured role:** enabled Tier-1 Gold source.  
**Recovered reality:** no source corpus, parser, crosswalk, or usable acquisition path is present in the repository. The configured `web3bugs.yaml` path does not exist.  
**Historical target effect:** none can be established from the active historical export.  
**Semantic rule:** absence of the source must remain `UNAVAILABLE`, never a ten-class negative vector.

## DISL

**Configured role:** enabled Bronze source used only for NonVulnerable pool.  
**Recovered reality:** the configured Etherscan connector is a stub and raises `NotImplementedError`; the older export audit records DISL as skipped because no preprocessed source existed.  
**Semantic risk:** “unlabeled” was conceptually equated with NonVulnerable in the design. That is not class-specific negative evidence.  
**Historical target effect:** no executable/current contribution is proven from the tracked repository.

## DeFiHackLabs

**Configured state:** disabled.  
**Native assertion:** real exploit PoCs.  
**Recovered reality:** partial preprocessing only; Foundry/forge-std dependencies blocked the intended source path.  
**Phase-2 role:** historical evidence only; no active historical target transformation.

## BCCC

**Configured state:** deferred.  
**Recovered artifact:** `contracts_clean_v1.4.csv` with 67,311 rows and class-specific verified/provisional/best-effort decisions.  
**Important historical result:** major reductions in Reentrancy, CallToUnknown, ExternalBug and DoS labels; three classes remained provisional because Stage 5.5 ML propagation was never run.  
**Active-pipeline role:** not loaded by current source configuration.  
**Phase-2 rule:** BCCC v1.4 may explain historical decisions and the origin of the June DoS patch, but must not be silently treated as an active training source.

## Cross-source semantic invariant

A ten-class vector is a **representation produced by SENTINEL**, not a native assertion format shared by the sources. Therefore:

- a positive cell can have source-specific authority;
- a zero cell requires its own provenance;
- no-source, unsupported-class, dropped-category, unknown, and explicit-negative states must not be merged merely because they all serialize as `0` historically.
