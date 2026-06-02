# EXP-L6 — Counterfactual Contract Pairs

**Layer:** 3 — Behavioral Interpretability  **Priority:** P1  **Status:** FAIL (1/4 pass)  
**Date run:** 2026-05-30 (re-run after solc fix)  **Checkpoint:** GCB-P1-Run4 (ep32, F1=0.3362)

## Hypothesis

For matched vulnerable/safe contract pairs, the model should assign a higher vulnerability score to the vulnerable contract than to the safe one for the correct class (vulnerable_score > safe_score). This would demonstrate that the model detects actual vulnerability patterns rather than superficial code structure.

## Method

Four hand-crafted contract pairs are compiled and graph-extracted on the fly: CEI Reentrancy (class 6), IntegerUO (class 4), Timestamp (class 7), and UnusedReturn (class 9). Each pair consists of a minimally-different vulnerable and safe version. The model is run on both, and the predicted probability for the target class is compared. The experiment requires solc compilation via solc-select.

## Fixes applied before re-run

- **solc-select broken symlink** repaired: `~/.solc-select/artifacts/solc-0.8.35/solc-0.8.35` was a self-referential symlink. Binary copied from `agents/.venv/.solc-select/` to the global artifacts directory.
- **solc 0.7.6 installed** for `^0.7.0` contracts.
- **`integer_uo_vulnerable.sol` upgraded** from `pragma ^0.7.0` to `^0.8.0` with `unchecked {}` arithmetic (the correct 0.8.x way to demonstrate integer overflow without needing a separate solc version).
- All 12 test contracts now compile cleanly with solc 0.8.35.

## Results

| Pair | Target class | Vulnerable score | Safe score | Delta | Result |
|------|-------------|-----------------|------------|-------|--------|
| CEI Reentrancy | Reentrancy | 0.1808 | 0.1823 | -0.0015 | **FAIL** |
| IntegerUO | IntegerUO | 0.4146 | 0.4718 | -0.0572 | **FAIL** |
| Timestamp | Timestamp | 0.0393 | 0.0393 | 0.0000 | **FAIL** |
| UnusedReturn | UnusedReturn | 0.1348 | 0.0268 | +0.1080 | **PASS** |

**Pass criteria:** vulnerable_score > safe_score for all pairs  
**Overall: FAIL — 1/4 pairs pass**

## Key Findings

- **UnusedReturn PASS (+0.1080):** The model correctly assigns higher probability to the contract missing a return-value check. This is the one class where the GNN has a structural signal (CALL edges where return value is dropped) and the corpus has consistent labelling.
- **Reentrancy FAIL (−0.0015):** The model cannot distinguish a CEI violation from a safe checks-effects-interactions pattern on a minimal contract. This directly confirms FINDING-2 (Phase 2 CFG Δ=0.014) and FINDING-9 (GNN eye F1=0.182 for Reentrancy). The model is not detecting the re-entrancy path; it is relying on token-level or corpus-level statistics.
- **IntegerUO FAIL (−0.0572 — inverted):** The safe contract scores *higher* (0.4718) than the vulnerable one (0.4146). The vulnerability pattern (`unchecked {}` blocks with arithmetic) does not produce higher IntegerUO probability. The model likely responds to contract size or overall complexity rather than the `unchecked` syntax. The training corpus may not contain sufficient minimal `unchecked`-arithmetic contracts (Sol-2 injection is required).
- **Timestamp FAIL (0.0000):** Both the vulnerable contract (using `block.timestamp` in a branch condition) and the safe contract score identically (0.0393). This confirms FINDING-8 — the model's Timestamp signal is correlated with contract size, not the presence of `block.timestamp` in control flow. Both minimal contracts are small, so both score low regardless of content.

## Architecture Implications

The 1/4 pass result is highly informative and consistent with all validated findings from the interpretability suite:

1. **CEI auxiliary loss is essential (Interp-2):** The Reentrancy failure on a minimal hand-crafted pair with an explicit re-entrancy structure confirms Phase 2 does not carry the CEI signal. Without a supervision signal that forces Phase 2 to attend to the violation path, the model cannot detect it on novel contracts.
2. **Sol-2 injection required before IntegerUO generalises:** The inverted IntegerUO result (safe > vulnerable) shows the model has not learned to associate `unchecked {}` blocks with overflow risk. Injecting labelled `unchecked`-arithmetic contracts (Sol-2) is a prerequisite for this class to generalise.
3. **Timestamp gating required (Sol-3 / Interp-3):** Identical scores for both Timestamp contracts confirms the size-shortcut: both are small, so both score near the class mean for small contracts. Gating Timestamp labels to contracts where `block.timestamp` actually appears in a branch condition (Sol-3) and normalising for size (Interp-3) are both required.
4. **UnusedReturn is the most reliable class on novel contracts:** This is the only class where a structural pattern (dropped return value call edge) directly drives the prediction on out-of-distribution contracts.

## Caveats

- The counterfactual pairs are minimal by design. A more representative vulnerable IntegerUO contract (with complex control flow and multiple `unchecked` blocks) might produce a different result, but minimality is precisely the point of counterfactual testing.
- EXP-L9 (attention rollout on the same test contracts) uses on-the-fly Slither extraction without solc-select and remains the complementary experiment for understanding which nodes the model attends to.
- The PASS on UnusedReturn does not mean the model is robust — it means the structural signal for that specific class happens to survive the gap between training-corpus graphs and hand-crafted graphs.
