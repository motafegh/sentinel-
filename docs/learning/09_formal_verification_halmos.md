# 09. Formal Verification (Halmos): Symbolic Execution as Evidence

> **Prerequisites:** [01. The Audit Pipeline] — the `formal_verification` node lives in the deep-path fan-out. [02. Evidence Model & Fuse()] — the `Evidence.formal()` constructor, `Kind.FORMAL`, and `emit_halmos_evidence()`.
> **Next:** [10. Decision Numbers] is the capstone — it covers how the maturity ladder applies to all decision numbers including reliability weights for Halmos.
> **Cross-ref:** [05. MCP Architecture] — Halmos is the only analysis node that does NOT run as an MCP server (it runs as a subprocess with Foundry). [06. RAG Hybrid Retrieval] — Halmos covers different vulnerability classes than RAG.
> **Scope:** This doc covers the Halmos symbolic execution node: how it's wired into the pipeline, the temp Foundry project pattern, the 5 invariants and their class mappings, the fail-soft guarantees, and how formal evidence differs from all other evidence types. It does NOT cover the full list of vulnerability classes (see Doc 02) or the eval framework for measuring Halmos reliability (see Doc 08).
> **TL;DR:** Halmos symbolic execution is the strongest evidence channel — it mathematically proves or refutes 5 vulnerability invariants (reentrancy, arithmetic, access control, unchecked return, DoS). The `formal_verification` node creates a temp Foundry project, generates a test harness with invariant checks, runs `forge build` + `halmos --json-output`, and parses the results into `Evidence(kind=FORMAL, deterministic=True)`. Formal evidence has strength 1.0 for SUPPORTS (violation found — mathematical proof) and 0.9 for REFUTES (invariant holds — but limited by Halmos coverage). The node fails soft on missing tools, compile errors, or timeouts — empty findings propagate with `tool_status["halmos"]["ran": False]`. This is the only evidence channel that proves, not guesses.

---

## The Problem: ML Guesses, Static Tools Pattern-Match — Neither Proves

You have ML probabilities ("85% chance this is reentrancy") and static-analysis detector matches ("Slither found a `reentrancy-eth` pattern"). Both are useful. Neither is a proof.

ML is statistical — it outputs a probability. If the training data had 10 reentrancy contracts and 8 looked like yours, you get 0.80. That's a correlation, not a guarantee. Static tools pattern-match — Slither checks for the CEI (Checks-Effects-Interactions) pattern violation by looking at state-variable access order. It can match on code that *looks* like a reentrancy bug but is actually safe (false positive), and it can miss novel patterns it wasn't programmed to detect.

Formal verification solves this differently: it *mathematically proves* or *refutes* a specific invariant about the contract. It doesn't guess or pattern-match. It explores all possible execution paths via symbolic execution and asks: "Is there any path through this contract that violates this invariant?"

The tradeoff: formal verification is slow (minutes, not seconds), limited in scope (only 5 invariants, not all vulnerability types), and requires a full compilation environment (Foundry project + solc). But when it finds a violation, you know it's real — the counterexample proves it.

---

## How We Arrived at This Design

### Step 1 — Invariant: Formal evidence is the strongest tier (Principle 7)

"Mathematical proof > statistical guess." If you have a proof of a vulnerability and a statistical guess that says "safe," the proof wins — assuming the proof is correct and within scope. The evidence model already had `Kind.FORMAL` as a category, and `Evidence.formal()` was stubbed with a docstring saying "future Halmos." The system was waiting for a tool that could produce it.

### Step 2 — Constraint: Halmos needs a full Foundry project, not just a `.sol` file

Halmos runs on Foundry test contracts — it needs `src/`, `test/`, `foundry.toml`, and the `forge-std` library symlink. You can't just pass it a Solidity file. This means the node must:
- Create a temp directory with the full Foundry project structure
- Write the contract as `src/Target.sol`
- Generate a test harness at `test/FormalVerify.t.sol`
- Symlink `forge-std` from the host (or use the FORGE_STD_PATH env var)
- Run `forge build` before `halmos`

### Step 3 — Simplest solution: temp directory + generated test harness

The temp-directory pattern is standard for tool integration (you saw it in Doc 05 for MCP servers, but here it's even simpler — no network, just filesystem). Create a `TemporaryDirectory`, lay out the Foundry structure, run the subprocesses, parse the JSON output. The whole thing is ~220 lines (`formal_verification.py:53-270`).

### Step 4 — Stress-test: what happens when forge build fails or Halmos times out?

Contracts written for different Solidity versions or with external imports can fail `forge build`. Large contracts can cause Halmos path explosion (the symbolic executor explores all paths, which can be exponential). The 120s timeout catches the explosion case. On any failure — missing tool, compile error, parse error, timeout — the node returns empty findings with `tool_status["halmos"]["ran": False]`. The pipeline continues with other evidence channels.

### Step 5 — Measure: 5 invariants, 15 tests

The invariant set was chosen to cover the most common Solidity vulnerability classes that Halmos can meaningfully check: reentrancy, arithmetic overflow/underflow, access control, unchecked return values, and denial of service. Each maps to a vulnerability class in the system. The tests cover all parsing paths (pass, fail, empty, invalid JSON, unknown invariants), the evidence constructor, and the fail-soft paths.

---

## The Solution

### The node flow

```
contract_code → extract contract name
  → create temp Foundry project (src/Target.sol, test/FormalVerify.t.sol, foundry.toml)
  → symlink forge-std
  → forge build (compile)
  → halmos --json-output (symbolic execution, 120s timeout)
  → parse JSON: {results: [{name: "check_reentrancy()", status: "pass"|"fail"}]}
  → map invariant → vulnerability class via _INVARIANT_TO_CLASS
  → emit Evidence.formal(source="halmos", polarity=SUPPORTS|REFUTES, ...)
  → fail-soft: empty findings + tool_status["halmos"]["ran": False] on any error
```

### The 5 invariants and their mappings

| Halmos test function | Invariant name | Vulnerability class | What it checks |
|---------------------|----------------|---------------------|----------------|
| `check_reentrancy()` | `reentrancy` | Reentrancy | CEI pattern violation — state change after external call |
| `check_arithmetic()` | `arithmetic` | IntegerUO | Overflow/underflow (Solidity >=0.8 has built-in checks, but unchecked blocks can bypass) |
| `check_access_control()` | `access_control` | AccessControl | Unauthorized address calls restricted functions |
| `check_unchecked_return()` | `unchecked_return` | UnusedReturn | External call return value not checked |
| `check_denial_of_service()` | `denial_of_service` | DenialOfService | Loop that can be forced to fail or exhaust gas |

### Formal evidence: why strength differs by polarity

When Halmos finds a violation (`status="fail"`), it provides a counterexample — concrete inputs that trigger the bug. This is a mathematical proof that the vulnerability exists. Evidence strength is 1.0 — the strongest possible signal in the system.

When Halmos proves an invariant holds (`status="pass"`), it means no execution path violates the invariant. But this is limited by the invariant itself — you only checked *one specific property*, not "the contract has no bugs." A contract can pass `check_reentrancy()` but still have a logic bug. So evidence strength for REFUTES is 0.9 — strong, but not absolute safety.

```python
# evidence.py:159-184
@staticmethod
def formal(source: str, vuln_class: str, polarity: "Polarity",
           invariant: str, proven: bool, counterexample: str = "",
           reliability: float = 0.95) -> Evidence:
    return Evidence(
        source=source,
        vuln_class=vuln_class,
        polarity=polarity,
        strength=1.0 if polarity == Polarity.SUPPORTS else 0.9,
        reliability=round(float(reliability), 4),
        kind=Kind.FORMAL,
        deterministic=True,
        detail={
            "invariant": invariant,
            "proven": proven,
            "counterexample": counterexample[:200] if counterexample else "",
        },
    )
```

The `deterministic=True` flag is critical — formal verification is a deterministic computation (same contract + same solver → same result). This means formal evidence can enter the ZK-provable half of the dual verdict, unlike LLM debate evidence (`deterministic=False`).

---

## Key Code

### 1. The node entry point: `formal_verification()` — gate checks and orchestration

```python
# formal_verification.py:92-157
async def formal_verification(state: AuditState) -> dict[str, Any]:
    if os.getenv("SENTINEL_DETERMINISTIC"):
        return {"symbolic_findings": [],
                "tool_status": {"halmos": {"ran": False, "reason": "deterministic_mode"}}}

    contract_code = state.get("contract_code", "") or ""
    if not contract_code:
        return {"symbolic_findings": []}

    halmos_path = shutil.which("halmos")
    forge_path = shutil.which("forge")
    if not halmos_path or not forge_path:
        return {"symbolic_findings": [],
                "tool_status": {"halmos": {"ran": False, "reason": "not_installed"}}}

    contract_name_match = re.search(r"contract\s+(\w+)", contract_code)
    if not contract_name_match:
        return {"symbolic_findings": [],
                "tool_status": {"halmos": {"ran": False, "reason": "no_contract_name"}}}

    with step_timer("formal_verification", address=address, budget_s=120):
        try:
            findings = await _run_halmos(contract_code, contract_name, address)
            evidence = emit_halmos_evidence(findings)
            return {"symbolic_findings": findings, "evidence_list": evidence,
                    "tool_status": {"halmos": {"ran": True, "findings": len(findings)}}}
        except Exception as exc:
            return {"symbolic_findings": [],
                    "tool_status": {"halmos": {"ran": False, "reason": str(exc)[:200]}}}
```

This function does three gate checks before touching Halmos: (1) skip in deterministic mode (conservative — Foundry project setup has side effects), (2) skip if no contract code, (3) skip if halmos or forge aren't installed. Each skip returns `tool_status["halmos"]["ran": False]` with a reason. The `except Exception` catch-all is the fail-soft — anything that goes wrong (subprocess crash, OOM, disk full) returns empty findings, not a crash.

### 2. The temp project builder: `_run_halmos()` — creating an isolated Foundry environment

```python
# formal_verification.py:160-250
async def _run_halmos(contract_code, contract_name, address):
    with tempfile.TemporaryDirectory(prefix="sentinel_halmos_") as tmpdir:
        tmp = Path(tmpdir)
        src_dir = tmp / "src"; src_dir.mkdir(parents=True, exist_ok=True)
        test_dir = tmp / "test"; test_dir.mkdir(parents=True, exist_ok=True)

        (src_dir / "Target.sol").write_text(contract_code)
        test_code = _generate_test_harness(contract_name)
        (test_dir / "FormalVerify.t.sol").write_text(test_code)

        (tmp / "foundry.toml").write_text(
            '[profile.default]\nsrc = "src"\nout = "out"\ntest = "test"\n'
            'libs = ["lib"]\nsolc_version = "0.8.19"\nevm_version = "paris"\n'
        )

        (tmp / "lib").mkdir(exist_ok=True)
        forge_std = os.getenv("FORGE_STD_PATH",
            str(Path.home() / ".foundry" / "lib" / "forge-std"))
        if Path(forge_std).exists():
            ((tmp / "lib") / "forge-std").symlink_to(forge_std, target_is_directory=True)

        loop = asyncio.get_running_loop()
        forge_result = await loop.run_in_executor(None, lambda: subprocess.run(
            ["forge", "build", "--root", str(tmp)], capture_output=True, text=True, timeout=60))
        # ... elided: check forge_result.returncode, then run halmos, parse JSON output
```

The temp directory pattern isolates the Foundry project from the rest of the system. Each audit gets its own directory (prefix `sentinel_halmos_`), and `tempfile.TemporaryDirectory` cleans up on exit — even on exception. The `asyncio.get_running_loop().run_in_executor()` pattern runs blocking `subprocess.run()` calls in a thread pool to avoid blocking the event loop. `forge build` gets 60s, Halmos gets whatever `HALMOS_TIMEOUT_S` is set to (default 120s).

### 3. The invariant-to-class mapping: `_INVARIANT_TO_CLASS`

```python
# formal_verification.py:44-50
_INVARIANT_TO_CLASS = {
    "reentrancy":          "Reentrancy",
    "arithmetic":          "IntegerUO",
    "access_control":      "AccessControl",
    "unchecked_return":    "UnusedReturn",
    "denial_of_service":   "DenialOfService",
}
```

This dict is the bridge between Halmos's output domain (test function names like `check_reentrancy()`) and the system's vulnerability class taxonomy (10 classes from the ML model). The invariant name is extracted by stripping `check_` and `()` from the Halmos result name. If the extracted name has no entry in the dict, the finding is silently skipped (logged at debug). This is a known fragility — see Mistakes below.

### 4. The evidence emitter: `emit_halmos_evidence()`

```python
# emit.py:205-243
def emit_halmos_evidence(symbolic_findings):
    evidence = []
    for finding in symbolic_findings:
        if finding.get("tool", "") != "halmos":
            continue
        cls = finding.get("vulnerability_class", "")
        if not cls:
            continue
        proven = finding.get("proven", False)
        polarity = Polarity.REFUTES if proven else Polarity.SUPPORTS
        evidence.append(Evidence.formal(
            source="halmos", vuln_class=cls, polarity=polarity,
            invariant=finding.get("invariant", ""),
            proven=proven,
            counterexample=finding.get("counterexample", ""),
            reliability=get_reliability("halmos", cls),
        ))
    return evidence
```

Filters for `tool == "halmos"`, maps `proven=True` → `REFUTES` (invariant holds → safety), `proven=False` → `SUPPORTS` (invariant violated → vulnerability). Uses `get_reliability("halmos", cls)` so the reliability weight comes from the same L3→L1→hardcoded fallback chain as every other source.

---

## Design Decision: Bounded Model Checking (Halmos) vs Unbounded Proof (Certora) vs Fuzzing (Echidna)

| Criterion | Halmos (chose) | Certora Prover | Echidna / Foundry fuzz |
|-----------|---------------|----------------|----------------------|
| Proof power | Bounded — explores paths up to loop bound | Unbounded — true formal proof | None — tests random inputs |
| License | MIT | Commercial (paid) | MIT / Apache |
| Python integration | Subprocess CLI | Java + CLI, Python API limited | Subprocess CLI |
| Setup complexity | Temp Foundry project | Dedicated spec language (CVL) | Temp Foundry project |
| Coverage | 5 invariants (template) | Custom invariants per contract | Anything you can write as test |

**Decision:** Halmos — the bounded model checker is tractable for Solidity contracts (< 120s per contract), has an MIT license (no procurement), and runs from the CLI (simple subprocess integration). The 5-invariant template covers the most common vulnerability classes without requiring per-contract spec writing.

**When Halmos is WRONG:** If you need an industrial-grade formal verification for a high-value contract (e.g., a DeFi protocol holding $100M+), Certora's unbounded proof is the right choice — it can prove properties that Halmos can't (like "the protocol's net asset value never drops below zero"). But Certora requires writing CVL specs, paying for licenses, and waiting minutes per proof — overkill for the 61-contract eval corpus.

---

## Technology Choice: Halmos

**Category:** Bounded model checker / symbolic execution tool for EVM bytecode.

**Alternatives:**

| Tool | Strength | Weakness |
|------|----------|----------|
| **Halmos** | Python-native (pip install), MIT license, CLI-friendly | Bounded (loop unrolling), 5 template invariants only |
| **Certora Prover** | Unbounded proof, customizable invariants | Commercial license, requires CVL spec language, Java-based |
| **Echidna** | Grammar-based fuzzing, finds real bugs | No proof, only statistical coverage |
| **Foundry invariant tests** | No extra tool needed | No symbolic execution — only fuzz with `vm.assume()` |

**Why Halmos:** It's the only tool in the table that combines (1) symbolic execution (finds root causes, not just correlated patterns), (2) an MIT license (no procurement), (3) Python-native setup (`pip install halmos`), and (4) a simple JSON output format. It runs within the 120s deep-path budget and produces the only `deterministic=True` evidence outside the static-analysis tools.

**When you'd choose differently:**
- High-value contract with custom invariants → Certora (unbounded proof)
- Fuzzing-heavy workflow → Echidna (finds more real bugs, but no proof)
- No extra dependencies allowed → Foundry native invariant tests (weaker but zero-install)

**Migration trigger:** If the corpus grows to contracts where Halmos's bounded model checker regularly times out (path explosion on contracts with >1000 branches), you'd either tune the per-invariant timeout or switch to a symbolic executor that handles larger branching factors.

---

## Anti-Patterns

### ❌ Run Halmos on every contract unconditionally

**What it looks like:** `formal_verification` runs in the fast path too — every contract gets symbolic execution.

**Why someone would build this:** "Formal proof is the strongest evidence — why wouldn't we always run it?"

**Why it's wrong:** Halmos takes 30–120s per contract. On safe contracts (which are 70% of the audit corpus), this adds 2 minutes of latency for zero benefit — the contract is safe, and the formal proof just confirms what ML already said. Path explosion can make Halmos time out on large contracts, stalling the pipeline. The deep-path gating ensures Halmos only runs when ML or quick_screen flags a vulnerability.

**The right approach:** Halmos runs in the deep path only. Safe contracts (fast path) never pay the latency cost.

### ❌ Halmos as the sole verdict source

**What it looks like:** If Halmos passes all invariants, emit `SAFE`. If it fails one, emit `CONFIRMED`. Ignore ML, Slither, and Aderyn.

**Why someone would build this:** "Formal proof is the strongest — why would I trust a statistical guess over a proof?"

**Why it's wrong:** Halmos only covers 5 invariants. A contract can pass all 5 and still have logic bugs (incorrect interest calculation), economic exploits (flash loan attack), or governance attacks (vote manipulation). Halmos can't prove "this contract has no vulnerabilities" — only "this specific invariant holds." The fuse() function combines all evidence channels for a reason: defense in depth.

**The right approach:** Halmos is one evidence channel among 6. Its evidence is weighted by reliability (L3 fitted) and combined with ML, Slither, Aderyn, RAG, and debate evidence in fuse().

---

## Mistakes & Fixes

### Mistake: Invariant-to-class mapping silently drops unknown invariants

**What happened:** Halmos check functions that didn't match `_INVARIANT_TO_CLASS` (e.g., a custom invariant added by a future version) had their findings silently dropped — the function is `_parse_halmos_output():288-289`, which just `continue`s when `vuln_class` is empty.

**Why it happened:** The mapping is designed as an allowlist — only known invariants become findings. Unknown invariants are skipped because there's no class to map them to. This was intentional for safety (don't emit evidence for an unmapped class), but the skip is silent.

**How we found it:** Code review of `_parse_halmos_output()` revealed the `continue` with no log statement. The test `test_parse_unknown_invariant_skipped` explicitly asserts this behavior.

**The fix:** Added a `logger.warning()` for unknown invariants. The mapping function is:
```python
# formal_verification.py:286-289
invariant = name.replace("check_", "").replace("()", "")
vuln_class = _INVARIANT_TO_CLASS.get(invariant, "")
if not vuln_class:
    logger.warning("formal_verification | unknown invariant skipped: {}", name)
    continue
```
Now the system at least surfaces the fact that an invariant was dropped.

**The lesson:** Explicit mapping tables beat name parsing — but make sure skips are surfaced, not silent. `continue` without a log is a silent-failure risk (Rule 5C).

### Mistake: Halmos needs a full Foundry project, not just a `.sol` file

**What happened:** Initial version tried `halmos --file contract.sol`. It failed — Halmos requires a Foundry project structure with `src/`, `test/`, `foundry.toml`, and `forge-std`.

**Why it happened:** The documentation for Halmos's CLI focuses on `halmos --root <project>`, not standalone files. Most examples show running Halmos on existing Foundry projects, not on ad-hoc contracts.

**The fix:** `_run_halmos()` builds the full project structure in a temp directory — `src/Target.sol` for the contract, `test/FormalVerify.t.sol` for the test harness, `foundry.toml` for the config, and a symlink to the host's `forge-std` library.

**The lesson:** When integrating an external tool, confirm the exact CLI invocation path before writing the wrapper. Run `--help` first. The temp-project pattern is reusable for any tool that needs a specific project layout.

### Mistake: `forge build` failures are silent

**What happened:** A contract with Solidity version incompatibility (e.g., `pragma solidity ^0.5.0` but `foundry.toml` says `solc_version = "0.8.19"`) caused `forge build` to fail. The failure was caught by `forge_result.returncode != 0`, which returned `[]` — no evidence, but no signal to downstream about why.

**Why it happened:** The fail-soft design intentionally swallows errors and returns empty findings. But "empty findings" looks the same as "Halmos ran and found nothing" to downstream fuse() — a silent skip (Rule 5C violation).

**The fix:** `tool_status["halmos"]` now carries `ran=False` with a `reason` field. When `forge build` fails, the status is `{"ran": False, "reason": "forge_build_failed: <stderr[:200]>"}`. The eval framework's `_tool_ran()` checks this field and excludes the contract from Halmos's TP+FP+FN+TN counts — just like the Aderyn silent-skip fix from Rule 5C.

**The lesson:** Fail-soft must distinguish "ran and found nothing" from "didn't run at all." A `tool_status` entry with `ran: False` is the mechanism.

---

## What Would Break If You Removed This?

Remove the `formal_verification` node and its `formal_verification` → `audit_check` edge from the graph (`graph.py:178,207`). The pipeline still compiles, still produces reports — but the system loses its only evidence channel that can mathematically prove a vulnerability.

Without formal evidence:
- Every finding is statistical (ML), syntactic (Slither, Aderyn), or semantic (RAG, debate). None is a proof.
- The `verdict_provable` output loses a `deterministic=True` source. The ZK-provable half of the dual verdict gets less signal, making the overall ZK guarantee weaker.
- A reentrancy vulnerability that ML assigns 0.60 probability and Slither flags with a `reentrancy-eth` detector but the reviewer wants "prove it" — without Halmos, you can't. You can only say "Slither found the pattern." With Halmos, you can say "Halmos found a counterexample: deposit → withdraw → deposit before the state update."
- The "defense in depth has standalone value" principle (Principle 7) is weakened — you go from 6 evidence channels to 5, and the strongest one is gone.

---

## At Scale

*Scale metric: audit corpus size (baseline: 61 contracts)*

| Scale | What works | What breaks | Migration path |
|-------|-----------|-------------|----------------|
| Current (61) | All invariants checkable, <30s for most | Some contracts cause path explosion (120s timeout) | — |
| 10x (610) | Pipeline handles parallel Halmos jobs | Path explosion on complex contracts becomes frequent | Tune Halmos timeout per contract (smaller budget for large contracts) |
| 100x (6,100) | Temp-project isolation works | CPU contention (Halmos + forge are not GPU, but consume cores) | Limit concurrent Halmos workers; prefer contracts with shallow branching |
| 1000x (61,000) | Temp-project pattern works at scale | Temp dir disk I/O becomes significant; forge build for each contract is wasteful | Cache compiled bytecode; only run Halmos on pre-vetted "high risk" contracts |

---

## Try It Yourself

> TRY IT: cd agents && source .venv/bin/activate && python -m pytest tests/test_formal_verification.py -v

> TRY IT: which halmos && halmos --version   # check if Halmos is installed

> TRY IT: python3 -c "from src.orchestration.verdict.evidence import Evidence, Polarity, Kind; ev = Evidence.formal('halmos', 'Reentrancy', Polarity.SUPPORTS, 'reentrancy', False, '0xdead'); print(f'{ev.kind} | strength={ev.strength} | det={ev.deterministic}')"

---

## Limitations & What's Missing

- **Only 5 invariants.** Halmos can't check logic bugs (incorrect interest calculation), economic exploits (flash loan attacks), or governance attacks (vote manipulation). The invariant set covers the most common Solidity vulnerability classes but is not comprehensive.
- **120s timeout.** Large contracts with deep branching (>1000 paths) can cause Halmos to time out. The bounded model checker unrolls loops to a fixed depth, but even bounded symbolic execution can explode on contracts with many branches.
- **Generic test harness.** The generated test harness (`FormalVerify.t.sol`) is a template — it checks the same 5 invariants on every contract. A contract-specific harness would find more bugs (e.g., checking that `withdraw(amount)` always reverts when `amount > balance`).
- **SENTINEL_DETERMINISTIC skips it.** Conservative choice — Foundry project setup creates files, symlinks, and build artifacts, which are side effects that could potentially interfere with deterministic execution. This means Halmos never runs in deterministic mode, leaving a gap in the ZK-provable evidence set.
- **No incremental verification.** Every audit runs all 5 invariants from scratch. There's no caching mechanism to say "this contract's bytecode hasn't changed since the last run, skip verification."
- **Reliability not yet measured.** Halmos is added in P8a, so there are no fitted L3 reliability values yet. The 0.95 default in `Evidence.formal()` is an L1 hand-set estimate — it will be replaced with a measured value once the eval corpus includes enough Halmos results.

---

## Transferable Patterns

1. **Temp project pattern for tool integration** — `_run_halmos():170-250`
   - *Interview story:* "Halmos needs a full Foundry project, not a standalone Solidity file. We create a temp directory with the complete project structure — `src/`, `test/`, `foundry.toml`, `lib/forge-std` symlink — run the tool, and clean up. The tempdir is atomically cleaned on exit by `tempfile.TemporaryDirectory`, even if the subprocess crashes."
   - *When this pattern is WRONG:* If the tool has a long-lived state (e.g., a database embedded in the project directory), recreating from scratch every run is wasteful — incremental caching would be cheaper.

2. **Formal evidence as the strongest tier** — `Evidence.formal():159-184`
   - *Interview story:* "In SENTINEL, Evidence has 5 kinds — statistical, syntactic, semantic, formal, economic. Formal evidence (strength 1.0 for violations) is the only one that mathematically proves a finding. When Halmos finds a reentrancy counterexample, it's not a probability or a pattern match — it's a concrete execution path that proves the bug exists. The fuse() function weights it accordingly."
   - *When this pattern is WRONG:* If the formal tool has known blind spots (e.g., Halmos only checks 5 invariants), treating its "pass" as "safe" is dangerous. Formal evidence for safety (REFUTES) gets strength 0.9, not 1.0, because tool coverage is limited.

3. **Fail-soft with status propagation** — `formal_verification.py:110-157`
   - *Interview story:* "Halmos can fail for many reasons — not installed, forge build error, timeout, path explosion. On any failure, the node returns empty findings with `tool_status['halmos']['ran': False]`. The eval framework uses this field to exclude the contract from Halmos's confusion matrix counts (Rule 5C). Downstream fuse() sees zero formal evidence but doesn't treat it as 'Halmos found nothing' — it just proceeds with the other channels."
   - *When this pattern is WRONG:* If the tool is the only evidence source for a critical class (e.g., if Halmos is the only tool that checks access control), fail-soft means the class goes unanalyzed. Add a warning or a secondary analysis path in that case.

---

**Source files verified:**
- `src/orchestration/nodes/formal_verification.py:1-307` — full node: gating, temp project, subprocess execution, JSON parsing
- `src/orchestration/verdict/evidence.py:159-184` — `Evidence.formal()` constructor
- `src/orchestration/verdict/emit.py:205-243` — `emit_halmos_evidence()` filter and mapping
- `src/orchestration/graph.py:135-207` — `formal_verification` node registration and deep-path wiring
- `agents/tests/test_formal_verification.py:1-200` — 15 tests covering parser, emitter, constructor, node gating

**Verified against commit hash:** `c47898ea5`
