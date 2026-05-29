# SENTINEL Agent Module — Unified Proposal V3

**Status:** Source of truth for all agent-layer work going forward  
**Supersedes:** `AGENTS_PLAN_V2.md`, `AGENTS_MODULE_PROPOSAL.md`  
**Date:** 2026-05-29 (updated post-discussion)  
**Current implementation baseline:** Phase 0 + Steps A–E complete (see STATUS.md)

---

## 1. Where We Are

### 1.1 Implemented (all passing tests)

```
START → ml_assessment → evidence_router
                         │
            ┌────────────┼───────────────┐
       rag_research  static_analysis  graph_explain   (deep path only)
            └────────────┼───────────────┘
                    audit_check
                         │
                  cross_validator   (deep path only)
                         │
                    synthesizer → END

Shallow path: evidence_router → synthesizer → END
```

| Component | Port | What it does |
|-----------|------|--------------|
| inference_server (MCP) | 8010 | Calls ML predictor, returns three-tier result |
| rag_server (MCP) | 8011 | Chroma vector search over 44K-contract corpus |
| audit_server (MCP) | 8012 | Slither static analysis, gas estimates |
| graph_inspector_server (MCP) | 8013 | Function-level hotspots via Slither proxy |

**Routing:** tier thresholds TIER_CONFIRMED=0.55, TIER_SUSPICIOUS=0.25.  
**Verdict vocabulary:** CONFIRMED / LIKELY / DISPUTED / WATCH / SAFE (LLM-adjudicated by cross_validator).

### 1.2 What the old specs called for that is NOT built

From `AGENTS_MODULE_PROPOSAL.md` (Phases 3–4, never started):
- `/hotspots` ML inference endpoint — true GNN attention weights, not Slither proxy
- `econ_sim` node (port 8015) — price manipulation cost estimator
- Mythril scoped per-function execution
- Feedback loop — logging of disagreements between ML and cross_validator for retraining
- `batch_predict` — multi-contract batch API

From `AGENTS_PLAN_V2.md` (detail not yet implemented):
- `HIGH_VALUE_RAG_CLASSES` distinction in routing (rag only for specific classes, not all deep-path)

---

## 2. Critical Design Gap — The Shallow Path Blind Spot

The current shallow path (all ML probs < 0.25) routes directly to the synthesizer with **zero tool evidence**. The synthesizer then writes a "safe" report based solely on ML output. This is dangerous because:

- ML F1 on hard classes (TOD=0.235, ExternalBug=0.246, CallToUnknown=0.247) means real bugs regularly score below 0.25
- A tricky or obfuscated contract can suppress all ML probabilities while remaining genuinely vulnerable
- "Safe" currently means "ML said low confidence" not "two independent systems agreed"

**Fix:** Insert a `quick_screen` node that runs on **every** contract before routing. It calls Slither + Aderyn (fast, deterministic, ~10s total). The routing decision is then based on two independent signals:

```
                        ┌──────────────────┐
                        │  ml_assessment   │
                        └────────┬─────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │      quick_screen      │  ← ALWAYS runs
                    │  Slither + Aderyn      │  ← ~10 seconds, no LLM
                    └────────────┬───────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ evidence_router  │  routes on ML + quick_screen combined
                        └───────┬──────────┘
                                │
          ┌─────────────────────┼──────────────────────┐
          │ ML < 0.25           │ ML < 0.25             │ ML ≥ 0.25
          │ Slither/Aderyn clean│ Slither/Aderyn hit    │ (any)
          ▼                     ▼                       ▼
    fast synthesizer     investigator loop        investigator loop
    (two tools agree     (ML missed it,           (normal deep path)
     it's safe)           tools caught it)
```

"Safe" now requires **both** ML and quick_screen to agree — not just ML alone.

---

## 3. Tool Coverage and Defense in Depth

Different tools operate at different levels and catch different bug classes. No single tool dominates.

### 3.1 Tool landscape

| Tool | How it works | Speed | Key strength |
|------|-------------|-------|-------------|
| **Slither** | AST-level pattern matching | ~5s | High recall on common patterns |
| **Aderyn** | AST-level, different detector set (Cyfrin) | ~5s | Different false-negative profile vs Slither |
| **Mythril** | Symbolic execution on EVM bytecode | 30s–10min | Bytecode-level arithmetic, proxy tricks, assembly obfuscation |
| **Echidna** | Property fuzzing | minutes+ | Stateful multi-tx invariant violations |
| **Halmos** | Symbolic testing (Foundry) | minutes+ | Formal property violations |

Slither + Aderyn are practical for `quick_screen` (always-on). Mythril scoped to hot functions is practical for Tier 2. Echidna and Halmos require hand-written invariants and are too slow for synchronous use.

### 3.2 Bug class coverage matrix

```
                    ┌─────────────────┬────────┬────────┬──────────┐
                    │                 │Slither │Aderyn  │Mythril   │
                    ├─────────────────┼────────┼────────┼──────────┤
                    │ Reentrancy      │  ✅    │  ✅    │  ✅      │
                    │ Integer over/UF │  ⚠️    │  ⚠️    │  ✅ (EVM)│
                    │ Timestamp       │  ✅    │  ✅    │  ⚠️      │
                    │ TOD / frontrun  │  ⚠️    │  ⚠️    │  ⚠️      │
                    │ Access control  │  ⚠️    │  ✅    │  ✅      │
                    │ Proxy/delegatecall│ ⚠️   │  ⚠️    │  ✅      │
                    │ Assembly tricks │  ❌    │  ❌    │  ✅      │
                    │ Gas issues      │  ✅    │  ✅    │  ❌      │
                    │ Multi-tx state  │  ❌    │  ❌    │  ⚠️      │
                    │ Business logic  │  ❌    │  ❌    │  ❌      │
                    │ Flash loan      │  ❌    │  ❌    │  ❌      │
                    │ Cross-contract  │  ❌    │  ❌    │  ❌      │
                    └─────────────────┴────────┴────────┴──────────┘
```

### 3.3 Tiered tool invocation

Tools run in tiers to avoid paying for expensive analysis on clean contracts:

- **Tier 0 (always):** Slither + Aderyn via `quick_screen` — ~10s, no LLM
- **Tier 1 (deep path):** full Slither detectors + RAG + GNN attention hotspots via investigator loop
- **Tier 2 (contested only):** Mythril scoped to hot functions, bounded at 90s timeout. Triggered when ML and Tier 0 disagree, or class is in {Reentrancy, ExternalBug, TOD} with SUSPICIOUS tier.

Mythril runs **scoped** — only on the 2-3 functions the GNN attention identified as hot, not the full contract. This reduces its runtime from potentially 10 minutes to 30-90 seconds.

### 3.4 Audit server tool additions (to existing :8012)

```python
# existing
run_slither(contract_path, detectors=None)

# add
run_aderyn(contract_path)             # fast, different detector set
run_mythril(contract_path,            # scoped, bounded
            functions=["withdraw"],   # ← hot functions from GNN attention
            timeout=90)
```

### 3.5 Honest remaining blind spots

Even with all three tools layered plus LLM reasoning:

| Scenario | Covered? | Why |
|----------|----------|-----|
| Standard reentrancy | ✅ all three catch it | Well-studied pattern |
| Integer overflow via assembly | ✅ Mythril catches it | EVM bytecode level |
| Proxy pattern hiding logic | ✅ Mythril catches it | Bytecode ignores source tricks |
| Multi-tx state manipulation | ⚠️ Mythril partial | Requires state enumeration |
| Business logic bugs | ❌ | No code signature to match; requires understanding intent |
| Flash loan attacks | ❌ | Requires forked mainnet state with real DeFi protocol state |
| Cross-contract vulnerabilities | ❌ | All tools analyze single contracts |
| Adversarial obfuscation (expert) | ⚠️ | Mythril helps; determined attacker can still evade |

The bottom three rows are the genuine ceiling. See Section 9 for research on addressing these (tools, papers, practical tradeoffs per limitation).

---

## 4. What the Research Tells Us

Six findings from the 2024–2025 literature that should directly shape what we build next:

### 4.1 Slither alone is not a reliable confirmer

Slither achieves **F1=2.38%** (precision 1.23%, recall 36%) on real-world audit benchmarks (not toy corpora). The current design treats Slither output as "hard evidence" inside cross_validator. In practice it is a signal with high recall / low precision — useful for surfacing candidates, harmful if used to rule things out.

**Consequence:** the cross_validator should weight Slither as one noisy signal, not a binary gate. False negatives from Slither should not suppress ML findings.

### 4.2 The right LLM-vs-static ordering is the inverse of what we built

GPTScan (ICSE 2024) gets best results by: LLM identifies semantic candidates → static analysis verifies the machine-readable properties. Our current order is reversed: static runs first, LLM synthesizes last.

**Consequence:** a dedicated "candidate reasoning" step before static analysis would improve precision. The LLM should flag which functions look suspicious on semantic grounds; Slither then tries to confirm the specific property.

### 4.3 Adversarial multi-agent reasoning outperforms single-pass LLM verdict

VulTrial (2025) uses a prosecutor/defender/judge structure. The prosecutor argues the contract IS vulnerable, the defender argues it is NOT, the judge weighs both. This eliminates confirmation bias present when a single LLM both gathers evidence and renders a verdict. In their evaluation, contested verdicts resolved by the judge are more accurate than single-pass.

**Consequence:** replace the current single-LLM `cross_validator` with a three-role adversarial panel for deep-path analysis.

### 4.4 CPG-based code slicing dramatically improves LLM reasoning

Slicing the Code Property Graph down to the subgraph relevant to a specific vulnerability class reduces token count by 67–91% while improving detection F1 by 15–40%. Full-contract context causes LLM attention to diffuse; focused slices let the LLM reason precisely.

**Consequence:** `graph_inspector_server` Phase 2 should deliver sliced subgraphs, not just hotspot scores. The LLM nodes should receive sliced code + graph, not the full source.

### 4.5 Exploit generation is practically viable for the vulnerabilities we detect

PoCo/V2E-style exploit generation achieves ~92% success on single-contract reentrancy and integer overflow bugs. An exploit generation node would:
- Provide proof-of-concept for CONFIRMED findings
- Serve as a falsifiability check (if no exploit can be generated, downgrade confidence)
- Produce actionable output for the audit report

### 4.6 Graph RAG outperforms flat vector RAG for security cross-pattern queries

Standard Chroma vector search loses cross-pattern relationships (e.g., "contracts that share both Reentrancy and AccessControl misuse"). A graph-structured knowledge base over the audit corpus allows relationship-aware retrieval.

---

## 5. Architectural Vision

### 5.1 The core shift: linear pipeline → investigator loop

Current design is a **linear pipeline**: ML → route → gather tools → synthesize.  
The limitation: once routed to "deep", every node runs regardless of intermediate findings. There is no feedback between the reasoning steps.

The proposed design adds an **investigator loop**: after the initial ML signal, an LLM ReAct agent decides which tools to call next based on accumulated evidence. It can call tools multiple times, request a specific code slice, or escalate to the adversarial panel only when evidence is ambiguous.

```
                                               ┌──────────────────────────────┐
START → ml_assessment → quick_screen           │   Investigator Loop          │
              (Slither+Aderyn, always)          │                              │
                         │                     │  ┌─ code_slicer (CPG)        │
                         ▼                     │  ├─ rag_research              │
                  evidence_router              │  ├─ static_analysis (Slither) │
                  (ML + screen signals)        │  ├─ mythril_probe (Tier 2)    │
                         │                     │  ├─ graph_explain (attn)      │
               ┌─────────┴──────────┐          │  └─ exploit_probe (optional)  │
          clean (both)           hit (either)  │           │                   │
               │                    └──────────│   adversarial_panel           │
               ▼                               │  (prosecutor/defender/judge)  │
         fast synthesizer        synthesizer ←─┘                              │
                                               └──────────────────────────────┘
```

The investigator loop replaces the current static fan-out. `evidence_router` hands off to the investigator when either ML or quick_screen fires; the investigator runs until it either reaches confidence or exhausts its budget.

### 5.2 Adversarial panel replaces cross_validator

Three LLM roles, each with a different system prompt and evidence access:

| Role | Instruction bias | Output |
|------|-----------------|--------|
| Prosecutor | Argue the contract IS vulnerable. Find the strongest path to exploitation. | Argument + cited evidence |
| Defender | Argue the contract is NOT vulnerable. Find mitigations, guards, false-positive reasons. | Counter-argument |
| Judge | Weigh prosecutor vs defender. No prior bias. | Final verdict (CONFIRMED/LIKELY/DISPUTED/WATCH/SAFE) + confidence |

This structure is more expensive (3 LLM calls vs 1) but eliminates the single-LLM confirmation bias. It should only run on CONFIRMED-tier findings or on SUSPICIOUS findings where ML and Slither disagree.

### 5.3 True GNN attention replaces Slither proxy in graph_inspector

Currently graph_inspector_server runs Slither to score hotspots. The real hotspot signal is in the GNN's `prefix_attention_mean` tensor from the checkpoint.

Requires:
1. `/hotspots` endpoint on the ML inference API — expose per-node attention aggregated by function
2. graph_inspector_server Phase 2 — call `/hotspots`, return CPG-sliced subgraph per vulnerability class

This is what makes the "code_slicer" tool in the investigator loop concrete.

---

## 6. Implementation Plan

Ordered by impact and dependency. Items are independent within a phase.

### Phase 1 — Immediate (before Run 5)

These unblock data quality work and fix known gaps in the current agent layer.

| ID | Item | Effort | Blocking |
|----|------|--------|---------|
| A1 | `/hotspots` ML inference endpoint | M | graph_inspector Phase 2 |
| A2 | graph_inspector_server Phase 2 (real GNN attention) | M | A1 |
| A3 | `quick_screen` node: Slither + Aderyn always-on, update routing | M | none |
| A4 | Aderyn tool in audit_server (:8012) | S | A3 |
| A5 | End-to-end smoke test: all 4 MCP + real contract | S | none |
| A6 | `HIGH_VALUE_RAG_CLASSES` routing distinction | S | none |

**A1 — `/hotspots` endpoint spec:**
```
GET /hotspots?contract_path=<path>
Response:
{
  "function_hotspots": [
    {
      "fn_name": "withdraw",
      "attention_score": 0.82,
      "node_ids": [14, 15, 22],
      "dominant_edge_types": [3, 6],
      "vulnerability_classes": ["Reentrancy", "ExternalBug"]
    }, ...
  ],
  "graph_stats": {"num_nodes": 183, "num_edges": 441},
  "attention_source": "prefix_attention_mean"
}
```

**A3 — quick_screen spec:**  
New LangGraph node inserted between `ml_assessment` and `evidence_router`. Calls `audit_server:run_slither` + new `audit_server:run_aderyn`. Populates `state["quick_screen_hits"]: dict[str, list[str]]` (tool → detector names). evidence_router routing logic updated: `should_go_deep = ml_tier != "safe" OR len(quick_screen_hits) > 0`.

**A5 — smoke test criteria:**
- All 4 servers reachable at their ports
- Run the full LangGraph topology with `MockVault.sol` (existing fixture)
- Deep path triggered (Reentrancy flagged ≥ 0.25)
- `graph_explanations` and `verdicts` both present in final state
- `quick_screen_hits` present in final state
- No tool call raises an exception

### Phase 2 — After Run 5

Run 5 must complete with F1-macro > 0.38 before these are worth implementing (better attention weights = better CPG slices).

| ID | Item | Effort | Notes |
|----|------|--------|-------|
| B1 | CPG code slicer node | L | Extracts function subgraphs per vuln class; feeds LLM nodes sliced code, not full source |
| B2 | Investigator ReAct loop | L | Replaces static fan-out; uses LangGraph conditional edges with budget counter |
| B3 | Adversarial panel (prosecutor/defender/judge) | M | Replaces cross_validator for deep path; cross_validator kept as fast-path fallback |
| B4 | Graph RAG upgrade (Chroma → Neo4j or LlamaIndex graph store) | L | Requires re-ingestion of audit corpus |
| B5 | Mythril scoped tool in audit_server | M | `run_mythril(contract_path, functions, timeout=90)`; Tier 2 in investigator loop |
| B6 | GPTScan logic pattern tool in audit_server | M | `run_gptscan(contract_path)`; triggered for ExternalBug/CallToUnknown |
| B7 | FinDet bytecode entropy screen in quick_screen | M | Obfuscation signal; high entropy → escalate regardless of ML score |

**B2 — Investigator loop state additions:**
```python
class AgentState(TypedDict):
    # existing fields ...
    investigator_steps: list[dict]   # trace of tool calls + results
    investigator_budget: int         # decrements per tool call; stop at 0
    code_slices: dict[str, str]      # class → sliced source for LLM
    exploit_candidates: list[dict]   # from exploit_probe if run
```

**B3 — Adversarial panel trigger conditions:**
- Any class has CONFIRMED tier AND ML prob > 0.70 AND Slither found no matching detector  
  (potential false positive — worth defending)
- Any class has SUSPICIOUS tier AND class is in {Reentrancy, ExternalBug, TOD}  
  (high-value, ambiguous — worth adjudicating)
- Always skip panel for: GasException, DoS, IntegerUO when probability > 0.80  
  (high-confidence, mechanically verifiable — panel adds cost not accuracy)

### Phase 3 — After Adversarial Panel Stable

| ID | Item | Effort | Notes |
|----|------|--------|-------|
| C1 | Exploit generation node | L | PoCo-style Foundry test generation for CONFIRMED findings |
| C2 | DeFi sim server (port 8015) | L | FlashSyn/ItyFuzz flash loan synthesis; triggered for ExternalBug/TOD + DeFi protocol detected |
| C3 | SMARTINV invariant checker | M | Infer + check invariants for any CONFIRMED finding |
| C4 | Disagreement logger + feedback loop | M | Log ML vs panel disagreements to SQLite; feed back to data quality phase |
| C5 | M6 Integration API (`POST /v1/audit`) | L | Wires full stack; bearer token auth, 10/min rate limit |
| C6 | Multi-contract analysis (Clairvoyance) | XL | Cross-contract reentrancy; requires multi-file input support |

**C1 — Exploit node trigger:** only runs when adversarial panel returns CONFIRMED AND class in {Reentrancy, IntegerUO, ExternalBug}.  
Output: a Foundry test file (`.t.sol`), or failure reason if generation fails. Failure downgrades CONFIRMED → LIKELY.

---

## 7. What NOT to Build

| Item | Reason |
|------|--------|
| Triage agent as separate process | Current corpus size doesn't justify orchestration complexity |
| Foundry fuzz tests in agent synchronous loop | Too slow; consider as async background job if needed |
| Full symbolic execution (Manticore/Halmos) in loop | Cost per contract is minutes; appropriate as offline tool only |
| Cross-contract dependency graph (Phase 2+) | Requires multi-contract parsing not yet implemented |
| Per-class threshold tuning in routing | Fixed thresholds until Run 5 gate passes; post-Run 5 item |
| DEF_USE(10) edge type | v8-AB test showed dilution of Reentrancy CEI signal; do not reintroduce |
| Full Graph RAG before Run 5 | Current corpus labels are noisy; re-ingesting noise into graph store wastes effort |

---

## 8. Node Catalogue (Target State Post-Phase 3)

| Node | Phase | Type | Role |
|------|-------|------|------|
| `ml_assessment` | live | tool caller | Calls inference_server MCP, populates ML result |
| `quick_screen` | Phase 1 | tool caller | Slither + Aderyn + bytecode entropy on every contract |
| `evidence_router` | live → updated | conditional | Routes on ML + quick_screen combined signals |
| `investigator` | Phase 2 | ReAct agent | Plans and executes tool sequence based on accumulated evidence |
| `code_slicer` | Phase 2 | tool caller | Calls graph_inspector for CPG slices per class |
| `rag_research` | live | tool caller | Calls rag_server for relevant audit patterns |
| `static_analysis` | live | tool caller | Calls audit_server (Slither + gas) |
| `graph_explain` | live | tool caller | Calls graph_inspector for hotspots / attention |
| `mythril_probe` | Phase 2 | tool caller | Scoped Mythril on hot functions, 90s bounded |
| `logic_scanner` | Phase 2 | tool caller | GPTScan for business logic pattern matching |
| `exploit_probe` | Phase 3 | tool caller | PoCo Foundry test generation for CONFIRMED findings |
| `defi_sim` | Phase 3 | tool caller | FlashSyn/ItyFuzz flash loan attack simulation on fork |
| `cross_validator` | live | single LLM | Fast-path verdict (kept as fallback) |
| `adversarial_panel` | Phase 2 | multi-LLM | Prosecutor / defender / judge on contested findings |
| `synthesizer` | live | LLM | Assembles final report from all evidence |

MCP servers (target):
| Server | Port | Phase | Status |
|--------|------|-------|--------|
| inference_server | 8010 | live | ✅ live |
| rag_server | 8011 | live | ✅ live |
| audit_server | 8012 | live → extended | ✅ live; add Aderyn + Mythril + GPTScan tools |
| graph_inspector_server | 8013 | live → Phase 2 | ✅ Phase 1; Phase 2 pending A1 (/hotspots endpoint) |
| defi_sim_server | 8015 | Phase 3 | ⏳ not built |

---

## 9. Hard Limitations — Research Findings and Mitigations

These are the three bug classes where the entire Slither+Aderyn+Mythril+ML stack fails. Research from 2023–2025 shows each is partially solvable, but with tradeoffs.

### 7.1 Business Logic Bugs

**The gap:** No syntactic signature. A contract that distributes rewards incorrectly, or gates access wrongly, looks like valid code — only the intended behavior can flag it as a bug.

**Best available tools:**

| Tool | Approach | Precision | Automation level |
|------|----------|-----------|-----------------|
| **GPTScan** (ICSE 2024) | LLM matches code to *typed* vulnerability scenario descriptions; static confirms. ~$0.01/1K LOC, ~14s/scan | >90% on token contracts, 57% on large projects | Fully automated for known scenario types |
| **SMARTINV** (IEEE S&P 2024) | Infers invariants from source + NatSpec via multimodal LLM, then checks violations. 3.5× more bug-critical invariants than prior tools | Found 119 zero-days across 89K contracts | Fully automated |
| **LLM-SmartAudit** (2024–2025) | Multi-agent conversation + buffer-of-thought; 98% accuracy on common classes | High on known classes | Fully automated |
| **Certora Prover** (open-sourced Feb 2025) | Formal verification with CVL specs. AI Composer (Dec 2025) auto-generates CVL from source | Highest confidence | Spec required per contract (AI Composer reduces but doesn't eliminate this) |

**What's automatable:** GPTScan and SMARTINV cover a wide range of *known logic vulnerability patterns* fully automatically. Novel protocol-specific logic bugs (e.g., "rewards should equal 1% of deposited principal per epoch") are not automatable without a spec or NatSpec documentation.

**How to integrate:** GPTScan is open-source and can be called from an MCP tool. SMARTINV invariant checking can wrap the audit_server. These run in the investigator loop as Tier 1 tools when the ML signals ExternalBug or CallToUnknown.

### 7.2 Flash Loan Attacks (External State Required)

**The gap:** Exploits only materialize when $100M+ is borrowed from Aave/Uniswap within one transaction, manipulating pool price ratios. Requires forked mainnet state to simulate.

**Best available tools:**

| Tool | Approach | Coverage | Automation level |
|------|----------|----------|-----------------|
| **FlashSyn** (ICSE 2024) | Runs on Foundry fork. Approximates DeFi protocol behavior with polynomial regression, then optimizes attack parameters. Synthesized attacks in 16/18 historical victims | Flash loan oracle manipulation | Fully automated given Foundry fork |
| **ItyFuzz** (ISSTA 2023) | Snapshot-based hybrid fuzzer at bytecode level; seeds from historical on-chain transactions. 44% more bugs than Echidna, 2.5× faster | Broad exploit classes | Fully automated |
| **AiRacleX** (2025) | Three-stage LLM pipeline: mines domain knowledge → chain-of-thought prompts → code evaluation. 2.58× recall improvement over GPTScan on price oracle attacks | DeFi oracle manipulation | Fully automated |
| **DeFiTainter** (ISSTA 2023) | Inter-contract taint analysis; 96% precision, 91.3% recall on its evaluation set | Price data flow | Needs taint source/sink definitions |

**What's automatable:** FlashSyn and ItyFuzz fully automate attack synthesis for known flash loan patterns against a forked node. Requires an RPC endpoint (Alchemy/Infura) and Foundry. Novel DeFi primitive attacks may not be in the optimization search space.

**How to integrate:** This belongs in a dedicated `defi_sim_server` (extends the existing `econ_sim` plan, port 8015). The server accepts contract address + block height, spins a Foundry fork, runs FlashSyn/ItyFuzz, returns whether an attack was synthesized and estimated profit. Triggered only when ML flags ExternalBug or TOD on contracts that interact with known DeFi protocols (Uniswap/Aave/Curve function signatures detected).

### 7.3 Cross-Contract Vulnerabilities

**The gap:** All single-contract tools miss reentrancy and dependency bugs that span two or more contracts. Slither's cross-contract warnings have very high false positive rates without path feasibility checking.

**Best available tools:**

| Tool | Approach | Results | Automation level |
|------|----------|---------|-----------------|
| **Clairvoyance** (ASE 2020) | Cross-contract call chain analysis with path feasibility checking (5 PPTs). Found 101 unknown reentrancy bugs in 17K contracts | Best recall on cross-contract reentrancy | Fully automated given all source files |
| **DeFiTainter** (ISSTA 2023) | Resolves callee addresses from bytecode constants/storage, builds inter-contract call graph, tracks taint | 96% precision cross-contract | Automated; fails on fully dynamic dispatch |
| **ItyFuzz** | EVM bytecode fuzzer; automatically correlates contracts from on-chain state | Best for deployed systems | Fully automated |
| **Wake** (Ackee Blockchain 2023) | Python-based testing + static analysis with explicit cross-contract and cross-chain test support | Production use at IPOR, Axelar, Solady | Automated for known test patterns |
| **I-PDG** (2024) | Inter-contract program dependency graph + slicing + symbolic execution | Better than Slither on complex multi-contract systems | Automated given source scope |

**What's automatable:** Clairvoyance and ItyFuzz work fully automatically. The key requirement is scope definition — which contracts are "in scope" must be provided. For deployed upgradeable proxies, dynamic dispatch makes full automation harder.

**How to integrate:** Clairvoyance can be integrated as an additional Slither-adjacent tool in `audit_server` with a `run_cross_contract(contract_paths: list[str])` tool. The multi-contract input comes from the user or from on-chain address resolution. This is what the ROADMAP labels "C4 Multi-contract analysis" and is correctly deferred — it requires accepting multiple files as input, not just a single `.sol`.

### 7.4 Adversarial / Obfuscated Contracts

**The gap:** Contracts using indirect jumps, split logic, assembly obfuscation, or bytecode-level control flow flattening. Decompilers (Dedaub) can only analyze 4% of code in the most obfuscated MEV bots.

**Best available tools:**

| Tool | Approach | Results | Automation level |
|------|----------|---------|-----------------|
| **SKANF** (2025) | Control-flow deobfuscation + symbolic + concolic execution seeded from historical txs. Generated exploits for 394/1,030 vulnerable contracts | Best on closed-source obfuscated contracts | Fully automated; slow (minutes/contract) |
| **FinDet** (2025) | LLM semantic interpretation of EVM bytecode + fund-flow reachability. Balanced accuracy 0.9374 | Detected all 5 known + 20+ unreported adversarial contracts in 10-day deployment | Fully automated at inference time |
| **ByteEye** (2025) | GNN on bytecode structure graph. GNN harder to fool than pattern-matching (variable renaming, dead code insertion don't break the structural graph) | Strong on known obfuscation classes | Fully automated |
| **Heimdall-rs** | Rust decompiler: bytecode → Solidity-like. Enables running Slither on bytecode-only contracts | Good decompilation quality on non-adversarial bytecode | Fully automated; quality degrades on adversarial |

**What's automatable:** FinDet and ByteEye-style GNN approaches automate detection of known adversarial contract classes at inference speed. SKANF is best for exploit synthesis but is too slow for screening. For truly novel obfuscation, retraining is required.

**How to integrate:** FinDet-style bytecode analysis can serve as a Tier 0 screen for *suspiciously obfuscated* contracts — a signal that should escalate to Mythril and SKANF-style concolic execution regardless of ML score. This is a new routing signal: if bytecode complexity metrics exceed a threshold (entropy, indirect jump ratio), treat as adversarial and escalate.

### 7.5 Summary: What Gets Added to the Investigator Toolset

| Tool | Phase | Trigger condition | MCP home |
|------|-------|-----------------|----------|
| GPTScan (logic patterns) | Phase 2 | ML flags ExternalBug / CallToUnknown | audit_server :8012 |
| SMARTINV (invariant inference) | Phase 3 | Any CONFIRMED finding | audit_server :8012 |
| FlashSyn / ItyFuzz (flash loan sim) | Phase 3 | ExternalBug/TOD + DeFi protocol detected | defi_sim_server :8015 |
| Clairvoyance (cross-contract reentrancy) | Phase 3 | Multi-file input provided | audit_server :8012 |
| FinDet / bytecode entropy screen | Phase 2 | Always (Tier 0 addition) | audit_server :8012 |
| Heimdall-rs decompiler | Phase 2 | No source available (bytecode-only input) | audit_server :8012 |

**Honest automation ceiling (2025):** A pipeline running Slither + Aderyn + GPTScan + SMARTINV + ItyFuzz + FlashSyn covers the majority of known vulnerability patterns across all four categories automatically. The irreducible human requirement is writing specs for novel protocol logic, defining contract scope for multi-contract analysis, and reviewing LLM outputs for false positives (GPTScan precision drops to 57% on large complex projects).

---

## 10. Success Metrics

| Metric | Current (Phase 0–E) | Phase 2 target | Phase 3 target |
|--------|---------------------|----------------|----------------|
| False-positive rate on verified-clean contracts | unknown (no benchmark) | < 15% | < 10% |
| True-positive rate on SWC-registry test set | unknown | > 50% | > 70% |
| Mean latency deep-path audit | unknown | < 90s | < 120s (exploit gen adds ~30s) |
| Cross_validator / panel agreement with human auditor | unknown | > 60% | > 75% |
| Investigator tool calls per contract | N/A | ≤ 6 avg, ≤ 12 max | same |

Before Phase 2 ships, we need a benchmark harness. Minimum viable benchmark: 50 contracts from SWC-registry with ground-truth labels.

---

## 11. Key Design Constraints

- **ZKML proxy constraint:** `fusion_output_dim=128` is locked. Any new classifier head must read from the 128-dim fused representation. Do not change.
- **NUM_CLASSES=10 locked.** Adding new vulnerability classes requires a full retrain and schema version bump.
- **LM Studio port 1234** is the LLM endpoint. All LLM calls in agent nodes use `get_strong_llm()` which points there. The adversarial panel will need 3 concurrent calls — test LM Studio concurrency before committing to parallel execution.
- **SqliteSaver checkpoints** are per-conversation, not per-contract. The investigator loop's tool call trace lives in `AgentState` and is persisted automatically.
- **MCP over SSE** (not stdio). All new MCP servers must follow the same ASGI pattern as the existing four: `/health`, `/sse`, `/messages/` routes.

---

## 12. Migration Notes

This proposal does not break any existing code. The migration path is additive:

1. A1–A3 add a new endpoint and Phase 2 branch in graph_inspector_server; no existing node changes.
2. B2 (investigator loop) replaces the static fan-out in `evidence_router` routing logic and `graph.py`. The existing nodes (`rag_research`, `static_analysis`, `graph_explain`) become tools the investigator can call; they don't change internally.
3. B3 (adversarial panel) is a new node. `cross_validator` stays in the graph as the shallow/fast-path fallback. The `synthesizer` already reads from `state.get("verdicts", {})` so it handles both sources transparently.
4. The `cross_validator` tests remain valid throughout — that node doesn't get removed, just bypassed when the panel runs.
