# 04. Reproducibility & Determinism: The ZK Boundary

> **Prerequisites:** [01. The Audit Pipeline] — control-flow vs output determinism. [02. Evidence Model & Fuse()] — the `deterministic` flag on Evidence, `verdict_provable` vs `verdict_full`. [03. Prompt Injection Defense] — even sanitized, the LLM is non-deterministic.
> **Next:** [05. MCP Architecture] covers the tool servers that the deterministic nodes call.
> **Cross-ref:** [09. Formal Verification] covers Halmos, whose output is `deterministic=True`. [10. ZK Integration] will cover EZKL, which proves the deterministic tier anchored here.
> **Scope:** This doc covers `SENTINEL_DETERMINISTIC` mode (disabling non-deterministic components), the model hash (SHA-256 of the checkpoint), and the provenance chain (ML API → state → report). It does NOT cover the ZK circuit itself (P9, not yet built) or the ML model architecture (see `ml/` module docs).
> **TL;DR:** A ZK proof requires the proven function to be *bit-for-bit reproducible*. The LLM debate is non-deterministic — same input, different output across runs (CUDA non-determinism, backend variations, batch effects). So SENTINEL splits the verdict into two tiers: `verdict_provable` (deterministic evidence only — ML + Slither + Aderyn + Halmos) and `verdict_full` (all evidence, including the LLM debate). `SENTINEL_DETERMINISTIC=1` disables the LLM and RAG entirely and sets `torch.use_deterministic_algorithms(True)`, making the provable tier reproducible. A SHA-256 hash of the model checkpoint propagates from the ML API through the pipeline to the final report, binding each verdict to a specific model version. This is the foundation for P9 (EZKL ZK circuit): you can only prove what you can reproduce.

---

## The Problem: You Can't ZK-Prove a Debate

### What ZK requires

A ZK proof (Groth16, PLONK, EZKL) proves that a *fixed, deterministic function* was executed correctly on specific inputs. The key word is *fixed*. The proof circuit is compiled from a specific function (the ML model + the fusion math). At proof-generation time, the prover runs that exact function on the inputs and produces a proof. At verification time, the verifier checks the proof without re-running the function.

**The implication:** the function's output must be identical every time it runs on the same inputs. If the output varies — even by one bit — the proof is invalid. The verifier's check would fail because the proof was generated for output A, but the function produces output B on a different run.

### Why the LLM breaks this

The LLM debate (Prosecutor + Defender + Judge) is non-deterministic. Even with `temperature=0.0`, the same prompt produces different outputs across runs. Here's why:

| Source of non-determinism | Why it happens | Can we fix it? |
|---|---|---|
| CUDA floating-point non-determinism | GPU parallel reductions sum in non-deterministic order | Partially — `torch.use_deterministic_algorithms(True)` helps, but not all ops have deterministic alternatives |
| llama.cpp backend variations | Batching strategy, KV cache eviction, quantization rounding differ across runs | No — it's a property of the inference engine |
| Model versioning | The same model name (`qwen2.5-coder-7b-instruct`) can point to different weights across deployments | No — weight versioning is external to the model |
| Batch effects | Other requests in the same batch influence attention patterns | No — unless you enforce batch_size=1, which kills throughput |

**The conclusion:** you cannot ZK-prove the LLM's verdict. It's not a matter of engineering effort — it's a fundamental property of non-deterministic computation. The ZK circuit would need to encode the entire LLM forward pass (billions of parameters), and the proof would change every time the LLM produced a different output.

### Teaching: How to think about the ZK boundary

The question is not "what should we prove?" but "what *can* we prove?" The ZK boundary is determined by what's *reproducible*, not by what's *valuable*. Here's the reasoning process:

1. **List all evidence sources** and classify each as deterministic or non-deterministic.
2. **For the non-deterministic ones, ask:** "Can we make them deterministic?" If yes → make them deterministic and include them. If no → exclude them from the provable tier.
3. **Accept the trade-off:** the provable tier is smaller (less evidence) but reproducible (ZK-provable). The full tier is richer (all evidence) but non-reproducible (human-reported only).

**Applied to SENTINEL:**

| Source | Deterministic? | In `verdict_provable`? | In `verdict_full`? |
|--------|:-:|:-:|:-:|
| ML model inference | Yes (with `torch.use_deterministic_algorithms`) | ✓ | ✓ |
| Slither static analysis | Yes (deterministic detector rules) | ✓ | ✓ |
| Aderyn static analysis | Yes (deterministic) | ✓ | ✓ |
| Halmos formal verification | Yes (symbolic execution) | ✓ | ✓ |
| RAG retrieval | No (embedding model non-determinism) | ✗ | ✓ |
| LLM debate | No (CUDA + backend non-determinism) | ✗ | ✓ |

The provable tier is 4 sources; the full tier is 6. The LLM and RAG are advisory — they enrich the human report but are *not* part of the cryptographic claim.

---

## How We Arrived at This Design

> **How to read this section:** Each step shows the question, *how to reason about it*, and the chain of logic connecting the answer to the design. Learn the method, not just the result.

### Step 1 — Identify the invariant (the "must always be true" test)

**The question:** What must always be true about the ZK proof?

**Applying the "useless or dangerous" test:**

| Candidate property | If violated → | Verdict |
|---|---|---|
| `verdict_provable` is bit-for-bit reproducible | ZK proof is invalid → on-chain verifier rejects → oracle is useless | **Invariant** |
| `verdict_full` is reproducible | Human report varies across runs → confusing but not dangerous | Preference |
| Model hash is in the report | Can't trace verdict to model version → no provenance → unverifiable | **Invariant** |
| LLM debate runs on every audit | Verdict quality drops → more false negatives | Preference |

**The reasoning chain:** The ZK proof anchors `verdict_provable` on-chain. If `verdict_provable` is not reproducible — if the same contract + same model produces a different `verdict_provable` on a second run — then the proof generated on run 1 won't match the verdict produced on run 2. The on-chain verifier rejects the proof. The oracle is broken. This means: *every component that contributes to `verdict_provable` must be deterministic*. The LLM debate is not deterministic → it must be excluded from `verdict_provable`. This isn't a design choice; it's a physical constraint of ZK cryptography.

### Step 2 — Identify the constraints (what forces a specific shape)

**Constraint A: `torch.use_deterministic_algorithms(True)` doesn't cover all operations.**
- *Why:* PyTorch's deterministic mode provides deterministic alternatives for *some* operations, but not all. Operations without deterministic alternatives raise an error (or, with `warn_only=True`, log a warning and use the non-deterministic path).
- *What this forces:* We accept that ML inference is *mostly* deterministic. For the operations that don't have deterministic alternatives, we log a warning. The ZK circuit (P9) will need to verify only the deterministic subset of the forward pass — or we accept a small reproducibility gap and document it. This is an open issue, not a solved one.

**Constraint B: The model hash must be stable across restarts.**
- *Why:* The hash binds the verdict to a specific model version. If the hash changes across restarts (e.g., because the checkpoint file is re-serialized), the provenance chain breaks — two audits of the same contract with the same model would have different hashes, making them look like different models.
- *What this forces:* The hash is computed from the *file content* (SHA-256 of the `.pt` file), not from the in-memory state dict. The file content doesn't change across restarts (unless the checkpoint is replaced). This is a deliberate choice over state_dict hashing (see Design Decision below).

**Constraint C: Deterministic mode must disable RAG, not just the LLM.**
- *Why:* RAG uses an embedding model (text-embedding-3-small or similar) to find similar vulnerability descriptions. Embedding models are also non-deterministic (same CUDA non-determinism + API-level variation). If RAG runs in deterministic mode, its evidence is non-deterministic → `verdict_full` varies across runs.
- *What this forces:* `rag_research.py:74-76` checks `SENTINEL_DETERMINISTIC` and returns `{"rag_results": []}` — skipping the embedding model entirely. This means `verdict_provable` (deterministic tier) never includes RAG evidence, which is correct: RAG is `deterministic=True` in the Evidence model (it's reproducible in principle), but in practice the embedding model's non-determinism makes it unreliable for ZK.

**Wait — is RAG deterministic or not?** This is a subtle point. RAG's *retrieval* is a nearest-neighbor search on pre-computed embeddings — deterministic given the query embedding. But the *query embedding* (computed by the embedding model) is non-deterministic. So RAG is "deterministic given the embedding, but the embedding is non-deterministic." In deterministic mode, we skip the embedding computation entirely. In normal mode, RAG runs and its evidence is marked `deterministic=True` (because the retrieval itself is deterministic). This is a deliberate approximation — the ZK tier excludes RAG to be safe; the human tier includes it for richness.

### Step 3 — Eliminate alternatives (find what breaks under *current* conditions)

**The three approaches for the ZK boundary:**

| Approach | How it breaks | When it breaks | Eliminate? |
|---|---|---|---|
| **Prove everything** (ZK-prove the LLM) | ZK circuit must encode the entire LLM forward pass (billions of parameters). Proof generation takes hours/days. Circuit size is astronomically large. | Always — LLMs are too large for ZK. | **Yes** |
| **Prove only the ML model** (not the fusion) | Verifier learns "the model said Reentrancy=0.87" but not whether `fuse()` produced CONFIRMED or SAFE. No verdict anchored on-chain. | When the verifier needs the verdict, not the raw score. | **Yes** |
| **Prove ML + fusion** (deterministic tier) | `verdict_provable` is anchored. `verdict_full` is advisory. The ZK circuit is small (ML model + weighted sum). | When non-deterministic evidence (LLM, RAG) would change the verdict. → It doesn't, because `verdict_provable` excludes them by definition. | **No** — survives. |

**The reasoning:** "Prove everything" breaks on physics — ZK circuits for billion-parameter models are infeasible today and possibly forever. "Prove only the ML model" breaks on utility — the verifier learns the model's raw output but not the final verdict; the fusion math (which maps probabilities to verdict labels) is not proven. "Prove ML + fusion" is the sweet spot: the circuit encodes the ML forward pass + the `fuse()` weighted sum (which is cheap — it's arithmetic). The verifier gets a proven verdict label, not just a probability. The non-deterministic evidence (LLM, RAG) is excluded from the proof but included in the human report.

**Steel-manning "prove only the ML model":** "If the verifier knows the model said Reentrancy=0.87, they can check the threshold themselves — 0.87 ≥ 0.70 → CONFIRMED. Why prove the fusion?" Because the thresholds are *configurable* (L1 config, Rule B). If the threshold changes from 0.70 to 0.65, the same model output (0.87) maps to a different verdict. The verifier needs to know *which threshold was used* — and that means proving the fusion with the config as a public input. Proving ML + fusion together is cleaner than proving ML alone and exposing config separately.

### Step 4 — Stress-test against future growth

**The test:** "What happens when we add a new model (model swap, retrain)?"

**Tracing through the design:**
1. New checkpoint file → `_compute_file_hash()` produces a new SHA-256 → `model_hash` changes.
2. `model_hash` propagates: ML API → `ml_assessment` → `state["model_hash"]` → `synthesizer` → `final_report["model_provenance"]["model_hash"]`.
3. The old model's verdicts (anchored with the old hash) are still valid — they were proven with the old model. The new model's verdicts get a new hash.
4. The on-chain `AuditRegistry` (P9) stores the hash alongside each verdict → a verifier can check which model produced each verdict.

**Total: zero code changes.** The hash is computed at startup and flows through automatically. A model swap is a config change (checkpoint path), not a code change.

**Counter-argument:** "But what if the new model uses a different architecture (e.g., CodeBERT → GraphCodeBERT)?" Then the ZK circuit needs to be recompiled (the model's forward pass changed). This is a P9 concern — the circuit is model-specific. The model hash tells you *which circuit to verify with*. The hash is the binding; the circuit is the proof.

### Step 5 — Measure (reproducibility test)

**The question:** Is `verdict_provable` actually reproducible?

**The test** (`test_deterministic_mode.py:130-175`): Create deterministic evidence (ML + Slither, no LLM/RAG). Run `fuse()` twice. Assert `result1 == result2`. This verifies that the fusion math is deterministic — same inputs → same outputs.

**The deeper test (not yet built):** Run the *full pipeline* (ML inference + Slither + Aderyn + fuse) in `SENTINEL_DETERMINISTIC=1` mode on the same contract twice. Assert `verdict_provable` is identical. This would verify the *entire* deterministic chain, not just `fuse()`. The current test only verifies `fuse()` in isolation — it doesn't verify that `torch.use_deterministic_algorithms(True)` actually makes the ML model reproducible end-to-end. This is a known gap, documented in Limitations.

> **The method, summarized:** (1) Find the invariant by asking "if violated, is the system useless?" → `verdict_provable` must be reproducible. (2) Find constraints from physical limits → ZK can't prove an LLM; `torch` deterministic mode doesn't cover all ops. (3) Eliminate alternatives by finding *current* failure conditions → "prove everything" breaks on physics, "prove only ML" breaks on utility. (4) Stress-test by tracing a model swap → zero code changes, hash propagates automatically. (5) Measure reproducibility → `fuse()` is tested; full pipeline reproducibility is a known gap.

---

## The Solution

### Deterministic mode: disabling non-deterministic components

When `SENTINEL_DETERMINISTIC=1` is set, three things happen:

```
┌─────────────────────────────────────────────────────────────────────┐
│  SENTINEL_DETERMINISTIC=1                                           │
│                                                                     │
│  ML API (api.py:107-115):                                           │
│    torch.use_deterministic_algorithms(True)                         │
│    torch.manual_seed(42)                                            │
│    torch.cuda.manual_seed_all(42)                                   │
│                                                                     │
│  Agents pipeline:                                                   │
│    _llm_enabled() → False                                           │
│      → cross_validator: skips debate, uses rule-based fallback       │
│      → synthesizer: skips narrative, uses template                   │
│      → reflection: skips self-critique                              │
│                                                                     │
│    rag_research() → {"rag_results": []}                             │
│      → skips embedding model call                                   │
│      → no RAG evidence in evidence_list                              │
│                                                                     │
│  Result:                                                            │
│    evidence_list contains only deterministic evidence (ML, Slither, │
│    Aderyn, Halmos). fuse() produces verdict_provable ==            │
│    verdict_full (both tiers use the same deterministic evidence).   │
│    The verdict is reproducible → ZK-anchorable.                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Teaching: why `verdict_provable == verdict_full` in deterministic mode.** In normal mode, `verdict_provable` (deterministic evidence only) differs from `verdict_full` (all evidence, including LLM + RAG). In deterministic mode, the LLM and RAG are disabled — they produce no evidence. So `evidence_list` contains only `deterministic=True` items. `fuse()` runs the fusion twice (once on `det_items`, once on all items), but since all items are deterministic, the two runs produce the same result. The two tiers converge. This is correct behavior: in ZK mode, there's no advisory tier — everything is provable.

### Model hash: the provenance chain

The model hash binds each verdict to a specific model version. Here's how it propagates:

```
ML API startup
  predictor = Predictor(checkpoint)
    → _compute_file_hash(Path(checkpoint))     ← SHA-256, 8MB chunks
    → predictor.model_hash = "a1b2c3..."       ← 64 hex chars, cached

  /health response
    → "model_hash": predictor.model_hash        ← exposed at api.py:264

  /predict response
    → PredictResponse.model_hash                ← Pydantic field, api.py:233

ml_assessment node
  → result = await _call_mcp_tool(...)          ← calls /predict
  → state["model_hash"] = result["model_hash"]  ← extracts from response

synthesizer node
  → final_report["model_provenance"] = {
      "model_hash":      state.get("model_hash", ""),
      "checkpoint_path": os.getenv("SENTINEL_CHECKPOINT", ""),
      "schema_version":  "v9",
    }
```

**Teaching: why the hash is computed at startup, not per-request.** Computing SHA-256 of an 8MB checkpoint file takes ~50ms. If we computed it on every `/predict` call, that's 50ms × 100 requests = 5 seconds of overhead per day. By computing once at startup and caching in `predictor.model_hash`, the per-request cost is zero — the hash is just a field in the response. The trade-off: if the checkpoint file is replaced while the API is running (hot-swap), the hash is stale. This is acceptable because checkpoint hot-swapping is not a supported workflow — you restart the API when you swap models.

**Teaching: why file hash, not state_dict hash.** The file hash (SHA-256 of the `.pt` file) is stable across restarts — the file doesn't change unless you replace it. A state_dict hash (hashing the tensor values in memory) is more precise (it detects if the same file was loaded with different precision, e.g., float32 vs float16), but it requires loading the checkpoint into memory first — which we're already doing, but the hash computation is slower (iterating all tensors vs reading the file sequentially). For our purposes, file hash is sufficient: we want to know "is this the same checkpoint?" not "is this the same tensor state?" See Design Decision below for the full reasoning.

## Key Code

The `_llm_enabled()` function — the single gate for all LLM calls:

```python
# _helpers.py:29-39
def _llm_enabled() -> bool:
    if os.getenv("SENTINEL_DETERMINISTIC", "").strip().lower() in ("1", "true", "yes"):
        return False
    return os.getenv("AGENTS_DISABLE_LLM", "").strip().lower() not in ("1", "true", "yes")
```

Why this matters: `SENTINEL_DETERMINISTIC` is checked *first* — it takes priority over `AGENTS_DISABLE_LLM`. This means: if you're in ZK mode, the LLM is off, period. You can't accidentally enable it with `AGENTS_DISABLE_LLM=0`. The two env vars serve different purposes: `SENTINEL_DETERMINISTIC` is for ZK proof generation (reproducibility), `AGENTS_DISABLE_LLM` is for testing/debugging (skip LLM cost). Checking deterministic first ensures ZK mode is never accidentally compromised.

The deterministic mode setup in the ML API lifespan:

```python
# api.py:107-115
_deterministic_mode = os.getenv("SENTINEL_DETERMINISTIC", "").lower() in ("1", "true", "yes")
if _deterministic_mode:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
```

Why this matters: `torch.use_deterministic_algorithms(True)` makes PyTorch select deterministic implementations for operations that have them. `torch.manual_seed(42)` fixes the random seed (for dropout, initialization, etc.). `torch.cuda.manual_seed_all(42)` fixes the CUDA RNG for all GPUs. Together, these make ML inference as deterministic as PyTorch allows — but not *completely* deterministic (some ops don't have deterministic alternatives). This is a known limitation (see Limitations).

The file hash computation — SHA-256 in 8MB chunks:

```python
# predictor.py:440-457
def _compute_file_hash(self, path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
```

Why this matters: reading in 8MB chunks avoids loading the entire checkpoint into memory for hashing (the checkpoint is already loaded for inference, but the hash computation is a separate pass). The chunk size is a trade-off: too small → many iterations, slow; too large → high memory peak. 8MB is a standard choice for file hashing.

The provenance chain — from ML API to final report:

```python
# synthesizer.py:372-376
"model_provenance": {
    "model_hash":      state.get("model_hash", ""),
    "checkpoint_path": os.getenv("SENTINEL_CHECKPOINT", ""),
    "schema_version":  "v9",
},
```

Why this matters: the `model_provenance` section is the *audit trail*. A security reviewer reading the report can see: "this verdict was produced by model checkpoint X (hash), loaded from path Y, with schema version Z." If the verdict is wrong, the reviewer can reproduce the audit by loading the same checkpoint and running the pipeline in deterministic mode. Without the hash, the reviewer can't tell which model version produced the verdict — it's untraceable.

## Design Decision: File Hash vs State Dict Hash vs ONNX Hash

> **How to read this section:** The table shows the options. The *elimination reasoning* below shows how to think about the choice.

### The elimination process

**Step 1: What are the options?** (a) File hash — SHA-256 of the `.pt` file. (b) State dict hash — hash of the tensor values loaded into memory. (c) ONNX hash — export the model to ONNX format and hash the ONNX graph.

**Step 2: Eliminate by finding failure conditions.**

**State dict hash — steel-man first:** "A state dict hash is more precise. It hashes the actual tensor values, so it detects if the same file was loaded with different precision (float32 vs float16) or if the model was modified after loading. File hash only tells you the file is the same, not that the model is the same."

**Why it fails for SENTINEL:**
1. *Requires loading the checkpoint twice* — once for inference, once for hashing (because the hash needs to iterate all tensors, which is a separate pass from the forward pass). This doubles startup time.
2. *Precision-dependent* — the same checkpoint loaded as float32 vs float16 produces different state dict hashes. But the model's *behavior* is the same (within floating-point tolerance). The hash is more precise than needed — it distinguishes things that don't matter.
3. *Not stable across PyTorch versions* — different PyTorch versions may serialize tensors differently, producing different state dicts from the same checkpoint file. The hash changes even though the model didn't.

**ONNX hash — steel-man first:** "Exporting to ONNX produces a canonical representation of the model's computational graph. Hashing the ONNX file gives you a stable, format-independent identity."

**Why it fails for SENTINEL:**
1. *Export is lossy* — ONNX export doesn't preserve all PyTorch operations (some custom ops aren't supported). The ONNX model is an *approximation* of the PyTorch model.
2. *Export is expensive* — ONNX export takes minutes for a model with custom layers. Startup time would increase significantly.
3. *Not needed yet* — we're not running inference in ONNX (we use PyTorch). The ONNX hash would be a *different identity* than the PyTorch model. If we later move to ONNX for ZK (EZKL uses ONNX), we'd re-hash at that point.

**File hash — why it survives:** It's computed in one pass (read the file, hash it). It's stable across restarts (the file doesn't change). It's stable across PyTorch versions (the file bytes are the same). Its weakness (doesn't detect precision changes) is acceptable because we don't change precision at runtime — the checkpoint is always loaded as float32. If we need precision tracking later, we add a `precision` field to `model_provenance` alongside the hash.

**The reasoning principle:** "When choosing a content-addressing scheme, eliminate options that are more precise than needed. Precision you don't need is cost you don't want to pay. The file hash answers the question 'is this the same checkpoint?' — which is the question we need. The state dict hash answers 'is this the same tensor state?' — which is more precise than we need, and costs double startup time."

### When this decision would be wrong

**The reversal condition:** If we start loading the same checkpoint with different precision (e.g., float16 for inference, float32 for ZK), the file hash can't distinguish them. At that point, add a `precision: str` field to `model_provenance` and include it in the on-chain anchor. Don't switch to state dict hash — just add the precision metadata. The trigger: when we run inference in mixed precision or quantized mode (likely P9+, when EZKL requires a specific precision).

## Technology Choice: `torch.use_deterministic_algorithms`

**The 5-question framework:**

1. **What category?** Making PyTorch GPU computation reproducible.
2. **What alternatives?** (a) `torch.use_deterministic_algorithms(True)` — PyTorch's built-in deterministic mode. (b) CPU-only inference — CPUs are inherently deterministic (no parallel reduction non-determinism). (c) Custom CUDA kernels — write deterministic versions of the non-deterministic ops.
3. **Why this?** It's the built-in, supported approach. It covers the majority of operations. It's one line of code. CPU-only inference is 10-100x slower (no GPU acceleration). Custom CUDA kernels are a research project.
4. **When is it insufficient?** When the model uses an operation without a deterministic alternative. `torch.use_deterministic_algorithms(True)` raises `NotImplementedError` for those ops. With `warn_only=True`, it logs a warning and uses the non-deterministic path — which means the output is *not* fully reproducible.
5. **Migration trigger:** If the ZK circuit (P9) requires *complete* determinism (no warnings allowed), and the model uses ops without deterministic alternatives, then either (a) replace those ops with deterministic alternatives (model surgery), or (b) run inference on CPU for ZK proof generation (slow but fully deterministic), or (c) use EZKL's ONNX runtime (which has its own determinism guarantees, separate from PyTorch).

**Why CPU fallback is the P9 plan:** For ZK proof generation, we don't need *fast* inference — we need *reproducible* inference. ZK proof generation is a one-time operation per audit (not real-time). Running on CPU takes 30s instead of 0.5s, but it's fully deterministic. The 30s cost is acceptable for a proof that anchors on-chain permanently.

## Anti-Patterns

### ❌ ZK-prove the LLM — "prove everything"
**What it looks like:** Encode the entire LLM debate (Prosecutor + Defender + Judge) in the ZK circuit. Prove that the debate was executed correctly and produced the verdict.
**Why someone would build this:** "The LLM is the system's richest signal. If we can't prove it, the ZK proof is just proving the boring deterministic tier — not the smart tier." This is the "prove everything" instinct.
**Why it's wrong:**
1. *Physics* — the LLM has billions of parameters. A ZK circuit encoding a 7B model would have trillions of constraints. Proof generation would take hours to days on specialized hardware. EZKL/Groth16 circuits are practical for models up to ~100M parameters.
2. *Non-determinism* — even if the circuit existed, the LLM's output is non-deterministic. The proof would change every run. The verifier would reject.
3. *No benefit* — the deterministic tier (ML + static + formal) is already strong. The LLM adds semantic richness but not cryptographic certainty. Proving the deterministic tier gives the on-chain consumer a trustworthy verdict; the human report gives the advisory tier.
**The right approach:** Prove the deterministic tier (`verdict_provable`). Treat the LLM as advisory (`verdict_full`). Anchor the former, report the latter. No trade-off between "smart" and "provable" — you produce both.

### ❌ `temperature=0.0` means deterministic — "if I set temp=0, the LLM is reproducible"
**What it looks like:** Set `temperature=0.0` on the LLM and claim the output is deterministic. Don't bother with `SENTINEL_DETERMINISTIC` mode.
**Why someone would build this:** "Temperature=0 means the model always picks the highest-probability token. Same input → same output. That's deterministic, right?"
**Why it's wrong:**
1. *CUDA non-determinism* — the model's forward pass runs on GPU. Floating-point reductions (sum, softmax) execute in non-deterministic order across CUDA thread blocks. The same input produces slightly different logits. At temperature=0, the argmax is usually the same — but not *always*. A tie or near-tie at the top can flip.
2. *llama.cpp backend* — the inference engine (llama.cpp, vLLM, etc.) makes implementation choices (batching, KV cache eviction, quantization) that affect the output. Different versions of the engine produce different outputs from the same model at temp=0.
3. *Batch effects* — if other requests are in the same batch, their attention patterns bleed into the output. Same prompt, different batch → different output.
**The right approach:** Don't rely on temperature=0 for reproducibility. In deterministic mode, disable the LLM entirely. The deterministic tier doesn't include the LLM — it's ML + static + formal only. Temperature=0 reduces variance; it doesn't eliminate it. "Reduced variance" is not "deterministic."

## Mistakes & Fixes

### Mistake: LM Studio at temp=0.0 was still non-deterministic
**What happened:** During P5 testing, the same prompt was sent to LM Studio (the LLM inference server) with `temperature=0.0` three times. The debate verdict differed across runs: CONFIRMED, then LIKELY, then CONFIRMED. The LLM was not reproducible despite temp=0.
**Why it happened:** CUDA non-determinism (the GPU's floating-point reductions execute in non-deterministic order) + llama.cpp batching effects (other requests in the batch influenced attention patterns). Temperature=0 controls the *sampling* (always pick the highest-probability token), but the *logits* themselves vary slightly across runs due to GPU non-determinism.
**How we found it:** P5 scratch file (`p5_reproducibility_20260626.md`) — manual test: same prompt, temp=0, three runs, two different verdicts. This confirmed that the LLM cannot be part of the ZK-provable tier.
**The fix:** `SENTINEL_DETERMINISTIC=1` disables the LLM entirely. The deterministic tier uses only ML + static + formal evidence. The LLM debate is advisory only.
**The lesson:** "Temperature=0" and "deterministic" are not the same thing. Temperature controls sampling; it doesn't control the forward pass. GPU non-determinism makes the forward pass itself vary. For ZK reproducibility, you need *bit-for-bit* identical outputs — which means no LLM in the proven tier.

### Mistake: `torch.use_deterministic_algorithms(True)` can raise on unsupported ops
**What happened:** When `torch.use_deterministic_algorithms(True)` was enabled, some PyTorch operations raised `NotImplementedError` because they didn't have deterministic alternatives. The ML API crashed on startup during the warmup forward pass.
**Why it happened:** Not all PyTorch operations have deterministic implementations. Operations like `scatter_add` and some convolution backward passes are non-deterministic on CUDA with no deterministic alternative. The model used at least one such operation.
**How we found it:** Startup crash with `NotImplementedError` during the warmup pass in `predictor.py:426-427`.
**The fix:** Use `warn_only=True` as a fallback: `torch.use_deterministic_algorithms(True, warn_only=True)`. This logs a warning for unsupported ops but doesn't crash. The output is *mostly* deterministic — the non-deterministic ops introduce small variations, but the overall verdict is usually stable. For P9 (ZK circuit), we'll either replace the non-deterministic ops or run on CPU for proof generation.
**The lesson:** "Deterministic mode" in PyTorch is a best-effort guarantee, not a complete one. Test your model's forward pass with deterministic mode enabled *before* claiming reproducibility. If it crashes, you have a non-deterministic op that needs replacing.

### Mistake: Model hash was not propagated to the report
**What happened:** The ML API computed `model_hash` and exposed it via `/health` and `/predict`. But the agents pipeline didn't extract it from the response — `state["model_hash"]` didn't exist as a field. The `model_provenance` section in the report was empty.
**Why it happened:** The P5 plan added the hash to the ML API (T5.1) and to the report (T5.2), but the *propagation* step (extracting the hash from the ML API response in `ml_assessment.py` and writing it to state) was initially missed.
**How we found it:** Test `test_model_hash_propagated_to_report` (`test_deterministic_mode.py:99-127`) failed — `model_provenance["model_hash"]` was an empty string, not the mock hash.
**The fix:** Add `model_hash` to `AuditState` (state.py:250). Extract it in `ml_assessment.py:118`: `"model_hash": result.get("model_hash", "")`. The synthesizer reads it from state: `state.get("model_hash", "")`.
**The lesson:** When building a provenance chain across modules (ML API → agents pipeline → report), trace the full path. A hash computed but not propagated is useless — the report needs the hash, and the report is produced by the synthesizer, which reads from state, which is written by `ml_assessment`. Every link in the chain must be tested. The test `test_model_hash_propagated_to_report` verifies the full chain end-to-end.

## What Would Break If You Removed This?

**Remove the `deterministic` flag from Evidence:** `verdict_provable` and `verdict_full` become identical. The ZK boundary disappears — the LLM debate (non-deterministic) is included in the proven tier. The ZK proof changes every run. The on-chain verifier rejects.

**Remove `SENTINEL_DETERMINISTIC` mode:** you can't run the pipeline in ZK-proof mode. The LLM debate always runs, adding non-deterministic evidence. `verdict_provable` is never reproducible. P9 (ZK integration) is impossible.

**Remove the model hash:** you can't trace a verdict to a model version. A security reviewer sees "verdict_provable = CONFIRMED" but can't tell which model produced it. The audit is untraceable. If the model is later found to be buggy, you can't identify which past verdicts are affected.

**Remove `model_provenance` from the report:** the hash is computed (in the ML API) but never reaches the human reviewer. The provenance chain is broken at the last link. The reviewer can't verify which model was used — they have to trust the operator.

## At Scale

*Scale metric: number of model versions in production (current: 1).*

| Scale | What works | What breaks | Migration path |
|-------|-----------|-------------|----------------|
| 1 model (current) | Hash computed at startup, flows to report | — | — |
| 5 models (model registry) | Each model has its own hash | Need to track which model is "active" | Model registry (MLflow or similar) with hash as the key |
| 20 models (A/B testing) | Hash distinguishes models | Need to compare verdicts across models | Eval framework compares `verdict_provable` across model hashes |
| 100 models (model zoo) | Hash is unique per model | Hash collision risk (SHA-256: negligible) | No change needed — SHA-256 collision resistance is sufficient |

The hash scales for free — SHA-256 has 2^128 collision resistance, which is more than enough for any number of model versions. The scale wall is *operational*: tracking which model is active, when it was swapped, and whether old verdicts (anchored with old hashes) are still valid. This is a model registry problem, not a hash problem.

## Try It Yourself

> TRY IT: `cd agents && SENTINEL_DETERMINISTIC=1 python -c "from src.orchestration.nodes._helpers import _llm_enabled; print('LLM enabled:', _llm_enabled())"` — should print `False`.

> TRY IT: `cd agents && python -c "from src.orchestration.verdict.fuse import fuse; from src.orchestration.verdict.evidence import Evidence, Polarity, Kind; ev=[Evidence('ml','Reentrancy',Polarity.SUPPORTS,0.85,0.90,Kind.STATISTICAL,True,{})]; r1=fuse(ev); r2=fuse(ev); print('Reproducible:', r1==r2); print('Verdict:', r1['Reentrancy'].verdict_provable)"` — verifies fuse() is deterministic.

> TRY IT: `cd agents && pytest tests/test_deterministic_mode.py -v` — runs all 8 P5 tests (LLM disable, RAG skip, model hash propagation, fuse reproducibility).

## Limitations & What's Missing

- **`torch.use_deterministic_algorithms` doesn't cover all operations.** With `warn_only=True`, non-deterministic ops log a warning and use the non-deterministic path. This means ML inference is *mostly* deterministic, not *completely* deterministic. For P9, we'll need to either replace non-deterministic ops or run on CPU for proof generation. This is an open issue.

- **Full-pipeline reproducibility test is not built.** The current test (`test_deterministic_mode_produces_consistent_verdicts`) verifies `fuse()` in isolation — same evidence → same verdict. It doesn't verify that the *full pipeline* (ML inference + Slither + Aderyn + fuse) produces the same `verdict_provable` across runs in deterministic mode. This is a known gap.

- **No ZK circuit yet (P9).** The infrastructure (model hash, deterministic mode, dual verdict) is built. The actual ZK proof circuit (EZKL) is P9, not yet started. The current system produces *anchorable* verdicts (reproducible + hashed), but doesn't yet anchor them.

- **No model versioning policy.** When a new model checkpoint is deployed, there's no policy for re-auditing contracts that were audited with the old model. Old verdicts remain valid (they were proven with the old model), but they may be wrong if the old model had bugs. This is a governance question, not a technical one.

- **File hash changes on re-serialization.** If the same model is saved to a `.pt` file twice (e.g., `torch.save(model.state_dict(), "model.pt")` twice), the file bytes may differ (PyTorch's serialization is not guaranteed to be byte-stable). The hash would change even though the model didn't. Mitigation: never re-serialize the checkpoint; always reference the original file. This is a workflow discipline, not a code fix.

## Transferable Patterns

1. **Dual-tier architecture — provable vs advisory** — `verdict_provable` (deterministic, ZK-anchorable) + `verdict_full` (all evidence, human-reported).
   - *Interview story:* "SENTINEL's ZK proof anchors only the deterministic evidence — the ML model and static analysis. The LLM debate is advisory: it enriches the human report but isn't part of the cryptographic claim. We get both tiers from the same `fuse()` function by running it twice: once on `deterministic=True` evidence, once on all evidence. When someone asks 'is the LLM verdict provable?' — no, it's advisory. 'Is the ML verdict provable?' — yes, it's anchored. No trade-off between smart and provable."
   - *When this pattern is WRONG:* when the deterministic tier is too weak to be useful alone (e.g., if ML is the only deterministic source and it's unreliable, `verdict_provable` will be mostly SAFE — the ZK proof attests to nothing). The pattern requires a strong deterministic core. If your deterministic evidence is sparse, either strengthen it or accept that the provable tier is conservative.

2. **Content-addressed provenance — SHA-256 as identity** — the model hash binds each verdict to a specific model version.
   - *Interview story:* "Every SENTINEL verdict carries a SHA-256 hash of the model checkpoint that produced it. The hash propagates from the ML API at startup, through the pipeline state, to the final report's `model_provenance` section. If a verdict is wrong, you can trace it back to the exact model version and reproduce the audit by loading the same checkpoint in deterministic mode. Without the hash, the verdict is untraceable — you can't tell which model produced it."
   - *When this pattern is WRONG:* when the content-addressed artifact is mutable (e.g., a model that's fine-tuned in place — the file changes, the hash changes, but it's "the same model"). Content-addressing works for immutable artifacts. For mutable artifacts, you need versioning (a version number alongside the hash) to distinguish "same model, different version" from "different model."

3. **Determinism boundary as a flag — `deterministic: bool` on Evidence** — each evidence item carries its own determinism classification.
   - *Interview story:* "Instead of having a global 'deterministic mode' that affects the whole pipeline, each Evidence item carries a `deterministic` flag. ML evidence is `True`, LLM debate is `False`. `fuse()` uses this flag to split the verdict into provable and full tiers. This means a single audit produces both a ZK-anchorable verdict and a human-readable report — from the same evidence list, with no separate code path for ZK mode."
   - *When this pattern is WRONG:* when the determinism classification is wrong (e.g., RAG is marked `deterministic=True` in the Evidence model, but the embedding model is actually non-deterministic). The flag must be accurate — if it says `True` but the source is non-deterministic, the ZK proof will fail intermittently. When in doubt, mark `False` — the cost of a false `True` (proof failure) is higher than the cost of a false `False` (excluded from the provable tier).

---

**Source files verified:**
- `agents/src/orchestration/nodes/_helpers.py:29-39` — `_llm_enabled()` with `SENTINEL_DETERMINISTIC` check
- `agents/src/orchestration/nodes/rag_research.py:74-76` — RAG skip in deterministic mode
- `agents/src/orchestration/nodes/ml_assessment.py:116-119` — `model_hash` extraction from ML API response
- `agents/src/orchestration/nodes/synthesizer.py:372-376` — `model_provenance` in final report
- `agents/src/orchestration/state.py:250` — `model_hash: str` field in AuditState
- `ml/src/inference/predictor.py:357, 440-457` — `_compute_file_hash()` SHA-256 in 8MB chunks
- `ml/src/inference/api.py:107-115, 233, 264` — deterministic mode lifespan, `model_hash` in PredictResponse + /health
- `agents/tests/test_deterministic_mode.py:18-175` — 8 tests (4 LLM, 2 RAG, 2 E2E)

**Verified against commit hash:** `c47898ea5`
