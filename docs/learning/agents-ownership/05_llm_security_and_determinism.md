# Agents Module Ownership — 05: LLM, Prompt Security, and Determinism

## Ownership Target

Understand where LLMs may influence the human report, how untrusted Solidity source is prepared for prompts, and what deterministic mode disables.

## Source Reading Order

1. `agents/src/orchestration/nodes/_helpers.py` — LLM enablement and MCP helper
2. `agents/src/llm/client.py`
3. `agents/src/security/comment_strip.py`
4. `agents/src/security/injection_detect.py`
5. `agents/src/security/prompt_delimit.py`
6. `agents/src/security/prompt_sanitize.py`
7. `agents/src/orchestration/nodes/cross_validator.py`
8. `agents/src/orchestration/nodes/synthesizer.py`
9. `agents/src/orchestration/nodes/rag_research.py`

## Items to Own

- Which nodes call an LLM and which routing decisions must remain LLM-independent.
- The three prompt-defense layers and their required order.
- Where injection detections enter state and final reports.
- The difference between a deterministic fused verdict and an LLM-assisted narrative/debate result.
- Effects of `AGENTS_DISABLE_LLM` and `SENTINEL_DETERMINISTIC`.
- Fallback behavior when an LLM is disabled or fails.
- Why raw contract text is data, never instructions.

## Prompt Boundary Exercise

Trace raw `contract_code` from input to each LLM prompt. Record the sanitizing function, maximum source excerpt, state update, and fallback behavior.

## Verification

```bash
cd agents
TMP=/tmp TEMP=/tmp TMPDIR=/tmp poetry run pytest \
  tests/test_comment_strip.py \
  tests/test_injection_detect.py \
  tests/test_prompt_delimit.py \
  tests/test_prompt_sanitize.py \
  tests/test_deterministic_mode.py \
  tests/test_routing_isolation.py -q
```

## Completion Check

- Can an LLM change graph routing? Why or why not?
- Which layers run before source reaches an LLM?
- What does deterministic mode intentionally omit?
- Where is an injection detection preserved for later audit?

## Intentionally Out of Scope

- Prompt wording optimization and model selection.
- Reliability fitting and static-tool internals.
- HTTP server and database implementation.
