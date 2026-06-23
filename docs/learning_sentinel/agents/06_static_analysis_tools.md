# Step 6 — Static Analysis Tools (Slither & Aderyn)

## Why static analysis exists in the pipeline

ML probabilities and LLM debate opinions are both "soft" — wrong-able, sometimes
non-reproducible, can hallucinate. You need at least one HARD source:
deterministic (same input → same output every time), fast, incapable of making
things up. That's static analysis. But both static tools were silently producing
NOTHING for the module's entire history until this week — so the hard foundation
wasn't actually there.

## "Static" defined

Analyzing code WITHOUT running it — reading structure/source only (like
proofreading an essay vs. performing it). Opposite = "dynamic" (actually execute
and watch). Static = fast + safe (never runs untrusted code) but only catches
structurally-visible bugs, not runtime-only ones.

## Slither (primary, Python lib, by Trail of Bits)

Compiles Solidity, builds an internal model, runs ~100 "detectors" — each a small
specialized check for one bug pattern (`reentrancy-eth`, `tx-origin`,
`unchecked-lowlevel`, ...).

Used by two nodes:
- `static_analysis` (deep path): runs SCOPED — filters ~100 detectors down to only
  those relevant to ML-flagged classes (via CLASS_TO_DETECTORS, Step 3).
- `quick_screen` (every contract): own separate "escalate-worthy" detector list.

## Each finding's shape

A dict in `state["static_findings"]`: `tool`, `detector`, `impact`
(High/Medium/Low), `description`, `lines`. The `tool` field is what lets the
consensus vote say "Slither AND Aderyn both found this" — stronger than either alone.

## THE BUG — the core trust-calibration lesson

`slither contract.sol` on the CLI works fine. But agents uses Slither as a Python
LIBRARY: `Slither(file)` directly. Trap: **the `Slither()` constructor registers
ZERO detectors** — the check list starts empty. The CLI has an extra explicit step
(loop all ~100 detector classes, call `register_detector()` on each) that the
library constructor does NOT do for you. Agents skipped it. So every call ran an
empty detector list → found nothing → and because the node fails SOFT (never
crashes, returns [] on any problem), it looked exactly like "checked, contract is
clean." A silent false-clean — the most dangerous bug class for a security tool,
given the false-negative asymmetry (a missed vuln costs millions).

Aderyn (2nd tool, Rust, by Cyfrin) had its OWN version: invoked with a file where
a directory was required + wrong output flag + parser expecting a non-existent JSON
shape. Three compounding mistakes, all swallowed by the same fail-soft pattern.

## How the fix was proven (internalize this method)

Did NOT trust the code or the existing tests — those tests MOCKED the tools
(replaced them with fakes returning canned answers), so they could never catch a
real-tool failure. Instead: ran Slither + Aderyn by hand on a contract we'd read
ourselves and knew had a reentrancy bug, confirmed each found it, then confirmed
the fixed node matched. New tests run the REAL un-mocked tools so it can't silently
regress.

## The permanent takeaway

Static findings are now real + trustworthy (deterministic, verified end-to-end).
But the lesson generalizes: "the tool returned nothing" is only meaningful if
you've verified the tool actually RUNS. A fail-soft "nothing" and a real "nothing"
are indistinguishable from outside — stay suspicious of any silent "nothing" in
this module.

→ You now know: static analysis is the only hard/deterministic evidence source,
it's the heavyweight witness in consensus voting, and it was silently empty until
this week — proving that "no findings" must always be distinguished from "tool
didn't run."
