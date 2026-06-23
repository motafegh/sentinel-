# RAG — Dedicated Build Plan (separate from the redesign)

**Status:** planning only. Split out from `01_MASTER_PLAN.md` per Ali's call
(2026-06-21): *"right now we have no RAG at all, literally it's just a name and even
worse it makes us misleading maybe — write down a complete separated plan to fully
build it at its time."*

**Why separate:** RAG is not a bug fix or a tweak to the existing pipeline — it's a
data-acquisition + indexing sub-project with its own lifecycle (sourcing, licensing,
parsing, quality control, refresh cadence). Bolting it into the redesign workstreams
would mix a fast correctness cleanup with a slow data project. This doc is where the
real RAG lives when its time comes.

---

## What "RAG" is, in one paragraph (so the plan is self-contained)

RAG = Retrieval-Augmented Generation. The idea: before asking the LLM to reason
about a contract, *retrieve* the most similar known real-world vulnerabilities from
a knowledge base and put them in the prompt, so the model reasons "this looks like
the Euler hack" instead of from generic training memory. It only works if the
knowledge base is (a) real, (b) relevant to the contract being audited, and (c)
retrievable by actual similarity — none of which is currently true.

## Honest current state (the thing this plan replaces)

- **726 real chunks** from DeFiHackLabs (a real GitHub repo of historical hack
  post-mortems) — but in our 2 real test runs it matched ZERO of the test
  contracts' actual content; it's a corpus of *famous DeFi hacks*, not of the
  general vulnerability classes the ML model detects.
- **24 fake chunks** I hand-wrote during Phase A (Code4rena/Sherlock/Solodit/
  Immunefi/SWC placeholders) — not fetched from anywhere, and one directly caused
  a hallucinated verdict. **These get removed immediately** (Redesign Workstream 2).
- **Net:** after removing the fakes, RAG is one narrow real corpus that rarely
  matches. Effectively "no RAG" for the general case, as Ali said.

---

## Step 0 — Decide whether RAG is worth it at all (a real gate, not a formality)

Before any build effort, answer: **does retrieval actually improve verdict quality
over the LLM reasoning from the contract source alone?** This is measurable once
the redesign's evaluation framework (Phase C / C.2) exists — run the benchmark with
RAG on vs. off and compare. If RAG-off is as good or better, RAG is a distraction;
stop here. **Do not build RAG before this gate, or before there's a way to measure
it.** (This is the same "validate before trusting" discipline the 4-eyes and
placeholder-RAG lessons taught.)

---

## Step 1 — Define what we actually want to retrieve

Two distinct knowledge-base types serve different purposes; decide which (or both):

- **A. Canonical weakness definitions** — SWC registry (37 fixed entries),
  per-class descriptions + remediation. Small, fixed, real, license-clean, no
  refresh problem. **Cheapest real win.** Grounds the LLM in "what IS reentrancy"
  rather than "what famous hack resembles this."
- **B. Real-world finding corpus** — Code4rena / Sherlock / Solodit / Immunefi
  audit findings. Large, high-value, but the hard part: sourcing, licensing,
  parsing inconsistent formats, and keeping current. This is the real sub-project.

Recommendation: ship A first (it's nearly free and never misleads), gate B behind
Step 0's evaluation result.

---

## Step 2 — Source acquisition (only if Step 0 says go, for type B)

Per source, resolve BEFORE writing a fetcher:
- **Access path:** public API? bulk dataset export (e.g. a published Hugging Face
  dataset of Code4rena findings)? scrape (last resort, fragile)?
- **License:** can we legally store + use it? (audit findings are often
  CC-licensed or public, but verify per source.)
- **Format:** what fields exist, how consistent, what's the parse target.
- **Volume + refresh:** one-time export vs. ongoing ingestion.

Only after these four are answered does a fetcher get written. The Phase A fetcher
*interfaces* are reusable (real, tested code) — they just need real data behind
them instead of placeholders.

---

## Step 3 — Relevance, not just retrieval (the lesson from the hallucination)

The current index retrieves by embedding similarity and returns the top-K no matter
how weak the match. The Multicall hallucination happened because a barely-related
chunk was retrieved with full confidence and the LLM treated it as evidence about
THIS contract. Requirements for the rebuild:
- **A relevance floor:** below a similarity threshold, return NOTHING rather than
  the least-bad irrelevant chunk. An empty retrieval is correct when nothing
  matches.
- **Per-class filtering:** retrieve within the vulnerability class(es) the ML/tools
  actually flagged, not globally.
- **Prompt labeling:** retrieved content is always framed as "general reference,
  may not apply to this contract" — never as a finding about the audited code.
  (This labeling fix partially shipped 2026-06-21; keep it.)

---

## Step 4 — Indexing & refresh infrastructure

Already exists and works (`build_index.py`: FAISS vector index + BM25 keyword index
+ Reciprocal Rank Fusion, atomic writes, checksums). This part is NOT the problem —
it's solid. The rebuild reuses it as-is; only the *data flowing into it* changes.
Add: a scheduled refresh if type-B sources are live-ingested.

---

## Step 5 — Validate, then wire in

Re-run Step 0's evaluation with the real corpus. Only promote RAG into the live
debate/narrative prompts if it measurably helps. Otherwise keep it built but
disabled, documented as "available, not enabled — didn't beat no-RAG on the
benchmark."

---

## Sequencing relative to the redesign

- **Now (part of Redesign WS2):** remove the 24 fakes, neuter the placeholder
  fetchers to no-ops. RAG becomes "DeFiHackLabs only," honestly labeled.
- **Blocked until Redesign Phase C/C.2 exists:** Step 0's go/no-go gate needs the
  evaluation framework to be answerable.
- **Then, if go:** Step 1 (ship SWC) → Step 2-3 (real type-B sourcing, the actual
  sub-project) → Step 5 (validate).

**Effort honesty:** Step 1 (SWC) is small. Steps 2-3 (real audit-finding corpus)
are a genuine multi-week data project, and the bulk of the cost is sourcing/
licensing/parsing, not code. Do not underestimate it by looking at the existing
fetcher code — that code is the easy 10%.
