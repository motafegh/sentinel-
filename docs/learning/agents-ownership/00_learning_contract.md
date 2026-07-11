# Agents Module Ownership — Learning Contract

## Purpose

This directory is a navigation system for owning `agents/`. It identifies what to learn, which executable files establish the truth, and how to verify understanding. It is not a tutorial, architecture specification, or implementation plan.

## Working Rules

- Treat Python source and tests as authoritative; use prose only as supporting context.
- Learn one bounded module at a time and trace its inputs, outputs, failure behavior, and downstream consumers.
- Do not attempt to memorise APIs, prompts, detector names, or every helper function.
- Before changing a source file, identify its focused tests and the state fields it reads and writes.
- Record unanswered questions in the progress log before moving to a different topic.

## Ownership Standard

An item is owned when Ali can explain, using the source:

1. its responsibility and boundary;
2. its inputs and outputs;
3. its failure/degraded behavior;
4. the tests that protect it; and
5. how a safe change would be verified.

## Learning Sequence

| Order | Artifact | Ownership target | Status |
|---|---|---|---|
| 1 | `01_request_lifecycle.md` | Follow one audit request through the graph | Ready |
| 2 | Future artifact | LangGraph state, reducers, and routing policy | Not created |
| 3 | Future artifact | Evidence emission, reliability, and `fuse()` | Not created |
| 4 | Future artifact | ML, static-analysis, RAG, and formal-tool boundaries | Not created |
| 5 | Future artifact | LLM, prompt security, and deterministic mode | Not created |
| 6 | Future artifact | MCP, gateway, persistence, tests, and safe changes | Not created |
| 7 | Future artifact | End-to-end ownership review | Not created |

## Progress Log

Add entries only after a learning session.

| Date | Artifact | Owned | Open questions | Evidence reviewed |
|---|---|---|---|---|
| — | — | — | — | — |
