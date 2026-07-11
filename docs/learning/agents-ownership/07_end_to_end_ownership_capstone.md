# Agents Module Ownership — 07: End-to-End Ownership Capstone

## Ownership Target

Demonstrate safe ownership of the Agents module by tracing, verifying, and explaining one audit without changing production behavior.

## Required Inputs

- One Solidity contract with known expected behavior.
- A working Agents environment and Linux temporary-directory settings.
- The completed artifacts `01` through `06`.

## Capstone Tasks

1. Identify the entry state and expected route before execution.
2. Trace state updates through every executed node.
3. List all `tool_status` entries and explain any degraded result.
4. List all evidence for one reported vulnerability class.
5. Explain the difference between the class's provable and full verdicts.
6. Identify the model hash and whether deterministic mode was enabled.
7. Explain the final recommendation using source-backed state, not prose documentation.
8. Choose one hypothetical safe change and name its affected source files, tests, and validation command.

## Verification

```bash
cd agents
TMP=/tmp TEMP=/tmp TMPDIR=/tmp poetry run pytest tests/ -q
```

## Completion Check

Ali owns the module when he can:

- draw the audit path for fast and deep cases;
- identify every relevant state and evidence boundary;
- distinguish observed evidence from unavailable evidence;
- explain why the final verdict is reproducible or non-reproducible;
- review an AI-proposed change by naming its downstream effects and tests.

## Intentionally Out of Scope

- Refactoring for style alone.
- Training a new ML model.
- Deploying contracts or publishing a production audit.
