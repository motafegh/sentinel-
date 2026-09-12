# Security Policy

SENTINEL is a public smart-contract security research/engineering project under active development. Security reports are welcome, but the repository does not currently represent a production deployment or a formally supported commercial service.

## Supported versions

Security fixes are evaluated against the current `main` branch. Historical branches, archived experiments, superseded plans, and historical model/data artifacts are preserved for auditability and are not independently supported unless an issue in them also affects current code, current documentation, or current artifact handling.

No stable production release is currently declared.

## Reporting a vulnerability

Please do **not** open a public issue containing exploit details, credentials, private keys, secrets, private endpoints, or other sensitive material.

Preferred reporting path:

1. Use GitHub's private vulnerability reporting / security-advisory flow for this repository when that option is available.
2. Include the affected path/component, impact, reproduction conditions, and the smallest proof needed to demonstrate the issue.
3. Do not include unrelated personal data or secrets in the report.

If private vulnerability reporting is not available, open a minimal public issue titled **`Security contact request`** without exploit details or sensitive data. A private reporting path can then be coordinated before technical details are shared.

There is no formal response-time SLA for this personal research project. Reports are triaged according to severity, reproducibility, and whether they affect the current supported state.

## What is in scope

Security-relevant reports may include, for example:

- smart-contract authorization, upgrade, replay, signature, stake, or verifier failures;
- unintended signing or transaction-authority exposure;
- secret, credential, private-key, mnemonic, or private-endpoint leakage;
- prompt-injection or hostile-input paths that bypass deterministic policy/routing controls;
- analyzer/tool failures that are incorrectly represented as successful or clean results;
- provenance, artifact-binding, or evidence-integrity failures;
- unsafe deserialization, command execution, path traversal, or similar host-impacting behavior;
- vulnerabilities in the current gateway, MCP services, ML inference boundary, DATA pipeline, ZKML tooling, or contract integration.

The canonical trust model and current limitations are documented in [`docs/handbook/12_security_and_trust.md`](docs/handbook/12_security_and_trust.md).

## Known boundaries are not vulnerability claims

Documented limitations should not be reported merely because they exist as declared research boundaries. Examples include:

- the retained EZKL configuration still using `check_mode="UNSAFE"` as a known production-assurance blocker;
- no production signer/broadcaster being claimed;
- the live audit MCP being intentionally read-only;
- historical Run12 remaining a historical operational baseline rather than repaired R4 model truth;
- missing or unavailable historical local artifacts in a fresh clone.

A bypass, escalation, incorrect claim, or implementation defect involving one of those boundaries is still in scope.

## Handling leaked secrets

If you discover an actual secret in the repository or its published artifacts:

1. do not copy it into a public issue or discussion;
2. identify the file/path and commit without repeating the secret value;
3. treat the credential as compromised and rotate/revoke it where applicable;
4. report whether the value appears usable and what scope it may expose.

Repository ignore rules are defense-in-depth only; they do not replace credential rotation or secret scanning.

## Good-faith research

Good-faith testing is welcome when it avoids privacy violations, data destruction, service disruption, social engineering, and access to data or systems you do not own or have permission to test.

Please minimize impact, stop once the issue is demonstrated, and provide enough evidence to reproduce the problem safely.
