# SENTINEL R0 containment implementation plan

**Status:** `APPROVED_FOR_LOCAL_IMPLEMENTATION`

**Plan baseline:** `1256d9aab45add9cf2d23fe33aaa944303259012`

**Integration branch:** `codex/r0-containment`

**Authorization:** Ali approved this R0 plan on 2026-07-14 with the existing mandatory acceptance-matrix and measured before/after evidence condition. Approval authorizes local implementation and isolated testing only; deployment and external mutations remain separately gated.

**Mandatory closure condition:** R0 closes only through the approved D2 acceptance matrix and committed, measured before/after evidence.

## 1. Outcome

R0 produces a safe containment release on the current architecture line. It must:

1. stop ML and chain failures from becoming successful evidence;
2. prevent report and archive paths from escaping their job workspaces;
3. make gateway and tool services private and authenticated by default;
4. remove raw signing authority from analysis/MCP processes;
5. disable misleading legacy V2 writes by default and make any explicitly enabled compatibility path truthful and fail closed;
6. block DATA, teacher, proxy, proof, and promotion workflows when their complete content is not authenticated before load;
7. preserve old records and local artifacts as explicitly legacy/untrusted without deleting or reinterpreting them.

R0 is not permission to claim production readiness. Durable job control remains R2; complete deterministic release types remain R1; replay-safe V3 identity, quorum, governance, and on-chain finality remain R3; scientific and operational acceptance remains R4.

## 2. Non-negotiable execution contract

- No runtime source, test, configuration, artifact, circuit, or contract change begins before plan approval.
- Each package has a separate branch, focused plan section, test set, measurement record, rollback unit, and review commit. The integration branch accepts a package only after its package gates pass.
- Before evidence is captured from the immutable baseline before that package's first code change. After evidence reruns the same command, fixture, environment manifest, and assertion against the candidate commit.
- A unit-test pass is necessary but cannot close a security, scientific, reproducibility, or operational row by itself.
- No threshold, timeout, capacity, rate, body, storage, confirmation, or gas number changes without a versioned measurement record and Ali's explicit decision on the proposed value.
- Production mode never auto-enables mock transport, dotenv override, untrusted artifacts, wildcard binding, unauthenticated access, or legacy submission.
- Rollback may disable a capability or select the last safe image/config. It may never restore silent mock fallback, public unauthenticated mutation, raw keys in MCP, or false successful transaction status.
- Live deployments, key migration, contract pause/upgrade, model promotion, and external RPC transactions require a separate explicit deployment authorization after code acceptance.

## 3. Locked before state

The authoritative D2 evidence is the initial R0 baseline. Focused probes were repeated from `1256d9aab` without mutating canonical data or services.

| Invariant | Measured before state |
|---|---|
| ML outage truth | A forced connect failure with `_MOCK_MODE=False` returned `has_error=False`, `label=safe`, ten probabilities, a `mock_model_hash_*`, and no mock marker. |
| Gateway authentication | An unauthenticated `POST /audit` returned HTTP 202, had no `WWW-Authenticate` header, and queued work. |
| Report containment | D2's isolated report-root probe wrote outside the root using an address-derived filename. |
| ZIP containment | `../repo_evil/pwned.txt` produced no exception and wrote to the sibling directory in an isolated temporary root. |
| DATA release integrity | The tracked manifest-tamper test passes by design; D2 also measured successful verification after a warmed-cache shard deletion. |
| Proof identity | D2's isolated Anvil probe accepted the same proof/signals for two targets and two model hashes. |
| Transaction truth | D2 measured a nonzero V2 estimate of 1,016,818 gas; the fixed 1,000,000-gas transaction reverted while source would report `submitted`. |
| Service/signer boundary | D2 traced seven wildcard bind sites, no application authentication boundary, and a raw operator key used inside audit MCP submission code. |
| ML artifact trust | Source loads unrestricted pickle before computing/reporting an observed digest; no expected authenticated identity is checked first. |
| Promotion governance | Missing label evidence, thresholds, baseline, or registry comparison can warn or silently skip rather than block promotion. |

Before/after raw records will be committed under a dated R0 evidence package. Secrets, absolute private endpoints, and payload source are redacted; command, commit, environment, result, and artifact digests are retained.

## 4. Containment architecture decisions

### 4.1 Runtime profile and execution status

Introduce a typed runtime profile (`test`, `development`, `production`) and one canonical execution-status contract. Production startup fails on mock mode, missing required endpoints/artifacts, unknown configuration, or dotenv override. Mock results exist only in test/development and always carry `MOCK`; they can never satisfy deterministic, proof, submission, or finality gates.

Every selected dependency/tool records a status from `SUCCEEDED`, `CLEAN`, `DEGRADED`, `FAILED`, `SKIPPED_POLICY`, `UNAVAILABLE`, or `MOCK`, plus attempted/ran, reason code, detail, dependency identity, input/output digest, timing, and attempt. R0 propagates this through state, final report, gateway response, evaluation input, and health/readiness. R1 later completes the canonical evidence/result schema.

### 4.2 Logical identity and storage identity

An Ethereum address is validated and stored only as domain data. A server-generated job ID selects storage. Reports use an isolated per-job directory, contained resolved paths, temporary-file plus atomic replace publication, and explicit persistence status. Address-derived legacy reports remain readable through an index/adapter; new writes never use an address as a filename.

ZIP extraction uses `Path.is_relative_to`-equivalent resolved containment, rejects absolute paths, traversal, links, devices and other special entries, extracts into a temporary workspace, and atomically promotes only a complete extraction. Production extraction requires measured limits for member count, total expanded bytes, per-file bytes, path depth, and compression behavior; no guessed defaults are introduced.

### 4.3 Authentication, service identity, and limits

All gateway and MCP entry points default to loopback. Production refuses non-loopback startup unless an authentication configuration is valid.

The application validates asymmetric JWTs from a configured issuer/audience/public-key or JWKS trust source. Claims map to a principal, tenant, and explicit capability. Scopes are separated for read, analyze, prove, legacy-submit, and admin operations. Health liveness is non-mutating and minimal; readiness detail requires service/admin identity. MCP tool invocation requires service identity. Development bypass is loopback-only, explicit, and cannot start under the production profile.

Body, request-rate, concurrent-job, tenant-storage, LLM-cost, and proof-capacity enforcement reads a versioned limits file. R0 first measures current latency, memory, burst, saturation, and failure behavior; proposed production values stop at a policy decision checkpoint for Ali. Tests may use tiny fixture-only values that are never production defaults.

Every accept/reject/limit/signer decision emits a structured audit event with request, principal, tenant, capability, result, and reason—never raw tokens, source payloads, or signing secrets.

### 4.4 Policy signer and legacy V2 boundary

Remove `SENTINEL_OPERATOR_KEY`, account construction, signing, and arbitrary transaction building from gateway, orchestration, and MCP processes. A minimal private signer process is the sole key owner. Its production key provider is an interface suitable for HSM/KMS; an environment-backed provider is allowed only for isolated development tests.

The R0 signer accepts a versioned `LegacySubmissionEnvelopeV1`, not calldata. The envelope binds chain, registry, target, source digest, teacher digest, proxy bundle digest, proof digest, public-signal digest/layout, normalized-score digest, request/job identity, expiry, and idempotency key. The signer constructs only the allowlisted V2 call, verifies artifact eligibility and all equality constraints, estimates gas under an approved measured policy, uses pending nonce state, checks receipt status and confirmations, and writes a signer decision record. It rejects arbitrary selectors/calldata and all mismatches.

Legacy V2 submission is absent from advertised MCP tools and disabled by default. If separately enabled for a controlled compatibility test, it requires the legacy-submit scope and returns `claim_semantics=legacy_proxy_only`. It must never return or expose a V3-style `verified audit` claim. V1/V2 reads remain available through a versioned adapter and retain their historical semantics.

### 4.5 Authenticated artifact descriptors

Introduce canonical JSON descriptors with closed, sorted file inventories, media type, size, and SHA-256 content digest. Descriptor signatures use an offline Ed25519 release key; runtime processes contain only the configured verification key. Descriptor bytes and every member are authenticated before any unrestricted deserialization or promotion.

R0 applies this gate to:

- DATA exports, including semantic manifest core, splits, class order, schema, sources, indices, and exact file set;
- teacher checkpoints and companion thresholds, temperatures, drift baseline, probe evidence, DATA release identity, and code/config identity;
- proxy, ONNX/external data, EZKL settings, compiled circuit, proving/verification keys, verifier identity, public-signal layout, and proof bundle;
- promotion evidence, whose records must name the exact checkpoint/release digest they evaluated.

The filesystem stat cache remains a performance hint only. Additions, deletions, same-size/same-mtime replacements, descriptor mutation, wrong checkpoint, missing probes, or signature mismatch fail before load. Existing exports/checkpoints/proof artifacts are retained unchanged as `legacy/untrusted`, cannot be promoted or submitted in production, and may be opened only in an explicit isolated development migration path whose outputs remain ineligible for finality.

### 4.6 Code ownership boundaries

| Concern | Canonical owner | Boundary rule |
|---|---|---|
| Runtime profile and execution status | AGENTS config/contracts packages | Nodes and transports consume the type; they do not invent local status vocabularies. |
| HTTP authentication and authorization | AGENTS security package | Gateway, MCP ASGI apps, and signer reuse one verifier/policy interface; route files do not parse tokens themselves. |
| Report persistence | Focused AGENTS persistence package | Synthesizer builds report data; persistence owns paths, containment, atomic writes, and migration lookup. |
| Policy signing | Standalone minimal signer package/process | Analysis code can submit typed envelopes only and cannot import key-provider or transaction-signing implementations. |
| Archive extraction | DATA ingestion security helper | Manual connector selects policy and calls the extractor; it does not implement path checks inline. |
| Release/artifact trust | Versioned R0 JSON schemas plus module-owned thin adapters | DATA, ML, AGENTS, and ZKML validate against identical golden vectors without importing one another's heavyweight runtime. |
| Legacy chain compatibility | Versioned AGENTS chain adapter | V1/V2 reads and opt-in V2 submission semantics remain isolated from future V3 names/types. |

New files remain focused below the Rule 5A size boundary. Existing large transport or submission files are reduced by moving policy, persistence, artifact, and signer responsibilities to their owners instead of appending more branches.

## 5. Work packages

### R0.0 — Evidence harness and configuration boundary

**Branch:** `codex/r0-0-evidence-harness`

**Purpose:** make every later result reproducible and comparable.

Implementation:

- add the dated evidence package, command manifest, environment redaction, JSON result schema, and matrix row IDs;
- add typed runtime profile/config loading with production unknown-key and dotenv-override rejection;
- preserve the baseline probes above as executable isolated tests;
- add a matrix coverage validator that fails when an R0 invariant lacks before and after records.

Exit gate: the harness reproduces the known failing baseline and cannot mark a row passed without two evidence records, test references, candidate commit, and reviewer decision.

### R0.1 — Fail-closed evidence and dependency health

**Branch:** `codex/r0-1-fail-closed-evidence`

**IDs:** `D2-AGT-001`, containment slices of `D2-AGT-004`, `D2-AGT-012`, `D2-AGT-016`.

Implementation:

- add the canonical execution-status type and validators;
- remove inference timeout/request-error mock fallback in live transport;
- remove audit RPC absence/init-failure auto-mock behavior;
- require prediction/chain result provenance and reject missing, mock, malformed, or degraded evidence at orchestration, report, evaluation, proof, and submission gates;
- preserve complete tool/dependency status in final report and gateway response;
- split liveness from readiness and report `live`, `degraded`, `mock`, or `unavailable` truthfully.

Required tests/evidence:

- connection refusal, timeout, HTTP failure, malformed body, startup-init failure, explicit test mock, and recovery;
- state → report → gateway/eval preservation;
- proof/submission refusal for mock/degraded/missing status;
- before/after assertion changes from plausible prediction plus `ran=true` to an explicit unavailable/terminal result with no probabilities, proof, or finality output.

Rollback: disable the affected dependency/capability and report unavailable; never restore transparent fallback.

### R0.2 — Filesystem and archive containment

**Branch:** `codex/r0-2-filesystem-containment`

**IDs:** `D2-AGT-002`, `D2-DATA-001`.

Implementation:

- validate canonical chain addresses and keep logical identifiers out of paths;
- create job-scoped report/visualization workspaces with resolved containment and atomic publication;
- add a legacy address-report lookup adapter without legacy address-named writes;
- harden ZIP entry validation and special-file rejection;
- measure accepted archive characteristics, propose versioned limits, and enable production extraction only after the policy checkpoint.

Required tests/evidence:

- `..`, sibling-prefix, absolute POSIX/Windows paths, alternate separators, Unicode normalization/confusables, NUL/invalid names, collisions, pre-existing symlinks, ZIP symlinks, devices/special files, deep paths, excess members, expanded size, ratio, truncation, and interrupted extraction;
- no write outside a temporary root and no partial promoted workspace;
- permission, disk-full/replace, and cleanup failures produce structured persistence/extraction status rather than log-only success;
- concurrent same-address jobs remain separate by job ID.

Rollback: disable disk report/visualization publication and ZIP ingestion. Preserve API result delivery and existing read-only legacy reports.

### R0.3 — Authenticated services and signer isolation

**Branch:** `codex/r0-3-service-signer-boundaries`

**IDs:** `D2-X-001`, `D2-AGT-011`, `D2-ZKC-014`.

Implementation:

- default every service host to loopback and add non-loopback startup guards;
- add JWT authentication, tenant/capability authorization, service identity, structured audit logging, and sanitized liveness/readiness;
- benchmark admission/resource behavior and stop for approval of derived limits before changing production policy numbers;
- enforce approved body/rate/concurrency/storage/cost/proof limits from versioned configuration;
- create the isolated policy-signer service and remove all raw-key access from analysis/MCP code and environment examples.

Required tests/evidence:

- 401 missing/invalid/expired/wrong issuer or audience; 403 wrong scope/tenant; allowed request; service-token separation; bypass and forwarded-header attacks;
- body, burst, concurrent, tenant-storage, and saturation behavior against the approved limits;
- source/config/process environment scan shows no signing secret in analysis services;
- signer rejects arbitrary calldata, wrong chain/registry/target/model/proxy/proof/signals/job/expiry and ineligible artifacts;
- non-loopback production startup fails without valid auth configuration.

Rollback: bind loopback and disable mutations/signer. Never restore public unauthenticated routes or place a key back in MCP.

### R0.4 — Legacy write containment and truthful transaction states

**Branch:** `codex/r0-4-legacy-write-containment`

**IDs:** containment slices of `D2-ZKC-001`, `D2-ZKC-002`; direct closure of `D2-ZKC-003`.

Implementation:

- remove/disable the MCP submission entry point by default and label the opt-in compatibility path legacy/proxy-only;
- use isolated per-job proof workspaces and authenticated proof/proxy descriptors;
- require equality of requested, ML-returned, descriptor, proof, and signer identities;
- submit only through the policy signer;
- estimate gas under the approved policy, use current fee fields, pending nonce state, single-flight containment until R2's durable nonce allocator, idempotency key, and explicit built/signed/broadcast/mined/finalized/failed states;
- require `receipt.status == 1`; a revert, timeout, replacement, or confirmation failure cannot become submitted/finalized.

Required tests/evidence:

- cross-target/model/chain/registry/job/proof/signal replay cannot create a V3-style verified claim;
- missing identity or status blocks proof and submission;
- fixed-gas behavior is absent; estimated-gas and deliberate status-zero receipts surface failure;
- duplicate request is idempotent; pending/mined/finalized remain distinguishable;
- no live transaction is sent by the acceptance suite; local isolated chain only.

Rollback: disable legacy writes and retain read compatibility. R0 does not roll back to direct MCP signing.

### R0.5 — Unsafe release and promotion freeze

**Branch:** `codex/r0-5-artifact-release-freeze`

**IDs:** `D2-DATA-002`, `D2-ML-004`, `D2-ML-005`, containment slice of `D2-ML-013`.

Implementation:

- implement signed closed-inventory descriptors and pre-load verification;
- make DATA verification structured and fail closed; deprecate the ambiguous boolean-only release gate;
- authenticate teacher/proxy/proof bytes before any unsafe load;
- make promotion registry errors and all missing/mismatched evidence terminal;
- bind behavioral probes, label quality, thresholds, temperatures, drift baseline, DATA release, source commit, and checkpoint digest to the candidate;
- preserve existing local artifacts as legacy/untrusted and produce no automatic deletion or conversion.

Required tests/evidence:

- deletion, addition, manifest mutation, index reorder, same-size/same-mtime replacement, truncated file, wrong checkpoint, wrong descriptor, bad signature, missing probe, probe from another checkpoint, missing threshold/temperature/baseline, and registry-unavailable cases all fail before load/promotion;
- malicious-pickle sentinel proves rejected bytes are not deserialized;
- exact descriptor/signature golden vectors and deterministic inventory order;
- a complete synthetic signed bundle passes without requiring private production artifacts.

Rollback: keep promotion and production load disabled; select only a previously accepted complete descriptor. Never treat the current unbound bundle as trusted.

### R0.6 — Integration, compatibility, and wave closure

**Branch:** `codex/r0-6-containment-integration`

**Purpose:** prove the packages compose without overstating later-wave completion.

Implementation and evidence:

- run focused package suites, full available DATA/ML/AGENTS/ZKML/Foundry suites, secret scans, dependency/config validation, and a clean-worktree/fresh-start containment rehearsal;
- compare every R0 before/after probe, suite delta, and unavailable prerequisite explicitly;
- exercise production startup with unavailable ML/RPC/artifact/auth/signer dependencies and prove fail-closed readiness;
- exercise development/test mock mode and prove its output cannot cross report/proof/submission/finality gates;
- validate migration adapters and rollback drills;
- append an R0 closure section to the approved acceptance matrix without rewriting its D2 baseline history;
- keep partial-scope findings open for R1/R3 and link their R0 containment evidence.

Exit gate: every R0 row below is `PASS` with immutable evidence, or R0 remains open. `BLOCKED`, `SKIPPED`, `UNAVAILABLE`, historical-only, or local-untracked evidence cannot close the wave.

## 6. Acceptance mapping

| Mandatory R0 matrix invariant | Package | Closure evidence |
|---|---|---|
| ML/chain outage never becomes successful evidence | R0.1 | Fault injection preserves explicit status through report/API/eval; no proof/finality output. |
| Report path cannot escape workspace | R0.2 | Traversal/symlink/absolute/Unicode/collision suite; zero outside writes. |
| Archive extraction cannot escape workspace | R0.2 | ZIP traversal/link/special/limit/interruption suite; zero outside or partial promoted writes. |
| Dataset release commitment binds semantics and exact files | R0.5 | Closed inventory and signed descriptor mutation/add/delete/cache tests fail before use. |
| Public mutation/expensive routes require auth, scope, tenant, and measured limits | R0.3 | 401/403/tenant/limit/saturation evidence under approved versioned policy. |
| Analysis process has no raw signing key | R0.3 | Source/config/process scan plus policy-signer rejection suite. |
| Proof cannot support a cross-identity V3-style verified claim | R0.4 | Cross-domain replay matrix plus default-disabled/legacy-only semantics. Full V3 replay prevention remains R3. |
| Failed/reverted transaction cannot be reported submitted | R0.4 | Estimate/receipt/status-state tests and isolated local-chain evidence. |

## 7. Measurement and review artifacts

The R0 evidence package will contain:

- `baseline_manifest.json`: baseline/candidate commits, dirty-state guard, environment and dependency versions;
- one `before.json` and `after.json` per matrix invariant with identical probe IDs and commands;
- raw focused/full test summaries with pass/fail/skip/error counts and classifications;
- security adversarial results for auth, signer, path, archive, artifact, and replay cases;
- resource measurements used to propose capacity/archive/gas policies;
- migration and rollback rehearsal results;
- an acceptance ledger mapping every row to evidence digest, commit, reviewer, and decision;
- a limitations ledger retaining R1–R4 blockers.

Evidence is committed before any package is merged. Any changed probe, fixture, environment, or assertion creates a new measurement series rather than overwriting the baseline.

## 8. Compatibility and migration

- Public response additions are versioned and additive where possible. Missing status from an old producer is treated as unavailable, not successful.
- New reports are job-keyed; legacy address-keyed reports are read-only through an adapter/index.
- V1/V2 chain history remains readable and is labeled by its actual semantics. No historical record is upgraded in meaning.
- Existing DATA/model/proof artifacts remain in place but are legacy/untrusted. Production refuses them until a complete authenticated bundle exists.
- Development mock and legacy-artifact paths require explicit non-production configuration and produce ineligible outputs.
- The signer can be deployed dark and tested against a local chain before any key migration or external transaction authorization.

## 9. Review decisions requested

Approval of this plan authorizes implementation and isolated tests on the R0 package branches. It does **not** authorize deployment, live-chain writes, key movement, contract administration, artifact deletion, or model promotion.

During implementation, work must pause for Ali at these policy checkpoints:

1. derived archive extraction limits;
2. derived gateway/body/rate/concurrency/storage/cost/proof limits;
3. derived transaction gas/confirmation policy;
4. any scientific threshold or model/calibration behavior change discovered while containing the release path.

After approval, execution order is:

`R0.0 → R0.1 → R0.2 → R0.5 → R0.3 → R0.4 → R0.6`

Implementation remains sequential unless Ali separately authorizes parallel agent work.

## 10. Review record

- Decision: `APPROVE_R0_PLAN`
- Reviewer: Ali
- Date: 2026-07-14
- Approved scope: local implementation and isolated tests for R0.0 through R0.6
- Mandatory condition retained: every package and the R0 wave close only through the approved acceptance matrix and committed, measured before/after evidence
- Not authorized: deployment, live-chain writes, key movement, contract administration, artifact deletion, or model promotion
