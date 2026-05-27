
---

## The Fundamental Problem the Agents Module Solves

Start here: what does the ML module produce, and what does a real user actually need?

ML output: 10 probabilities. `{reentrancy: 0.87, timestamp: 0.12, ...}`

What a developer or auditor needs:
- **Where** exactly in the code is the problem (line, function, specific pattern)
- **Why** this is dangerous, not just that it scores high
- **How** to fix it, with reference to canonical patterns
- **How confident** to be, given model uncertainty
- **Precedent** — has this exact pattern appeared in a real exploit?
- **Priority** — if 3 contracts are flagged, which one first?

The gap between "probability vector" and "actionable audit finding" is too large for a rule-based pipeline to cross. It requires reasoning, synthesis across multiple signals, retrieval from external knowledge, and iterative investigation. That is exactly the space where LLM agents operate.

This is the agent module's core value proposition: **it converts statistical signals into reasoned conclusions.**

---

## Agent Paradigm Decision

There are three main patterns, and the choice matters:

**Single monolithic agent:** one LLM loop that does everything — fetches contract, runs ML, explains findings, suggests fixes. Simple to build. Fails at scale and specialization. The context window fills with everything simultaneously, which degrades reasoning quality on each individual task.

**Pipeline (sequential, fixed steps):** predefined stages that always run in order (triage → analyze → explain → fix). Fast and predictable. But can't adapt — a contract that is clearly safe still runs through all stages, and a deeply ambiguous case doesn't get extra investigation.

**Multi-agent with orchestrator:** a coordinator agent that decides *which* specialist agents to invoke, in *what order*, with *what inputs*. Each specialist is narrow and deep. The orchestrator reasons about the audit goal; specialists reason about their specific domain.

**The right choice for SENTINEL: multi-agent with orchestrator.** Here's why:

The workload is fundamentally heterogeneous. An obvious false positive needs 2 seconds of triage. A contract with high eye disagreement between GNN and CodeBERT needs deep attribution analysis, similarity search, and historical context. A fixed pipeline treats both identically, which wastes compute on the former and underserves the latter. Only an orchestrator with the ability to *decide* what depth to apply can handle this correctly.

---

## The Agent Roster

Five specialized agents, one orchestrator. Each has a precise mandate.

---

### 1. Orchestrator Agent

**What it does:** Receives the audit request, reads the triage summary from the ML module, decides which specialists to invoke, assembles their outputs into a unified finding, and manages the audit session state.

**What it knows:** The full ML output for every contract in scope — probabilities, eye breakdown, JK phase weights, confidence signals. It does not do analysis itself. It reads signals and dispatches.

**Key decisions it makes:**
- Which contracts in a batch need deep analysis vs fast pass?
- Which vulnerability classes within a flagged contract are worth investigating?
- Has eye disagreement exceeded the threshold that justifies the Debate pattern?
- Is the finding strong enough to surface, or should it be escalated to human review?
- In what order do findings get presented to the user?

**The reasoning principle behind this design:** the orchestrator is a *reasoning* agent, not an *analysis* agent. You want it focused on strategy, not drowning in technical details. This is the same principle behind why CEOs don't write code — cognitive separation of strategic reasoning from execution.

---

### 2. Triage Agent

**What it does:** Rapid first pass. For every contract in scope, it reads the probability vector and assigns one of three categories: clear-safe (no further analysis), investigate (pass to analysis agent), escalate (high-stakes, human review needed).

**Its threshold logic is not fixed.** The orchestrator passes it the context: is this a production protocol with $100M TVL, or a student project? The same 0.65 reentrancy score means different things in different contexts. The triage agent reasons about *deployment risk*, not just raw probability.

**What it does NOT do:** access the code, run tools, retrieve history. Pure classification from the existing signal.

**Why it exists as a separate agent:** speed and cost. Most contracts in a large batch will be clear-safe. You do not want to pay for LLM reasoning on those. The triage agent is cheap and fast; it filters before the expensive specialists engage.

---

### 3. Analysis Agent

**What it does:** Deep investigation of a single flagged vulnerability in a single contract. This is the technical core of the system.

**Tools it uses:**
- `get_node_attributions(contract_hash, vuln_class)` — GradCAM output: which nodes were most important for this prediction?
- `get_eye_disagreement(contract_hash)` — did GNN and CodeBERT agree? High disagreement means the pattern may be structural but not semantic (or vice versa), which is a specific signal.
- `get_jk_phase_weights(contract_hash)` — which execution phase dominated the GNN's attention? Phase 2 (CONTROL_FLOW) dominating means the vulnerability is in execution order. Phase 1 (structural) dominating means it's about what's connected, not when.
- `find_similar_contracts(gnn_embedding)` — vector similarity search: what are the 5 most structurally similar contracts in the audited library?
- `get_source_location(contract_hash, node_id)` — map a graph node ID back to a source line

**What it produces:** A structured evidence package — a set of claims, each backed by a specific tool output. Not a narrative yet. Raw evidence.

**The key insight behind the evidence package design:** separate evidence collection from narrative generation. The analysis agent should be a careful investigator, not a storyteller. The storytelling is the Explanation Agent's job. Mixing them produces confident-sounding but sloppy explanations.

---

### 4. Explanation Agent

**What it does:** Takes the evidence package from the Analysis Agent and converts it into a human-readable audit finding with three components: what the vulnerability is, where specifically it occurs (line references from source location tool), and why the evidence supports this conclusion.

**It has access to the RAG knowledge base.** Before writing the explanation, it retrieves:
- The canonical description of the vulnerability class
- The most relevant historical exploit that matches this pattern
- The standard fix pattern

**The output format matters.** It should produce:
- A severity rating (with reasoning, not just a number)
- A short executive summary (for non-technical stakeholders)
- A technical finding with line references (for the developer)
- The evidence chain: GNN said X because of nodes A, B, C; CodeBERT said Y because tokens P, Q, R

**Why the historical exploit reference is non-negotiable:** humans don't update on probabilities, they update on stories. "This pattern matches The DAO exploit, which lost $60M in 2016" lands differently than "reentrancy: 0.87." The RAG retrieval is not just context — it is the persuasion mechanism.

---

### 5. Fix Agent

**What it does:** Given a confirmed finding, proposes a specific code-level remediation.

**This is the most dangerous agent to design poorly.** If the Fix Agent hallucinates a "fix" that actually introduces a new vulnerability, or removes the pattern without understanding why it's dangerous, it causes harm. The design must reflect this.

**Constraints on the Fix Agent:**
- It should know the canonical fix patterns per vulnerability class: CEI for reentrancy, using `block.number` rather than `block.timestamp` for timing, proper access control patterns.
- It should be conservative: propose the minimal change, not a full refactor.
- It should explain *why* the proposed fix removes the vulnerability, not just provide a diff.
- It should flag cases where the fix is non-obvious — not all reentrancy vulnerabilities have the same safe fix. Context matters.

**What it does NOT do:** verify its own fix. That is a separate concern.

---

### 6. (Optional) Debate Agent Pair

This is the most architecturally interesting optional component. Invoke it when eye disagreement is high, or when the probability is in the ambiguous 0.45–0.70 range.

**The pattern:** two agents, same evidence package, adversarial mandates.

- Agent Red: "Make the strongest possible case that this contract IS vulnerable. Find every piece of supporting evidence."
- Agent Blue: "Make the strongest possible case that this contract IS NOT vulnerable. Find every piece of counter-evidence."

Their arguments are presented to the Orchestrator, which synthesizes the disagreement into a *confidence-adjusted* finding.

**Why this is better than a single agent for ambiguous cases:** a single agent anchors on the initial signal. If the ML model said 0.72, the agent will probably confirm it because the framing primes confirmation. The debate pattern forces explicit consideration of the contrary case, which surfaces the actual uncertainty.

**The tradeoff:** expensive. Two full agent runs plus synthesis. Reserve for genuinely ambiguous cases where the cost of a false positive or false negative is high.

---

## Memory Architecture

The agents need four distinct kinds of memory. These are not implementation details — they are architectural decisions that determine the system's long-term value.

**In-context memory (ephemeral):** everything that happened in this audit session. Which contracts were analyzed, findings so far, tool outputs. Lives in the LLM context window. Expires when the session ends. Cheap and fast but not persistent.

**Episodic memory (session-level persistence):** audit history for a project. "We audited this repository last month and flagged these findings. This PR changes these three functions. Which findings are still relevant?" Stored in a database, indexed by project and contract hash. Enables incremental auditing — you don't re-analyze unchanged code.

**Semantic memory (knowledge base):** the RAG layer. Vulnerability descriptions, exploit post-mortems, fix patterns, EIPs, audit reports. This is the "what does this vulnerability mean" layer. Stored in a vector database. Queried by embedding similarity. Updated independently of the models.

**Procedural memory (baked in):** how to perform an audit, what steps to take for each vulnerability class, when to escalate. This lives in the agent system prompts and tool schemas. It is the accumulated operational knowledge of the system. Changes require deliberate prompt engineering, not data updates.

**The key architectural insight:** most "memory" discussions in agent design conflate these four. They have different storage requirements, different update frequencies, and different retrieval mechanisms. Treating them as one thing leads to systems that try to put exploit history in the context window, which is both slow and expensive.

---

## Workflow Design — What Happens When You Submit a Contract

Let me trace a single audit from start to finish.

**1. Ingestion.** Contract code arrives. Source can be: raw Solidity text, contract address (agent fetches bytecode + verified source from Etherscan via tool), Git repository path, or PR diff.

**2. ML pre-computation.** Before any agent runs, the ML module processes the contract and produces the full payload: 10 probabilities, per-eye scores, JK phase weights, GNN embedding, cross-attention weights. This is synchronous — agents block until the payload is ready. This is correct because every agent downstream depends on it.

**3. Triage.** The Orchestrator invokes the Triage Agent with the full payload. Triage classifies: clear-safe, investigate, escalate. For a batch of 100 contracts, this might classify 70 as clear-safe and 30 as needing investigation. Cost: minimal.

**4. Dispatch.** For each "investigate" contract, the Orchestrator decides which vulnerability classes to pursue (not all 10 — only those above threshold) and invokes the Analysis Agent per class. These can run in parallel — no dependency between analyzing reentrancy and analyzing timestamp in the same contract.

**5. Evidence synthesis.** Each Analysis Agent returns an evidence package. The Orchestrator checks if any package has high enough eye disagreement to trigger the Debate pattern. If yes, invokes the Debate Agent pair. If no, passes directly to the Explanation Agent.

**6. Explanation and report generation.** The Explanation Agent pulls from the RAG knowledge base and produces structured findings. The Fix Agent generates remediation for each confirmed finding.

**7. Output assembly.** The Orchestrator assembles all findings into a unified audit report. Ranking by severity, deduplication if multiple vulnerability classes overlap, confidence calibration from eye disagreement signals.

**8. Feedback loop.** User reviews the report and marks findings as confirmed or false positive. These go to the active learning queue for the ML module. The episodic memory is updated with the session results.

---

## The Cross-Contract Analysis Capability

This is the most important capability the Agents module enables that the ML module cannot provide by design.

The ML module analyzes one contract in isolation. But real DeFi vulnerabilities often require understanding relationships between contracts.

The agent can:
1. Analyze Contract A individually
2. Identify that Contract A calls Contract B (via the graph edges or from source analysis)
3. Analyze Contract B individually
4. Reason about the interaction: "Contract A passes user-controlled data to Contract B's privileged function. Contract B has no validation. Even if neither contract is individually vulnerable, the combination is."

This is not something the GNN can do — its Phase 3 (REVERSE_CONTAINS) operates within a contract's scope, not across contract boundaries. The agent is the layer that crosses that boundary.

**Implementation concept:** build a dependency graph of contracts in scope at the agent level. Not a GNN — a simple directed graph maintained in the Orchestrator's state. Nodes are contracts, edges are CALL relationships. The Orchestrator uses this map to decide which contracts to analyze together and to reason about cross-contract interaction patterns.

---

## Interface Design

**Inbound interfaces — how contracts enter the system:**

The agent must handle multiple submission modes because different users have different workflows:
- **Single contract:** paste code or contract address. Direct analysis.
- **Project batch:** a directory of Solidity files. The agent builds the inter-contract dependency graph first, then audits in topological order (dependencies before dependents).
- **PR diff integration:** GitHub webhook → only the changed functions need full re-analysis; unchanged functions reuse cached results from episodic memory.
- **Continuous monitoring:** deployed contract address → re-analyze on a schedule or when the on-chain monitoring module detects unusual transaction patterns.

**Outbound interfaces — what the agent produces:**

Different consumers need different formats:
- **JSON structured report:** machine-readable for downstream tools, dashboards, integration with CI/CD. The canonical format that all others derive from.
- **Markdown report:** human-readable, can be posted directly to GitHub PR reviews or documentation.
- **Inline annotations:** map findings to specific lines, suitable for IDE integrations.
- **Executive summary:** single paragraph, severity rating, top 3 findings. For stakeholders who do not read code.

**The single source of truth principle:** the JSON report is generated first. All other formats derive from it via rendering. Never generate the human-readable report first and try to extract structured data from it — LLMs hallucinate when reversing that direction.

---

## Confidence Architecture — Richer Than Probabilities

The raw probability score is a poor communication mechanism for several reasons: it doesn't convey what kind of evidence supports it, it doesn't account for model uncertainty, and humans do not naturally calibrate against probability scores.

The agent should produce a multi-dimensional confidence signal:

**Evidence breadth:** how many independent signals point to this vulnerability? GNN said yes, CodeBERT said yes, 3 similar contracts in the library were exploited, JK Phase 2 dominated. That is 4 independent signals. One signal alone with high probability is weaker than 4 moderate signals.

**Model agreement:** did all three eyes agree? GNN=0.91, Transformer=0.89, Fused=0.87 is stronger than GNN=0.91, Transformer=0.34, Fused=0.72. High inter-eye disagreement should lower the stated confidence regardless of the maximum eye's value.

**Historical precedent:** was a similar contract actually exploited? A finding that maps to a known CVE or post-mortem is qualitatively different from a novel pattern.

**Coverage confidence:** was the contract fully analyzable? If the contract was 2,400 tokens long and required 5 windows, and the model has seen very few 5-window contracts in training, the prediction is less reliable. This is epistemic uncertainty from out-of-distribution inputs.

These four dimensions together produce a richer confidence signal than any single number. The Explanation Agent should always communicate at least two of them.

---

## Escalation Tiers — Managing Cost

Not every contract justifies a full multi-agent audit. Design tiers explicitly:

**Tier 0 — Fast pass:** all probabilities < 0.35 and all eyes agree. The Triage Agent returns "clear-safe" and no specialist is invoked. Produces a one-line negative finding. Cost: one LLM call (triage).

**Tier 1 — Standard analysis:** at least one probability > 0.35. The Analysis and Explanation agents run. No debate, no RAG-heavy retrieval unless confidence is ambiguous. Cost: 3–4 LLM calls.

**Tier 2 — Deep analysis:** probability > 0.65 OR high eye disagreement. Full pipeline including RAG retrieval and Fix Agent. Cost: 6–8 LLM calls.

**Tier 3 — Human escalation:** high TVL protocol OR the Debate agents significantly disagree OR the finding matches a critical CVE class. System flags for human expert review and does not generate an auto-finding. The agent produces a briefing document for the human, not a conclusion.

**The principle behind tiers:** the cost of a false positive in Tier 3 (a human expert falsely told a secure protocol is critical-vulnerable) is much higher than in Tier 0. Calibrate the depth of analysis to the cost of being wrong.

---

## Technology Stack Decisions

**Agent framework:** Given this project already uses Claude Code (Anthropic SDK) and the MCP layer is designed for Claude's tool-use protocol, the natural choice is to build agents on top of the Anthropic SDK's tool-use primitive directly, or use LangGraph for orchestration state management if the workflow complexity grows. LangGraph is worth knowing because it treats agent workflows as explicit state machines — you can visualize exactly which agent is in which state, which is valuable for debugging.

**Message passing between agents:** define a strict schema for the payload that moves between agents. The evidence package from Analysis Agent to Explanation Agent should be a typed data structure, not free text. This prevents the Explanation Agent from hallucinating details that weren't in the evidence.

**Idempotency:** the same contract hash should produce the same finding if run twice (given the same model weights). Cache ML payloads by contract hash. Cache agent outputs by (contract_hash, model_version) pair. This is critical for incremental auditing in the PR diff use case.

**Observability:** every tool call the agent makes should be logged with: timestamp, agent identity, tool name, input hash, output hash, latency. This is how you debug why an agent reached a wrong conclusion — you trace its tool call history and find where the evidence diverged from the truth.

---

## The Key Design Tension

Here is the honest architectural tradeoff you need to decide early:

**Autonomous agents vs guided agents.**

Autonomous: the orchestrator decides everything. Given a contract, it determines depth, runs all necessary specialists, produces the report. Zero human intervention. Maximum throughput, but confidence in conclusions is harder to convey and errors can cascade.

Guided: the agent produces a recommendation at each tier and waits for human confirmation before proceeding to deeper analysis. Slower, but every human confirmation point is a quality gate and a training signal.

For a **learning/research project**: guided makes more sense. Human checkpoints produce labeled data, surface model weaknesses, and teach you where the agent's reasoning breaks down. The checkpoints are features, not friction.

For **production at scale**: autonomous makes more sense. Human checkpoints become bottlenecks at 10,000 contracts/day.

SENTINEL today is a learning project. Design for guided, with the explicit intention of making checkpoints configurable so they can be removed as confidence in the system grows. Do not bake the assumption of human availability into the core logic — parameterize it.

---

That's the full architecture of the Agents module with reasoning behind each decision. The structure is:

```
Orchestrator
├── Triage Agent          (classify depth needed)
├── Analysis Agent        (evidence collection per vuln per contract)
├── Debate Agent Pair     (optional, high-uncertainty cases)
├── Explanation Agent     (evidence → human-readable finding)
└── Fix Agent             (confirmed finding → remediation)

Cross-cutting:
├── Memory Layer          (in-context, episodic, semantic/RAG, procedural)
├── Confidence Layer      (evidence breadth, eye agreement, precedent, coverage)
├── Escalation Tiers      (cost vs depth calibration)
└── Interface Layer       (ingest modes, output formats)
```

Where do you want to go deeper — the memory architecture, the cross-contract analysis workflow, the confidence model, or the interface design?