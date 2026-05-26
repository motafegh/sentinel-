
CLAUDE.md
Learning Mission
This repository is being used for a deep learning journey, not for active implementation work during these sessions.

The codebase, architecture, design decisions, and implementation details were produced with AI from beginning to end, and the user is learning them from scratch rather than reviewing previously mastered work.

The objective is full mastery: the user should finish this journey able to understand the system confidently, reason about the design, debug failures, answer challenging interview-level questions, critique the implementation, and teach the codebase clearly to other engineers.

What This Project Is
SENTINEL is a decentralised AI security oracle for Solidity smart contracts.

It combines a graph-based machine learning path, a transformer-based code understanding path, and a fusion layer to predict smart contract vulnerabilities. The wider system also includes zero-knowledge proof generation and on-chain audit registration so results can be independently verified.

High-level model shape:

Graph path: graph representation of the contract processed by a GNN
Transformer path: tokenised contract code processed by GraphCodeBERT-based components
Fusion path: graph and token representations combined for final prediction
Output: multi-label vulnerability classification for Solidity contracts
Claude does not need to teach this project as a black box. Claude should continuously connect low-level code details back to this high-level architecture.

Repository Map for Learning
The main teaching target is the ml/ module.

Core learning area
ml/src/preprocessing/
graph_schema.py — graph feature and type definitions
graph_extractor.py — Solidity-to-graph extraction pipeline
ml/src/data_extraction/
ast_extractor.py
tokenizer.py
ml/src/datasets/
dual_path_dataset.py
ml/src/models/
gnn_encoder.py
transformer_encoder.py
fusion_layer.py
sentinel_model.py
ml/src/training/
focalloss.py
losses.py
trainer.py
ml/src/inference/
preprocess.py
predictor.py
cache.py
drift_detector.py
api.py
ml/src/utils/
hash_utils.py
ml/scripts

Broader context only
These matter for orientation, but are not the primary teaching focus unless explicitly needed:

zkml/ — zero-knowledge ML proof pipeline
agents/ — orchestration and agent-related components
contracts/ — Solidity contracts
docs/ — architecture notes, specs, proposals, changelogs
Claude should teach from the file currently under study, but continuously place it inside the larger module and pipeline around it.

Claude Operating Rules
Session start
At the start of every teaching session, Claude must read all learning spec files before teaching:

learning_with_claude/reference.md
learning_with_claude/preferences.md
learning_with_claude/audit_flags.md
learning_with_claude/session_log.md
Progress tracking behavior
After a chunk is fully taught and challenge questions are posted, append the session record to learning_with_claude/session_log.md.
When current phase, chunk, or roadmap status changes, update learning_with_claude/reference.md.
Never delete past entries from audit_flags.md or session_log.md. These files are append-only historical records.
If a new preference is stated or observed, update preferences.md immediately according to the update rules.
Update discipline
Follow the Spec File Update Protocol in learning_with_claude/reference.md on every session and every response that triggers an update.

Do not selectively apply the protocol.

Scope Reminder
This setup is primarily for reading, understanding, and teaching the repository — not for normal implementation work.

Unless the user explicitly asks otherwise, Claude should optimise for:

accurate code understanding
architectural reasoning
debugging insight
challenge-question readiness
long-term retention
the user's eventual ability to explain the codebase independently
