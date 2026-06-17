Research Synthesis: What's Out There (Beyond the Proposal)
I'll split this into two sections: things genuinely not in your proposal (the new discoveries) and meaningful updates to what's already in there.

NEW — Not in the proposal at all
1. VISION: Counterfactual Augmentation via LLM (AAAI/AIES 2025)
This is the most directly relevant new finding. The proposal talks about CGBC for noisy labels. VISION is a completely different angle: it uses an LLM to generate counterfactual code examples (same structure, opposite vulnerability label) and trains the GNN on those paired examples to break spurious correlations. Results on CWE-20: accuracy jumped from 51.8% → 97.8%. This hits the ExternalBug FP problem at the root — the model that trained on s_Form001 (your confirmed FP) learned spurious structural features. VISION forces the model to learn what actually changes the label. The proposal doesn't have anything like this.

2. GRPO / DAPO — Modern Post-Training Pipeline (2025–2026 mainstream)
Your proposal recommends CPPO (from 2017 ICML). The field has moved significantly past PPO. GRPO (Group Relative Policy Optimization, DeepSeek R1) eliminates the critic network entirely — instead of computing advantages against a value function, it generates K responses per prompt and normalizes within the group. No critic = ~40% less VRAM, more stable training. DAPO (another 2026 variant) adds dynamic advantage clipping. The paper "From SFT to RL for LLM-based Vulnerability Detection" (arXiv:2602.14012) maps this pipeline directly onto vulnerability detection. The insight: SFT cold-start → RL with verifiable rewards (can you correctly identify the vulnerability class?) outperforms either alone. This is a more modern, more stable path than CPPO.

3. KAN (Kolmogorov-Arnold Networks) applied to GNNs (ICLR 2025 + Nature MI 2025)
KANs replace fixed activation functions on nodes with learnable 1D spline functions on edges. A KAN-GNN variant was just published in Nature Machine Intelligence (2025) for molecular property prediction — the same graph-structured, multi-label problem as SENTINEL. The interesting part for you: KAN-based classifier heads or aux heads would be more interpretable (the learned spline functions are visualizable) and the research found in the noisy label paper that DCM (Dynamic Connection Masking) is already compatible with KAN. You could swap your Linear(512→256→9) classifier for a KAN layer and gain interpretability. No one in the smart contract vulnerability space has done this yet.

4. MAVUL: Multi-Agent Vulnerability Detection (arXiv:2510.00317)
Multi-agent system where different specialized agents collaborate via contextual reasoning + interactive refinement. The analogy to SENTINEL's four-eye architecture is direct — your four eyes already function like independent agents. MAVUL's idea is to let them communicate beyond the fusion layer — agents can challenge each other's outputs, request additional information, refine their votes. This is a fundamentally different inference paradigm from the current concatenate-and-classify approach.

5. Test-Time Adaptation (TTA) — Shift-ACT, DynaTTA (2025–2026)
Nothing in the proposal addresses what happens when deployed SENTINEL encounters a distribution shift mid-deployment. TTA methods adapt model parameters using only unlabeled test data at inference time. For SENTINEL this is extremely practical: new Solidity versions, new DeFi patterns, new contract styles will all cause distribution shift. Instead of waiting for Run 14, TTA could adapt the model live. DynaTTA (ICML 2025) estimates distribution shift relative to source and adapts accordingly.

6. Program Slicing + Neural: GraphFVD / SLICEMATE (2025)
The proposal's "multi-task function-level" section treats function-level detection as a research project. These 2025 papers show it's much closer than that. GraphFVD (ScienceDirect 2025) does cross-function program slicing → BERT features → GNN detection. SLICEMATE (arXiv:2507.18957) uses an LLM agent to perform accurate program slicing. For SENTINEL: your PDG builder and CFG builder already exist in data_module/sentinel_data/representation/. Function-level slices are one step away. This is more tractable than the proposal suggests.

7. Continual Learning with I-LoRA / L-MoE (NeurIPS 2025, ICLR 2026)
The proposal mentions EWC (2017) for continual learning. I-LoRA (2025) is much more practical: train new LoRA adapters on new vulnerability types, iteratively merge them without task-order dependency. L-MoE adds a gating network that dynamically composes adapter outputs per input. This is directly applicable to SENTINEL: as new vulnerability classes emerge (or as BCCC ME increases), you don't retrain from scratch — you add a new LoRA adapter and merge.

8. CausalCode: Causal Learning for Source Code Models (ACM SE 2025)
Different from VISION's data augmentation approach. CausalCode modifies the learning process itself using causal inference principles — it identifies causal vs. spurious features during training. The architecture includes a causal feature extractor and a causal intervention module. Published in ACM on Software Engineering 2025 — directly targeting the same ExternalBug-style FP problem SENTINEL has.

9. Graph Foundation Models (GFMs) as Backbone (Survey: IEEE TPAMI 2025)
Instead of training your 8-layer GAT from scratch every run, you could initialize from a pre-trained GFM backbone. The 2025 IEEE TPAMI paper and the comprehensive 2025 survey cover this. The practical path: a GFM pre-trained on diverse graph domains could provide better initial node representations than random init. This is a step beyond graph contrastive pre-training (which pre-trains on your own unlabeled data) — GFMs are pre-trained on massive cross-domain graph corpora.

10. Flow Matching for Synthetic Graph Generation (NeurIPS 2025 — 30+ papers)
Not in the proposal at all. Flow matching is replacing GANs for generative modeling. For SENTINEL: you could train a graph flow matching model on your vulnerable contract graphs to generate additional synthetic training examples for rare classes (DoS: 243 positives). This is more stable than GAN-based augmentation and the field exploded in 2025. TabDiff, VFMol show the principle works for structured discrete data.

Updates to what's already in the proposal
Noisy label learning: Beyond CGBC, there's now Dynamic Connection Masking (DCM, 2025) which adaptively masks unimportant graph edges during training to mitigate gradient errors from noisy labels. Works on the GNN itself, not on the dataset. Complementary to CGBC.

Formal verification hybrid: The proposal has this as Tier 4. 2025 papers show symbolic execution → semantic graph → GNN hybrids are practical now (not research-phase). GNNSE integrates symbolic execution directly as a preprocessing step.

Active learning: The proposal doesn't mention this much. DALL (2026) combines data programming + active learning + LLMs for labeling. Given your ExternalBug problem, targeted active learning (query for the most uncertain ExternalBug predictions and have a human annotate them) could correct the label noise faster than CGBC's automated approach.

