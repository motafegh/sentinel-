# D1.2b — ML Module Doc

**Doc target:** `docs/handbook/03_ml_module.md`
**Estimated time:** 1h
**Rule:** Every claim verified against source code.

---

## Source files to read before writing (10 files)

1. `ml/src/models/sentinel_model.py:155-310,525-593`
   - SentinelModel.__init__ params: fusion_output_dim, gnn_hidden_dim, gnn_num_layers, gnn_heads, gnn_prefix_k, lora_r, lora_alpha, use_edge_attr, etc.
   - The 4 eyes: gnn_eye_proj, transformer_eye_proj, cfg_eye_proj, classifier (4*eye_dim → 256 → num_classes)
   - forward(): return_aux=True path → (logits, aux_dict)
   - aux_dict keys: gnn, transformer, fused, phase2, jk_entropy, fusion_embedding
   - `:592` — fusion_embedding = fused_eye (the ZKML boundary, [B, 128])

2. `ml/src/models/gnn_encoder.py:161,404-410`
   - GNNEncoder: 8-layer GAT, gnn_hidden_dim=256, use_edge_attr=True
   - edge_attr check: `:407` — raises ValueError if use_edge_attr=True but edge_attr=None
   - Returns: (node_embs [N, hidden], batch [N], jk_attn_weights)

3. `ml/src/models/transformer_encoder.py:152,166`
   - TransformerEncoder: GraphCodeBERT (microsoft/graphcodebert-base) + LoRA (r=16, alpha=32, target=query+value)
   - WindowAttentionPooler: window_size=512, prefix_k=gnn_prefix_k
   - SDPA active (flash-attn unsupported on RTX 3070)

4. `ml/src/models/fusion_layer.py:198-240`
   - CrossAttentionFusion: node_dim=256, token_dim=768, attn_dim=256, heads=8, output=128
   - forward(): node_embs [N,256] + token_embs [B,W*512,768] → [B,128]
   - key_padding_mask: [B, W*L] (flattened), True=IGNORE

5. `ml/src/inference/predictor.py:293,459-590,830-910`
   - `:293` — `self.model.float()` (normalize BF16 to FP32)
   - predict_source(): windowed inference, _score_windowed
   - _score_windowed(): [1,4,512] format, model(return_aux=True), sigmoid, _format_result
   - predict_fusion_embedding(): returns 128-dim vector (P11 addition)
   - _format_result(): three-tier output (confirmed/suspicious/safe)
   - model_hash: _compute_file_hash (SHA-256 of checkpoint file)

6. `ml/src/inference/api.py:74-80,107-114,199-240,380-510`
   - `:74-80` — CHECKPOINT path (3-level: SENTINEL_CHECKPOINT env > mlops_config.json > default)
   - `:107-114` — SENTINEL_DETERMINISTIC mode (torch.use_deterministic_algorithms, manual_seed(42))
   - `:199-233` — PredictResponse Pydantic model (all fields)
   - `:235-240` — FusionEmbeddingResponse Pydantic model
   - `:380-463` — /hotspots endpoint
   - `:467-510` — /fusion-embedding endpoint (P11)

7. `ml/src/inference/preprocess.py:217`
   - ContractPreprocessor: graph extraction + tokenization
   - process_source_windowed(): returns (graph, windows)

8. `ml/src/datasets/sentinel_dataset.py` — training data loading (reference only, see 02_data_module.md)

9. `ml/src/training/trainer.py` — training loop (reference only, not deep dive)

10. `ml/checkpoints/README.md` — checkpoint metadata, Run 12 details

---

## Sections to write

**1. TL;DR** (5 lines)
```
What: GNN+GraphCodeBERT dual-path model, 10-class vulnerability classifier
Architecture: four_eye_v8, ~127M params, fusion_output_dim=128
Checkpoint: ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt (269 MB)
Server: source ml/.venv/bin/activate && uvicorn ml.src.inference.api:app --port 8001
Tests: source ml/.venv/bin/activate && python -m pytest ml/tests/ (214 passed)
```

**2. Architecture** (~1.5 pages)
- The 4 eyes (verify from `sentinel_model.py:250-310`):
  - GNN eye: global_max_pool + global_mean_pool → [B, 2*256] → gnn_eye_proj → [B, 128]
  - Transformer eye: WindowAttentionPooler → [B, 768] → transformer_eye_proj → [B, 128]
  - Fused eye: CrossAttentionFusion(node_embs, token_embs) → [B, 128]
  - CFG eye: pool Phase 2 embeddings over CFG nodes → [B, 2*256] → cfg_eye_proj → [B, 128]
- Main classifier: cat([4 eyes]) → [B, 512] → Linear(512→256) → ReLU → Dropout → Linear(256→10)
- CrossAttentionFusion (verify from `fusion_layer.py:198-240`):
  - node_proj: 256→256, token_proj: 768→256
  - Node→Token cross-attention + Token→Node cross-attention
  - Output: [B, 128] — THE ZK BOUNDARY
- The `return_aux=True` path (verify from `sentinel_model.py:586-593`):
  - aux_dict keys: gnn, transformer, fused, phase2, jk_entropy, fusion_embedding
  - `fusion_embedding` = fused_eye = [B, 128] — this is what ZKML proves
- ASCII diagram of data flow: input → GNN + Transformer → Fusion → 4 eyes → classifier → 10 logits

**3. Inference API** (~1 page)
- 4 endpoints (verify each route decorator from `api.py`):
  - `GET /health` — status, predictor_loaded, checkpoint, architecture, model_hash, thresholds_loaded
  - `POST /predict` — full prediction: label, probabilities, confirmed, suspicious, thresholds, eye_predictions, model_hash
  - `POST /hotspots` — per-function GNN attention scores + full prediction
  - `POST /fusion-embedding` — 128-dim CrossAttentionFusion output (P11, for ZKML)
- PredictResponse fields (verify from `api.py:199-233`): label, probabilities, confirmed, suspicious, vulnerabilities (legacy), tier_thresholds, thresholds, truncated, windows_used, num_nodes, num_edges, eye_predictions, model_hash
- FusionEmbeddingResponse fields (verify from `api.py:235-240`): fusion_embedding (128 floats), num_nodes, num_edges, model_hash, windows_used
- Model hash: `predictor.py:_compute_file_hash()` — SHA-256 of checkpoint file, stable across restarts

**4. Three-tier output** (~0.5 page)
- Verify thresholds from `predictor.py:783-813`:
  - CONFIRMED: prob >= per-class threshold (loaded from `{checkpoint}_thresholds.json`, min=0.050, max=0.500)
  - SUSPICIOUS: tier_suspicious_threshold (default 0.25) <= prob < confirmed threshold
  - NOTEWORTHY: prob < 0.25 (included in probabilities dict only, not in tier lists)
- Label values: "safe" | "suspicious" | "confirmed_vulnerable" | "no_contracts_found"

**5. Deterministic mode** (~0.5 page)
- `SENTINEL_DETERMINISTIC=1` (verify from `api.py:107-114`):
  - `torch.use_deterministic_algorithms(True)`
  - `torch.manual_seed(42)` + `torch.cuda.manual_seed_all(42)`
  - In agents: disables LLM (`_helpers.py:_llm_enabled` returns False) + skips RAG (`rag_research.py` returns empty)
- Why: ZK proof requires reproducibility. LLM debate is non-deterministic even at temp=0.

**6. Deep reference**
- → `docs/learning/04_reproducibility_determinism.md`
- → `ml/CLAUDE.md` (module-specific Claude rules)
- → source: `sentinel_model.py`, `predictor.py`, `api.py`, `fusion_layer.py`

---

## Verification checklist
- [ ] Every endpoint path matches `api.py` route decorators (`@app.post`, `@app.get`)
- [ ] PredictResponse field list matches the Pydantic model at `api.py:199-233`
- [ ] FusionEmbeddingResponse fields match `api.py:235-240`
- [ ] `_compute_file_hash` exists in `predictor.py` and returns 64-char hex
- [ ] Architecture params match checkpoint config (gnn_prefix_k=48, fusion_output_dim=128, num_classes=10)
- [ ] aux_dict keys match `sentinel_model.py:586-593` (gnn, transformer, fused, phase2, jk_entropy, fusion_embedding)
- [ ] Three-tier thresholds match `predictor.py:799-803` logic
- [ ] Test command produces 214 passed
