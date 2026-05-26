"""
show_full_input.py — Show exactly what goes into the model for one contract.
Both the graph tensor AND the token tensor.
"""
import os, sys
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer
from ml.src.preprocessing.graph_extractor import GraphExtractionConfig, extract_contract_graph
from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM, NODE_TYPES, EDGE_TYPES

ID_TO_NODE = {v: k for k, v in NODE_TYPES.items()}
ID_TO_EDGE = {v: k for k, v in EDGE_TYPES.items()}
MAX_TYPE_ID = float(max(NODE_TYPES.values()))

# Use contract 01 (classic reentrancy) as the example
SOL = __import__("pathlib").Path(__file__).parent / "test_contracts" / "01_reentrancy_classic.sol"

# ── 1. GRAPH ─────────────────────────────────────────────────────────────────
print("=" * 80)
print("CONTRACT:", SOL.name)
print("=" * 80)

cfg = GraphExtractionConfig()
g = extract_contract_graph(SOL, cfg)

print(f"\n━━━ GRAPH INPUT (→ GNNEncoder) ━━━")
print(f"g.x shape:          {g.x.shape}   ← [N={g.num_nodes}, {NODE_FEATURE_DIM} features per node]")
print(f"g.edge_index shape: {g.edge_index.shape}   ← [2, E={g.num_edges}] COO format")
print(f"g.edge_attr shape:  {g.edge_attr.shape}   ← [E] integer edge type IDs")
print()
print("g.x (full feature matrix — what the GNN actually sees):")
print(f"  {'idx':>3}  {'node_type':<18}  {'feat[0]':>7}  {'feat[1]':>7}  {'feat[2]':>7}  {'feat[3]':>7}  "
      f"{'feat[4]':>7}  {'feat[5]':>7}  {'feat[6]':>7}  {'feat[7]':>7}  {'feat[8]':>7}  {'feat[9]':>7}  {'feat[10]':>7}")
print(f"  {'':>3}  {'':^18}  {'type_id':>7}  {'visible':>7}  {'blk_gl':>7}  {'view':>7}  "
      f"{'payable':>7}  {'complex':>7}  {'loc':>7}  {'ret_ign':>7}  {'ctt':>7}  {'loop':>7}  {'extcall':>7}")
print("  " + "-"*112)
for i in range(g.num_nodes):
    ntype = ID_TO_NODE.get(int(round(g.x[i, 0].item() * MAX_TYPE_ID)), "?")
    vals  = [f"{v:+.4f}" for v in g.x[i].tolist()]
    print(f"  {i:>3}  {ntype:<18}  {'  '.join(vals)}")

print()
print("g.edge_index + g.edge_attr (every edge the GNN message-passes over):")
print(f"  {'e':>3}  {'etype':<20}  src→dst  (src_type → dst_type)")
print("  " + "-"*70)
for e in range(g.num_edges):
    src = int(g.edge_index[0, e])
    dst = int(g.edge_index[1, e])
    et  = ID_TO_EDGE.get(int(g.edge_attr[e]), "?")
    st  = ID_TO_NODE.get(int(round(g.x[src, 0].item() * MAX_TYPE_ID)), "?")
    dt  = ID_TO_NODE.get(int(round(g.x[dst, 0].item() * MAX_TYPE_ID)), "?")
    print(f"  {e:>3}  {et:<20}  {src}→{dst}  ({st} → {dt})")

# ── 2. TOKENS ─────────────────────────────────────────────────────────────────
print(f"\n━━━ TOKEN INPUT (→ TransformerEncoder / CodeBERT) ━━━")
source = SOL.read_text(encoding="utf-8")
print(f"\nRaw source ({len(source)} chars, {source.count(chr(10))+1} lines):")
print("-" * 60)
print(source)
print("-" * 60)

tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
encoded = tok(source, max_length=512, truncation=True, padding="max_length", return_tensors="pt")

input_ids      = encoded["input_ids"]      # [1, 512]
attention_mask = encoded["attention_mask"] # [1, 512]

n_real   = int(attention_mask.sum())
full_len = len(tok.encode(source, add_special_tokens=True))
truncated = full_len > 512

print(f"\ninput_ids shape:      {input_ids.shape}  (long tensor)")
print(f"attention_mask shape: {attention_mask.shape}  (long tensor)")
print(f"Real tokens (non-pad): {n_real}")
print(f"Full token count:      {full_len}")
print(f"Truncated:             {truncated}")

print(f"\nDecoded tokens (what CodeBERT actually reads):")
real_ids   = input_ids[0, :n_real].tolist()
token_strs = tok.convert_ids_to_tokens(real_ids)
print(f"  Total: {len(token_strs)} tokens")
print()
# Print in groups of 20 per line
for i in range(0, len(token_strs), 20):
    chunk = token_strs[i:i+20]
    print(f"  [{i:>3}]  " + "  ".join(f"{t:<12}" for t in chunk))

print(f"\nToken IDs (first 20): {real_ids[:20]}")
print(f"Token IDs (last  20): {real_ids[-20:]}")

print(f"\nattention_mask: {attention_mask[0, :n_real+3].tolist()}  ... {attention_mask[0, -3:].tolist()}")

# ── 3. HOW THEY COMBINE ───────────────────────────────────────────────────────
print(f"\n━━━ HOW THESE TWO INPUTS COMBINE IN THE MODEL ━━━")
print("""
SentinelModel.forward(batch, input_ids, attention_mask):

  ┌─────────────────────────────────────────────────────────────────────┐
  │  GNN path (graph input):                                            │
  │    batch.x          [N=15, 11]  ─→  GNNEncoder  ─→  gnn_out [B,256]│
  │    batch.edge_index [2, 24]          8-layer GAT                   │
  │    batch.edge_attr  [24]             JK attention                   │
  │    GNN prefix: 48 prefix tokens injected into CodeBERT attention    │
  │                                                                     │
  │  Transformer path (token input):                                    │
  │    input_ids        [1, 512]    ─→  CodeBERT   ─→  tf_out  [B,768] │
  │    attention_mask   [1, 512]        (frozen +                       │
  │                                      LoRA r=16)                    │
  │                                                                     │
  │  CrossAttentionFusion:                                              │
  │    query = gnn_out [B,256]  (graph asks: "what in the text matters")│
  │    key/value = tf_out [B,768] (CodeBERT answers)                   │
  │    fused_out [B,128]                                                │
  │                                                                     │
  │  Three-Eye Classifier:                                              │
  │    [gnn_out[B,256] ‖ tf_out[B,768] ‖ fused_out[B,128]] = [B,1152] │
  │      wait — actually:                                               │
  │    [gnn_eye[B,128] ‖ tf_eye[B,128] ‖ fused_eye[B,128]] = [B,384]  │
  │    → Linear(384,192) → ReLU → Linear(192,10) → logits [B,10]       │
  └─────────────────────────────────────────────────────────────────────┘

  Final output: 10 raw logits, one per class.
  sigmoid(logits) → probabilities.
  Apply per-class thresholds → binary predictions.
""")

print("Class order (matches logit positions 0-9):")
from ml.src.training.trainer import CLASS_NAMES
for i, cn in enumerate(CLASS_NAMES):
    print(f"  logit[{i}] → {cn}")

print(f"\n━━━ WHAT THE GNN 'SEES' vs WHAT CODEBERT 'READS' ━━━")
print(f"""
GNN sees (graph):
  - 15 nodes, each a 11-dim float vector
  - Node types: CONTRACT, STATE_VAR, FUNCTION, CFG_NODE_*, RECEIVE
  - Key signals from features:
      withdraw(): call_typed=0.0 (raw addr call), extcall=0.228
      CFG_NODE_CALL[10] → CONTROL_FLOW → CFG_NODE_WRITE[12]  ← CEI violation
  - The GNN does NOT know the variable is named "balances" or the function "withdraw"
  - It knows: "a function with raw-addr external call has CALL before WRITE in its CFG"

CodeBERT reads (tokens, {n_real} tokens):
  - The actual Solidity source code as subword tokens
  - It reads identifiers: "balances", "withdraw", "msg.sender", "call"
  - It reads patterns: "call{{value: amount}}" before "balances[msg.sender] = 0"
  - It knows: "this code pattern is textually similar to reentrancy examples in training"
  - But it does NOT see the graph structure

CrossAttentionFusion:
  - GNN's graph embedding asks CodeBERT: "given this structural pattern, what source
    tokens are most relevant to the vulnerability?"
  - CodeBERT provides token-level context to sharpen the GNN's structural finding
  - Result: a combined embedding that uses BOTH structure AND code semantics
""")
