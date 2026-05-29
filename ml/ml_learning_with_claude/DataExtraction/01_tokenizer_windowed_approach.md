# Data Extraction — Chunk 1: Tokenization & the Sliding Window Approach

> **Files:** `ml/src/data_extraction/tokenizer.py` (simple pipeline) + `ml/scripts/retokenize_windowed.py` (production pipeline)
> **What you'll learn:** How raw Solidity source is converted to BERT-compatible token tensors, why sliding windows replace hard truncation, the `init_worker` pattern for expensive multiprocessing initialization, and how truncation/padding creates uniform tensor shapes.
> **Time:** ~25 minutes
> **Interview relevance:** ML (text preprocessing, tokenization), AI (BERT internals), MLOps (data pipeline design)

---

## 1. Why Tokenize at All?

The GNN path reads Solidity through **structure** (graphs). The Transformer path reads it through **text** — but neural networks don't understand raw characters. They need numbers.

**Tokenization** converts source code text into a sequence of integer IDs:
```
"function withdraw(uint amount)"
        ↓ CodeBERT tokenizer
[0, 2270, 10440, 1006, 21183, 3815, 1007, 2]
```

Each integer is an index into CodeBERT's vocabulary (~50K tokens). The model looks up a 768-dimensional embedding vector for each token ID.

CodeBERT (`microsoft/codebert-base`) is a variant of RoBERTa pretrained on code from GitHub across 6 programming languages. It already "understands" common code patterns at a subword level — `function`, `require`, `msg.sender` are all meaningful units.

---

## 2. The Core Problem: Solidity Contracts Are Long

CodeBERT has a maximum sequence length of **512 tokens**. But:

```
Median Solidity contract = 2,469 tokens
Single window at 512 tokens = 21% of the median contract
```

The original simple approach (`tokenizer.py`):
```python
encoded = tokenizer(
    code,
    max_length=512,
    truncation=True,    # just cut after 512 tokens
    padding="max_length",
    return_tensors="pt"
)
# input_ids:      [512]
# attention_mask: [512]
```

**Problem:** For a 2,000-token contract, you see the first ~21% and discard the rest. The vulnerability might be at line 300 (which is past token 512). You'd completely miss it.

---

## 3. The Sliding Window Solution

The windowed tokenizer (`retokenize_windowed.py`) divides the contract into **overlapping windows**:

```
WINDOW_SIZE = 512   # tokens per window
STRIDE      = 256   # start of next window (overlap = 512-256 = 256)
MAX_WINDOWS = 4     # cap for very long contracts
```

For a contract of ~2,000 tokens:
```
Window 1: tokens 0   → 511   (contains first third)
Window 2: tokens 256 → 767   (overlaps with W1 by 256)
Window 3: tokens 512 → 1023  (overlaps with W2 by 256)
Window 4: tokens 768 → 1279  (overlaps with W3 by 256)
```

Coverage: `4 windows × 512 = 2048 token positions` → **62% of the median contract** vs 21% before.

The overlap (stride < window_size) ensures there are **no gaps** — every token position is covered by at least one window.

**The HuggingFace `return_overflowing_tokens` parameter does all the work:**
```python
encoded = tokenizer(
    code,
    max_length=WINDOW_SIZE,
    padding="max_length",
    truncation=True,
    stride=STRIDE,
    return_overflowing_tokens=True,   # ← produce multiple windows
    return_tensors="pt",
)
# encoded["input_ids"]:      [W, 512]  — W windows
# encoded["attention_mask"]: [W, 512]
```

> 🎯 **INTERVIEW FOCUS:** "How do you handle long documents with BERT (which has 512-token limit)?" — Sliding window tokenization with overlap. Overlap prevents edge effects where important context is split exactly at a window boundary.

---

## 4. Window Sub-sampling for Very Long Contracts

Some contracts are extremely long — a naive sliding window would produce W=15 or more windows. With `MAX_WINDOWS=4`, you need to pick 4 representative windows:

```python
def _select_windows(all_input_ids, all_attention_masks, max_windows):
    W = len(all_input_ids)
    if W <= max_windows:
        return all_input_ids, all_attention_masks
    
    # linspace: pick max_windows indices spread evenly across 0..W-1
    indices = [round(i) for i in np.linspace(0, W - 1, max_windows)]
    sel_ids   = [all_input_ids[i]       for i in indices]
    sel_masks = [all_attention_masks[i] for i in indices]
    return sel_ids, sel_masks
```

**`np.linspace(0, W-1, 4)`** generates 4 evenly-spaced floating-point values between 0 and W-1. For W=12: `[0.0, 3.67, 7.33, 11.0]` → rounded to `[0, 4, 7, 11]`.

This sub-sampling strategy is better than just taking the first 4 windows because it covers:
- **Window 0**: beginning of the contract (constructor, state variable declarations)
- **Windows 4, 7**: middle sections (core logic)
- **Window 11**: end of contract (may contain vulnerable functions added later)

---

## 5. Uniform Output Shape — Why It Matters for DataLoader

```python
# Always output [MAX_WINDOWS, 512] regardless of contract length

# Short contract (1 real window):
input_ids = [[...512 real tokens...], [0,0,...,0], [0,0,...,0], [0,0,...,0]]
#             window 0 (real)          window 1 (pad) window 2 (pad) window 3 (pad)
attention_mask = [[1,1,...,1], [0,0,...,0], [0,0,...,0], [0,0,...,0]]
#                 all real      all masked   all masked   all masked

# Long contract (4+ windows, subsampled):
input_ids = [[...], [...], [...], [...]]  # 4 real windows
attention_mask = [[1,...,1], [1,...,1], [1,...,1], [1,...,1]]  # all active
```

**Why uniform shape?**
PyTorch's `DataLoader` batches tensors with `torch.stack()`. Stack requires **identical shapes** across all samples in a batch. If sample A has shape `[3, 512]` and sample B has shape `[4, 512]`, they can't be stacked.

The zero-padding solves this: every sample outputs exactly `[4, 512]`, so `torch.stack()` always produces `[B, 4, 512]`.

**The padding windows are masked out** (`attention_mask=0`) — the model's cross-attention (and the CrossAttentionFusion) uses the attention mask to ignore zero-padded positions.

```python
while len(all_ids) < max_windows:
    all_ids.append([pad_id] * WINDOW_SIZE)
    all_masks.append([0] * WINDOW_SIZE)    # ← key: masked out
```

---

## 6. The `init_worker` Pattern — Critical Multiprocessing Pattern

Loading the CodeBERT tokenizer takes ~500MB of memory and several seconds. Without the `init_worker` pattern:

```python
# BAD: each of 68,000 contracts triggers tokenizer load
def tokenize_single_contract(path):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")  # 500MB, every time!
    ...
```

**With `init_worker`:**
```python
_tokenizer = None  # module-level global

def _init_worker():
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(...)
    # This runs ONCE per worker process

with mp.Pool(processes=11, initializer=_init_worker) as pool:
    # 11 workers × 1 load = 11 total loads
    # 68,000 contracts × 1 tokenizer call each (no loading)
    results = pool.imap(tokenize_single_contract, paths)
```

**`initializer=_init_worker`**: each worker calls this function once when it starts, before processing any work items. The loaded tokenizer lives in the worker's global scope, reused for all its contracts.

**Why a module-level global?** The worker function (`tokenize_single_contract`) and the init function (`_init_worker`) are both in the same module. The global `_tokenizer` variable is the communication channel between them. There's no way to pass the tokenizer object from `init_worker` to `tokenize_single_contract` through `pool.imap` (which only sends the work item argument).

> 🎯 **INTERVIEW FOCUS:** "How do you efficiently process 68,000 files with a model that takes 3 seconds to load?" — `initializer=init_worker` in `mp.Pool`. Load once per worker process, reuse for all work items assigned to that worker.

---

## 7. Special Tokens: [CLS] and [SEP]

CodeBERT (like all BERT variants) wraps every sequence with special tokens:
```
[CLS] token_1 token_2 ... token_N [SEP]
 id=0                               id=2
```

- `[CLS]` (id=0): Classification token. Its final embedding is what the model uses for classification tasks. Position 0 in the output.
- `[SEP]` (id=2): Separator token. Marks end of sequence.
- `[PAD]` (id=1): Padding token. Fills positions up to 512. `attention_mask=0` for these.

**Why this matters for the SENTINEL prefix injection:**
```python
# Position IDs in TransformerEncoder:
prefix_positions = 1  # RoBERTa's padding slot — avoids 0 (BOS) and 2 (EOS)
code_positions = 3..3+code_budget-1
```

The prefix tokens (GNN embeddings injected before the code) must use position IDs that don't collide with the special token positions (0=BOS/CLS, 1=padding, 2=EOS/SEP). Position 1 (the padding slot) is repurposed for prefix tokens because RoBERTa never puts real content there.

---

## 8. Attention Mask — What It Actually Does

```python
attention_mask = [1, 1, 1, 1, 1, 0, 0, 0, 0, ...]
#                  real tokens  ← padding tokens →
```

The attention mask tells the model which positions contain real content (1) vs padding (0). In the self-attention computation:

```
softmax(QK^T / √d_k + mask_bias)
```

Where `mask_bias = 0` for real tokens and `mask_bias = -∞` for masked tokens. Softmax of `-∞` is 0 — padding positions contribute zero to the weighted average. The model "sees through" padding.

**For multi-window tokens:** The attention mask also marks entire zero-padding windows as masked (`[0,0,...,0]` for padding windows). The CrossAttentionFusion uses this to ignore padding windows in cross-attention.

---

## 9. Truncation Awareness

```python
# Old tokenizer.py approach:
truncated = num_real_tokens >= (MAX_LENGTH - 2)  # if all 510 content positions filled
```

This is an **approximation**: if the sequence uses all 510 non-special-token positions, it was probably truncated. True truncation detection would require tokenizing without truncation and checking if len > 512.

The windowed approach effectively **measures** truncation through `num_windows` statistics:
```python
stats = {"w1": 0, "w2": 0, "w3": 0, "w4plus": 0}
# w4plus: contracts that needed max_windows windows (possibly still truncated)
```

A `w4plus` contract had >4 windows before sub-sampling. The last window covers tokens 768–1279 at minimum; anything past that is lost. This is the "truncation rate" in the windowed pipeline.

---

## 10. Content-Addressed Token Files

```python
# The MD5 hash matches the graph file name exactly
filename = get_filename_from_hash(token_data["contract_hash"])  # e.g., "abc123.pt"
torch.save(token_data, output_dir / filename)
```

Token files and graph files share the same MD5-based filename. The `DualPathDataset` loads both using the same hash stem — this is the pairing mechanism. If a contract's token file is `abc123.pt`, its graph file must also be `abc123.pt`.

**The MD5 is always the path-based MD5** (relative to project root), not the content MD5. This is important: the same Solidity code in two different directories gets two different hashes, because the path is different.

---

## 11. What's Saved to Disk

Each `.pt` token file is a Python dict:
```python
{
    "input_ids":              tensor([W, 512], dtype=long),     # token IDs
    "attention_mask":         tensor([W, 512], dtype=long),     # 1=real, 0=pad
    "num_windows":            3,                                 # real windows (≤4)
    "stride":                 256,
    "contract_hash":          "abc123def456",
    "contract_path":          "/path/to/contract.sol",
    "num_tokens":             int,                               # total real tokens
    "tokenizer_name":         "microsoft/codebert-base",
    "max_length":             512,
    "feature_schema_version": "v8",                             # schema at extraction
}
```

The `feature_schema_version` in the token file is the graph schema version at tokenization time. It's included for traceability — if you retokenize after a schema bump, you can tell from the file which schema was current.

---

## 12. Evolution: tokenizer.py → retokenize_windowed.py

| | `tokenizer.py` | `retokenize_windowed.py` |
|--|----------------|--------------------------|
| Coverage | 21% of median contract | 62% of median contract |
| Output shape | `[512]` | `[W, 512]` (W=4) |
| Long contracts | Hard truncation | linspace sub-sampling |
| Short contracts | Padded to 512 | Padded to `[4, 512]` |
| Data source | `contracts_metadata.parquet` | `multilabel_index_deduped.csv` |

The windowed version is the production pipeline. The simple one remains for reference and backward compatibility with old checkpoints.

---

## 13. Summary

| Concept | What you learned |
|---------|----------------|
| Tokenization | Text → integer IDs → embedding lookups |
| [CLS], [SEP], [PAD] | Special tokens in BERT-family models |
| Attention mask | Tells model which positions are real vs padding |
| Sliding window | Overcomes 512-token limit by creating overlapping windows |
| `return_overflowing_tokens` | HuggingFace parameter to auto-generate windows |
| linspace sub-sampling | Covers start/middle/end of very long contracts |
| Uniform output shape | Required for DataLoader batch collation |
| `init_worker` pattern | Load expensive resources once per worker, not per item |
| Content-addressed naming | MD5 hash matches graph file for pairing |

---

## Interview Questions

1. **"BERT has a 512-token limit. How do you handle documents that are longer?"**
   → Sliding window tokenization: produce multiple overlapping 512-token windows, each with stride < window_size to ensure no gaps. Pad short contracts with masked zero-windows to keep a uniform tensor shape for batching.

2. **"Why do BERT-family models use attention masks?"**
   → To distinguish real content from padding tokens. During attention computation, masked positions receive `-∞` logits before softmax, producing zero attention weights. Without masking, padding tokens would incorrectly influence the representations of real tokens.

3. **"How do you efficiently run a function that requires a 500MB model across 68,000 inputs in parallel?"**
   → `mp.Pool(processes=N, initializer=init_worker)`: load the model once in each worker process via the `initializer` callback. Reuse the worker-level global for all items processed by that worker. Total loads = N workers, not N×items.

4. **"What is `np.linspace` useful for in data processing?"**
   → Generating N evenly-spaced values across a range. Used here for sub-sampling: pick 4 windows spread evenly across a contract's W total windows, covering beginning, middle, and end instead of just truncating.

---

**Next:** `Datasets/01_dual_path_dataset_and_dataloader.md`
