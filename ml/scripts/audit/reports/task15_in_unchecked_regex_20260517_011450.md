# Task 15: in_unchecked Feature & Regex Audit

## Dead Feature Confirmation

**CONFIRMED: The `in_unchecked` feature [9] is DEAD.**

No graph .pt file out of the sampled set has `in_unchecked = 1.0` on any node. This means:
- The Slither `NodeType.STARTUNCHECKED` check is not firing
- The fallback regex `\bunchecked\s*\{` is also not matching in function source_mapping content
- Either contracts in the dataset don't use `unchecked{}` blocks, or the source_mapping content is unavailable for these functions

## Regex False Positive Tests

Regex pattern: `\bunchecked\s*\{`

| Test Case | Source | Match? | Result |
|-----------|--------|--------|--------|
| Comment | `// unchecked { this is a comment }` | YES | **FALSE POSITIVE** |
| String | `string memory s = "unchecked {"` | YES | **FALSE POSITIVE** |
| Real code | `unchecked { return a + b; }` | YES | Correct (match) |

### False Positive Analysis

The regex `\bunchecked\s*\{` produces false positives when:
- `unchecked {` appears in a **comment** — the regex matches text that is not executable code
- `unchecked {` appears in a **string literal** — the regex matches text inside string constants

However, the graph_extractor.py first tries Slither's `NodeType.STARTUNCHECKED` which only fires on real AST nodes. The regex fallback uses `func.source_mapping.content` which contains only the function body (not comments outside the function). But comments **inside** the function body would still cause false positives.

## Contract Counts

| Category | Count |
|----------|-------|
| Total .sol files scanned | 5000 |
| Solidity >= 0.8.0 contracts | 52 |
| Contracts with 'unchecked' anywhere | 15 |
| Contracts with 'unchecked' AND >= 0.8.0 | 0 |
| Contracts with 'unchecked' in code (not just comments/strings) | 0 |

## Impact Assessment

Since `in_unchecked` is never activated:

1. **Feature [9] is always 0.0** for all nodes in all graphs
2. This wastes one feature dimension — the GNN cannot learn anything from it
3. For IntegerUO (integer overflow/underflow) detection, this feature was supposed to be a direct signal for unchecked arithmetic blocks
4. The model must rely entirely on other features (complexity, external_call_count, has_loop) to detect IntegerUO patterns

### Possible Causes

- Slither `NodeType.STARTUNCHECKED` may not be triggered for the contracts in the dataset
- The `source_mapping.content` for functions may be empty/unavailable, causing the regex fallback to be skipped
- The dataset may genuinely contain no contracts using `unchecked{}` blocks (possible for older Solidity or conservative code)

