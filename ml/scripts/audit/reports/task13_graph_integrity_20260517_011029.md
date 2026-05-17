# Task 13: Graph Integrity Audit
**Sample size:** 500  
**Successfully loaded:** 500  
**Skipped (load errors):** 0  
**Stale v5 (x.shape[1] != 12):** 0

## Check Results
| Check | Pass | Fail | Rate |
|-------|------|------|------|
| x_shape_12 | 500 | 0 | 100.0% |
| x_dtype_float32 | 500 | 0 | 100.0% |
| edge_index_shape | 500 | 0 | 100.0% |
| edge_index_dtype | 500 | 0 | 100.0% |
| edge_index_in_range | 500 | 0 | 100.0% |
| edge_attr_1d | 500 | 0 | 100.0% |
| edge_attr_range | 500 | 0 | 100.0% |
| no_nan_inf | 500 | 0 | 100.0% |
| contains_cfg_targets | 500 | 0 | 100.0% |
| cf_cfg_nodes | 500 | 0 | 100.0% |
| at_least_1_edge | 496 | 4 | 99.2% |
| no_cf_self_loops | 500 | 0 | 100.0% |

## Edge Type Distribution
| Edge Type ID | Name | Count |
|--------------|------|-------|
| 0 | CALLS | 5974 |
| 1 | READS | 8495 |
| 2 | WRITES | 8622 |
| 5 | CONTAINS | 48175 |
| 6 | CONTROL_FLOW | 41233 |

## Node Type Distribution
| Type ID | Name | Count |
|---------|------|-------|
| 0 | STATE_VAR | 5010 |
| 1 | FUNCTION | 8982 |
| 2 | MODIFIER | 785 |
| 3 | EVENT | 1839 |
| 4 | FALLBACK | 230 |
| 6 | CONSTRUCTOR | 767 |
| 7 | CONTRACT | 500 |
| 8 | CFG_NODE_CALL | 4418 |
| 9 | CFG_NODE_WRITE | 8370 |
| 10 | CFG_NODE_READ | 6013 |
| 11 | CFG_NODE_CHECK | 2695 |
| 12 | CFG_NODE_OTHER | 26679 |

## Failures Detail (first 30)
- **7319be6ba90861d17ee9615295ac8b43**: graph has 0 edges
- **12a9a38cccf60d2f23cd7d810838999a**: graph has 0 edges
- **5d686fe64106e05772ef8720fab9316a**: graph has 0 edges
- **1fd0a54cdb2ec7c35d95bcc6bc28defc**: graph has 0 edges
