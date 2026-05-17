# Task 14: Sub-Sample Coverage Audit

**Contracts analyzed:** 10  
**Max windows (sub-sampled):** 4  
**Window size:** 512 tokens, stride 256

## Per-Contract Analysis

### Contract 1: `85807b84227d66f1a22e4ca5...`

- **Vulnerability classes:** Timestamp
- **File size:** 797,742 bytes
- **Full windows:** 1862
- **Sub-sampled windows:** 4
- **Vuln-relevant windows (full):** 4
- **Vuln-relevant windows (sub):** 1
- **Survival rate:** 100.0%

- **Survived patterns:** require() check

### Contract 2: `2bbd5702d76cdb87a319e870...`

- **Vulnerability classes:** ExternalBug, IntegerUO, TransactionOrderDependence
- **File size:** 710,234 bytes
- **Full windows:** 1663
- **Sub-sampled windows:** 4
- **Vuln-relevant windows (full):** 4
- **Vuln-relevant windows (sub):** 1
- **Survival rate:** 100.0%

- **Survived patterns:** require() check

### Contract 3: `baf41f29b115946c5736c275...`

- **Vulnerability classes:** ExternalBug, Reentrancy, UnusedReturn
- **File size:** 709,039 bytes
- **Full windows:** 1661
- **Sub-sampled windows:** 4
- **Vuln-relevant windows (full):** 4
- **Vuln-relevant windows (sub):** 1
- **Survival rate:** 100.0%

- **Survived patterns:** require() check

### Contract 4: `1e70a2b90b517e7d88655ac4...`

- **Vulnerability classes:** IntegerUO, TransactionOrderDependence
- **File size:** 708,889 bytes
- **Full windows:** 1662
- **Sub-sampled windows:** 4
- **Vuln-relevant windows (full):** 4
- **Vuln-relevant windows (sub):** 1
- **Survival rate:** 100.0%

- **Survived patterns:** require() check

### Contract 5: `243e75bcf9abcdf101b8aa19...`

- **Vulnerability classes:** IntegerUO, MishandledException, Timestamp
- **File size:** 704,320 bytes
- **Full windows:** 1652
- **Sub-sampled windows:** 4
- **Vuln-relevant windows (full):** 4
- **Vuln-relevant windows (sub):** 1
- **Survival rate:** 100.0%

- **Survived patterns:** require() check

### Contract 6: `049e0f7fc92134f52de6c1fb...`

- **Vulnerability classes:** IntegerUO
- **File size:** 703,858 bytes
- **Full windows:** 1658
- **Sub-sampled windows:** 4
- **Vuln-relevant windows (full):** 4
- **Vuln-relevant windows (sub):** 1
- **Survival rate:** 100.0%

- **Survived patterns:** require() check

### Contract 7: `bc44d92b25565c0750b9f4f2...`

- **Vulnerability classes:** GasException, IntegerUO
- **File size:** 702,647 bytes
- **Full windows:** 1656
- **Sub-sampled windows:** 4
- **Vuln-relevant windows (full):** 4
- **Vuln-relevant windows (sub):** 1
- **Survival rate:** 100.0%

- **Survived patterns:** require() check

### Contract 8: `8b0d4be2e9f7dd1a157bc957...`

- **Vulnerability classes:** IntegerUO
- **File size:** 702,530 bytes
- **Full windows:** 1653
- **Sub-sampled windows:** 4
- **Vuln-relevant windows (full):** 4
- **Vuln-relevant windows (sub):** 1
- **Survival rate:** 100.0%

- **Survived patterns:** require() check

### Contract 9: `2592264dc4d7b06450f319f1...`

- **Vulnerability classes:** ExternalBug, GasException, IntegerUO
- **File size:** 701,928 bytes
- **Full windows:** 1652
- **Sub-sampled windows:** 4
- **Vuln-relevant windows (full):** 4
- **Vuln-relevant windows (sub):** 1
- **Survival rate:** 100.0%

- **Survived patterns:** require() check

### Contract 10: `8c1cd97c2bddfe1c0c33cbe6...`

- **Vulnerability classes:** IntegerUO
- **File size:** 700,658 bytes
- **Full windows:** 1657
- **Sub-sampled windows:** 4
- **Vuln-relevant windows (full):** 4
- **Vuln-relevant windows (sub):** 1
- **Survival rate:** 100.0%

- **Survived patterns:** require() check

## Summary

- **Average survival rate:** 100.0%
- **Min survival rate:** 100.0%
- **Max survival rate:** 100.0%

## Practical Impact Assessment

### How Sub-Sampling Works

The windowed tokenizer (`retokenize_windowed.py`) uses `linspace` sub-sampling:
- Contracts with ≤ 4 windows: all windows retained (no loss)
- Contracts with > 4 windows: 4 evenly-spaced windows selected via `np.linspace(0, W-1, 4)`
  - This covers start, ~1/3, ~2/3, and end of the contract

### Key Findings

- **LOW RISK**: Average survival rate ≥ 90%. Most vulnerability-relevant code survives sub-sampling.
- The `linspace` strategy biases toward start and end coverage, which may miss vulnerability code in the middle of very long contracts.
- Consider increasing `MAX_WINDOWS` from 4 to 6-8 if VRAM permits.
