# Task 24: Token-Graph-Source Alignment Audit

**Paired stems available:** 44470  
**Sample size:** 100  
**.sol files resolved:** 100/100

## 1. Hash Alignment (Graph ↔ Token)

| Metric | Count | Rate |
|--------|-------|------|
| Hash match | 0 | 0.0% |
| Hash mismatch | 0 | 0.0% |
| Hash missing | 100 | — |

## 2. Path Alignment (Graph ↔ Token)

| Metric | Count | Rate |
|--------|-------|------|
| Path match | 0 | 0.0% |
| Path mismatch | 95 | 100.0% |
| Path missing (one or both) | 5 | — |
| Resolved via md5_to_path (match) | 0 | — |
| Resolved via md5_to_path (mismatch) | 5 | — |

## 3. Filename Stem ↔ Path Hash Verification

| Metric | Count | Rate |
|--------|-------|------|
| Stem matches path hash | 100 | 100.0% |
| Stem mismatches path hash | 0 | 0.0% |
| Unresolvable | 0 | — |

## 4. Decode Verification (10 stems, first 200 tokens)

| Stem | Status | Token Match Rate | Token Match | Prefix Overlap | Decoded Len | Sol Len |
|------|--------|-----------------|-------------|----------------|-------------|--------|
| 6bf376ffdc35... | ok | 100.00% | 200/200 | 27.47% | 1820 | 8915 |
| 83f02c1fef86... | ok | 100.00% | 200/200 | 31.02% | 1612 | 9076 |
| 38d962126ced... | ok | 100.00% | 200/200 | 32.20% | 1553 | 4962 |
| 0ba5f2b441de... | ok | 100.00% | 200/200 | 30.03% | 1665 | 28290 |
| 919f9eb8b4f4... | ok | 100.00% | 200/200 | 29.59% | 1690 | 14255 |
| 8bd6ef3964c6... | ok | 100.00% | 200/200 | 32.07% | 1559 | 2991 |
| 75e21030c6b3... | ok | 100.00% | 200/200 | 43.22% | 1157 | 11805 |
| 8fe7b4fa4430... | ok | 100.00% | 200/200 | 31.53% | 1586 | 6338 |
| 6c71edd825d6... | ok | 100.00% | 200/200 | 39.06% | 1280 | 20249 |
| 0f0a184a54a7... | ok | 100.00% | 200/200 | 25.71% | 1945 | 5952 |

**Decode pass rate (≥90% token match):** 10/10

## 5. Mismatches Detail (first 30)

- **6bf376ffdc35333c55ae6a4833761aca**: missing hash: graph_hash=None, token_hash=6bf376ffdc35333c55ae6a4833761aca
- **6bf376ffdc35333c55ae6a4833761aca**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/IntegerUO/710f2b6e6f10b20347c445fe511d05998f9dffb9aaddecd0c2ebfe2ae8df6338.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/IntegerUO/710f2b6e6f10b20347c445fe511d05998f9dffb9aaddecd0c2ebfe2ae8df6338.sol
- **83f02c1fef8679a8684d8f00773b96e4**: missing hash: graph_hash=None, token_hash=83f02c1fef8679a8684d8f00773b96e4
- **83f02c1fef8679a8684d8f00773b96e4**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/NonVulnerable/0ac04e30e37cb806eea2ff8bfe1b9ffaf8d93af4846401dd63a4fb615032a939.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/NonVulnerable/0ac04e30e37cb806eea2ff8bfe1b9ffaf8d93af4846401dd63a4fb615032a939.sol
- **38d962126ced7812697a994c8a1d31b3**: missing hash: graph_hash=None, token_hash=38d962126ced7812697a994c8a1d31b3
- **38d962126ced7812697a994c8a1d31b3**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/NonVulnerable/a16cf28e9195ead62091d0d7612a7e860567077665df1202dd46319e09cfcc21.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/NonVulnerable/a16cf28e9195ead62091d0d7612a7e860567077665df1202dd46319e09cfcc21.sol
- **0ba5f2b441de44fd0bed67d8cc9c01f7**: missing hash: graph_hash=None, token_hash=0ba5f2b441de44fd0bed67d8cc9c01f7
- **0ba5f2b441de44fd0bed67d8cc9c01f7**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/MishandledException/7e651f23bb593bf9c2e5db2916f4123003debaf384beacadb2de3bf091136559.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/MishandledException/7e651f23bb593bf9c2e5db2916f4123003debaf384beacadb2de3bf091136559.sol
- **919f9eb8b4f42ce431c27e33f72c181e**: missing hash: graph_hash=None, token_hash=919f9eb8b4f42ce431c27e33f72c181e
- **919f9eb8b4f42ce431c27e33f72c181e**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/Reentrancy/0a04b7966715ae71ce0cef9731f45b23fd4525c32c8eaf6e6a54b2c9c3a25a81.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/Reentrancy/0a04b7966715ae71ce0cef9731f45b23fd4525c32c8eaf6e6a54b2c9c3a25a81.sol
- **8bd6ef3964c6979feab5bc87b5be7924**: missing hash: graph_hash=None, token_hash=8bd6ef3964c6979feab5bc87b5be7924
- **8bd6ef3964c6979feab5bc87b5be7924**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/NonVulnerable/4e4264c37ae6db7284d414f93cdde06aaa8468aa4223872ef75f548df3e67811.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/NonVulnerable/4e4264c37ae6db7284d414f93cdde06aaa8468aa4223872ef75f548df3e67811.sol
- **75e21030c6b3700490c0621918a0e191**: missing hash: graph_hash=None, token_hash=75e21030c6b3700490c0621918a0e191
- **75e21030c6b3700490c0621918a0e191**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/NonVulnerable/3919547a0e3b07c6cdf7d7b0cf52879f6f7ff0a4f927a48b4ed59d2e8e970a76.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/NonVulnerable/3919547a0e3b07c6cdf7d7b0cf52879f6f7ff0a4f927a48b4ed59d2e8e970a76.sol
- **8fe7b4fa4430f2caa553bd918533a3ac**: missing hash: graph_hash=None, token_hash=8fe7b4fa4430f2caa553bd918533a3ac
- **8fe7b4fa4430f2caa553bd918533a3ac**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/NonVulnerable/832b67cc9f51c748441d0897b18549b699399081e4490ee53f0b49829a0b4a62.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/NonVulnerable/832b67cc9f51c748441d0897b18549b699399081e4490ee53f0b49829a0b4a62.sol
- **6c71edd825d63d0ec8dd95e0bb2857dc**: missing hash: graph_hash=None, token_hash=6c71edd825d63d0ec8dd95e0bb2857dc
- **6c71edd825d63d0ec8dd95e0bb2857dc**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/IntegerUO/d5827567d41aa99384abe1696c4becfa7a991f23ded2c17cc00d0600a90a4ccd.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/IntegerUO/d5827567d41aa99384abe1696c4becfa7a991f23ded2c17cc00d0600a90a4ccd.sol
- **0f0a184a54a7c9354138241f088ddf4c**: missing hash: graph_hash=None, token_hash=0f0a184a54a7c9354138241f088ddf4c
- **0f0a184a54a7c9354138241f088ddf4c**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/MishandledException/2084b5c7603a6568c5096bbc7a7e6b4c82cfc40c22340d7e4f63978f5a6ee9d5.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/MishandledException/2084b5c7603a6568c5096bbc7a7e6b4c82cfc40c22340d7e4f63978f5a6ee9d5.sol
- **75dc0d41a39763832e3ea04608ab8471**: missing hash: graph_hash=None, token_hash=75dc0d41a39763832e3ea04608ab8471
- **75dc0d41a39763832e3ea04608ab8471**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/Timestamp/299d71210aab9cb8c3a8df295de7d1e2dfc998e29368a2859e1df8bade5d3862.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/Timestamp/299d71210aab9cb8c3a8df295de7d1e2dfc998e29368a2859e1df8bade5d3862.sol
- **4d7a890e020cbbd9fca879816b69ed4d**: missing hash: graph_hash=None, token_hash=4d7a890e020cbbd9fca879816b69ed4d
- **4d7a890e020cbbd9fca879816b69ed4d**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/NonVulnerable/a009b7244579508ea1479a8aabe286e0a0bf6207835fad65fdc3ae80b83646f2.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/NonVulnerable/a009b7244579508ea1479a8aabe286e0a0bf6207835fad65fdc3ae80b83646f2.sol
- **2ab3ec5892984708f85ecbe302190bad**: missing hash: graph_hash=None, token_hash=2ab3ec5892984708f85ecbe302190bad
- **2ab3ec5892984708f85ecbe302190bad**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/IntegerUO/652a1700cc710b786ead3fb37e2933ad5de18a6db97945d5f096160f5260e6bc.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/IntegerUO/652a1700cc710b786ead3fb37e2933ad5de18a6db97945d5f096160f5260e6bc.sol
- **270e3fee74886aa7b310524b39860cc1**: missing hash: graph_hash=None, token_hash=270e3fee74886aa7b310524b39860cc1
- **270e3fee74886aa7b310524b39860cc1**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/Timestamp/e9b99453fea102f08cd4ba190d9cddc472bdf44ae9050be05ca81b7fcd56cd82.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/Timestamp/e9b99453fea102f08cd4ba190d9cddc472bdf44ae9050be05ca81b7fcd56cd82.sol
- **19ed9bb20779101b8dd932566063afdf**: missing hash: graph_hash=None, token_hash=19ed9bb20779101b8dd932566063afdf
- **19ed9bb20779101b8dd932566063afdf**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/UnusedReturn/2e968fbb73e49621353da39e91d1a7672601f1594043c0752adb998f2eff21b8.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/UnusedReturn/2e968fbb73e49621353da39e91d1a7672601f1594043c0752adb998f2eff21b8.sol
- ... and 170 more

## 6. Summary

- **Path alignment rate:** 0.0%
- **Stem↔hash verification rate:** 100.0%
- **Decode verification rate:** 10/10
