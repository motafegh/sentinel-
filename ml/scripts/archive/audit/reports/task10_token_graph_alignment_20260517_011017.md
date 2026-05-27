# Task 10: Token–Graph Alignment Audit
**Paired stems available:** 44470  
**Sample size:** 100
## Hash Alignment
| Metric | Count | Rate |
|--------|-------|------|
| Hash match | 1 | 100.0% |
| Hash mismatch | 0 | 0.0% |
| Missing hash | 99 | — |
| Missing graph file | 0 | — |
| Missing token file | 0 | — |

## Path Alignment
| Metric | Count | Rate |
|--------|-------|------|
| Path match | 0 | 0.0% |
| Path mismatch | 92 | 100.0% |

## Decode Verification (10 stems)
| Metric | Count | Rate |
|--------|-------|------|
| Decode match (≥90%) | 10 | 100.0% |
| Decode mismatch | 0 | — |
| Skipped | 0 | — |

## Hash/Path Mismatches Detail
- **debcbc567b2c687666f8b3dc8869b9ce**: missing hash: graph_hash=None, token_hash=debcbc567b2c687666f8b3dc8869b9ce
- **debcbc567b2c687666f8b3dc8869b9ce**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/NonVulnerable/2808bd3254a95d1173ad620070356e7dc03a6043d0415e5d54c495cd506c4f4e.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/NonVulnerable/2808bd3254a95d1173ad620070356e7dc03a6043d0415e5d54c495cd506c4f4e.sol
- **f99ff89d7251aa26295378db3d86317f**: missing hash: graph_hash=None, token_hash=f99ff89d7251aa26295378db3d86317f
- **f99ff89d7251aa26295378db3d86317f**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/NonVulnerable/cda71feaeab67ca5abef22032b8d69414f82558091a207293b4f6fa0b9cd8ead.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/NonVulnerable/cda71feaeab67ca5abef22032b8d69414f82558091a207293b4f6fa0b9cd8ead.sol
- **484058571489dda2a4f2e4f340e55bea**: missing hash: graph_hash=None, token_hash=484058571489dda2a4f2e4f340e55bea
- **484058571489dda2a4f2e4f340e55bea**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/ExternalBug/e5b41da3b8642fc855c0989d0ed39cba5a518a075a6b4b267a5c2a14c962cae0.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/ExternalBug/e5b41da3b8642fc855c0989d0ed39cba5a518a075a6b4b267a5c2a14c962cae0.sol
- **6a46469b85e2c823c3772bc07a8a3914**: missing hash: graph_hash=None, token_hash=6a46469b85e2c823c3772bc07a8a3914
- **6a46469b85e2c823c3772bc07a8a3914**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/NonVulnerable/bcc08d2f4704f35d7e60cdccb2fac12db3eed7a1a8db543d29bf589231284548.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/NonVulnerable/bcc08d2f4704f35d7e60cdccb2fac12db3eed7a1a8db543d29bf589231284548.sol
- **c932ca338ff8b9145e57af88e41f7fa9**: missing hash: graph_hash=None, token_hash=c932ca338ff8b9145e57af88e41f7fa9
- **c932ca338ff8b9145e57af88e41f7fa9**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/NonVulnerable/70d7fab384792e0b11675ba60487dfb2803a4a0f74e193952cc2b03655425c27.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/NonVulnerable/70d7fab384792e0b11675ba60487dfb2803a4a0f74e193952cc2b03655425c27.sol
- **dbcea3455f1c26a2fde0630fb9a829aa**: missing hash: graph_hash=None, token_hash=dbcea3455f1c26a2fde0630fb9a829aa
- **dbcea3455f1c26a2fde0630fb9a829aa**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/IntegerUO/4a892f0c48ef0f87679d3db5e418ba05cd521e74784b65b71bdc405ae7ce21a1.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/IntegerUO/4a892f0c48ef0f87679d3db5e418ba05cd521e74784b65b71bdc405ae7ce21a1.sol
- **4db35d9214d5b4723b21f7ebb715d791**: missing hash: graph_hash=None, token_hash=4db35d9214d5b4723b21f7ebb715d791
- **4db35d9214d5b4723b21f7ebb715d791**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/IntegerUO/7b8d192bc89ebafd6f79a82be8334bfb80ceba59e17ba853d7a42b6f98cd966d.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/IntegerUO/7b8d192bc89ebafd6f79a82be8334bfb80ceba59e17ba853d7a42b6f98cd966d.sol
- **c65792897b31025b8289ff10ecadf70f**: missing hash: graph_hash=None, token_hash=c65792897b31025b8289ff10ecadf70f
- **c65792897b31025b8289ff10ecadf70f**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/IntegerUO/3f25e6a3c6ef0fed7621e5bc467c4a8500f3cc69e61460f4ebc47a5fb2622493.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/IntegerUO/3f25e6a3c6ef0fed7621e5bc467c4a8500f3cc69e61460f4ebc47a5fb2622493.sol
- **d2299cd1a59d2451a25696061ab6a546**: missing hash: graph_hash=None, token_hash=d2299cd1a59d2451a25696061ab6a546
- **d2299cd1a59d2451a25696061ab6a546**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/NonVulnerable/97583949d28434c41210b2d60ada733a8318af0e3948dbd299eea971790f7c5c.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/NonVulnerable/97583949d28434c41210b2d60ada733a8318af0e3948dbd299eea971790f7c5c.sol
- **1a3799460cdf7eb30d0c1bdaa5f2f1a9**: missing hash: graph_hash=None, token_hash=1a3799460cdf7eb30d0c1bdaa5f2f1a9
- **1a3799460cdf7eb30d0c1bdaa5f2f1a9**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/IntegerUO/f9efc9ee62339dfdd76df443194123fadad2d89612fb122fdc422cbd38fad21d.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/IntegerUO/f9efc9ee62339dfdd76df443194123fadad2d89612fb122fdc422cbd38fad21d.sol
- **c4b80a26e8eafebb27d58e9b83e3a932**: missing hash: graph_hash=None, token_hash=c4b80a26e8eafebb27d58e9b83e3a932
- **c4b80a26e8eafebb27d58e9b83e3a932**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/NonVulnerable/4e930ae0462770c8f9887414b9f984297dc99ca873fe286cd7c206ad2534410e.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/NonVulnerable/4e930ae0462770c8f9887414b9f984297dc99ca873fe286cd7c206ad2534410e.sol
- **20fdb0e8a0eb2522627294bfd3c18f59**: missing hash: graph_hash=None, token_hash=20fdb0e8a0eb2522627294bfd3c18f59
- **20fdb0e8a0eb2522627294bfd3c18f59**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/Reentrancy/8651e5d81ed74010c672e109b956a7bfed4227b43b9a1683fb2dd42452df1333.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/Reentrancy/8651e5d81ed74010c672e109b956a7bfed4227b43b9a1683fb2dd42452df1333.sol
- **403a98b839802358876a544ef5107c23**: missing hash: graph_hash=None, token_hash=403a98b839802358876a544ef5107c23
- **403a98b839802358876a544ef5107c23**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/UnusedReturn/6d0cebbb20559b8b1921e8b42ef508078e8a7a661903c5330304f6e15aafc44e.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/UnusedReturn/6d0cebbb20559b8b1921e8b42ef508078e8a7a661903c5330304f6e15aafc44e.sol
- **918c5c21359b2894f6888bde1648a2e7**: missing hash: graph_hash=None, token_hash=918c5c21359b2894f6888bde1648a2e7
- **918c5c21359b2894f6888bde1648a2e7**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/NonVulnerable/2261ffd2d2da27c0e0b192c438ec8e147e97a3d9447bfe1d5faa806430593dea.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/NonVulnerable/2261ffd2d2da27c0e0b192c438ec8e147e97a3d9447bfe1d5faa806430593dea.sol
- **295a8238d6e158ad68b8ce0777ea5e93**: missing hash: graph_hash=None, token_hash=295a8238d6e158ad68b8ce0777ea5e93
- **295a8238d6e158ad68b8ce0777ea5e93**: path mismatch: graph=BCCC-SCsVul-2024/SourceCodes/NonVulnerable/9fbfb6d6647aeb428244e882fccbf135223944b7e1f5695af7d237ccb7627e91.sol, token=/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/NonVulnerable/9fbfb6d6647aeb428244e882fccbf135223944b7e1f5695af7d237ccb7627e91.sol
- ... and 161 more
