# D1.6 — Final Review Pass

**Target:** All 11 handbook docs in `docs/handbook/`
**Estimated time:** 1h
**Rule:** Every doc verified against source code and live testing.

---

## Checklist (run against every doc)

### Content checks
- [ ] TL;DR is understandable in 30 seconds by someone who's never seen SENTINEL
- [ ] TL;DR has: what, where (ports/paths), test command, key files
- [ ] Tour section is 1-2 pages, scannable, has ASCII diagram where useful
- [ ] Deep reference section links to `docs/learning/` and source files — no content duplication
- [ ] No doc exceeds 5 pages total (TL;DR + tour + deep reference links)

### Source verification
- [ ] Every `file:line` reference points to actual code at that line (not stale)
- [ ] Every constant/number matches source (not from memory)
- [ ] Every function name matches the actual function in source
- [ ] Every test command passes when copy-pasted into a fresh terminal

### Cross-references
- [ ] Cross-references between docs use consistent format: `→ 04_zkml.md §2`
- [ ] No broken references (every referenced doc/section exists)
- [ ] The 3 learning paths from 00_README cover all content with no gaps

### Formatting
- [ ] ASCII diagrams use box-drawing characters, render correctly in monospace
- [ ] Tables are properly formatted Markdown
- [ ] Code blocks have correct language tags (```bash, ```python, ```solidity)
- [ ] No duplication of `docs/learning/` content — only links to it

### Live verification
- [ ] A developer can start the full system using only `10_operations.md`
- [ ] A developer can understand the ML→ZK→Chain flow using only `07_cross_module.md`
- [ ] All 5 test commands in `10_operations.md` produce their stated pass counts
- [ ] All ports in `01_architecture.md` match actual running services

---

## Review order

1. **00_README.md** — verify glossary terms, learning paths
2. **01_architecture.md** — verify ports, test counts, diagram arrows
3. **02_data_module.md** — verify CLASS_NAMES, split counts, schema constants
4. **03_ml_module.md** — verify endpoints, response fields, architecture params
5. **04_zkml_module.md** — verify CIRCUIT_VERSION, param count, EZKL steps, signal count
6. **05_contracts_module.md** — verify guards, struct fields, test counts, MIN_STAKE
7. **06_agents_module.md** — verify node names, MCP tools, report fields, evidence kinds
8. **07_cross_module.md** — verify boundaries, constants table, failure modes
9. **08_security.md** — verify 8 patterns, isolation tests, Rule 5C locations
10. **09_evaluation.md** — verify Fbeta formula, 9 gates, fallback chain, α=5
11. **10_operations.md** — verify every command live, every port, every test count

---

## Final gate

After all 11 docs pass the checklist:
- [ ] Commit all docs to git
- [ ] Push to GitHub
- [ ] Update MEMORY.md with a pointer to the handbook
- [ ] Verify: a fresh clone + `docs/handbook/10_operations.md` = working system
