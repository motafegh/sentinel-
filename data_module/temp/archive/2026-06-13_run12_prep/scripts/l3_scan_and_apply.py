"""L3 dedup scan + selective application.

1. Normalize all preprocessed .sol files (strip comments + collapse whitespace).
2. Group by normalized hash → L3 groups.
3. For each group with 2+ members, check label consistency from merged labels.
4. Save dedup_groups_l3_candidates.json with full analysis.
5. Apply L3 to label-consistent groups (0 conflicts) → update dedup_groups_graph_hash.json.
"""

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DATA = Path("/home/motafeq/projects/sentinel/data_module/data")
PREPROCESSED_SOURCES = ["dive", "solidifi", "smartbugs_curated"]

_BLOCK_CMT = re.compile(r'/\*.*?\*/', re.DOTALL)
_LINE_CMT  = re.compile(r'//[^\n]*')
_WHITESPACE = re.compile(r'\s+')


def normalize(content: str) -> str:
    out = _BLOCK_CMT.sub('', content)
    out = _LINE_CMT.sub('', out)
    out = _WHITESPACE.sub(' ', out).strip()
    return out


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def positive_classes(label_path: Path) -> frozenset:
    """Return the set of class names with value=1 in a merged .labels.json."""
    try:
        d = json.loads(label_path.read_text())
        return frozenset(k for k, v in d["classes"].items() if v["value"] == 1)
    except Exception:
        return frozenset()


# ── Step 1: load all preprocessed .sol files ──────────────────────────────────
print("Loading preprocessed .sol files...", flush=True)
norm_hash_to_contracts: dict[str, list[dict]] = defaultdict(list)
total = 0

for source in PREPROCESSED_SOURCES:
    src_dir = DATA / "preprocessed" / source
    if not src_dir.exists():
        print(f"  SKIP {source} (not found)")
        continue
    sol_files = sorted(src_dir.glob("*.sol"))
    for sol_path in sol_files:
        sha = sol_path.stem
        content = sol_path.read_text(errors="replace")
        nh = sha256(normalize(content))
        norm_hash_to_contracts[nh].append({"sha256": sha, "source": source})
        total += 1
    print(f"  {source}: {len(sol_files)} contracts")

print(f"Total: {total} contracts, {len(norm_hash_to_contracts)} unique norm-hashes")

# ── Step 2: identify L3 groups (norm-hash with 2+ members) ────────────────────
l3_groups_raw = {nh: members for nh, members in norm_hash_to_contracts.items()
                 if len(members) >= 2}
print(f"L3 groups (2+ members): {len(l3_groups_raw)}")
contracts_in_groups = sum(len(v) for v in l3_groups_raw.values())
print(f"Contracts in L3 groups: {contracts_in_groups}")

# ── Step 3: check label consistency ───────────────────────────────────────────
merged_label_dir = DATA / "labels" / "merged"

groups_detail = []
n_conflicts = 0
n_consistent = 0
n_no_labels = 0
consistent_groups = []   # norm_hash where labels agree

for nh, members in l3_groups_raw.items():
    sources_present = sorted(set(m["source"] for m in members))
    is_cross_source = len(sources_present) > 1

    label_sets = []
    for m in members:
        lp = merged_label_dir / f"{m['sha256']}.labels.json"
        if lp.exists():
            label_sets.append(positive_classes(lp))
        else:
            label_sets.append(None)   # no merged label

    # Filter out None to check consistency among those with labels
    real_label_sets = [ls for ls in label_sets if ls is not None]
    has_conflict = len(set(frozenset(ls) for ls in real_label_sets)) > 1 if real_label_sets else False

    if not real_label_sets:
        n_no_labels += 1
        status = "no_labels"
    elif has_conflict:
        n_conflicts += 1
        status = "conflict"
    else:
        n_consistent += 1
        status = "consistent"
        consistent_groups.append(nh)

    # Canonical = first sha256 in sorted order
    canonical = sorted(m["sha256"] for m in members)[0]

    groups_detail.append({
        "norm_hash": nh,
        "canonical_sha256": canonical,
        "member_count": len(members),
        "members": members,
        "sources": sources_present,
        "is_cross_source": is_cross_source,
        "label_sets": [sorted(ls) if ls is not None else None for ls in label_sets],
        "status": status,
    })

print(f"\nConsistent groups (apply L3): {n_consistent}")
print(f"Conflicting groups (skip):    {n_conflicts}")
print(f"No-labels groups (skip):      {n_no_labels}")
print(f"FP rate (conflicts/total):    {n_conflicts / len(l3_groups_raw) * 100:.1f}%")

# ── Step 4: save l3 candidates JSON ───────────────────────────────────────────
candidates_path = DATA / "dedup_groups_l3_candidates.json"
candidates = {
    "n_l3_groups": len(l3_groups_raw),
    "n_contracts_in_l3_groups": contracts_in_groups,
    "n_consistent": n_consistent,
    "n_conflicts": n_conflicts,
    "n_no_labels": n_no_labels,
    "fp_rate_pct": round(n_conflicts / len(l3_groups_raw) * 100, 1),
    "consistent_norm_hashes": consistent_groups,
    "groups": groups_detail,
}
candidates_path.write_text(json.dumps(candidates, indent=2))
print(f"\nSaved {candidates_path}")

# ── Step 5: apply consistent L3 groups to dedup_groups_graph_hash.json ────────
graph_hash_path = DATA / "dedup_groups_graph_hash.json"
orig = json.loads(graph_hash_path.read_text())
groups_map: dict[str, str] = orig["groups"]  # sha256 → canonical sha256

# For each consistent L3 group, reassign all members to the canonical sha256.
# Canonical = the group's current canonical (the one that already has the earliest
# position in the existing graph-hash groups map, if any member is already canonical
# there, or just sorted-first among members).
n_reassigned = 0
for nh in consistent_groups:
    detail = next(g for g in groups_detail if g["norm_hash"] == nh)
    members_sha = [m["sha256"] for m in detail["members"]]
    # Pick canonical: whichever member is already a canonical in graph-hash groups,
    # preferring the one with the fewest characters (tiebreak: alphabetically first)
    current_canonicals = [sha for sha in members_sha if groups_map.get(sha) == sha]
    if current_canonicals:
        l3_canonical = sorted(current_canonicals)[0]
    else:
        # All members are non-canonical in graph-hash; use graph-hash canonical of first member
        l3_canonical = groups_map.get(members_sha[0], members_sha[0])

    for sha in members_sha:
        old_canonical = groups_map.get(sha, sha)
        if old_canonical != l3_canonical:
            groups_map[sha] = l3_canonical
            n_reassigned += 1

print(f"Reassigned {n_reassigned} contracts to new L3 canonicals")

# Recompute stats
all_canonicals = set(groups_map.values())
canonical_counts: dict[str, int] = defaultdict(int)
for v in groups_map.values():
    canonical_counts[v] += 1

n_unique = len(all_canonicals)
n_singleton = sum(1 for c, cnt in canonical_counts.items() if cnt == 1)
n_dup = n_unique - n_singleton
n_in_dup = sum(cnt for cnt in canonical_counts.values() if cnt > 1)

orig["n_unique_groups"] = n_unique
orig["n_singleton_groups"] = n_singleton
orig["n_dup_groups"] = n_dup
orig["n_contracts_in_dup_groups"] = n_in_dup
orig["groups"] = groups_map
orig["l3_applied"] = True
orig["l3_consistent_groups_applied"] = n_consistent

graph_hash_path.write_text(json.dumps(orig, indent=2))
print(f"Updated {graph_hash_path}")
print(f"  n_contracts:         {orig['n_contracts']}")
print(f"  n_unique_groups:     {n_unique}")
print(f"  n_singleton_groups:  {n_singleton}")
print(f"  n_dup_groups:        {n_dup}")
print(f"  n_in_dup_groups:     {n_in_dup}")
print(f"\nStep B scan + apply DONE.")
