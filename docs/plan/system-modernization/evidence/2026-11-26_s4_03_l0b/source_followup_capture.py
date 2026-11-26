from pathlib import Path
import json,collections as C,subprocess as sp,csv,hashlib
base=Path("data_module/data");ev=Path("docs/plan/system-modernization/evidence/2026-11-26_s4_03_l0b")
raw=json.loads((base/"raw/dive/ingestion_manifest.json").read_text())
claimed=set()
for line in (base/"r4-v2-build/source_claims.jsonl").open():
 x=json.loads(line)
 if x["source"]=="dive":claimed.add(x["original_path"])
omitted=sorted({x["path"] for x in raw["files"]}-claimed)
# Historical reports only; never review candidate packets.
for name in ["docs/plan/ml-R4/runs/2026-08-15_PHASE8_repaired_data_acceptance_and_launch_decision.md","docs/plan/ml-R4/adrs/ADR-R4-008-repaired-v2-data-acceptance-and-phase8-no-launch.md"]:
 p=Path(name)
 print(name)
 for i,l in enumerate(p.read_text().splitlines(),1):
  if any(x in l.lower() for x in ("22,330","22,308","22 failed","22 normalized","22 rejected","22 compile","22,054","22 failures","rejection")): print(i,l[:300])
logs={}
for cat in ["Re-entrancy","Unhandled-Exceptions"]:
 counts=C.Counter()
 for p in sorted((base/"raw/solidifi/repo/buggy_contracts"/cat).glob("*.csv")):
  with p.open(newline="",encoding="utf-8-sig") as f:counts.update(row.get("bug type") for row in csv.DictReader(f))
 logs[cat]=dict(counts)
out={"omitted_dive_manifest_paths":omitted,"omitted_reason":"No per-path rejection evidence checked in this slice; not negative labels and not silent replacement.","native_log_literal_categories":logs,"interpretation":"Re-erntrancy is a spelling difference; +AC0- appears to be an encoding-like variant; semantic equivalence and line coordinate interpretation not established. Do not count literal string mismatch as verified wrong label.","class_policy":{c:json.loads(Path("docs/plan/ml-R4/specs/data_vnext_policy_v1.json").read_text())["class_supervision"][c] for c in ["GasException","UnusedReturn","DenialOfService"]},"prior_ci":{"head_sha":"ca7e47c6ce7a1aeaf49f9ee5689e0862bf0dbf06","runs":[{"name":"Portfolio showcase","id":45867843798,"conclusion":"success"},{"name":"Security hygiene","id":45867844339,"conclusion":"success"}]}}
(ev/"source_followup.json").write_text(json.dumps(out,indent=2)+"\n")
print("omitted",len(omitted),"native strings",logs)
