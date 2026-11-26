from pathlib import Path
import json,hashlib,csv,collections as C,subprocess as sp
import pyarrow.parquet as pq
root=Path("docs/plan/system-modernization");ev=root/"evidence/2026-11-26_s4_03_l1a"
ev.mkdir(exist_ok=False)
(ev/"WORKING_RECORD.md").write_text("""# S4-03-L1A native annotation calibration and recovery triage

Baseline 604d882dac7958baed50c3cafb957b5d060ae28e. Freeze small deterministic non-blind calibration and DIVE triage before running analyzers. No source code redistribution, no new truth labels. Current taxonomy.yaml and R4-D-016 govern; UnusedReturn includes any discarded value, even internal calls, not void calls; MishandledException is external failure ignored; overlap allowed. DoS requires attacker-induced persistent nonfunctionality, not merely a loop or failed transaction; GasException concerns out-of-gas failures, overlap possible but no automatic class mapping. Do not narrow definitions to detector behavior. Review rubric: exact source/component identity, source range, return type and consumption/dataflow or attacker path + essential operation + durable blocking mechanism, counterevidence, coverage limits, candidate state only. Analyzers are triage, absence never negative. Confirmed negatives and target changes remain R4-gated.

Calibration: first two lexicographic SmartBugs DoS and unchecked_low_level_calls files; first two lexicographic SolidiFI Re-entrancy and Unhandled-Exceptions files. Read annotations and logs, test 1-based bounds, raw/normalized alignment and inspect targeted examples. No inference of correctness across the corpus. SolidiFI range convention [loc, loc+length), upstream inspection script; literal labels retained. No fuzzy line mapping admitted as exact.

DIVE: two deterministic representatives per class x native positive/no mapped claim x compiler era (<0.8 / >=0.8), 16 maximum; source groups all distinct, TRAIN_UNLABELED only, exclude 200 R4 reserved group IDs from committed V3 queue (metadata only, no candidate material). Restrict normalized source <=80 KiB, <=1500 lines, installed exact cached compiler; size convenience bias recorded. Reserve selected groups in pilot manifest for review only, no optimizer use in successors pending owner decision. Select before detector results, no replacement on failure. Native no-claim strata not negative controls. Use actual Slither on copied normalized files with exact compiler, network namespace, 30s per case, bounded parallelism. Detectors: unused-return, unchecked-lowlevel, unchecked-send, calls-loop, costly-loop, return-bomb; record tool support, findings and source locations; no class truth from detector output. Compare native claim vs tool observations only. Actual manual adjudication and independent review separate L1B/L2. This is method feasibility, not label confirmation; L-G1 open.
""")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=Path("data_module/data")
export=base/"exports/sentinel-r4-vnext-v3"
m=json.loads(Path("docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/logical_v3_manifest.json").read_text())
assert sha(export/"ml_targets.parquet")==m["artifacts"]["ml_targets"]["sha256"]
assert sha(base/"r4-v2-build/source_claims.jsonl")==m["artifacts"]["claims"]["sha256"]
q=Path("docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/confirmed_negative_review_queue_v1.json")
reserved=set(json.loads(q.read_text())["reserved_group_ids"])
assert len(reserved)==200
claim=C.defaultdict(set)
for line in (base/"r4-v2-build/source_claims.jsonl").open():
 x=json.loads(line)
 if x["source"]=="dive" and x["native_category"] in ("Unchecked Return Values","DoS"):claim[x["artifact_id"]].add(x["native_category"])
targets=pq.read_table(export/"ml_targets.parquet",columns=["contract_id","source","role","group_id","target_1","target_9","effective_loss_mask_1","effective_loss_mask_9","outcome_state_1","outcome_state_9"]).to_pylist()
pools=C.defaultdict(list); excluded=C.Counter()
for x in targets:
 if x["source"]!="dive" or x["role"]!="TRAIN_UNLABELED":continue
 if x["group_id"] in reserved:excluded["r4_reserved"]+=1;continue
 aid=x["contract_id"];mp=base/"sentinel-preprocessed-r4-v2/dive"/(aid+".meta.json");sol=mp.with_name(aid+".sol")
 meta=json.loads(mp.read_text());v=meta["solc_version"];compiler=Path.home()/".solc-select/artifacts"/("solc-"+v)/("solc-"+v)
 if not compiler.is_file():excluded["compiler_missing"]+=1;continue
 if sol.stat().st_size>81920 or meta["n_normalized_lines"]>1500:excluded["size"]+=1;continue
 era="modern" if tuple(map(int,v.split(".")))>=(0,8,0) else "legacy"
 for c,n,i in [("UnusedReturn","Unchecked Return Values",9),("DenialOfService","DoS",1)]:
  assert x["target_"+str(i)] is None and not x["effective_loss_mask_"+str(i)]
  strat=(c,"native_positive" if n in claim[aid] else "no_mapped_claim",era)
  pools[strat].append({"contract_id":aid,"group_id":x["group_id"],"class":c,"native_claim":strat[1],"era":era,"role":x["role"],"outcome_state":x["outcome_state_"+str(i)],"source_path":str(sol),"source_sha256":aid,"compiler_version":v,"lines":meta["n_normalized_lines"],"bytes":sol.stat().st_size})
cases=[];used=set()
for k,items in sorted(pools.items()):
 n=0
 for x in sorted(items,key=lambda x:hashlib.sha256(("S4-L1A-v1|"+str(k)+"|"+x["group_id"]+"|"+x["contract_id"]).encode()).hexdigest()):
  if x["group_id"] in used:continue
  assert sha(Path(x["source_path"]))==x["source_sha256"]
  x["case_id"]=f"L1A-{len(cases)+1:02d}";x["reservation"]="REVIEW_ONLY_DO_NOT_ADD_TO_FUTURE_OPTIMIZER_PENDING_OWNER_DECISION"
  used.add(x["group_id"]);cases.append(x);n+=1
  if n==2:break
 assert n==2,k
assert len(cases)==16
sb=Path("ml/data/smartbugs-curated");ann=sb/"vulnerabilities.json";cal=[]
for cat in ("denial_of_service","unchecked_low_level_calls"):
 for a in sorted((a for a in json.loads(ann.read_text()) if Path(a["path"]).parent.name==cat),key=lambda x:x["path"])[:2]:
  cal.append({"source":"smartbugs_curated","raw_path":str(sb/a["path"]),"manifest_path":"repo/"+a["path"].removeprefix("dataset/"),"annotation_path":str(ann),"annotation_sha256":sha(ann),"raw_sha256":sha(sb/a["path"]),"category":cat,"annotations":a["vulnerabilities"]})
sf=base/"raw/solidifi/repo/buggy_contracts"
for cat in ("Re-entrancy","Unhandled-Exceptions"):
 for p in sorted((sf/cat).glob("*.sol"))[:2]:
  log=p.parent/("BugLog_"+p.stem.split("_")[-1]+".csv")
  cal.append({"source":"solidifi","raw_path":str(p),"manifest_path":"repo/buggy_contracts/"+cat+"/"+p.name,"annotation_path":str(log),"annotation_sha256":sha(log),"raw_sha256":sha(p),"category":cat})
out={"baseline":sp.check_output(["git","rev-parse","HEAD"],text=True).strip(),"taxonomy_sha256":sha(Path("data_module/sentinel_data/labeling/schema/taxonomy.yaml")),"queue_sha256":sha(q),"excluded_reserved_groups":len(reserved),"excluded_contract_counts":dict(excluded),"stratum_pool_contract_counts":{"|".join(k):len(v) for k,v in pools.items()},"stratum_pool_group_counts":{"|".join(k):len({x["group_id"] for x in v}) for k,v in pools.items()},"cases":cases,"calibration":cal,"limits":"Pre-analysis freeze. No negatives, no truth promotion. Selected groups reserved for review only in this program; existing frozen roles unchanged."}
(ev/"pilot_manifest.json").write_text(json.dumps(out,indent=2)+"\n")
print("frozen",len(cases),"calibration",len(cal),"pools",out["stratum_pool_group_counts"],"excluded",dict(excluded))
