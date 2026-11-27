from pathlib import Path
import json,hashlib,collections as C
import pyarrow.parquet as pq
p=Path("docs/plan/system-modernization/evidence/2026-11-27_s4_03_l1b");prev=p.parent/"2026-11-26_s4_03_l1a"
m={c["case_id"]:c for c in json.loads((prev/"pilot_manifest.json").read_text())["cases"]}
r={x["case_id"]:x["inventory"] for x in json.loads((p/"inventory_results.json").read_text())["results"]}
def sha(x):return hashlib.sha256(x.read_bytes()).hexdigest()
base=Path("data_module/data");export=base/"exports/sentinel-r4-vnext-v3";manifest=Path("docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/logical_v3_manifest.json")
auth=json.loads(manifest.read_text());assert sha(export/"ml_targets.parquet")==auth["artifacts"]["ml_targets"]["sha256"];assert sha(base/"r4-v2-build/source_claims.jsonl")==auth["artifacts"]["claims"]["sha256"]
wide={x["contract_id"]:x for x in pq.read_table(export/"ml_targets.parquet",columns=["contract_id","group_id","role","target_1","target_9","effective_loss_mask_1","effective_loss_mask_9","outcome_state_1","outcome_state_9"]).to_pylist()}
claims=C.defaultdict(set);ids={x["contract_id"] for x in m.values()}
for line in (base/"r4-v2-build/source_claims.jsonl").open():
 x=json.loads(line)
 if x["artifact_id"] in ids and x["source"]=="dive":claims[x["artifact_id"]].add(x["native_category"])
# Source-grounded primary observations, not label adjudications.
notes=[
("L1A-01","DenialOfService","INSUFFICIENT",[[142,163]],"Owner-restricted batch distribution and a single-recipient alternative. Cost alone does not establish attacker-induced permanent loss of function.","NOT_ESTABLISHED","Structural inventory plus targeted source review; other paths not exhaustively proven."),
("L1A-02","DenialOfService","INSUFFICIENT",[[558,568],[764,770],[788,802]],"Loop requires owner; token-check modifier contains restrictions but no demonstrated attacker-controlled permanent blocking mechanism in reviewed evidence.","NOT_ESTABLISHED","Structural inventory plus targeted source review; other paths not exhaustively proven."),
("L1A-03","DenialOfService","INSUFFICIENT",[[1045,1066],[1207,1250],[1289,1308],[1375,1380]],"Receiver callback and enumeration writes occur in mint-related loops. Supply and non-owner mint amount limits; owner airdrop. No durable attacker-induced essential-function blockage established.","NOT_ESTABLISHED","Structural inventory plus targeted source review; no full environmental proof."),
("L1A-04","DenialOfService","INSUFFICIENT",[[656,665],[711,720],[726,757]],"No requested DoS detector hit. Parsed loop in owner-only fee-exclusion batch; external router setup and ordinary fee logic do not establish permanent blocking. Other paths unresolved.","NOT_ESTABLISHED","Structural inventory and selected loop/call paths reviewed; not complete semantic assessment."),
("L1A-08","UnusedReturn","SUPPORTS_CLASS_CANDIDATE",[[24,24],[108,115],[328,337]],"Typed approve boolean and router three-value result are discarded in standalone call statements; literal class applies at these sites. Router nested arguments make helper conservatively omit that call.","NOT_ESTABLISHED","Existential positive witness; all components parsed, not full semantic review."),
("L1A-09","UnusedReturn","INSUFFICIENT",[[1,102]],"Read all source; low-level success is checked with require, reseed is void, lucky-number result consumed/returned. No supported discarded nonvoid result identified by review or helper. Builtin and complete semantic coverage not independently established.","NOT_ESTABLISHED","Full source read and structural inventory, no negative adjudication."),
("L1A-10","UnusedReturn","INSUFFICIENT",[[1,140],[145,233]],"Reviewed typed callback declaration is void, raw call result checked by assert, transfer helpers returned, value transfer has no return. No supporting witness identified; unreviewed/ambiguous scope prevents negative.","NOT_ESTABLISHED","Near-complete source read, structural inventory, no independent assessment."),
("L1A-11","UnusedReturn","SUPPORTS_CLASS_CANDIDATE",[[84,91],[260,269],[399,402],[425,436],[526,529]],"Typed router triple result and internal basic-transfer boolean discarded. Internal function returns true on successful completion; ignoring a constant-success return does not itself establish security harm.","NOT_ESTABLISHED","Existential witness; full parsed inventory, targeted call and declaration review."),
("L1A-12","UnusedReturn","SUPPORTS_CLASS_CANDIDATE",[[827,842]],"Low-level call returns a tuple; second component discarded, success assigned but no subsequent read in the complete enclosing function. Search of source shows only declaration and assignment of success. Direct-expression helper excludes assignment; primary source witness outside its scope.","NOT_ESTABLISHED","Targeted full enclosing function and references read; no exploitability conclusion."),
("L1A-01","UnusedReturn","SUPPORTS_CLASS_CANDIDATE",[[118,118],[133,140],[208,208],[216,221]],"Internal distr function declares bool and returns true on success; five standalone calls ignore that return. Literal class applicability supported, not security significance.","NOT_ESTABLISHED","Post hoc cross-class candidate, targeted source declaration and calls."),
("L1A-02","UnusedReturn","SUPPORTS_CLASS_CANDIDATE",[[553,568]],"Internal transfer returns bool; owner-only batch caller discards return. Implementation returns true on successful completion. Literal discarded-return observation, no demonstrated material risk.","NOT_ESTABLISHED","Post hoc cross-class candidate, targeted source declaration and call.")]
rows=[]
for cid,cls,disposition,ranges,reason,impact,scope in notes:
 a=m[cid];w=wide[a["contract_id"]];i=1 if cls=="DenialOfService" else 9;native="DoS" if i==1 else "Unchecked Return Values";src=Path(a["source_path"])
 assert sha(src)==a["source_sha256"]==r[cid]["source_sha256"] and w["group_id"]==a["group_id"] and w["role"]=="TRAIN_UNLABELED"
 assert w["target_"+str(i)] is None and not w["effective_loss_mask_"+str(i)]
 n=len(src.read_text().splitlines());assert all(1<=lo<=hi<=n for lo,hi in ranges)
 rows.append({"case_id":cid,"class":cls,"class_index":i,"contract_id":a["contract_id"],"group_id":a["group_id"],"source_sha256":a["source_sha256"],"source_path":a["source_path"],"compiler_version":a["compiler_version"],"native_claim_present":native in claims[a["contract_id"]],"prior_outcome":w["outcome_state_"+str(i)],"prior_target":None,"effective_loss_mask":False,"primary_disposition":disposition,"evidence_ranges":ranges,"reasoning":reason,"security_impact":impact,"review_scope":scope,"independent_review":"PENDING","admission":"NOT_ADMITTED","selection":"POST_HOC_CROSS_CLASS" if cid in ("L1A-01","L1A-02") and cls=="UnusedReturn" else "L1B_PLANNED","target_change_authorized":False})
assert len({(x["contract_id"],x["class_index"]) for x in rows})==len(rows)
out={"baseline":"820bc1457f3260c54c946c704f7695240b19f1d4","taxonomy_sha256":sha(Path("data_module/sentinel_data/labeling/schema/taxonomy.yaml")),"v3_manifest_sha256":sha(manifest),"pilot_manifest_sha256":sha(prev/"pilot_manifest.json"),"records":rows,"counts":dict(C.Counter(x["primary_disposition"] for x in rows)),"admitted":0,"limitations":"Primary non-blind, partial scope where stated; no negative, no candidate accepted, no optimizer role change."}
(p/"primary_reviews.json").write_text(json.dumps(out,indent=2)+"\n")
print("inventory",len(r),"functions",sum(len(x["functions"]) for x in r.values()),"direct sites",sum(len(x["direct_candidates"]) for x in r.values()),"direct contracts",sum(bool(x["direct_candidates"]) for x in r.values()))
print("reviews",out["counts"],"positive native claim",dict(C.Counter(str(x["native_claim_present"]) for x in rows if x["primary_disposition"]=="SUPPORTS_CLASS_CANDIDATE")))
