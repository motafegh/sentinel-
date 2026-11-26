from pathlib import Path
import hashlib,json,subprocess as sp,collections
import pyarrow.parquet as pq
root=Path("data_module/data"); p=root/"exports/sentinel-r4-vnext-v3"
out=Path("docs/plan/system-modernization/evidence/2026-11-26_s4_01_003")
manifest_path=Path("docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/logical_v3_manifest.json")
m=json.loads(manifest_path.read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
paths={"label_states":p/"label_states.parquet","ml_targets":p/"ml_targets.parquet","claims":root/"r4-v2-build/source_claims.jsonl","source_evidence_ledger":root/"r4-v2-build/evidence_ledger_v2.parquet","source_evidence_ledger_manifest":root/"r4-v2-build/evidence_ledger_v2_manifest.json","grouping":root/"r4-v3-logical-build/grouping.json","policy":Path(m["artifacts"]["policy"]["path"])}
checks={}
for k,f in paths.items():
 v=digest(f) if f.is_file() else None
 checks[k]={"path":str(f),"actual_sha256":v,"expected_sha256":m["artifacts"][k]["sha256"],"matches":v==m["artifacts"][k]["sha256"]}
assert all(c["matches"] for c in checks.values()),checks
assert digest(p/"manifest.json")==digest(manifest_path)
w=pq.read_table(paths["ml_targets"]).to_pylist();wide={r["contract_id"]:r for r in w}
assert len(w)==len(wide)==22540
classes=m["class_names"]; rows=collections.Counter();by=collections.defaultdict(collections.Counter);src=collections.defaultdict(collections.Counter);cross=collections.defaultdict(collections.Counter);groups=collections.defaultdict(lambda:collections.defaultdict(set));claim=collections.defaultdict(collections.Counter);seen=set()
mismatch=[];role_groups=collections.defaultdict(set)
for r in w:role_groups[r["group_id"]].add(r["role"])
for batch in pq.ParquetFile(paths["label_states"]).iter_batches(batch_size=8192):
 for r in batch.to_pylist():
  c=r["class_name"];i=r["class_index"];cid=r["contract_id"];q=wide[cid];key=(cid,i)
  assert key not in seen and classes[i]==c
  seen.add(key); k=str(i)
  for field,wfield in [("target_value","target_"),("outcome_state","outcome_state_"),("training_strength","strength_"),("loss_eligible","source_loss_eligible_"),("effective_loss_mask","effective_loss_mask_"),("policy_decision_id","policy_decision_id_")]:
   if r[field]!=q[wfield+k]:mismatch.append((cid,i,field))
  if q["outcome_metric_mask_"+k] and not r["outcome_metric_eligible"]:mismatch.append((cid,i,"metric_eligibility"))
  hist=r["historical_state"];state=r["outcome_state"];target=r["target_value"];loss=r["effective_loss_mask"];metric=q["outcome_metric_mask_"+k];source=q["source"];role=q["role"]
  for ct in (rows,by[c],src[source],cross[(source,c)]):
   ct["cells"]+=1;ct["outcome:"+state]+=1;ct["historical:"+hist]+=1;ct["target:"+str(target)]+=1;ct["strength:"+r["training_strength"]]+=1;ct["effective_loss"]+=int(loss);ct["outcome_metric"]+=int(metric)
   if hist=="HISTORICAL_POSITIVE" and target is None:ct["historical_positive_masked"]+=1
   if not r["source_claims"]:ct["no_source_claims"]+=1
   if not r["evidence_ids"]:ct["no_evidence_ids"]+=1
  for typ,flag in [("target_positive",target==1),("effective_loss",loss),("outcome_metric",metric),("confirmed_negative",state=="CONFIRMED_NEGATIVE")]:
   if flag:groups[c][typ].add(q["group_id"])
  for sc in r["source_claims"] or []:
   ct=claim[str(sc.get("source"))];ct["claim_occurrences"]+=1;ct["state:"+str(sc.get("source_claim_state"))]+=1;ct["reason:"+str(sc.get("reason_code"))]+=1
   ref=sc.get("evidence_ref") or {}
   ct["raw_sha256_present"]+=int(bool(ref.get("raw_sha256")));ct["flattened_sha256_present"]+=int(bool(ref.get("flattened_sha256")))
assert len(seen)==225400 and not mismatch,mismatch[:5]
assert rows["effective_loss"]==m["effective_loss_cells"] and rows["outcome_metric"]==m["outcome_metric_cells"] and rows["outcome:CONFIRMED_NEGATIVE"]==0
assert all(len(v)==1 for v in role_groups.values())
out_data={"source_sha":sp.check_output(["git","rev-parse","HEAD"],text=True).strip(),"manifest_sha256":digest(manifest_path),"artifact_checks":checks,"total":dict(rows),"per_class":{c:{**by[c],"independent_group_counts":{k:len(v) for k,v in groups[c].items()}} for c in classes},"per_primary_source":src,"per_source_class":{f"{s}/{c}":v for (s,c),v in cross.items()},"source_claim_occurrences":claim,"role_contract_counts":collections.Counter(r["role"] for r in w),"primary_source_contract_counts":collections.Counter(r["source"] for r in w),"groups":len(role_groups),"cross_role_groups":0,"wide_long_mismatches":len(mismatch),"limitations":"Hash equality against frozen manifest, exact label/target projection checks; no source-content verification, class truth adjudication, compiler audit or physical representation rehash. Group counts are distinct recorded groups, not statistically independent samples."}
(out/"label_inventory.json").write_text(json.dumps(out_data,indent=2,sort_keys=True)+"\n")
print("TOTAL",dict(rows));print("SOURCES",out_data["primary_source_contract_counts"])
for c in classes:print(c,dict(by[c]),out_data["per_class"][c]["independent_group_counts"])
print("CLAIMS",[(k,v["claim_occurrences"],v["raw_sha256_present"],v["flattened_sha256_present"]) for k,v in claim.items()])
