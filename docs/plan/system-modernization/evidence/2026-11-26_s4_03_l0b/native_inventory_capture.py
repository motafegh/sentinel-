from pathlib import Path
import collections as C,csv,json,hashlib,subprocess as sp,time
from sentinel_data.vnext.r4_source_claims import claims_for_meta,load_dive_labels
root=Path("data_module/data"); ev=Path("docs/plan/system-modernization/evidence/2026-11-26_s4_03_l0b")
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def textsha(s):return hashlib.sha256(s.encode("utf-8")).hexdigest()
def key(x):return json.dumps(x,sort_keys=True,separators=(",",":"))
def dump(x): (ev/"native_inventory.json").write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
policy_path=Path("docs/plan/ml-R4/specs/data_vnext_policy_v1.json")
policy=json.loads(policy_path.read_text())
m=json.loads(Path("docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/logical_v3_manifest.json").read_text())
csvp=root/"raw_staging/dive_labels/DIVE_Labels.csv"
dive=load_dive_labels(csvp)
claims_path=root/"r4-v2-build/source_claims.jsonl"
assert sha(claims_path)==m["artifacts"]["claims"]["sha256"]
assert sha(policy_path)==m["artifacts"]["policy"]["sha256"]
native=C.Counter()
for v in dive.values():native.update(v)
claimed=C.Counter(); by_source=C.defaultdict(C.Counter); paths_by_source=C.defaultdict(set); ids_by_source=C.defaultdict(set); by_category=C.defaultdict(C.Counter)
for line in claims_path.open():
 x=json.loads(line);claimed[key(x)]+=1;s=x["source"];cat=x["native_category"] or "(none)"
 by_source[s]["claims"]+=1;by_source[s]["positive_claims"]+=int(x["source_claim_state"]=="POSITIVE")
 paths_by_source[s].add(x["original_path"]);ids_by_source[s].add(x["source_record_id"])
 by_category[s][cat]+=1
out={"source_sha":sp.check_output(["git","rev-parse","HEAD"],text=True).strip(),"policy_sha256":sha(policy_path),"claims_sha256":sha(claims_path),"dive_csv":{"path":str(csvp),"sha256":sha(csvp),"rows":len(dive),"native_positive_counts":dict(native),"no_positive_rows":sum(not v for v in dive.values())},"claims":{"rows":sum(claimed.values()),"exact_distinct_rows":len(claimed),"duplicates":sum(v-1 for v in claimed.values()),"by_source":{s:{**ct,"distinct_paths":len(paths_by_source[s]),"distinct_source_ids":len(ids_by_source[s]),"categories":dict(by_category[s])} for s,ct in by_source.items()}},"sources":{}}
replay=C.Counter()
for s in ("dive","smartbugs_curated","solidifi"):
 base=root/"raw"/s; mp=base/"ingestion_manifest.json";j=json.loads(mp.read_text());entries={e["path"]:e for e in j["files"]}
 assert len(entries)==len(j["files"])
 ct=C.Counter(); mismatch=[]; meta_mismatch=[];meta_paths=set(); compiler=C.Counter(); statuses=C.Counter(); meta_ids=set()
 for path,e in entries.items():
  p=base/path
  if not p.is_file():ct["missing_raw"]+=1;mismatch.append(path);continue
  b=p.read_bytes()
  if hashlib.sha256(b).hexdigest()!=e["sha256"] or len(b)!=e["size_bytes"]:ct["raw_hash_or_size_mismatch"]+=1;mismatch.append(path)
  else:ct["raw_exact"]+=1
 for p in sorted((root/"sentinel-preprocessed-r4-v2"/s).glob("*.meta.json")):
  x=json.loads(p.read_text());ct["meta_files"]+=1;aid=x["sha256"];meta_ids.add(aid)
  compiler[str(x.get("solc_version"))]+=1;statuses[str(x.get("compile_status"))]+=1
  normalized=p.with_name(p.name.removesuffix(".meta.json")+".sol")
  if normalized.is_file() and sha(normalized)==aid==x["normalized_text_sha256"]:ct["normalized_exact"]+=1
  else:ct["normalized_mismatch"]+=1;meta_mismatch.append([p.name,"normalized"])
  for r in x["source_records"]:
   path=r["original_path"];meta_paths.add(path);ct["source_record_occurrences"]+=1
   if r["ingestion_entry"]!=entries.get(path):ct["ingestion_ref_mismatch"]+=1;meta_mismatch.append([p.name,"ingestion"])
   raw=base/path
   if raw.is_file():
    raw_hash=textsha(raw.read_text(encoding="utf-8",errors="replace"))
    if raw_hash==r["raw_sha256"] and textsha(s+"\0"+path+"\0"+raw_hash)==r["source_record_id"]:ct["source_identity_exact"]+=1
    else:ct["source_identity_mismatch"]+=1;meta_mismatch.append([p.name,"identity"])
  for x in claims_for_meta(s,x,policy,dive_labels=dive if s=="dive" else None):replay[key(x)]+=1
 source_root=base/"repo" if s=="solidifi" else Path("ml/data/smartbugs-curated") if s=="smartbugs_curated" else None
 origin={"manifest_sha256":sha(mp),"manifest_pin":j.get("resolved_pin"),"manifest_count":len(entries)}
 if source_root:
  origin["local_git_head"]=sp.check_output(["git","-C",str(source_root),"rev-parse","HEAD"],text=True).strip()
  origin["tracked_dirty"]=bool(sp.check_output(["git","-C",str(source_root),"status","--porcelain","--untracked-files=no"],text=True).strip())
  lic=source_root/"LICENSE"
  origin["license_file_sha256"]=sha(lic) if lic.is_file() else None
  origin["license_scope"]="Repository notice present; per-contract original license obligations not adjudicated."
 else:origin["license_scope"]="No portable archive locator/license verification established."
 out["sources"][s]={"origin":origin,"counts":dict(ct),"unique_meta_artifacts":len(meta_ids),"unique_meta_paths":len(meta_paths),"unrepresented_manifest_paths":len(set(entries)-meta_paths),"claim_meta_path_difference":len(meta_paths^paths_by_source[s]),"compiler_metadata_counts":dict(compiler),"compile_status_metadata_counts":dict(statuses),"raw_mismatch_examples":mismatch[:10],"meta_mismatch_examples":meta_mismatch[:10]}
 print(s,dict(ct),"manifest",len(entries),"unrepresented",len(set(entries)-meta_paths),flush=True)
out["claim_replay"]={"actual_rows":sum(replay.values()),"missing":sum((claimed-replay).values()),"extra":sum((replay-claimed).values()),"exact_multiset_match":replay==claimed}
# Native annotation availability and category cross-check, not truth.
sb=Path("ml/data/smartbugs-curated");p=sb/"vulnerabilities.json";ann=json.loads(p.read_text());counts=C.Counter();diff=[]
for a in ann:
 path=a["path"];parts=Path(path).parts;folder=parts[-2];v=a.get("vulnerabilities",[])
 counts["records"]+=1;counts["annotation_entries"]+=len(v)
 if not (sb/path).is_file():counts["missing_source"]+=1;continue
 counts["source_present"]+=1
 n=len((sb/path).read_text(encoding="utf-8",errors="replace").splitlines())
 for e in v:
  counts["category:"+str(e["category"])]+=1
  if e["category"]!=folder:counts["folder_category_disagreement"]+=1;diff.append({"path":path,"folder":folder,"annotation":e["category"]})
  if e.get("lines"):counts["with_line_spans"]+=1
  if any(type(i)!=int or not 1<=i<=n for i in e.get("lines",[])):counts["out_of_bounds_line_entry"]+=1
out["smartbugs_native_annotations"]={"path":str(p),"sha256":sha(p),"counts":dict(counts),"disagreement_examples":diff[:10]}
sf=root/"raw/solidifi/repo/buggy_contracts";ct=C.defaultdict(C.Counter)
for p in sorted(sf.glob("*/*.sol")):
 cat=p.parent.name; c=ct[cat];c["sources"]+=1
 log=p.parent/("BugLog_"+p.stem.split("_")[-1]+".csv")
 if not log.is_file():c["missing_log"]+=1;continue
 c["log_present"]+=1
 with log.open(newline="",encoding="utf-8-sig") as f:
  for row in csv.DictReader(f):
   c["injection_records"]+=1
   if row.get("bug type")!=cat:c["category_disagreements"]+=1
out["solidifi_native_injection_logs"]={k:dict(v) for k,v in ct.items()}
out["limitations"]=["Compiler counts from historical metadata, not compilation rerun.","Local Git pin and license notice do not establish external redistributability of all contracts.","Exact builder replay is reproducibility, not label truth.","Native source categories and annotation lines are candidate evidence, not independently verified findings.","DIVE source labels are automated, original archive locator remains unverified; no zero is negative."]
dump(out)
print("DIVE",out["dive_csv"],"CLAIMS",out["claims"],"REPLAY",out["claim_replay"],"SMARTBUGS",out["smartbugs_native_annotations"],"SOLIDIFI",out["solidifi_native_injection_logs"],flush=True)
