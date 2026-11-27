from pathlib import Path
import collections as C
import hashlib,json,urllib.request,io,zipfile,time,subprocess as sp
import pyarrow as pa
import pyarrow.parquet as pq
from sentinel_data.vnext.dive_observations import *

ROOT=Path("docs/plan/system-modernization/evidence")
EV=ROOT/"2026-11-27_s4_03_l3b0b"
BASE=Path("data_module/data")
OUT=BASE/"s4-dive-observations-v1"
V3=BASE/"exports/sentinel-r4-vnext-v3"
SNAP=Path("docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3")
def read(p):return json.loads(p.read_text())
def sha(p):return digest(p.read_bytes())
def write(p,obj):p.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
start=time.monotonic()
assert not OUT.exists()
assert sp.run(["git","check-ignore","-q",str(OUT/"manifest.json")]).returncode==0
m=read(SNAP/"logical_v3_manifest.json")
claims=BASE/"r4-v2-build/source_claims.jsonl"
targets=V3/"ml_targets.parquet"
def guard():
    assert sha(claims)==m["artifacts"]["claims"]["sha256"]
    assert sha(targets)==m["artifacts"]["ml_targets"]["sha256"]
guard()
prev=read(ROOT/"2026-11-27_s4_03_l3b/dive_label_comparison.json")
f=prev["expected_archive"]
with urllib.request.urlopen(urllib.request.Request(f["links"]["self"],headers={"User-Agent":"Sentinel-public-metadata-research/1.0"}),timeout=15) as r:
    b=r.read(f["size"]+1)
assert len(b)==f["size"] and digest(b)==prev["archive_sha256"]
with zipfile.ZipFile(io.BytesIO(b)) as z:
    q=z.read("Labels/Tool_Results.csv")
    v=z.read("Labels/DIVE_Labels.csv")
assert digest(q)=="dd4226b8024ad1b9a9f8d12d43de2ea4125ed63d3548229678fdfcf67161fb70" and digest(v)==prev["published_csv_sha256"]
rows=list(observation_rows(q));labels=published_labels(v)
view=list(category_view(rows,labels))
assert len(rows)==26071 and len(labels)==22330 and len(view)==178640
assert len({x["row_id"] for x in rows})==len(rows)
print("parsed rows and category view",len(rows),len(view),flush=True)
target_rows={r["contract_id"]:r for r in pq.read_table(targets).to_pylist()}
# Exact original record path linkage from accepted claims. No source file contents from reserved groups.
links={};raws=BASE/"raw_staging/dive/__source__"
for line in claims.open():
    x=json.loads(line)
    if x["source"]!="dive":continue
    p=Path(x["original_path"]);cid=p.stem
    assert p.parent.as_posix()=="repo/__source__" and p.suffix==".sol" and cid in labels
    e=x["evidence_ref"]["ingestion_entry"]
    assert e["path"]==x["original_path"]
    a=x["artifact_id"];t=target_rows[a]
    assert t["source"]=="dive"
    record={"native_id":cid,"source_record_id":x["source_record_id"],"raw_sha256":e["sha256"],"raw_bytes":e["size_bytes"],"contract_id":a,"group_id":t["group_id"],"role":t["role"],"link_state":"ACCEPTED_SOURCE_RECORD_PATH"}
    if cid in links and links[cid]!=record:raise ValueError("ambiguous path link")
    links[cid]=record
assert len(links)==22308
# Check on-disk filename set; original hash provenance established L0B, do not read reserved raw content.
assert {p.stem for p in raws.glob("*.sol")}==set(labels)
all_links=[links.get(cid,{"native_id":cid,"link_state":"NOT_IN_ACCEPTED_SOURCE_RECORDS"}) for cid in sorted(labels,key=int)]
assert len({(x["native_id"],x.get("source_record_id")) for x in all_links})==22330
reserved=set(read(SNAP/"confirmed_negative_review_queue_v1.json")["reserved_group_ids"])
l1a={x["group_id"] for x in read(ROOT/"2026-11-26_s4_03_l1a/pilot_manifest.json")["cases"]}
assert len(reserved)==200 and len(l1a)==16 and not reserved&l1a
prior=reserved|l1a
quotas=[("time_unsupported_positive",9),("ur_zero_positive",4),("dos_zero_positive",4),("ur_zero_no_positive",2),("dos_zero_no_positive",2)]
pools=C.defaultdict(list);excluded=C.defaultdict(C.Counter);stratum_counts=C.Counter()
meta_cache={}
for x in view:
    cat=x["native_category"];pub=x["published_annotation"];pos=x["positive_observation_tools"]
    if cat=="Time manipulation" and pub==1 and pos==0:st,idx="time_unsupported_positive",7
    elif cat=="Unchecked Return Values" and pub==0:st,idx=("ur_zero_positive" if pos else "ur_zero_no_positive"),9
    elif cat=="DoS" and pub==0:st,idx=("dos_zero_positive" if pos else "dos_zero_no_positive"),1
    else:continue
    stratum_counts[st]+=1
    cid=x["native_id"];l=links.get(cid)
    if l is None:excluded[st]["no_accepted_link"]+=1;continue
    if l["group_id"] in prior:excluded[st]["reserved"]+=1;continue
    t=target_rows[l["contract_id"]]
    if t["role"]!="TRAIN_UNLABELED":excluded[st]["role"]+=1;continue
    if t[f"target_{idx}"] is not None or t[f"effective_loss_mask_{idx}"]:excluded[st]["target_or_mask"]+=1;continue
    aid=l["contract_id"]
    if aid not in meta_cache:
        mp=BASE/"sentinel-preprocessed-r4-v2/dive"/(aid+".meta.json")
        meta=read(mp);sol=mp.with_name(aid+".sol")
        ver=meta["solc_version"]
        compiler=Path.home()/".solc-select/artifacts"/("solc-"+ver)/("solc-"+ver)
        meta_cache[aid]=(sol,ver,meta["n_normalized_lines"],sol.stat().st_size,compiler.is_file())
    sol,ver,lines,size,compiler_ok=meta_cache[aid]
    if not compiler_ok:excluded[st]["compiler_missing"]+=1;continue
    if size>81920 or lines>1500:excluded[st]["size"]+=1;continue
    pools[st].append({**l,"class_index":idx,"class_name":m["class_names"][idx],"native_category":cat,"published_annotation":pub,"positive_observation_tools":pos,"observation_row_ids":x["observation_row_ids"],"source_path":str(sol),"compiler_version":ver,"lines":lines,"bytes":size,"prior_outcome":t[f"outcome_state_{idx}"],"prior_target":None,"prior_effective_loss_mask":False,"admission":"NOT_ADMITTED","independent_review":"PENDING"})
assert stratum_counts["time_unsupported_positive"]==9 and stratum_counts["ur_zero_positive"]==880 and stratum_counts["dos_zero_positive"]==2289
selected=pick_groups(pools,quotas,prior,"S4-L3B0B-v1")
new_groups={x["group_id"] for x in selected}
assert len(new_groups)==len(selected) and not new_groups&prior
for i,x in enumerate(selected,1):
    assert sha(Path(x["source_path"]))==x["contract_id"]
    x["case_id"]=f"L3B0B-{i:02d}";x["reservation"]="REVIEW_ONLY_NO_FUTURE_OPTIMIZER_PENDING_OWNER_DECISION"
print("linked",len(links),"artifacts",len({x["contract_id"] for x in links.values()}),"selected",C.Counter(x["stratum"] for x in selected),flush=True)
OUT.mkdir(mode=0o700,exist_ok=False)
# Explicit nullable types, no target or loss columns in observation views.
row_schema=pa.schema([("row_id",pa.string()),("row_ordinal",pa.int32()),("native_id",pa.string())]+[(k,pa.int8()) for k in COLUMNS])
link_schema=pa.schema([(k,pa.int64() if k=="raw_bytes" else pa.string()) for k in ("native_id","source_record_id","raw_sha256","raw_bytes","contract_id","group_id","role","link_state")])
for name,data,schema in [("tool_rows.parquet",rows,row_schema),("native_category_view.parquet",view,None),("source_links.parquet",all_links,link_schema)]:
    table=pa.Table.from_pylist(data,schema=schema)
    pq.write_table(table,OUT/name,compression="zstd")
    assert pq.read_table(OUT/name).to_pylist()==table.to_pylist()
write(OUT/"review_queue.json",{"schema":"s4-dive-disagreement-queue-v1","cases":selected,"quotas":dict(quotas),"reserved_group_ids":sorted(prior|new_groups),"scope":"Review only; no label, optimizer, or successor role authorization."})
write(OUT/"manifest.json",{"schema":SCHEMA,"archive_sha256":digest(b),"tool_sha256":digest(q),"label_sha256":digest(v),"v3_manifest_sha256":sha(SNAP/"logical_v3_manifest.json"),"input_claims_sha256":sha(claims),"input_targets_sha256":sha(targets),"artifacts":{p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in OUT.iterdir()},"source":"https://zenodo.org/records/18519253","license":"CC BY 4.0 dataset; third-party source rights separate","training_authorized":False})
guard()
assert_optimizer_reservations([x["group_id"] for x in target_rows.values() if x["role"] in ("TRAIN_STRONG","TRAIN_WEAK")],prior|new_groups)
try:assert_optimizer_reservations(new_groups,prior|new_groups)
except ValueError:guard_reject=True
else:raise AssertionError("reservation check failed")
summary={"schema":SCHEMA,"artifact_manifest_sha256":sha(OUT/"manifest.json"),"artifacts":read(OUT/"manifest.json")["artifacts"],"rows":len(rows),"ids":len(labels),"category_cells":len(view),"literal_tool_cells":len(rows)*48,"null_tool_cells":sum(x[k] is None for x in rows for k in COLUMNS),"one_tool_cells":sum(x[k]==1 for x in rows for k in COLUMNS),"zero_tool_cells":sum(x[k]==0 for x in rows for k in COLUMNS),"conflicting_native_cells":sum(x["conflicting_tools"]>0 for x in view),"source_records_linked":len(links),"unlinked_native_ids":len(labels)-len(links),"distinct_linked_artifacts":len({x["contract_id"] for x in links.values()}),"distinct_linked_groups":len({x["group_id"] for x in links.values()}),"strata_total":dict(stratum_counts),"strata_eligible":{k:len(v) for k,v in pools.items()},"strata_eligible_groups":{k:len({x["group_id"] for x in v}) for k,v in pools.items()},"exclusions":{k:dict(v) for k,v in excluded.items()},"selected":dict(C.Counter(x["stratum"] for x in selected)),"prior_reserved_groups":len(prior),"new_reserved_groups":len(new_groups),"combined_reserved_groups":len(prior|new_groups),"guard_rejects_overlap":guard_reject,"accepted_v3_hashes_unchanged":True,"labels_admitted":0,"training_authorized":False,"elapsed_seconds":round(time.monotonic()-start,3),"limitations":"Source record join, not original analyzer source/run binding; 22 unlinked IDs retained. Local-only ledger/queue; guard not wired into future builder. No review or source execution."}
write(EV/"ledger_summary.json",summary)
print(json.dumps(summary,indent=2),flush=True)
