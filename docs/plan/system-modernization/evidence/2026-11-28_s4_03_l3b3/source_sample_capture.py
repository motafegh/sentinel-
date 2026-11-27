from pathlib import Path
import json,hashlib,urllib.request,urllib.parse,re,collections as C,concurrent.futures as cf
EV=Path("docs/plan/system-modernization/evidence/2026-11-28_s4_03_l3b3")
LOCAL=Path("data_module/data/s4-forge-preflight-v1")
BASE=Path("data_module/data")
m=json.loads((LOCAL/"tree_selection.json").read_text())
finds={x["case_id"]:x for x in json.loads((LOCAL/"finding_selection.json").read_text())}
bindings=json.loads((LOCAL/"source_tree_bindings.json").read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def get(url,limit):
 with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"Sentinel-public-metadata-research/1.0","Accept-Encoding":"identity"}),timeout=15) as r:b=r.read(limit+1)
 assert len(b)<=limit
 return b
def rawurl(repo,pin,path):return "https://raw.githubusercontent.com/"+repo+"/"+pin+"/"+urllib.parse.quote(path,safe="/")
raw=C.defaultdict(set)
for p in (BASE/"raw").glob("*/ingestion_manifest.json"):
 for x in json.loads(p.read_text())["files"]:raw[x["sha256"]].add(p.parent.name)
man=json.loads(Path("docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/logical_v3_manifest.json").read_text())
claims=BASE/"r4-v2-build/source_claims.jsonl"
assert sha(claims.read_bytes())==man["artifacts"]["claims"]["sha256"]
norm={json.loads(l)["artifact_id"] for l in claims.open()}
def run(x):
 cid=x["case_id"];f=finds[cid];out={"case_id":cid,"finding_status":f["status"],"source_files_staged":0,"admission":"NOT_ADMITTED"}
 if not f.get("selected"):return out
 cand=x.get("contract",{}).get("matches",[]);rawc=x.get("raw",{}).get("matches",[])
 out.update(curated_path_matches=len(cand),raw_path_matches=len(rawc),raw_gitlinks=x.get("raw",{}).get("gitlinks"),upstream_metadata_attempts=len(x.get("upstream",[])),upstream_metadata_successes=sum(bool(u.get("sha")) for u in x.get("upstream",[])),project_scope_solidity_paths=x.get("contract",{}).get("solidity_paths"))
 if len(cand)!=1:
  out["source_status"]="CURATED_PATH_UNRESOLVED";return out
 p=cand[0];out["curated_git_blob"]=p["sha"];out["curated_raw_blob_match"]=len(rawc)==1 and rawc[0]["sha"]==p["sha"]
 out["upstream_matching_claimed_commits"]=sum(u.get("sha")==p["sha"] for u in x["upstream"])
 out["upstream_differing_claimed_commits"]=sum(bool(u.get("sha")) and u["sha"]!=p["sha"] for u in x["upstream"])
 try:
  assert p["mode"]=="100644" and 0<p["size"]<=1024*1024
  b=get(rawurl("shenyimings/FORGE-Curated",m["pin"],p["path"]),p["size"])
  assert len(b)==p["size"] and hashlib.sha1(("blob "+str(len(b))+"\0").encode()+b).hexdigest()==p["sha"],(len(b),p["size"])
  t=b.decode("utf-8-sig");h=sha(b);n=len(t.splitlines())
  locs=f["selected"]["locations"];name=Path(f["selected"]["file"]).name
  matching=[l for l in locs if isinstance(l,str) and l.split("::")[0].split("#")[0]==name]
  numeric=[]
  for l in matching:
   z=re.search(r"#L?(\d+)(?:-L?(\d+))?$",l)
   if z:numeric.append((int(z[1]),int(z[2] or z[1])))
  out.update(source_status="VERIFIED_BLOB",sha256=h,bytes=len(b),lines=n,spdx=sorted(set(re.findall(r"(?m)SPDX-License-Identifier:\s*([^\r\n*]+)",t))),imports=len(re.findall(r"(?m)^\s*import\s",t)),pragma_solidity=sorted(set(re.findall(r"pragma\s+solidity\s+([^;]+);",t))),raw_manifest_sources=sorted(raw.get(h,[])),exact_normalized_artifact=h in norm,location_strings=len(locs),same_file_location_strings=len(matching),numeric_locations=len(numeric),numeric_in_bounds=sum(1<=a<=z<=n for a,z in numeric))
 except Exception as e:out.update(source_status="FAILED",error_type=type(e).__name__,error=str(e)[:180])
 print(cid,out.get("source_status"),"upstream matches",out.get("upstream_matching_claimed_commits"),"SPDX",out.get("spdx"),"locations",out.get("numeric_locations"),out.get("numeric_in_bounds"),flush=True)
 return out
with cf.ThreadPoolExecutor(max_workers=3) as ex:results=list(ex.map(run,bindings))
out={"pin":m["pin"],"tree_binding_sha256":sha((LOCAL/"source_tree_bindings.json").read_bytes()),"records":results,"source_files_staged":0,"source_executed":0,"labels_admitted":0,"limitations":"One file per frozen report. Git blob equality with one claimed audit commit is file identity only, not verified vulnerable revision or complete project build. Raw gitlink is not a retrieved project. Bounds only, no semantic localization or PDF cross-check. No SPDX-based blanket license approval. No group/family admission."}
(EV/"source_preflight.json").write_text(json.dumps(out,indent=2)+"\n")
print("verified sources",sum(x.get("source_status")=="VERIFIED_BLOB" for x in results))
