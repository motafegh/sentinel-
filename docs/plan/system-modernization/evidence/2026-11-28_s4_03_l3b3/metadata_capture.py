from pathlib import Path,PurePosixPath
import json,hashlib,urllib.request,urllib.parse,re,collections as C,concurrent.futures as cf
EV=Path("docs/plan/system-modernization/evidence/2026-11-28_s4_03_l3b3")
LOCAL=Path("data_module/data/s4-forge-preflight-v1")
m=json.loads((LOCAL/"tree_selection.json").read_text());pin=m["pin"]
def sha(b):return hashlib.sha256(b).hexdigest()
def get(x):
 p=x["path"];assert x["mode"]=="100644" and 0<x["size"]<1000000
 url="https://raw.githubusercontent.com/shenyimings/FORGE-Curated/"+pin+"/"+urllib.parse.quote(p,safe="/")
 with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"Sentinel-public-metadata-research/1.0","Accept-Encoding":"identity"}),timeout=15) as r:b=r.read(x["size"]+1)
 assert len(b)==x["size"] and hashlib.sha1(("blob "+str(len(b))+"\0").encode()+b).hexdigest()==x["sha"],(len(b),x["size"])
 return b
def lst(x):return x if isinstance(x,list) else [x] if isinstance(x,str) and x.strip() else []
def sev(f):
 return min(({"critical":0,"high":1,"medium":2,"low":3,"informational":4,"info":4}.get(x.lower(),5) for x in lst(f.get("severity")) if isinstance(x,str)),default=5)
def run(pair):
 i,x=pair;out={"case_id":f"FORGE-P{i:02d}","path":x["path"],"expected_git_blob":x["sha"]}
 try:
  b=get(x);j=json.loads(b);p=j["project_info"];find=j["findings"]
  assert isinstance(find,list) and all(isinstance(f,dict) for f in find)
  pairs=[(f,fn) for f in find for fn in lst(f.get("files")) if isinstance(fn,str) and fn.lower().endswith(".sol") and not PurePosixPath(fn).is_absolute() and ".." not in PurePosixPath(fn).parts]
  chosen=sorted(pairs,key=lambda x:(sev(x[0]),str(x[0].get("id")),x[1]))[:1]
  out.update(status="VERIFIED_BLOB",sha256=sha(b),bytes=len(b),report_path=j.get("path"),project_urls=lst(p.get("url")),claimed_commits=lst(p.get("commit_id")),project_path=p.get("project_path"),audit_date=p.get("audit_date"),findings=len(find),severity_counts=dict(C.Counter(sev(f) for f in find)),cwe_counts=dict(C.Counter(c for f in find for v in f.get("category",{}).values() for c in lst(v) if isinstance(c,str))),finding_ids_unique=len({str(f.get("id")) for f in find})==len(find),findings_with_sol_mapping=len({str(f.get("id")) for f,_ in pairs}))
  if chosen:
   f,fn=chosen[0];out["selected"]={"finding_id":f.get("id"),"file":fn,"severity":sev(f),"cwe":f.get("category"),"locations":lst(f.get("location"))}
  else:out["selected"]=None
 except Exception as e:out.update(status="FAILED",error_type=type(e).__name__,error=str(e)[:200])
 return out
with cf.ThreadPoolExecutor(max_workers=3) as ex:r=list(ex.map(run,enumerate(m["selected"],1)))
(LOCAL/"finding_selection.json").write_text(json.dumps(r,indent=2)+"\n")
profile={"pin":pin,"subtrees":m["trees"],"report_extensions":m["report_extensions"],"report_auditors":m["report_auditors"],"freeze_sha256":sha((LOCAL/"tree_selection.json").read_bytes()),"finding_selection_sha256":sha((LOCAL/"finding_selection.json").read_bytes()),"selected_reports":len(r),"verified_reports":sum(x["status"]=="VERIFIED_BLOB" for x in r),"findings_total":sum(x.get("findings",0) for x in r),"findings_with_sol_mapping":sum(x.get("findings_with_sol_mapping",0) for x in r),"selected_solidity_files":sum(bool(x.get("selected")) for x in r),"severity_counts":dict(sum((C.Counter(x.get("severity_counts",{})) for x in r),C.Counter())),"cwe_counts":dict(sum((C.Counter(x.get("cwe_counts",{})) for x in r),C.Counter())),"limits":"Sample only, not corpus-wide finding distributions. Severity rank 0 critical 1 high 2 medium 3 low 4 informational 5 other. One mapped Solidity path per report, not a validated vulnerability."}
(EV/"metadata_profile.json").write_text(json.dumps(profile,indent=2)+"\n")
print(json.dumps(profile,indent=2))
for x in r: print(json.dumps({k:x.get(k) for k in ("case_id","status","path","report_path","project_urls","claimed_commits","project_path","selected","error")}))
