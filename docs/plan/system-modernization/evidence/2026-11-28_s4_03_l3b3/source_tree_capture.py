from pathlib import Path,PurePosixPath
import json,hashlib,urllib.request,urllib.parse,concurrent.futures as cf,re,collections as C
EV=Path("docs/plan/system-modernization/evidence/2026-11-28_s4_03_l3b3")
LOCAL=Path("data_module/data/s4-forge-preflight-v1")
m=json.loads((LOCAL/"tree_selection.json").read_text())
rows=json.loads((LOCAL/"finding_selection.json").read_text())
pin=m["pin"]
def getj(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"Sentinel-public-metadata-research/1.0","Accept":"application/vnd.github+json"}),timeout=15) as r:b=r.read(12000001)
 assert len(b)<=12000000
 return json.loads(b)
def tree(sha):return getj("https://api.github.com/repos/shenyimings/FORGE-Curated/git/trees/"+sha+"?recursive=1")
def run(x):
 out={"case_id":x["case_id"],"finding_status":x["status"]}
 if not x.get("selected"):return out
 path=x["selected"]["file"];out["mapped_file"]=path
 dirs={k:[z for z in m[k+"_dirs"] if z["path"].replace(".pdf-source","").replace(".md-source","")==Path(x["path"]).stem] for k in ("contract","raw")}
 for k,ds in dirs.items():
  try:
   assert len(ds)==1
   j=tree(ds[0]["sha"]);assert not j["truncated"]
   a=j["tree"];folder="contracts" if k=="contract" else "contracts-raw"
   matches=[{"path":f"dataset-curated/{folder}/{ds[0]['path']}/{t['path']}","sha":t["sha"],"size":t.get("size"),"mode":t["mode"]} for t in a if t["type"]=="blob" and (t["path"]==path or t["path"].endswith("/"+path))]
   out[k]={"tree_sha":ds[0]["sha"],"entries":len(a),"solidity_paths":sum(t["path"].endswith(".sol") for t in a),"gitlinks":sum(t["mode"]=="160000" for t in a),"license_files":sum(t["type"]=="blob" and bool(re.match(r"^(LICENSE|COPYING)(\.|$)",Path(t["path"]).name,re.I)) for t in a),"matches":matches}
  except Exception as e:out[k]={"error_type":type(e).__name__,"error":str(e)[:180]}
 out["upstream"]=[]
 # All claimed commits; no assumption which is vulnerable vs fixed.
 urls=x["project_urls"];commits=x["claimed_commits"]
 if len(urls)==1:
  repo=urllib.parse.urlparse(urls[0]).path.strip("/")
  if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",repo):
   out["repo"]=repo.lower()
   for commit in commits:
    if not re.fullmatch(r"[0-9a-fA-F]{40}",commit):continue
    try:
     j=getj("https://api.github.com/repos/"+repo+"/contents/"+urllib.parse.quote(path.split("/",1)[1],safe="/")+"?ref="+commit)
     out["upstream"].append({"commit":commit,"sha":j.get("sha"),"size":j.get("size"),"type":j.get("type")})
    except Exception as e:out["upstream"].append({"commit":commit,"error_type":type(e).__name__,"error":str(e)[:120]})
 print(out["case_id"],[(k,len(out.get(k,{}).get("matches",[])),out.get(k,{}).get("error")) for k in ("contract","raw")],[(u.get("sha"),u.get("error")) for u in out["upstream"]],flush=True)
 return out
with cf.ThreadPoolExecutor(max_workers=3) as ex:results=list(ex.map(run,rows))
(LOCAL/"source_tree_bindings.json").write_text(json.dumps(results,indent=2)+"\n")
print("tree bindings captured",len(results))
