from pathlib import Path
import json,hashlib,subprocess as sp,tempfile,shutil,concurrent.futures,time
ev=Path("docs/plan/system-modernization/evidence/2026-11-26_s4_03_l1a")
m=json.loads((ev/"pilot_manifest.json").read_text());repo=Path.cwd();scratch=Path(tempfile.mkdtemp(prefix="sentinel-l1a-"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(c):
 p=scratch/c["case_id"];p.mkdir();home=p/"home";home.mkdir()
 src=p/"target.sol";shutil.copyfile(repo/c["source_path"],src);assert sha(src)==c["source_sha256"]
 v=c["compiler_version"];compiler=Path.home()/".solc-select/artifacts"/("solc-"+v)/("solc-"+v)
 env={"HOME":str(home),"PATH":"/usr/local/bin:/usr/bin:/bin","LANG":"C.UTF-8","PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","CUDA_VISIBLE_DEVICES":""}
 cmd=["unshare","-Urn","--",str(repo/"agents/.venv/bin/python"),"/tmp/s4-l1a-detectors.py","--compiler",str(compiler),"--source",str(src)]
 t=time.monotonic()
 out={"case_id":c["case_id"],"command":cmd,"source_sha256":sha(src),"compiler_sha256":sha(compiler)}
 try:
  r=sp.run(cmd,cwd=p,env=env,text=True,capture_output=True,timeout=30)
  out.update(exit_code=r.returncode,stderr_sha256=hashlib.sha256(r.stderr.encode()).hexdigest(),stdout_sha256=hashlib.sha256(r.stdout.encode()).hexdigest())
  if r.returncode==0:out["tool"]=json.loads(r.stdout.splitlines()[-1])
  else:out["tool"]={"status":"PROCESS_FAILED"}
 except sp.TimeoutExpired:out["tool"]={"status":"TIMEOUT"};out["exit_code"]=None
 out["wall_seconds"]=round(time.monotonic()-t,3)
 print(c["case_id"],c["class"],c["native_claim"],c["compiler_version"],out["tool"]["status"],[(x["detector"],len(x["lines"])) for x in out["tool"].get("findings",[])],flush=True)
 return out
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,m["cases"]))
(ev/"detector_results.json").write_text(json.dumps({"manifest_sha256":sha(ev/"pilot_manifest.json"),"environment":"isolated per-case scratch HOME and source; network namespace; existing AGENTS venv; no live services; bounded 3 workers; source snippets and raw tool errors not published, hashes only","results":results},indent=2)+"\n")
for name in ["freeze","calibrate","detectors","run"]:shutil.copyfile("/tmp/s4-l1a-"+name+".py",ev/(name+"_capture.py"))
