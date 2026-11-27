from pathlib import Path
import json,hashlib,subprocess as sp,tempfile,shutil,concurrent.futures,time
repo=Path.cwd();ev=repo/"docs/plan/system-modernization/evidence/2026-11-27_s4_03_l1b";prev=ev.parent/"2026-11-26_s4_03_l1a"
m=json.loads((prev/"pilot_manifest.json").read_text());scratch=Path(tempfile.mkdtemp(prefix="sentinel-l1b-"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(c):
 p=scratch/c["case_id"];p.mkdir();home=p/"home";home.mkdir()
 src=p/"target.sol";shutil.copyfile(repo/c["source_path"],src);assert sha(src)==c["source_sha256"]
 v=c["compiler_version"];compiler=Path.home()/".solc-select/artifacts"/("solc-"+v)/("solc-"+v)
 env={"HOME":str(home),"PATH":"/usr/local/bin:/usr/bin:/bin","LANG":"C.UTF-8","PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1"}
 cmd=["unshare","-Urn","--",str(repo/"agents/.venv/bin/python"),str(ev/"return_inventory_capture.py"),"--compiler",str(compiler),"--source",str(src)]
 t=time.monotonic();out={"case_id":c["case_id"],"command":cmd,"source_sha256":sha(src)}
 try:
  r=sp.run(cmd,cwd=p,env=env,text=True,capture_output=True,timeout=30)
  out.update(exit_code=r.returncode,stderr_sha256=hashlib.sha256(r.stderr.encode()).hexdigest(),stdout_sha256=hashlib.sha256(r.stdout.encode()).hexdigest())
  if r.returncode==0:out["inventory"]=json.loads(r.stdout.splitlines()[-1])
  else:out["inventory"]={"status":"PROCESS_FAILED"}
 except sp.TimeoutExpired:out["inventory"]={"status":"TIMEOUT"};out["exit_code"]=None
 out["wall_seconds"]=round(time.monotonic()-t,3);i=out["inventory"]
 print(c["case_id"],i["status"],"functions",len(i.get("functions",[])),"direct candidates",len(i.get("direct_candidates",[])),"detector",len(i.get("unused_return_detector_findings",[])),i.get("error_type",""),flush=True)
 return out
fixture={"case_id":"fixture","source_path":str(ev/"fixture.sol"),"source_sha256":sha(ev/"fixture.sol"),"compiler_version":"0.8.22"}
f=run(fixture);e=json.loads((ev/"fixture_expectations.json").read_text())
actual={x["function"].split(".")[-1].split("(")[0] for x in f["inventory"].get("direct_candidates",[])}
f["expected_direct_functions"]=e["direct_discard_expected"];f["actual_direct_functions"]=sorted(actual);f["pass"]=actual==set(e["direct_discard_expected"]) and f["inventory"]["status"]=="COMPLETED"
(ev/"fixture_result.json").write_text(json.dumps(f,indent=2)+"\n")
assert f["pass"],"fixture failure, no corpus run"
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,m["cases"]))
(ev/"inventory_results.json").write_text(json.dumps({"manifest_sha256":sha(prev/"pilot_manifest.json"),"helper_sha256":sha(ev/"return_inventory_capture.py"),"results":results},indent=2)+"\n")
