from pathlib import Path
import subprocess as sp, tempfile, json, shutil, os, hashlib
repo=Path.cwd();ev=repo/"docs/plan/system-modernization/evidence/2026-11-26_s4_01_003"
scratch=Path(tempfile.mkdtemp(prefix="sentinel-s4-utility-"));snap=scratch/"source";snap.mkdir()
a=sp.Popen(["git","archive","HEAD","agents/src","agents/configs","schemas"],stdout=sp.PIPE)
r=sp.run(["tar","-xf","-","-C",str(snap)],stdin=a.stdout);a.stdout.close();assert a.wait()==0 and r.returncode==0
home=scratch/"home";home.mkdir()
compiler=Path.home()/".solc-select/artifacts/solc-0.8.22/solc-0.8.22"
env={"HOME":str(home),"PATH":"/usr/local/bin:/usr/bin:/bin","LANG":"C.UTF-8","PYTHONPATH":str(snap/"agents")+":"+str(snap),"PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","CUDA_VISIBLE_DEVICES":"","AGENTS_DISABLE_LLM":"1","AUDIT_MOCK":"true","SEPOLIA_RPC_URL":"","ETHERSCAN_API_KEY":"","OMP_NUM_THREADS":"1"}
results=[]
for c in json.loads((ev/"utility_cases.json").read_text())["cases"]:
 cmd=["unshare","-Urn","--",str(repo/"agents/.venv/bin/python"),"/tmp/s4-utility-pilot.py","--cases",str(ev/"utility_cases.json"),"--compiler",str(compiler),"--case",c["id"]]
 try:
  r=sp.run(cmd,cwd=snap/"agents",env=env,text=True,capture_output=True,timeout=30)
  entry={"command":cmd,"exit_code":r.returncode,"stderr":r.stderr,"stdout":r.stdout}
  if r.returncode==0:entry["result"]=json.loads(r.stdout.splitlines()[-1])
 except sp.TimeoutExpired:entry={"command":cmd,"exit_code":None,"status":"TIMEOUT"}
 results.append(entry)
 print(c["id"],entry.get("exit_code"),entry.get("result",{}).get("rubric"),entry.get("stderr","")[-600:],flush=True)
(ev/"utility_results.json").write_text(json.dumps({"source_sha":sp.check_output(["git","rev-parse","HEAD"],text=True).strip(),"case_manifest_sha256":hashlib.sha256((ev/"utility_cases.json").read_bytes()).hexdigest(),"environment":"tracked source snapshot; network namespace; existing AGENTS venv; CPU only; no live services","results":results},indent=2)+"\n")
shutil.copyfile("/tmp/s4-utility-pilot.py",ev/"utility_pilot.py")
shutil.copyfile("/tmp/s4-l0-inventory.py",ev/"label_inventory_capture.py")
