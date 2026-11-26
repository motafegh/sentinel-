from pathlib import Path
import os, json, subprocess as sp, time
repo=Path.cwd();ev=repo/"docs/plan/system-modernization/evidence/2026-11-26_s4_00_002";p=ev/"module_baseline.json";d=json.loads(p.read_text())
first=d["results"]["AGENTS"];cmd=first["command"]
scratch=Path(cmd[cmd.index("--basetemp")+1]).parent;snap=scratch/"source"
home=scratch/"home-AGENTS-retry";home.mkdir();tmp=scratch/"tmp-AGENTS-retry";tmp.mkdir()
cmd=cmd.copy();cmd[cmd.index("--basetemp")+1]=str(scratch/"pytest-AGENTS-retry")
env={"HOME":str(home),"PATH":"/usr/local/bin:/usr/bin:/bin","LANG":"C.UTF-8","TMPDIR":str(tmp),"PYTHONPATH":os.pathsep.join([str(snap/"agents"),str(snap)]),"PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1","PYTEST_DISABLE_PLUGIN_AUTOLOAD":"1","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","CUDA_VISIBLE_DEVICES":"","AGENTS_DISABLE_LLM":"1","AUDIT_MOCK":"true","SEPOLIA_RPC_URL":"","ETHERSCAN_API_KEY":"","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
t=time.monotonic();r=sp.run(cmd,cwd=snap/"agents",env=env,capture_output=True,text=True,timeout=120)
result={"command":cmd,"cwd":"agents","pythonpath":env["PYTHONPATH"],"exit_code":r.returncode,"elapsed_seconds":round(time.monotonic()-t,3),"stdout":r.stdout,"stderr":r.stderr}
d["attempts"]={"AGENTS_initial":first}
d["results"]["AGENTS"]=result
d["harness_adjustment"]="Initial AGENTS collection included ml/src as top-level src because the ML root was on PYTHONPATH. Retry removes DATA/ML module roots, keeps only AGENTS and snapshot root. No source/dependency change. Initial error preserved."
p.write_text(json.dumps(d,indent=2)+"\n")
print(json.dumps(result,indent=2))
