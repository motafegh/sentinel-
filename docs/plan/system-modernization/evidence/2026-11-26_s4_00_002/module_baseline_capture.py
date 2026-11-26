from pathlib import Path
import subprocess as sp, tempfile, os, shutil, json, time, hashlib, datetime, concurrent.futures
P=Path;repo=P.cwd();ev=repo/"docs/plan/system-modernization/evidence/2026-11-26_s4_00_002";assert (ev/"g4_repair.json").exists()
sha=sp.check_output(["git","rev-parse","HEAD"],text=True).strip()
scratch=P(tempfile.mkdtemp(prefix="sentinel-s4-002-")); snap=scratch/"source";snap.mkdir()
paths=["data_module/sentinel_data","data_module/tests","data_module/data/exports/sentinel-r4-vnext-v1/ml_targets.parquet","data_module/pyproject.toml","ml/__init__.py","ml/src","ml/tests","ml/pyproject.toml","agents/src","agents/tests","agents/configs","agents/pyproject.toml","zkml","contracts","docs/plan/ml-R4/specs","schemas","pyproject.toml"]
archive=sp.Popen(["git","archive",sha,*paths],stdout=sp.PIPE)
extract=sp.run(["tar","-xf","-","-C",str(snap)],stdin=archive.stdout,capture_output=True,text=True)
archive.stdout.close();assert archive.wait()==0 and extract.returncode==0,extract.stderr
# Libraries are local prerequisites, not tracked. Copy, not symlink; no downloads.
lib=repo/"contracts/lib";shutil.copytree(lib,snap/"contracts/lib",ignore=shutil.ignore_patterns(".git","node_modules","out","cache"))
compiler=P.home()/".solc-select/artifacts/solc-0.8.22/solc-0.8.22";forge=P.home()/".foundry/bin/forge"
# Safe baseline: no application .env, private checkpoints, physical roots or real services in snapshot.
assert not (snap/"ml/checkpoints").exists() and not (snap/"agents/data/index").exists()
jobs={
"DATA":("data_module",["data_module/tests/test_archive_safety.py","data_module/tests/test_preprocessing/test_r4_grouping_v3.py","data_module/tests/test_vnext/test_policy.py"],snap),
"ML":("ml",["ml/tests/test_vnext_phase8_compat.py"],snap),
"AGENTS":("agents",["tests/test_audit_mcp_readonly.py","tests/test_audit_server.py","tests/test_policy_signer_v3.py"],snap/"agents"),
"ZKML":("ml",["zkml/tests/test_validate_bundle.py","zkml/tests/test_setup_circuit_lineage.py"],snap)}
def run(name,cmd,cwd,pythonpath,limit=120):
 home=scratch/("home-"+name);home.mkdir()
 tmp=scratch/("tmp-"+name);tmp.mkdir()
 env={"HOME":str(home),"PATH":"/usr/local/bin:/usr/bin:/bin","LANG":"C.UTF-8","TMPDIR":str(tmp),"PYTHONPATH":pythonpath,"PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1","PYTEST_DISABLE_PLUGIN_AUTOLOAD":"1","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","CUDA_VISIBLE_DEVICES":"","AGENTS_DISABLE_LLM":"1","AUDIT_MOCK":"true","SEPOLIA_RPC_URL":"","ETHERSCAN_API_KEY":"","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
 full=["unshare","-Urn","--",*cmd];t=time.monotonic()
 try:
  r=sp.run(full,cwd=cwd,env=env,text=True,capture_output=True,timeout=limit)
  result={"command":full,"cwd":str(cwd.relative_to(snap)),"exit_code":r.returncode,"elapsed_seconds":round(time.monotonic()-t,3),"stdout":r.stdout,"stderr":r.stderr}
 except sp.TimeoutExpired as e:
  result={"command":full,"cwd":str(cwd.relative_to(snap)),"exit_code":None,"error":"timeout","elapsed_seconds":round(time.monotonic()-t,3),"stdout":(e.stdout or b"").decode(errors="replace") if isinstance(e.stdout,bytes) else e.stdout or "","stderr":(e.stderr or b"").decode(errors="replace") if isinstance(e.stderr,bytes) else e.stderr or ""}
 except OSError as e:result={"command":full,"cwd":str(cwd.relative_to(snap)),"exit_code":None,"error":str(e),"elapsed_seconds":round(time.monotonic()-t,3)}
 print(name,result["exit_code"],result.get("stdout","")[-1000:],result.get("stderr","")[-400:],flush=True)
 return result
def pyjob(item):
 name,(module,tests,cwd)=item
 exe=repo/module/".venv/bin/python"
 cmd=[str(exe),"-m","pytest","-q","-c","/dev/null","--noconftest","-p","no:cacheprovider","--basetemp",str(scratch/("pytest-"+name))]
 if name=="AGENTS":cmd+=["-p","pytest_asyncio.plugin","-o","asyncio_mode=auto"]
 cmd+=tests
 pp=os.pathsep.join([str(snap),str(snap/module),str(snap/"data_module"),str(snap/"ml")])
 return name,run(name,cmd,cwd,pp)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=dict(pool.map(pyjob,jobs.items()))
# Only tracked source tests requiring neither generated proof fixture nor external digest env.
cmd=[str(forge),"test","--offline","--use",str(compiler),"--skip","ZKMLVerifierRealProof","--skip","AuditRegistryV3GoldenDigest","--match-contract","^(AuditRegistryTest|AuditRegistryV2Test|AuditRegistryV3Test|AuditRegistryV3UpgradeTest|SentinelTokenTest)$","--no-match-test","^invariant_"]
results["CONTRACTS"]=run("CONTRACTS",cmd,snap/"contracts","",120)
data={"source_sha":sha,"observed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"scratch_source":"git archive selected tracked paths; local Foundry libs copied excluding .git/node_modules/out/cache","network_isolation":"unshare -Urn (verified available); allowlisted env and temporary HOME/TMPDIR; not a complete filesystem sandbox","env_note":"existing module venv interpreters, no install; CUDA hidden, HF offline, AGENTS LLM disabled, RPC/key blank, no protected roots/checkpoint/RAG index present","selected_tests":{k:v[1] for k,v in jobs.items()},"foundry_compiler":{"version":"0.8.22","path":str(compiler),"sha256":hashlib.sha256(compiler.read_bytes()).hexdigest()},"results":results}
(ev/"module_baseline.json").write_text(json.dumps(data,indent=2)+"\n")
print("SAVED",ev/"module_baseline.json",flush=True)
