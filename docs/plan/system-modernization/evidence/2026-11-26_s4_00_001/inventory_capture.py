from pathlib import Path
import subprocess as sp, os, sys, json, hashlib, re, collections, tomllib, time, datetime, shutil
P=Path
root=P("docs/plan/system-modernization/evidence/2026-11-26_s4_00_001")
root.mkdir(parents=True,exist_ok=False)
def git(*a):return sp.check_output(["git",*a],text=True).strip()
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()
def run(a,cwd=".",timeout=40):
 t=time.monotonic()
 try:
  r=sp.run(a,cwd=cwd,text=True,capture_output=True,timeout=timeout)
  return {"command":a,"cwd":cwd,"exit_code":r.returncode,"elapsed_seconds":round(time.monotonic()-t,3),"stdout":r.stdout,"stderr":r.stderr}
 except (OSError,sp.TimeoutExpired) as e:return {"command":a,"cwd":cwd,"exit_code":None,"error":type(e).__name__,"elapsed_seconds":round(time.monotonic()-t,3)}
def rel(p):return str(p)
tracked=[P(os.fsdecode(x)) for x in sp.check_output(["git","ls-files","-z"]).split(b"\0") if x]
trackedset=set(tracked)
untracked=[P(os.fsdecode(x)) for x in sp.check_output(["git","ls-files","--others","--exclude-standard","-z"]).split(b"\0") if x and not os.fsdecode(x).startswith(str(root)+"/")]
e={"schema":"s4-baseline-inventory-v1","slice":"S4-00-001","observed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"source_sha":git("rev-parse","HEAD"),"branch":git("branch","--show-current"),"remote_main":git("ls-remote","origin","refs/heads/main").split()[0],"tracked_clean":not bool(git("status","--porcelain","--untracked-files=no")),"tracked_paths":len(tracked),"top_level_counts":dict(collections.Counter(p.parts[0] for p in tracked)),"docs_bin_paths":sum(str(p).startswith("docs/.bin/") for p in tracked),"untracked_paths":[{"path":str(p),"kind":"file" if p.is_file() else "other","size_bytes":p.stat().st_size if p.is_file() else None,"disposition":"PRESERVE_UNREVIEWED"} for p in untracked],"modules":{},"environments":{},"tools":{},"workflows":[],"artifacts":[]}
mods={"DATA":("data_module/sentinel_data","data_module/tests","data_module"),"ML":("ml/src","ml/tests","ml"),"AGENTS":("agents/src","agents/tests","agents"),"ZKML":("zkml/src","zkml/tests","zkml"),"CONTRACTS":("contracts/src","contracts/test","contracts")}
for k,(src,tests,base) in mods.items():
 fs=[p for p in tracked if str(p).startswith(src+"/") and p.suffix in (".py",".sol")]
 ts=[p for p in tracked if str(p).startswith(tests+"/") and p.suffix in (".py",".sol")]
 e["modules"][k]={"source_root":src,"test_root":tests,"source_files":len(fs),"source_physical_lines":sum(len(p.read_bytes().splitlines()) for p in fs),"test_source_files":len(ts),"source_subareas":dict(collections.Counter(p.relative_to(src).parts[0] if len(p.relative_to(src).parts)>1 else "(root files)" for p in fs)),"tracked_subareas":dict(collections.Counter(p.relative_to(base).parts[0] for p in tracked if str(p).startswith(base+"/"))),"source_inventory":[{"path":str(p),"lines":len(p.read_bytes().splitlines())} for p in fs]}
for base in (".","data_module","ml","agents"):
 pp=P(base)/"pyproject.toml"; lp=P(base)/"poetry.lock"; lock=tomllib.loads(lp.read_text()); manifest=tomllib.loads(pp.read_text())
 locked={p["name"].lower():p["version"] for p in lock["package"]}
 v={"manifest":str(pp),"manifest_sha256":sha(pp),"lock":str(lp),"lock_sha256":sha(lp),"declared_python":manifest["tool"]["poetry"]["dependencies"]["python"],"locked_packages":locked,"poetry_check":run(["poetry","check","--lock"],base)}
 if base!=".":
  exe=str((P.cwd()/base/".venv/bin/python"))
  r=run([exe,"-c","import sys,json,importlib.metadata as m; print(json.dumps({'python':sys.version.split()[0],'packages':{d.metadata['Name'].lower():d.version for d in m.distributions()}}))"])
  if r.get("exit_code")==0:
   inst=json.loads(r["stdout"]); v["installed"]=inst;v["different_common_versions"]={k:{"locked":locked[k],"installed":inst["packages"][k]} for k in sorted(set(locked)&set(inst["packages"])) if locked[k]!=inst["packages"][k]}
  else:v["installed_probe"]=r
 e["environments"][base]=v
for name,args in {"python3":["python3","--version"],"poetry":["poetry","--version"],"git":["git","--version"],"forge":[str(P.home()/".foundry/bin/forge"),"--version"],"docker":["docker","--version"],"uv":["uv","--version"],"ruff":["ruff","--version"],"nvidia-smi":["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"]}.items():e["tools"][name]=run(args,timeout=10)
for p in sorted(P(".github/workflows").glob("*.yml")):
 t=p.read_text();e["workflows"].append({"path":str(p),"sha256":sha(p),"triggers":re.findall(r"^  (push|pull_request|workflow_dispatch|schedule):",t,re.M),"uses":re.findall(r"uses:\s*([^\s#]+)",t)})
for aid,manifest in [("D-011","docs/plan/ml-R4/evidence/2026-09-02_v10_v26_physical_acceptance/acceptance.json"),("D-013","docs/plan/ml-R4/evidence/2026-09-29_guarded_token_physical_acceptance/acceptance.json")]:
 p=P(manifest);v=json.loads(p.read_text())["accepted_lineage"];q=P(v["physical_root"]);e["artifacts"].append({"id":aid,"path":str(q),"exists":q.is_dir(),"recorded_binding_digest":v["binding_digest_sha256"],"acceptance_record":manifest,"acceptance_record_sha256":sha(p),"verification":"existence_only_no_rehash"})
for n in ["ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt","agents/data/index/faiss.index","agents/data/index/bm25.pkl","agents/data/index/chunks.pkl","agents/data/index/index_metadata.json","zkml/models/proxy_best.pt","zkml/ezkl/model.compiled","zkml/ezkl/settings.json","zkml/ezkl/srs.params","zkml/ezkl/proving_key.pk","zkml/ezkl/verification_key.vk","docs/plan/ml-R4/review_bundles/r4_gap007_candidate2_independent_review_v2.zip","contracts/foundry.lock"]:
 p=P(n);exists=p.is_file();e["artifacts"].append({"path":n,"exists":exists,"tracked":p in trackedset,"size_bytes":p.stat().st_size if exists else None,"sha256":sha(p) if exists and p in trackedset and p.stat().st_size<10*1024*1024 else None,"verification":"presence; small tracked file hash only"})
(root/"inventory.json").write_text(json.dumps(e,indent=2,sort_keys=True)+"\n")
print("INVENTORY",e["source_sha"],e["tracked_paths"],"modules",[(k,v["source_files"],v["test_source_files"]) for k,v in e["modules"].items()],flush=True)
for k,v in e["environments"].items():print("ENV",k,v["poetry_check"]["exit_code"],"locked",len(v["locked_packages"]),"installed",len(v.get("installed",{}).get("packages",{})),"different_common",len(v.get("different_common_versions",{})),flush=True)
print("ARTIFACTS",[(v.get("id",v["path"]),v["exists"]) for v in e["artifacts"]],flush=True)
print("TOOLS",[(k,v.get("exit_code"),v.get("stdout","").strip().splitlines()[:1]) for k,v in e["tools"].items()],flush=True)
