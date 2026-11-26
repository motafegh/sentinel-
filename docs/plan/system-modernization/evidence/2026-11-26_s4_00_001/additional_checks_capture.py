from pathlib import Path
import subprocess as sp,json,re,ast,hashlib,time,datetime,shutil
P=Path;root=P("docs/plan/system-modernization/evidence/2026-11-26_s4_00_001")
r=json.loads((root/"baseline_checks.json").read_text())
for k,a in {"g6_frozen":["python3","docs/plan/ml-R4/scripts/p6_validate_frozen_partitions.py"],"v3_snapshot":["python3","docs/plan/ml-R4/scripts/p8_verify_committed_logical_v3_snapshot.py"]}.items():
 t=time.monotonic()
 try:
  q=sp.run(a,capture_output=True,text=True,timeout=60);r["checks"][k]={"command":a,"cwd":"repository root","exit_code":q.returncode,"elapsed_seconds":round(time.monotonic()-t,3),"stdout":q.stdout,"stderr":q.stderr}
  print(k,q.returncode,q.stdout[-800:],q.stderr[-300:],flush=True)
 except (OSError,sp.TimeoutExpired) as e:r["checks"][k]={"command":a,"exit_code":None,"error":type(e).__name__}
(root/"baseline_checks.json").write_text(json.dumps(r,indent=2)+"\n")
inv=json.loads((root/"inventory.json").read_text())
scan={"source_sha":inv["source_sha"],"method":"AST broad Exception/BaseException or bare except containing direct pass; lexical sensitive load call site leads; no reachability/exposure conclusions","modules":{},"authority":{}}
for name,v in inv["modules"].items():
 if name=="CONTRACTS":continue
 ex=[];calls=[]
 for item in v["source_inventory"]:
  p=P(item["path"]);s=p.read_text();tree=ast.parse(s)
  for n in ast.walk(tree):
   if isinstance(n,ast.ExceptHandler) and (n.type is None or isinstance(n.type,ast.Name) and n.type.id in ("Exception","BaseException")) and any(isinstance(b,ast.Pass) for b in n.body):ex.append({"path":str(p),"line":n.lineno})
   if isinstance(n,ast.Call):
    f=ast.unparse(n.func)
    if f in ("torch.load","pickle.load","pickle.loads"):
     w=next((ast.unparse(k.value) for k in n.keywords if k.arg=="weights_only"),None)
     if f.startswith("pickle") or w!="True":calls.append({"path":str(p),"line":n.lineno,"call":f,"weights_only":w})
 scan["modules"][name]={"broad_except_pass":ex,"sensitive_load_sites":calls}
 print(name,"except",len(ex),"load leads",len(calls),flush=True)
for p in map(P,["data_module/CLAUDE.md","ml/CLAUDE.md","docs/handbook/16_current_status.md","docs/handbook/_meta/current_r4.json"]):
 s=p.read_text();scan["authority"][str(p)]={"decision_mentions":{x:x in s for x in ("R4-D-013","R4-D-015","R4-D-016")},"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"relevant_lines":[{"line":i,"text":l} for i,l in enumerate(s.splitlines(),1) if "candidate #1" in l.lower() or "physical build/acceptance remains pending" in l.lower() or "selector promotion remains unresolved" in l.lower()]}
(root/"source_leads.json").write_text(json.dumps(scan,indent=2)+"\n")
shutil.copyfile("/tmp/sentinel-s4-checks.py",root/"baseline_checks_capture.py")
