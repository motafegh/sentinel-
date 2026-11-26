from pathlib import Path
import subprocess as sp,json,re,hashlib,datetime,time,sys,shutil
P=Path
root=P("docs/plan/system-modernization/evidence/2026-11-26_s4_00_001")
inv=json.loads((root/"inventory.json").read_text())
assert inv["source_sha"]==sp.check_output(["git","rev-parse","HEAD"],text=True).strip()
def run(a,timeout=60,input_text=None):
 t=time.monotonic()
 try:
  r=sp.run(a,input=input_text,text=True,capture_output=True,timeout=timeout)
  return {"command":a,"cwd":"repository root","exit_code":r.returncode,"elapsed_seconds":round(time.monotonic()-t,3),"stdout":r.stdout,"stderr":r.stderr}
 except (OSError,sp.TimeoutExpired) as e:return {"command":a,"exit_code":None,"error":type(e).__name__,"elapsed_seconds":round(time.monotonic()-t,3)}
checks={}
for k,a in {"handbook_static":["python3","docs/handbook/tools/verify_handbook.py","static"],"r4_authority":["python3","docs/handbook/tools/verify_current_r4.py"],"handbook_unit":["python3","-m","unittest","discover","-s","docs/handbook/tools/tests","-p","test_*.py"],"showcase":["python3","tools/showcase_sentinel.py"],"g6_frozen":["python3","docs/plan/ml-R4/scripts/p6_validate_g6.py"]}.items():
 if k=="g6_frozen" and not P(a[-1]).exists():continue
 checks[k]=run(a);print(k,checks[k]["exit_code"],checks[k].get("stdout","").splitlines()[-3:],flush=True)
wf=P(".github/workflows/r4-phase4-g4.yml")
m=re.search(r"^          python - <<'PY'\n(.*?)^          PY$",wf.read_text(),re.M|re.S)
assert m
import textwrap
code=textwrap.dedent(m.group(1))
g4=run(["python3","-"],input_text=code)
g4["input_source"]=str(wf);g4["input_sha256"]=hashlib.sha256(code.encode()).hexdigest()
g4["assert_line_75"]=code.splitlines()[74]
checks["g4_replay"]=g4
print("G4",g4["exit_code"],g4["assert_line_75"],g4.get("stderr","")[-500:],flush=True)
reg=P("docs/audit/system-engineering/2026-07-14_SYSTEM_D2_source_engineering_audit/registry/2026-07-14_SYSTEM_D2_FINDINGS_registry.md")
rows=[]
for line in reg.read_text().splitlines():
 if not line.startswith("| D2-"):continue
 fields=[x.strip() for x in line.strip("|").split("|")]
 assert len(fields)==7,fields
 ident,severity,disposition,evidence,owner,related,wave=fields
 stage={"DATA":"S4-03","ML":"S4-04","AGT":"S4-05","ZKC":"S4-06","X":"S4-07"}[ident.split("-")[1]]
 rows.append(dict(id=ident,original_severity=severity,original_disposition=disposition,original_evidence=evidence,original_owner=owner,original_related=related,original_wave=wave,first_review_stage=stage,current_disposition="UNVERIFIED",review_state="QUEUED"))
print("D2 rows",len(rows),[(r["id"],r["original_disposition"]) for r in rows if "merg" in r["original_disposition"].lower()],flush=True)
assert len(rows)==86 and len({r["id"] for r in rows})==86
dups={"D2-X-009":"D2-AGT-001","D2-AGT-013":"D2-ZKC-004"}
for r in rows:
 if r["id"] in dups:r.update(current_disposition="NOT_APPLICABLE",review_state="HISTORICAL_DUPLICATE_TOMBSTONE",canonical_id=dups[r["id"]],rationale="Original D2 duplicate; reconcile canonical finding instead.")
q={"schema":"s4-d2-reconciliation-queue-v1","source_sha":inv["source_sha"],"historical_source":str(reg),"historical_source_sha256":hashlib.sha256(reg.read_bytes()).hexdigest(),"unique_items":84,"raw_rows":86,"rows":rows}
(root/"d2_reconciliation_queue.json").write_text(json.dumps(q,indent=2)+"\n")
(root/"baseline_checks.json").write_text(json.dumps({"source_sha":inv["source_sha"],"observed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"checks":checks},indent=2)+"\n")
shutil.copyfile("/tmp/sentinel-s4-inventory.py",root/"inventory_capture.py")
print("evidence written",[(p.name,p.stat().st_size) for p in root.iterdir()],flush=True)
