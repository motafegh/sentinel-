from pathlib import Path
import csv,io,zipfile,hashlib,json,urllib.request,collections as C
ev=Path("docs/plan/system-modernization/evidence/2026-11-27_s4_03_l3b0a");prev=ev.parent/"2026-11-27_s4_03_l3b"
m=json.loads((prev/"dive_label_comparison.json").read_text());f=m["expected_archive"]
with urllib.request.urlopen(urllib.request.Request(f["links"]["self"],headers={"User-Agent":"Sentinel-public-metadata-research/1.0"}),timeout=15) as r:b=r.read(f["size"]+1)
assert hashlib.sha256(b).hexdigest()==m["archive_sha256"]
with zipfile.ZipFile(io.BytesIO(b)) as z:
 tr=csv.DictReader(io.StringIO(z.read("Labels/Tool_Results.csv").decode("utf-8-sig")));rows=list(tr);cols=tr.fieldnames[1:]
 lr=csv.DictReader(io.StringIO(z.read("Labels/DIVE_Labels.csv").decode("utf-8-sig")));labels={x["contractID"]:x for x in lr}
by=C.defaultdict(list)
for x in rows:by[x["contractID"]].append(x)
tools=["MAIAN","Mythril","Semgrep","Slither","Solhint","VeriSmart"]
patterns=C.Counter();pertool=C.defaultdict(C.Counter);time=C.Counter();classes=C.defaultdict(C.Counter)
for cid,a in by.items():
 first=a[0];last=a[-1]
 def response(x,t):return any(x[t+"_"+str(i)]!="" for i in range(1,9))
 if len(a)>1:
  patterns["first="+",".join(t for t in tools if response(first,t))+";last="+",".join(t for t in tools if response(last,t))]+=1
  for t in tools:
   pertool[t][f"first_{int(response(first,t))}_last_{int(response(last,t))}"]+=1
 if labels[cid]["Time manipulation"]=="1" and all(first[t+"_8"]=="" for t in ("Mythril","Slither","Solhint")):
  u={k:next(iter({x[k] for x in a}-{""}),"") for k in cols}
  vals=[u[t+"_8"] for t in ("Mythril","Slither","Solhint")]
  time["published_positive_no_first_eligible"]+=1
  time["repeated_id" if len(a)>1 else "single_id"]+=1
  time["union_no_eligible_response" if not any(vals) else "union_positive_vote" if "1" in vals else "union_only_zero_votes"]+=1
  if any(vals):time["union_majority_positive" if vals.count("1")>=vals.count("0") else "union_majority_negative"]+=1
 for i,n in enumerate(list(labels[cid])[1:],1):
  vals={x[t+"_"+str(i)] for x in a for t in tools}
  pub=labels[cid][n]
  classes[n]["published_positive_no_any_tool_positive"]+=pub=="1" and "1" not in vals
  classes[n]["published_positive_no_any_tool_observation"]+=pub=="1" and vals=={""}
  classes[n]["published_zero_any_tool_positive"]+=pub=="0" and "1" in vals
out={"duplicate_tool_presence_patterns":dict(patterns),"duplicate_tool_presence_by_tool":pertool,"timestamp_no_first_response_breakdown":time,"all_tool_union_class_observations":classes,"limits":"No response or zero observations are not ground truth. Repeated-row semantics remain unproven; observed nonblank agreement permits a descriptive union only, not independent voting or label admission."}
(ev/"duplicate_and_missingness_followup.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
