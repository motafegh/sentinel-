from pathlib import Path
import csv,io,zipfile,hashlib,json,urllib.request,collections as C
ev=Path("docs/plan/system-modernization/evidence/2026-11-27_s4_03_l3b0a");prev=ev.parent/"2026-11-27_s4_03_l3b"
m=json.loads((prev/"dive_label_comparison.json").read_text());f=m["expected_archive"]
def sha(b):return hashlib.sha256(b).hexdigest()
with urllib.request.urlopen(urllib.request.Request(f["links"]["self"],headers={"User-Agent":"Sentinel-public-metadata-research/1.0"}),timeout=15) as r:b=r.read(f["size"]+1)
assert len(b)==f["size"] and sha(b)==m["archive_sha256"]
with zipfile.ZipFile(io.BytesIO(b)) as z:
 q=z.read("Labels/Tool_Results.csv");v=z.read("Labels/DIVE_Labels.csv")
assert sha(q)=="dd4226b8024ad1b9a9f8d12d43de2ea4125ed63d3548229678fdfcf67161fb70" and sha(v)==m["published_csv_sha256"]
tr=csv.DictReader(io.StringIO(q.decode("utf-8-sig"),newline=""));rows=list(tr);cols=tr.fieldnames[1:]
lr=csv.DictReader(io.StringIO(v.decode("utf-8-sig"),newline=""));labels={x["contractID"]:x for x in lr};names=lr.fieldnames[1:]
assert len(names)==8 and len(cols)==48 and all(x[k] in ("","0","1") for x in rows for k in cols)
by=C.defaultdict(list)
for x in rows:by[x["contractID"]].append(x)
assert set(by)==set(labels) and len(labels)==22330
dups={k:a for k,a in by.items() if len(a)>1}
def sets(a,k):return set(x[k] for x in a)-{""}
def nonempty(x):return sum(x[k]!="" for k in cols)
profile={"rows":len(rows),"distinct_ids":len(by),"repeated_ids":len(dups),"extra_rows":len(rows)-len(by),"multiplicity_histogram":dict(C.Counter(map(len,by.values()))),"exact_duplicate_extra_rows":sum(len(a)-len({tuple(x[k] for k in cols) for x in a}) for a in by.values()),"duplicate_ids_with_conflicting_nonblank":sum(any(sets(a,k)=={"0","1"} for k in cols) for a in dups.values()),"duplicate_ids_with_blank_and_nonblank":sum(any("" in {x[k] for x in a} and sets(a,k) for k in cols) for a in dups.values()),"duplicate_ids_all_rows_identical":sum(len({tuple(x[k] for k in cols) for x in a})==1 for a in dups.values()),"all_blank_rows":sum(nonempty(x)==0 for x in rows),"all_blank_first_ids":sum(nonempty(a[0])==0 for a in by.values()),"all_blank_last_ids":sum(nonempty(a[-1])==0 for a in by.values()),"all_blank_ids":sum(not any(nonempty(x) for x in a) for a in by.values()),"first_more_nonblank_than_last":sum(nonempty(a[0])>nonempty(a[-1]) for a in dups.values()),"last_more_nonblank_than_first":sum(nonempty(a[-1])>nonempty(a[0]) for a in dups.values()),"conflicting_tool_cells":sum(sets(a,k)=={"0","1"} for a in by.values() for k in cols)}
voters=[["Mythril","Slither","Solhint"],["MAIAN","Mythril","Semgrep","Slither","Solhint","VeriSmart"],["Mythril","Slither","VeriSmart"],["Mythril","Slither","Solhint"],["Mythril","Slither","Solhint"],["Mythril","Slither","Solhint"],["Mythril"],["Mythril","Slither","Solhint"]]
inverters={1:{"Slither":"Semgrep"},5:{"Slither":"MAIAN"},6:{"Slither":"Semgrep","Solhint":"Semgrep"}}
result={};obs={};trans={}
for mode in ("first","last","unambiguous_union"):
 counts={n:C.Counter() for n in names};coverage={n:C.Counter() for n in names};bydup={n:C.Counter() for n in names}
 for cid,a in by.items():
  vals={k:(a[0][k] if mode=="first" else a[-1][k] if mode=="last" else next(iter(sets(a,k))) if len(sets(a,k))==1 else "") for k in cols}
  for i,n in enumerate(names,1):
   raw=[vals[t+"_"+str(i)] for t in voters[i-1]]
   corrected=[("0" if vals[t+"_"+str(i)]=="1" and vals[inverters[i][t]+"_"+str(i)]=="1" else vals[t+"_"+str(i)]) if t in inverters.get(i,{}) else vals[t+"_"+str(i)] for t in voters[i-1]]
   yes=corrected.count("1");no=corrected.count("0");respond=yes+no;pub=int(labels[cid][n])
   # Reproduce literal tie-positive majority, including empty vote list of fixed named tools.
   pred=int(yes>=no) if i==8 else int(yes>0)
   key=f"replay_{pred}_published_{pub}";counts[n][key]+=1
   if cid in dups:bydup[n][key]+=1
   coverage[n]["no_eligible_response"]+=respond==0
   coverage[n]["no_eligible_response_published_positive"]+=respond==0 and pub==1
   coverage[n]["one_or_more_positive_before_inversion"]+=raw.count("1")>0
   coverage[n]["eligible_positive_after_inversion"]+=yes>0
   coverage[n]["eligible_disagreement"]+=yes>0 and no>0
   coverage[n]["inverted_vote"]+=raw!=corrected
   coverage[n]["majority_empty_tie_positive"]+=i==8 and respond==0
 result[mode]={n:dict(counts[n]) for n in names};obs[mode]={n:dict(coverage[n]) for n in names};trans[mode]={n:dict(bydup[n]) for n in names}
out={"archive_sha256":sha(b),"tool_sha256":sha(q),"label_sha256":sha(v),"category_order":names,"profile":profile,"voters":voters,"inverters":inverters,"replay_counts":result,"coverage":obs,"repeated_id_replay_counts":trans,"limits":"Hypothesis replay from published Table 8 and inspected current election.py; no exact original voting config or site-level refiner inputs. No eligible response remains unknown for Sentinel even if literal upstream tie would produce 1. Union conflict treated as missing only for sensitivity. Counts are annotation disagreement, not false positives/negatives."}
(ev/"tool_vote_reconstruction.json").write_text(json.dumps(out,indent=2)+"\n")
print("PROFILE",json.dumps(profile))
for mode,c in result.items():print(mode,[(n,x.get("replay_1_published_0",0),x.get("replay_0_published_1",0),x.get("replay_1_published_1",0)) for n,x in c.items()])
print("FIRST COVERAGE",json.dumps(obs["first"]))
