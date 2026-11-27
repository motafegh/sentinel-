from pathlib import Path
import urllib.request,urllib.parse,hashlib,json,re,concurrent.futures,collections as C,time
ev=Path("docs/plan/system-modernization/evidence/2026-11-27_s4_03_l3b")
m=json.loads((ev/"gas_gauge_selection.json").read_text());pin=m["pin"]
def sha(b):return hashlib.sha256(b).hexdigest()
def get(path,size,blob):
 u="https://raw.githubusercontent.com/gasgauge/gasgauge.github.io/"+pin+"/"+urllib.parse.quote(path,safe="/")
 with urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"Sentinel-public-metadata-research/1.0"}),timeout=12) as r:b=r.read(size+1)
 assert len(b)==size,(len(b),size)
 assert hashlib.sha1(("blob "+str(len(b))+"\0").encode()+b).hexdigest()==blob,"Git blob mismatch"
 return b
base=Path("data_module/data")
known=C.defaultdict(set)
for p in (base/"raw").glob("*/ingestion_manifest.json"):
 for e in json.loads(p.read_text())["files"]:known[e["sha256"]].add(p.parent.name)
# Compare exact raw source bytes only, not near duplicates.
def check(pair):
 cohort,x=pair;out={"cohort":cohort,"path":x["path"],"git_blob":x["sha"],"expected_size":x["size"],"linked_reports":len(x["reports"])}
 try:
  b=get(x["path"],x["size"],x["sha"]);t=b.decode("utf-8",errors="replace");h=sha(b)
  out.update(status="VERIFIED_BLOB",sha256=h,bytes=len(b),raw_exact_match_sources=sorted(known.get(h,[])),spdx=sorted(set(re.findall(r"SPDX-License-Identifier:\s*([^\r\n*]+)",t))),pragmas=sorted(set(re.findall(r"pragma\s+solidity\s+([^;]+);",t))),import_directive_count=len(re.findall(r"(?m)^\s*import\s",t)),copyright_notice_present=bool(re.search(r"(?i)copyright",t)))
  out["disposition"]="METADATA_ONLY_RIGHTS_NOT_ADJUDICATED"
 except Exception as e:out.update(status="UNAVAILABLE",error_type=type(e).__name__,error=str(e)[:180])
 # Only inspect historical per-source Slither reports for sample. No inferred truth.
 candidates=[r for r in x["reports"] if r["path"].startswith("Results/slither/")]
 out["slither_report"]={"status":"AMBIGUOUS_OR_MISSING","count":len(candidates)}
 if len(candidates)==1:
  r=candidates[0]
  try:
   b=get(r["path"],r["size"],r["sha"]);t=b.decode("utf-8",errors="replace")
   out["slither_report"]={"status":"VERIFIED_BLOB","path":r["path"],"sha256":sha(b),"bytes":len(b),"contains_case_stem":Path(x["path"]).stem.lower() in t.lower(),"mentions_gas":bool(re.search(r"(?i)\bgas\b",t)),"mentions_loop":bool(re.search(r"(?i)\bloop[s]?\b",t)),"mentions_error":bool(re.search(r"(?i)\b(error|failed|exception)\b",t))}
  except Exception as e:out["slither_report"]={"status":"UNAVAILABLE","error_type":type(e).__name__,"error":str(e)[:180]}
 print(cohort,out["status"],"reports",out["slither_report"]["status"],"spdx",out.get("spdx"),"overlap",out.get("raw_exact_match_sources"),flush=True)
 return out
pairs=[(c["k"],x) for c in m["cohorts"] for x in c["selected"]]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:rows=list(ex.map(check,pairs))
out={"selection_sha256":sha((ev/"gas_gauge_selection.json").read_bytes()),"pin":pin,"records":rows,"source_files_persisted":0,"source_files_executed":0,"labels_admitted":0,"notes":"Read public source bytes in memory for hash/header preflight only; no source staged or redistributed. Exact raw hash comparison against local ingestion manifests, not normalized or near duplicate checks. Keyword mentions are not findings or runtime evidence; named reports not exact source-bound."}
(ev/"gas_gauge_sample_preflight.json").write_text(json.dumps(out,indent=2)+"\n")
print("source verified",sum(r["status"]=="VERIFIED_BLOB" for r in rows),"reports verified",sum(r["slither_report"]["status"]=="VERIFIED_BLOB" for r in rows),"unique source blobs",len({x["git_blob"] for x in rows}))
