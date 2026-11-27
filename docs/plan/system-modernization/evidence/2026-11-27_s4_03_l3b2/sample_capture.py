from pathlib import Path, PurePosixPath
import json,hashlib,urllib.request,urllib.parse,re,collections as C,concurrent.futures as cf,time
EV=Path("docs/plan/system-modernization/evidence/2026-11-27_s4_03_l3b2")
LOCAL=Path("data_module/data/s4-cve-preflight-v1")
BASE=Path("data_module/data")
PIN="ed62986909d00f7260e38178fe9a6cd5064c9ef6"
selection=json.loads((LOCAL/"selection.json").read_text())
tree={x["path"]:x for x in json.loads((LOCAL/"pinned_blobs.json").read_text())}
assert len(selection)==5 and len(tree)==20 and all(x["mode"]=="100644" for x in tree.values())
def sha(b):return hashlib.sha256(b).hexdigest()
def get(path):
 x=tree[path];size=x["size"];assert 0<size<1024*1024 and not PurePosixPath(path).is_absolute() and ".." not in PurePosixPath(path).parts
 u="https://raw.githubusercontent.com/smartbugs/CVE-Smart-Contracts/"+PIN+"/"+urllib.parse.quote(path,safe="/")
 with urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"Sentinel-public-metadata-research/1.0","Accept-Encoding":"identity"}),timeout=15) as r:b=r.read(size+1)
 assert len(b)==size,(len(b),size)
 assert hashlib.sha1(("blob "+str(len(b))+"\0").encode()+b).hexdigest()==x["sha"]
 return b
def explorer_record(obj):
 if isinstance(obj,list) and len(obj)==1 and isinstance(obj[0],dict):return obj[0]
 if "SourceCode" in obj:return obj
 return explorer_record(obj["result"])
def source_units(source):
 s=source.strip()
 if s.startswith("{{") and s.endswith("}}"):s=s[1:-1]
 if s.startswith("{"):
  j=json.loads(s);units=j.get("sources",j)
  return {k:v["content"].encode() if isinstance(v,dict) else v.encode() for k,v in units.items()},j.get("settings")
 return {"source.sol":source.encode()},None
def screen(b):
 text=b.decode("utf-8-sig")
 return {"sha256":sha(b),"bytes":len(b),"line_count":len(text.splitlines()),"spdx":sorted(set(re.findall(r"(?m)SPDX-License-Identifier:\s*([^\r\n*]+)",text))),"copyright_notice":bool(re.search(r"(?i)copyright",text)),"pragma_solidity":sorted(set(re.findall(r"pragma\s+solidity\s+([^;]+);",text))),"import_directive_count":len(re.findall(r"(?m)^\s*import\s",text))}
raw=C.defaultdict(set)
for p in (BASE/"raw").glob("*/ingestion_manifest.json"):
 for x in json.loads(p.read_text())["files"]:raw[x["sha256"]].add(p.parent.name)
m=json.loads(Path("docs/plan/ml-R4/evidence/2026-08-15_phase8_logical_v3/logical_v3_manifest.json").read_text())
claims=BASE/"r4-v2-build/source_claims.jsonl"
assert sha(claims.read_bytes())==m["artifacts"]["claims"]["sha256"]
norm={json.loads(line)["artifact_id"] for line in claims.open()}
def compare_bytes(b):
 h=sha(b);return {"raw_manifest_sources":sorted(raw.get(h,[])),"exact_normalized_artifact":h in norm}
def run(c):
 paths={"source":c["source_path"],"explorer":c["explorer_path"],"runtime":c["runtime_path"],"cve":"data/cves/"+c["cve_id"]+".json"}
 result={"case_id":c["case_id"],"stratum":c["stratum"],"labels":c["labels"],"blobs":{},"admission":"NOT_ADMITTED","source_files_staged":False}
 values={}
 for k,p in paths.items():
  try:
   b=get(p);values[k]=b;result["blobs"][k]={"status":"VERIFIED_BLOB","git_blob":tree[p]["sha"],"sha256":sha(b),"bytes":len(b)}
  except Exception as e:result["blobs"][k]={"status":"UNAVAILABLE","error_type":type(e).__name__,"error":str(e)[:150]}
 try:
  src=values["source"];result["source"]=screen(src);result["source"].update(compare_bytes(src))
  e=explorer_record(json.loads(values["explorer"]));units,settings=source_units(e["SourceCode"])
  result["explorer"]={"compiler_version":e.get("CompilerVersion"),"contract_name_matches_catalog":e.get("ContractName")==c["source_contract"],"optimizer":e.get("OptimizationUsed"),"runs":e.get("Runs"),"evm_version":e.get("EVMVersion"),"license_type":e.get("LicenseType"),"unit_count":len(units),"settings_retained":settings is not None,"source_exact_unit_matches":sum(src==u for u in units.values()),"source_newline_normalized_unit_matches":sum(src.replace(b"\r\n",b"\n")==u.replace(b"\r\n",b"\n") for u in units.values()),"units":[{**screen(u),**compare_bytes(u)} for u in units.values()]}
  ranges=[]
  for l in c["locations"]:
   name=l["source_path"];f=l["function"];u=units.get(name)
   if u is None and len(units)==1 and name=="source.sol":u=next(iter(units.values()))
   n=len(u.decode("utf-8-sig").splitlines()) if u is not None else None
   ranges.append({"source_unit_found":u is not None,"range_in_bounds":n is not None and 1<=f["start_line"]<=f["end_line"]<=n,"source_unit_sha256":sha(u) if u is not None else None})
  result["localization_checks"]=ranges
  result["rights_disposition"]="EXPLICIT_SOURCE_NOTICE_REVIEW_REQUIRED" if result["source"]["spdx"] else "NO_SPDX_RIGHTS_UNRESOLVED"
 except Exception as e:result["source_analysis_error"]={"type":type(e).__name__,"error":str(e)[:180]}
 if "runtime" in values:
  t=values["runtime"].decode().strip().removeprefix("0x")
  result["runtime_hex_valid"]=len(t)%2==0 and bool(re.fullmatch(r"[0-9a-fA-F]+",t))
 if "cve" in values:
  j=json.loads(values["cve"]);result["cve_identity_matches"]=j.get("cveMetadata",{}).get("cveId")==c["cve_id"]
 print(c["case_id"],[(k,x["status"]) for k,x in result["blobs"].items()],"analysis_error",result.get("source_analysis_error"),"unit_matches",result.get("explorer",{}).get("source_exact_unit_matches"),"rights",result.get("rights_disposition"),"raw_overlap",result.get("source",{}).get("raw_manifest_sources"),flush=True)
 return result
start=time.monotonic()
with cf.ThreadPoolExecutor(max_workers=3) as ex:results=list(ex.map(run,selection))
out={"pin":PIN,"selection_sha256":sha((LOCAL/"selection.json").read_bytes()),"tree_subset_sha256":sha((LOCAL/"pinned_blobs.json").read_bytes()),"records":results,"elapsed_seconds":round(time.monotonic()-start,3),"source_files_staged":0,"runtime_reproduced":0,"labels_admitted":0,"limits":"Pinned blob bytes and location range bounds only; no AST localization or source-runtime recompilation, no current live-chain queries, no normalized/near-duplicate grouping. SPDX notices not full rights adjudication. Public evidence uses opaque cases and hashes; raw code not persisted."}
(EV/"sample_preflight.json").write_text(json.dumps(out,indent=2)+"\n")
print("verified",sum(x["status"]=="VERIFIED_BLOB" for r in results for x in r["blobs"].values()),"of",len(tree),flush=True)
