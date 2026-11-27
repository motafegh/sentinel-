from pathlib import Path
import json,hashlib,urllib.request,csv,io,collections as C
ev=Path("docs/plan/system-modernization/evidence/2026-11-27_s4_03_l3b2")
p=json.loads((ev/"catalog_profile.json").read_text())
def get(path):
 m=p["metadata_file_hashes"][path]
 u="https://raw.githubusercontent.com/smartbugs/CVE-Smart-Contracts/"+p["pin"]+"/"+path
 with urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"Sentinel-public-metadata-research/1.0","Accept-Encoding":"identity"}),timeout=15) as r:b=r.read(m["bytes"]+1)
 assert len(b)==m["bytes"] and hashlib.sha256(b).hexdigest()==m["sha256"], (path,len(b),m["bytes"])
 return b
from concurrent.futures import ThreadPoolExecutor
paths=["cve.json","data/index.csv","validation/validation.csv","labeling/labels.json","localization/locations.json"]
data={}
for path in paths:
 for attempt in range(2):
  try:
   data[path]=get(path);break
  except Exception as e:print("retry",path,attempt,type(e).__name__,str(e)[:150],flush=True)
 if path not in data:raise RuntimeError("verified metadata unavailable: "+path)
r=json.loads(data["cve.json"])["records"]
i={x["cve_id"]:x for x in csv.DictReader(io.StringIO(data["data/index.csv"].decode()))}
v={x["cve_id"]:x for x in csv.DictReader(io.StringIO(data["validation/validation.csv"].decode()))}
l=json.loads(data["labeling/labels.json"])["records"]
loc=json.loads(data["localization/locations.json"])
expected={k for k,x in i.items() if x["kind"]=="contract" and x["refuted"]=="f" and v[k]["classification"]=="match"}
out={"index_count":len(i),"validation_contract_count":len(v),"actual_label_record_count":len(l),"labels_cover_index":set(l)==set(i),"catalog_matches_selected_index_validation":set(r)==expected,"localizations_cover_catalog":set(loc)==set(r),"catalog_labels_equal_full_labels_count":sum(x["labels"]==l[k] for k,x in r.items()),"upstream_refuted_count":sum(x["refuted"]=="t" for x in i.values()),"initial_profile_label_records_field_error":"Initial capture counted two top-level JSON envelope keys, not records. Correct count is actual_label_record_count above. Original capture retained."}
assert len(l)==568 and set(l)==set(i) and set(r)==expected and set(loc)==set(r)
(ev/"metadata_reconciliation.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
