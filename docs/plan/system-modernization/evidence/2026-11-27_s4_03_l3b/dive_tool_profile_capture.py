from pathlib import Path
import urllib.request,hashlib,zipfile,io,csv,json,collections as C
ev=Path("docs/plan/system-modernization/evidence/2026-11-27_s4_03_l3b");m=json.loads((ev/"dive_label_comparison.json").read_text());f=m["expected_archive"]
with urllib.request.urlopen(urllib.request.Request(f["links"]["self"],headers={"User-Agent":"Sentinel-public-metadata-research/1.0"}),timeout=15) as r:b=r.read(f["size"]+1)
print("received bytes",len(b),"sha256",hashlib.sha256(b).hexdigest(),flush=True)
assert len(b)==f["size"] and hashlib.sha256(b).hexdigest()==m["archive_sha256"]
with zipfile.ZipFile(io.BytesIO(b)) as z:q=z.read("Labels/Tool_Results.csv")
reader=csv.DictReader(io.StringIO(q.decode("utf-8-sig"),newline=""));rows=list(reader)
print("columns",reader.fieldnames,"rows",len(rows))
out={"archive_sha256":m["archive_sha256"],"member":"Labels/Tool_Results.csv","sha256":hashlib.sha256(q).hexdigest(),"bytes":len(q),"rows":len(rows),"columns":reader.fieldnames,"column_profiles":{}}
for k in reader.fieldnames:
 c=C.Counter(r[k] for r in rows);out["column_profiles"][k]={"distinct":len(c),"empty":c.get("",0),"value_counts":dict(c) if len(c)<=16 else None}
print(json.dumps(out,indent=2)[:7500])
(ev/"dive_tool_results_profile.json").write_text(json.dumps(out,indent=2)+"\n")
