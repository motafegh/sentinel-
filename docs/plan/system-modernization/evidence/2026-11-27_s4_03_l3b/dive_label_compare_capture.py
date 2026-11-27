from pathlib import Path
import urllib.request,hashlib,zipfile,io,csv,json,collections as C
ev=Path("docs/plan/system-modernization/evidence/2026-11-27_s4_03_l3b")
m=json.loads((ev/"dive_preflight.json").read_text())
f=next(x for x in m["zenodo_record"]["data"]["files"] if x["key"]=="DIVE_Labels.zip")
def sha(b):return hashlib.sha256(b).hexdigest()
out={"expected_archive":f,"local_csv":m["local_csv"],"status":"NOT_RUN"}
try:
 with urllib.request.urlopen(urllib.request.Request(f["links"]["self"],headers={"User-Agent":"Sentinel-public-metadata-research/1.0"}),timeout=15) as r:
  b=r.read(f["size"]+1)
 assert len(b)==f["size"] and hashlib.md5(b).hexdigest()==f["checksum"].split(":")[1]
 out["archive_sha256"]=sha(b);out["archive_bytes"]=len(b);out["archive_md5_verified"]=True
 with zipfile.ZipFile(io.BytesIO(b)) as z:
  infos=z.infolist();out["members"]=[{"path":i.filename,"size":i.file_size} for i in infos]
  assert len(infos)<100 and sum(i.file_size for i in infos)<10000000
  candidates=[i for i in infos if Path(i.filename).name=="DIVE_Labels.csv" and "__MACOSX" not in i.filename]
  assert len(candidates)==1,candidates
  q=z.read(candidates[0])
 local=Path("data_module/data/raw_staging/dive_labels/DIVE_Labels.csv").read_bytes()
 def read(data):
  reader=csv.DictReader(io.StringIO(data.decode("utf-8-sig"),newline="")); rows=list(reader)
  assert reader.fieldnames and "contractID" in reader.fieldnames
  assert len({r["contractID"] for r in rows})==len(rows)
  return reader.fieldnames,{r["contractID"]:r for r in rows}
 a,x=read(local);c,y=read(q);shared=set(x)&set(y)
 changed=[k for k in shared if x[k]!=y[k]]
 out.update(status="COMPARED",published_csv_sha256=sha(q),published_csv_bytes=len(q),byte_identical=local==q,local_columns=a,published_columns=c,local_rows=len(x),published_rows=len(y),local_only=len(set(x)-set(y)),published_only=len(set(y)-set(x)),changed_shared_records=len(changed),changed_fields=dict(C.Counter(k for i in changed for k in set(x[i])|set(y[i]) if x[i].get(k)!=y[i].get(k))),same_keyed_csv=a==c and x==y)
except Exception as e:out.update(status="FAILED",error_type=type(e).__name__,error=str(e)[:300])
(ev/"dive_label_comparison.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
