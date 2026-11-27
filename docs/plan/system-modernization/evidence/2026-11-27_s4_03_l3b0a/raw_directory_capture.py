from pathlib import Path
import urllib.request,json,struct,hashlib,zlib,collections as C,stat,time
ev=Path("docs/plan/system-modernization/evidence/2026-11-27_s4_03_l3b0a")
m=json.loads((ev/"raw_archive_range_preflight.json").read_text())
p=ev.parent/"2026-11-27_s4_03_l3b";a=json.loads((p/"dive_preflight.json").read_text());f=next(x for x in a["zenodo_record"]["data"]["files"] if x["key"]=="DIVE_Raw_Data.zip")
offset=m["central_directory_offset"];size=m["central_directory_bytes"];n=f["size"];assert 0<size<=8*1024*1024 and offset+size<=n
out={"expected_archive_bytes":n,"archive_checksum_verified":False,"full_archive_downloaded":False,"comparison":"CRC32 and uncompressed size only, not cryptographic source equality"}
try:
 req=urllib.request.Request(f["links"]["self"],headers={"Range":f"bytes={offset}-{offset+size-1}","User-Agent":"Sentinel-public-metadata-research/1.0","Accept-Encoding":"identity"})
 with urllib.request.urlopen(req,timeout=15) as r:
  assert r.status==206 and r.headers.get("Content-Range")==f"bytes {offset}-{offset+size-1}/{n}"
  b=r.read(size+1)
 assert len(b)==size
 out["directory_sha256"]=hashlib.sha256(b).hexdigest()
 pos=0;members=[]
 while pos<len(b):
  fields=struct.unpack_from("<4s6H3L5H2L",b,pos)
  sig,vm,vn,flags,method,mtime,mdate,crc,comp,unc,nlen,xlen,clen,disk,internal,external,loc=fields
  assert sig==b"PK\x01\x02" and pos+46+nlen+xlen+clen<=len(b)
  name=b[pos+46:pos+46+nlen].decode("utf-8" if flags&0x800 else "cp437")
  members.append((name,crc,unc,external));pos+=46+nlen+xlen+clen
 assert pos==len(b) and len(members)==m["eocd_entries"]
 out["directory_entries"]=len(members)
 out["top_prefix_counts"]=dict(C.Counter(x[0].split("/")[0] for x in members))
 sol=[x for x in members if x[0].lower().endswith(".sol") and "__MACOSX" not in x[0].split("/") and not Path(x[0]).name.startswith("._")]
 out["solidity_member_count"]=len(sol);out["solidity_parents"]=dict(C.Counter(str(Path(x[0]).parent) for x in sol))
 src=Path("data_module/data/raw_staging/dive/__source__")
 assert src.is_dir()
 local={x.name:x for x in src.glob("*.sol")}
 assert len(local)==22330
 by=C.defaultdict(list)
 for x in sol:by[Path(x[0]).name].append(x)
 ambiguous=[k for k,v in by.items() if len(v)!=1]
 out.update(local_solidity_paths=len(local),unique_archive_solidity_basenames=len(by),ambiguous_archive_basenames=len(ambiguous),local_only=len(set(local)-set(by)),archive_only=len(set(by)-set(local)))
 counts=C.Counter()
 for k in sorted(set(local)&set(by)):
  if len(by[k])!=1:counts["ambiguous"]+=1;continue
  name,crc,unc,external=by[k][0]
  if stat.S_ISLNK(external>>16):counts["archive_symlink"]+=1;continue
  f=local[k];data=f.read_bytes()
  if len(data)==unc and zlib.crc32(data)&0xffffffff==crc:counts["size_crc_match"]+=1
  elif len(data)!=unc:counts["size_mismatch"]+=1
  else:counts["crc_mismatch"]+=1
 out["comparison_counts"]=dict(counts);out["status"]="DIRECTORY_COMPARISON_COMPLETE"
except Exception as e:out.update(status="UNAVAILABLE_OR_FAILED",error_type=type(e).__name__,error=str(e)[:300])
(ev/"raw_archive_directory_comparison.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2),flush=True)
