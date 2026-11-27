from pathlib import Path
import json,urllib.request,hashlib,struct
ev=Path("docs/plan/system-modernization/evidence/2026-11-27_s4_03_l3b0a");p=ev.parent/"2026-11-27_s4_03_l3b"
m=json.loads((p/"dive_preflight.json").read_text());f=next(x for x in m["zenodo_record"]["data"]["files"] if x["key"]=="DIVE_Raw_Data.zip")
n=f["size"];start=n-65557;url=f["links"]["self"]
out={"expected_size":n,"published_checksum":f["checksum"],"requested_range":[start,n-1],"max_bytes":65557,"full_archive_downloaded":False,"archive_checksum_verified":False,"source_equivalence":"NOT_ESTABLISHED"}
try:
 req=urllib.request.Request(url,headers={"Range":f"bytes={start}-{n-1}","User-Agent":"Sentinel-public-metadata-research/1.0","Accept-Encoding":"identity"})
 with urllib.request.urlopen(req,timeout=12) as r:
  out.update(http_status=r.status,content_range=r.headers.get("Content-Range"),content_length=r.headers.get("Content-Length"))
  if r.status==206 and r.headers.get("Content-Range")==f"bytes {start}-{n-1}/{n}":
   b=r.read(65558);assert len(b)==65557
   i=b.rfind(b"PK\x05\x06");assert i>=0 and i+22<=len(b)
   sig,disk,cd_disk,disk_entries,entries,cd_size,cd_offset,comment=struct.unpack_from("<4s4H2LH",b,i)
   assert i+22+comment==len(b) and disk==cd_disk==0 and disk_entries==entries
   out.update(disposition="ZIP_DIRECTORY_METADATA_ONLY",tail_sha256=hashlib.sha256(b).hexdigest(),eocd_entries=entries,central_directory_bytes=cd_size,central_directory_offset=cd_offset,zip64_possible=entries==65535 or cd_size==4294967295)
  else:out["disposition"]="RANGE_NOT_HONORED_BODY_NOT_READ"
except Exception as e:out.update(disposition="UNAVAILABLE",error_type=type(e).__name__,error=str(e)[:200])
(ev/"raw_archive_range_preflight.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
