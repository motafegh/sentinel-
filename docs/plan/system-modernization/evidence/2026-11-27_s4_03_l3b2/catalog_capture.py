from pathlib import Path
import json,csv,io,hashlib,urllib.request,collections as C
ev=Path("docs/plan/system-modernization/evidence/2026-11-27_s4_03_l3b2")
pin="ed62986909d00f7260e38178fe9a6cd5064c9ef6"
base="https://raw.githubusercontent.com/smartbugs/CVE-Smart-Contracts/"+pin+"/"
files={"cve.json":(967352,"c04a6e79b6d046134ebf8f17758ab29b4e78381a"),"data/index.csv":(99835,"0f9ff21e4b2e72c65e95f7ee2416fc17226518a1"),"validation/validation.csv":(88002,"5013dfe2c29cda8b2c83b36d606ca191e804eca3"),"metadata.json":(1699,"562dc077cca838aa7eaf841f5ee01ea978af541f"),"localization/locations.json":(398056,"41ec9de4461acddbcc886afb7b03ca17965dea1a"),"labeling/labels.json":(436437,"78637e0cbd227857b0548de183d49c12c2c7e9e7")}
def get(path,size,blob):
 with urllib.request.urlopen(urllib.request.Request(base+path,headers={"User-Agent":"Sentinel-public-metadata-research/1.0","Accept-Encoding":"identity"}),timeout=15) as r:b=r.read(size+1)
 assert len(b)==size and hashlib.sha1(("blob "+str(len(b))+"\0").encode()+b).hexdigest()==blob,(path,len(b),size)
 return b
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=3) as ex:data=dict(zip(files,ex.map(lambda p:get(p,*files[p]),files)))
j=json.loads(data["cve.json"]);r=j["records"];index=list(csv.DictReader(io.StringIO(data["data/index.csv"].decode())))
val=list(csv.DictReader(io.StringIO(data["validation/validation.csv"].decode())))
loc=json.loads(data["localization/locations.json"]);lab=json.loads(data["labeling/labels.json"]);m=json.loads(data["metadata.json"])
print("index columns",list(index[0]),"validation columns",list(val[0]),"loc keys count",len(loc),"label keys count",len(lab))
print("validation categorical",[(k,C.Counter(x[k] for x in val).most_common(8)) for k in val[0] if len({x[k] for x in val})<12])
print("primary SWC",C.Counter(x["labels"]["SWC"]["primary"]["id"] if x["labels"]["SWC"]["primary"] else "NONE" for x in r.values()))
print("primary IDN",C.Counter(x["labels"]["IulianoDiNucci2026"]["primary"]["name"] if x["labels"]["IulianoDiNucci2026"]["primary"] else "NONE" for x in r.values()))
print("labels shape",list(lab)[:3],"location shape",list(loc)[:3])
assert len(r)==491 and len(index)==568 and len({x["cve_id"] for x in index})==568
ix={x["cve_id"]:x for x in index}
assert all(ix[k]["refuted"]=="f" and ix[k]["kind"]=="contract" and ix[k]["artifact"]==v["artifacts"]["source"]["path"] for k,v in r.items())
def ids(x,t):
 a=x["labels"][t]
 return [b["id"] for b in ([a["primary"]] if a["primary"] else [])+a["secondary"]]
# Freeze first-priority quotas before individual source retrieval; never choose on source contents.
quotas=[("gas",("SWC-126","SWC-128"),3),("dos",("SWC-113",),2),("return",("SWC-104",),2),("arithmetic",("SWC-101",),1),("reentrancy",("SWC-107",),1),("other",(),1)]
used=set();selected=[];pools={}
for st,swcs,limit in quotas:
 candidates=[(k,x) for k,x in r.items() if (not swcs or set(ids(x,"SWC"))&set(swcs))]
 pools[st]=len(candidates)
 for k,x in sorted(candidates,key=lambda v:hashlib.sha256(("S4-L3B2-v1|"+st+"|"+v[0]).encode()).hexdigest()):
  if sum(z["stratum"]==st for z in selected)>=limit:break
  a=x["artifacts"]["source"];group=a["verified_source"]
  if group in used:continue
  used.add(group);selected.append({"case_id":f"CVE-P{len(selected)+1:02d}","cve_id":k,"stratum":st,"source_path":a["path"],"explorer_path":group,"runtime_path":x["artifacts"]["runtime"]["path"] if x["artifacts"]["runtime"] else None,"labels":{t:ids(x,t) for t in x["labels"]},"locations":x["localization"]["locations"],"source_contract":a["contract"],"chain":a["chain"],"index_source_address_matches":a["address"]==ix[k]["src_addr"].lower(),"index_deployment_address_matches":bool(x["artifacts"]["runtime"]) and x["artifacts"]["runtime"]["address"]==ix[k]["addr"].lower()})
# Only metadata and hashes committed; local case selection contains public addresses through paths.
out={"pin":pin,"metadata_file_hashes":{p:{"sha256":hashlib.sha256(b).hexdigest(),"git_blob":files[p][1],"bytes":len(b)} for p,b in data.items()},"metadata_coverage":m["coverage"],"index_kind_counts":dict(C.Counter(x["kind"] for x in index)),"refuted":sum(x["refuted"]=="t" for x in index),"catalog":len(r),"distinct_explorer_paths":len({x["artifacts"]["source"]["verified_source"] for x in r.values()}),"distinct_runtime_paths":len({x["artifacts"]["runtime"]["path"] for x in r.values() if x["artifacts"]["runtime"]}),"location_records":len(loc),"label_records":len(lab),"catalog_location_count":sum(len(x["localization"]["locations"]) for x in r.values()),"swc_primary":dict(C.Counter(x["labels"]["SWC"]["primary"]["id"] if x["labels"]["SWC"]["primary"] else "NONE" for x in r.values())),"swc_any":dict(C.Counter(t for x in r.values() for t in set(ids(x,"SWC")))),"selection_quotas":{s:n for s,_,n in quotas},"selection_pools":pools,"selected_counts":dict(C.Counter(x["stratum"] for x in selected)),"limits":"Unique explorer record as provisional sample unit, not independent project or leakage group. No claim truth."}
(ev/"catalog_profile.json").write_text(json.dumps(out,indent=2)+"\n")
local=Path("data_module/data/s4-cve-preflight-v1");assert not local.exists();local.mkdir(mode=0o700)
(local/"selection.json").write_text(json.dumps(selected,indent=2)+"\n")
print("profile",json.dumps(out,indent=2),"selected",[(x["case_id"],x["stratum"],x["labels"]["SWC"]) for x in selected])
