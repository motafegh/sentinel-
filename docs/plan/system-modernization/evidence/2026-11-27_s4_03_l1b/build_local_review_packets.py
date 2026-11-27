from pathlib import Path
import argparse,hashlib,json,zipfile,io,secrets,subprocess as sp
import yaml
ROOT=Path(__file__).resolve().parent
TAXONOMY=Path("data_module/sentinel_data/labeling/schema/taxonomy.yaml")
FILES={"source.sol","review_task.json","bundle_manifest.json","START-HERE.md"}
INSTRUCTIONS="""# Independent source review

Read only this packet in an isolated review environment. Do not inspect the project, native annotations, tools' previous findings, earlier reviewer conclusions, prior chats, or external source copies. Disclose prior familiarity or access to case-specific conclusions before starting; stop if contaminated. Use one isolated session per packet, especially if another packet may contain the same source under a different class. Do not execute or deploy the source.

Verify the full source SHA-256 against the manifest. Apply the exact supplied class description; do not redefine it based on risk or detector heuristics. Consider all source components and identify scope limits. For positive evidence identify concrete source lines, declaration/return use or persistent blocking mechanism as relevant, counterevidence, and reasoning. Separately report whether meaningful security impact is established. Return SUPPORTS_CLASS_CANDIDATE, NOT_CONFIRMED, or INSUFFICIENT; none is an accepted dataset label. NOT_CONFIRMED is not a confirmed negative. No source silence, missing label, or unavailable tool establishes a negative. Report what was inspected, uncertainty, independence declaration and source hash. Do not infer training eligibility. Return the review separately, not by editing the packet.
"""
def sha(b):return hashlib.sha256(b).hexdigest()
def js(o):return (json.dumps(o,sort_keys=True,indent=2)+"\n").encode()
def definitions():
    return {c["name"]:c for c in yaml.safe_load(TAXONOMY.read_text())["classes"]}

def verify(blob,expected=None):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names=z.namelist()
        assert len(names)==len(FILES) and set(names)==FILES
        data={n:z.read(n) for n in names}
    m=json.loads(data["bundle_manifest.json"]);t=json.loads(data["review_task.json"])
    assert set(m)=={"schema","packet_id","files","source_sha256"}
    assert set(t)=={"schema","packet_id","class_name","class_index","class_description","compiler_version","source_sha256","scope"}
    assert m["schema"]=="s4-neutral-review-packet-v1" and t["schema"]=="s4-neutral-review-task-v1"
    assert set(m["files"])==FILES-{"bundle_manifest.json"}
    assert all(m["files"][n]==sha(data[n]) for n in m["files"])
    assert sha(data["source.sol"])==m["source_sha256"]==t["source_sha256"]
    assert t["packet_id"]==m["packet_id"]
    assert data["START-HERE.md"]==INSTRUCTIONS.encode()
    d=definitions()[t["class_name"]];assert t["class_description"]==d["description"] and t["class_index"]==d["id"]
    if expected is not None:
        assert t["class_name"]==expected["class"] and t["source_sha256"]==expected["source_sha256"] and t["compiler_version"]==expected["compiler_version"]
    return t

def make(row):
    data=Path(row["source_path"]).read_bytes();assert sha(data)==row["source_sha256"]
    d=definitions()[row["class"]];pid=secrets.token_hex(12)
    t={"schema":"s4-neutral-review-task-v1","packet_id":pid,"class_name":row["class"],"class_index":d["id"],"class_description":d["description"],"compiler_version":row["compiler_version"],"source_sha256":sha(data),"scope":"Full normalized Solidity source of the specified artifact, all components; deployment, external dependency behavior and runtime context not supplied. Source-only review."}
    files={"source.sol":data,"review_task.json":js(t),"START-HERE.md":INSTRUCTIONS.encode()}
    files["bundle_manifest.json"]=js({"schema":"s4-neutral-review-packet-v1","packet_id":pid,"source_sha256":sha(data),"files":{k:sha(v) for k,v in files.items()}})
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for n,v in files.items():z.writestr(n,v)
    b=buf.getvalue();verify(b,row);return pid,b

def must_reject(b):
    try:verify(b)
    except (AssertionError,KeyError,ValueError,zipfile.BadZipFile):return True
    raise AssertionError("invalid packet accepted")

def mutate(b,extra=False):
    with zipfile.ZipFile(io.BytesIO(b)) as z:files={n:z.read(n) for n in z.namelist()}
    if extra:files["prior_review.json"]=b"{}"
    else:files["source.sol"]+=b" "
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w") as z:
        for n,v in files.items():z.writestr(n,v)
    return buf.getvalue()

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    assert a.output.parts[:2]==("data_module","data") and not a.output.is_absolute()
    assert ".." not in a.output.parts
    assert sp.run(["git","check-ignore","-q",str(a.output/"private_index.json")]).returncode==0
    rows=json.loads((ROOT/"primary_reviews.json").read_text())["records"]
    a.output.mkdir(parents=True,exist_ok=False,mode=0o700);(a.output/"packets").mkdir(mode=0o700)
    index=[];sample=None
    for row in rows:
        pid,b=make(row);path=a.output/"packets"/(pid+".zip");path.write_bytes(b);path.chmod(0o600)
        verify(path.read_bytes(),row)
        index.append({"packet_id":pid,"case_id":row["case_id"],"class":row["class"],"source_sha256":row["source_sha256"],"packet_sha256":sha(b),"group_id":row["group_id"]})
        if sample is None:sample=b
    tests={"modified_source_rejected":must_reject(mutate(sample)),"extra_prior_review_rejected":must_reject(mutate(sample,True))}
    private=a.output/"private_index.json";private.write_bytes(js({"packets":index,"access":"Coordinator only; never give to reviewer"}));private.chmod(0o600)
    public={"schema":"s4-local-packet-verification-v1","packet_count":len(index),"validated":len(index),"distinct_source_groups":len({x["group_id"] for x in index}),"checks":tests,"source_files_published":False,"independent_reviews_completed":0,"local_only":True,"private_index_and_packets_gitignored":True,"limitations":"Local packet verification only. Isolation is a procedural requirement, not ensured by ZIP. No upload or redistribution of DIVE sources."}
    (ROOT/"packet_verification.json").write_bytes(js(public));print(json.dumps(public))
if __name__=="__main__":main()
