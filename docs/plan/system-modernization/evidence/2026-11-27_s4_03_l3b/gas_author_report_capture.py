from pathlib import Path
import urllib.request,urllib.parse,hashlib,json,re,concurrent.futures
ev=Path("docs/plan/system-modernization/evidence/2026-11-27_s4_03_l3b")
m=json.loads((ev/"gas_gauge_selection.json").read_text())
files=[("finder/loop/Result_finder.txt",818153,"f0add0a9f84cfada4df821077788c16b94f437df"),("finder/noloop/Results.txt",321484,"4cbfca30b34dea72f572adf7d95ef1687623b2c6"),("fuzzer/Result_fuzzer.txt",1737399,"3e95585fb255120e7e989d6b8b22790c9583c8bb"),("threshold/Result_threshold.txt",2312151,"c33dc2bd05c27fb06753208236aa80c62f31b1b2")]
stems=[Path(x["path"]).stem.lower() for c in m["cohorts"] for x in c["selected"]]
def check(x):
 path,size,blob=x;path="Results/our results/"+path;u="https://raw.githubusercontent.com/gasgauge/gasgauge.github.io/"+m["pin"]+"/"+urllib.parse.quote(path,safe="/")
 out={"path":path,"expected_bytes":size,"git_blob":blob}
 try:
  with urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"Sentinel-public-metadata-research/1.0"}),timeout=15) as r:b=r.read(size+1)
  assert len(b)==size and hashlib.sha1(("blob "+str(len(b))+"\0").encode()+b).hexdigest()==blob
  t=b.decode("utf-8",errors="replace");lower=t.lower()
  out.update(status="VERIFIED_BLOB",sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),line_count=len(t.splitlines()),selected_stem_mentions=sum(s in lower for s in stems),per_selected_stem_counts={s:lower.count(s) for s in stems},contains_out_of_gas_text=bool(re.search(r"out.of.gas",lower)),contains_threshold_text="threshold" in lower)
  # Display format-only beginning lines to establish section structure, redact hexadecimal values.
  print(path,"format",[(re.sub(r"0x[0-9a-fA-F]+","<hex>",l)[:160]) for l in t.splitlines()[:5]],flush=True)
 except Exception as e:out.update(status="UNAVAILABLE",error_type=type(e).__name__,error=str(e)[:180])
 return out
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:r=list(ex.map(check,files))
(ev/"gas_gauge_author_report_preflight.json").write_text(json.dumps({"pin":m["pin"],"results":r,"limits":"Literal case-stem and keyword presence only; no authenticated source/run identity, execution settings or reproduced gas failure. Reports read in memory, no upstream source redistribution."},indent=2)+"\n")
print([(x["path"],x["status"],x.get("selected_stem_mentions")) for x in r])
