from pathlib import Path
import json,csv,hashlib,difflib,collections as C
ev=Path("docs/plan/system-modernization/evidence/2026-11-26_s4_03_l1a")
m=json.loads((ev/"pilot_manifest.json").read_text())
base=Path("data_module/data")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
results=[]
for c in m["calibration"]:
 p=Path(c["raw_path"]);a=Path(c["annotation_path"]);assert sha(p)==c["raw_sha256"] and sha(a)==c["annotation_sha256"]
 found=[]
 for meta in (base/"sentinel-preprocessed-r4-v2"/c["source"]).glob("*.meta.json"):
  x=json.loads(meta.read_text())
  if any(r["original_path"]==c["manifest_path"] for r in x["source_records"]):found.append(x)
 assert len(found)==1,(c,len(found))
 meta=found[0];norm=base/"sentinel-preprocessed-r4-v2"/c["source"]/(meta["sha256"]+".sol");assert sha(norm)==meta["sha256"]
 raw=p.read_text(encoding="utf-8",errors="replace").splitlines();normalized=norm.read_text().splitlines()
 # Align exact stripped nonempty text only; repeated identical lines are not unique mappings.
 counts_raw=C.Counter(l.strip() for l in raw if l.strip());counts_norm=C.Counter(l.strip() for l in normalized if l.strip())
 unique_norm={l.strip():i for i,l in enumerate(normalized,1) if l.strip() and counts_norm[l.strip()]==1}
 mapping={i:unique_norm[l.strip()] for i,l in enumerate(raw,1) if l.strip() and counts_raw[l.strip()]==1 and l.strip() in unique_norm}
 if c["source"]=="smartbugs_curated":
  spans=[{"literal_category":v["category"],"lines":v["lines"]} for v in c["annotations"]]
 else:
  with a.open(newline="",encoding="utf-8-sig") as f:
   spans=[{"literal_category":v["bug type"],"lines":list(range(int(v["loc"]),int(v["loc"])+int(v["length"])))} for v in csv.DictReader(f)]
 results.append({"source":c["source"],"manifest_path":c["manifest_path"],"artifact_sha256":meta["sha256"],"raw_lines":len(raw),"normalized_lines":len(normalized),"flatten_status":meta["flatten_status"],"spans":len(spans),"in_bounds_spans":sum(bool(x["lines"]) and all(1<=i<=len(raw) for i in x["lines"]) for x in spans),"annotated_line_occurrences":sum(len(x["lines"]) for x in spans),"same_line_exact_nonempty":sum(1 for x in spans for i in x["lines"] if 1<=i<=min(len(raw),len(normalized)) and raw[i-1].strip() and raw[i-1].strip()==normalized[i-1].strip()),"unique_exact_text_matches":sum(i in mapping for x in spans for i in x["lines"]),"unique_exact_text_shifted":sum(i in mapping and mapping[i]!=i for x in spans for i in x["lines"]),"example_spans":[{"category":x["literal_category"],"raw_lines":x["lines"],"unique_text_match_lines":[mapping.get(i) for i in x["lines"]]} for x in spans[:3]]})
 print(c["source"],c["manifest_path"],results[-1]["spans"],results[-1]["in_bounds_spans"],results[-1]["annotated_line_occurrences"],results[-1]["same_line_exact_nonempty"],results[-1]["unique_exact_text_matches"],flush=True)
 # Private source inspection: show locations only in tool transcript, not in committed evidence.
 for x in spans[:1]:
  for i in x["lines"][:5]:
   if 1<=i<=len(raw):print(i,raw[i-1].strip()[:150])
(ev/"coordinate_results.json").write_text(json.dumps({"manifest_sha256":sha(ev/"pilot_manifest.json"),"method":"1-based bounds; upstream SolidiFI inspection.py compares loc <= detector line < loc+length. Unique exact stripped nonempty text matches are candidate anchors, not proven source maps. Same-line equality is not semantic equivalence.","results":results},indent=2)+"\n")
