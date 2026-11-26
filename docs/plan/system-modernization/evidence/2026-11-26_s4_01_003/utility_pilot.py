from pathlib import Path
import argparse,json,hashlib,time,asyncio,importlib.metadata,subprocess as sp
from unittest.mock import patch
from slither.slither import Slither
from slither.detectors.operations.block_timestamp import Timestamp
from src.orchestration.verdict.emit import emit_static_evidence
from src.orchestration.verdict.evidence import Evidence
from src.orchestration.nodes.synthesizer import synthesizer
from src.orchestration.routing import compute_overall_verdict
from src.orchestration.verdict.fuse import fuse
p=argparse.ArgumentParser()
p.add_argument("--cases",type=Path,required=True);p.add_argument("--compiler",type=Path,required=True);p.add_argument("--case",required=True)
args=p.parse_args();c=next(x for x in json.loads(args.cases.read_text())["cases"] if x["id"]==args.case)
assert hashlib.sha256(c["source"].encode()).hexdigest()==c["source_sha256"]
src=Path("pilot-"+c["id"]+".sol");src.write_text(c["source"])
start=time.monotonic();findings=[];error=None
try:
 sl=Slither(str(src),solc=str(args.compiler));sl.register_detector(Timestamp)
 for group in sl.run_detectors():
  for f in group:
   lines=sorted({ln for e in f.get("elements",[]) for ln in e.get("source_mapping",{}).get("lines",[])})
   findings.append({"tool":"slither","detector":f["check"],"impact":f["impact"],"confidence":f["confidence"],"description":f["description"],"lines":lines})
 status="COMPLETED"
except Exception as e:
 status="FAILED";error=f"{type(e).__name__}: {e}"
elapsed=round(time.monotonic()-start,4)
direct=emit_static_evidence(findings)
async def project(items):
 state={"contract_code":c["source"],"analysis_path":"deep","no_llm":True,"tool_status":{"slither":{"ran":status=="COMPLETED","n_findings":len(findings),"scope":"timestamp detector only"},"aderyn":{"ran":False,"reason":"not_requested"}},"static_findings":findings,"evidence_list":items,"error":error}
 with patch("src.orchestration.nodes.synthesizer.persist_report",return_value={}):
  report=(await synthesizer(state))["final_report"]
 return {k:report[k] for k in ("overall_verdict","assessment","verdicts","recommendation","vulnerability_verdicts")}
base=asyncio.run(project(direct))
context=Evidence.rag("Reentrancy",1.0,1.0,chunk_id="synthetic-unrelated-context")
aug=asyncio.run(project(direct+[context]))
legacy=compute_overall_verdict({k:v.verdict_full for k,v in fuse(direct+[context]).items()})
expected_hit=any(f["detector"]==c["expected_detector"] and c["expected_line"] in f["lines"] for f in findings) if c["expected_detector"] else not findings
checks={"execution_status_matches_expected":status==c["expected_status"],"expected_detector_and_line_observed":expected_hit,"no_unsupported_safety_assurance":base["overall_verdict"]!="SAFE" and base["assessment"]["assurance"]=="NOT_ESTABLISHED" and (bool(findings) or base["overall_verdict"]=="INCONCLUSIVE"),"direct_observation_retained":not findings or ("Timestamp" in base["assessment"]["direct_signal_classes"] and base["verdicts"].get("Timestamp") not in ("SAFE","INCONCLUSIVE",None)),"context_only_not_promoted":aug["verdicts"].get("Reentrancy")=="INCONCLUSIVE" and all(aug["verdicts"].get(k)==v for k,v in base["verdicts"].items())}
print(json.dumps({"case":c["id"],"source_sha256":c["source_sha256"],"status":status,"findings":findings,"error":error,"analyzer_seconds":elapsed,"static_projection":base,"synthetic_context_projection":aug,"historical_raw_fusion_with_context":legacy,"rubric":checks,"slither_version":importlib.metadata.version("slither-analyzer"),"compiler_version":sp.check_output([str(args.compiler),"--version"],text=True),"compiler_sha256":hashlib.sha256(args.compiler.read_bytes()).hexdigest()}))
