from pathlib import Path
import argparse,json,hashlib,time,inspect,importlib.metadata
from slither.slither import Slither
from slither.detectors import all_detectors
from slither.detectors.abstract_detector import AbstractDetector
p=argparse.ArgumentParser();p.add_argument("--compiler",required=True);p.add_argument("--source",required=True);a=p.parse_args()
names={"unused-return","unchecked-lowlevel","unchecked-send","calls-loop","costly-loop","return-bomb"}
available={}
for name in dir(all_detectors):
 c=getattr(all_detectors,name)
 if inspect.isclass(c) and issubclass(c,AbstractDetector) and getattr(c,"ARGUMENT",None) in names:available[c.ARGUMENT]=c
start=time.monotonic()
out={"requested_detectors":sorted(names),"registered_detectors":[],"available_detectors":sorted(available),"missing_detectors":sorted(names-set(available)),"findings":[],"slither_version":importlib.metadata.version("slither-analyzer")}
try:
 sl=Slither(a.source,solc=a.compiler)
 for name,c in sorted(available.items()):sl.register_detector(c);out["registered_detectors"].append(name)
 for group in sl.run_detectors():
  for f in group:
   lines=sorted({ln for e in f.get("elements",[]) for ln in e.get("source_mapping",{}).get("lines",[]) if type(ln)==int})
   out["findings"].append({"detector":f["check"],"impact":f["impact"],"confidence":f["confidence"],"lines":lines,"description_sha256":hashlib.sha256(f.get("description","").encode()).hexdigest()})
 out["status"]="COMPLETED"
except Exception as exc:
 out["status"]="FAILED";out["error_type"]=type(exc).__name__;out["error_sha256"]=hashlib.sha256(str(exc).encode()).hexdigest()
out["seconds"]=round(time.monotonic()-start,4)
print(json.dumps(out))
