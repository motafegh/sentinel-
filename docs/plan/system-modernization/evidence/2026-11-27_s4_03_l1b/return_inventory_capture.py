import argparse, json, hashlib, time, importlib.metadata
from pathlib import Path
from collections import Counter
from slither.slither import Slither
from slither.core.cfg.node import NodeType
from slither.core.expressions.call_expression import CallExpression
from slither.core.declarations.function import Function
from slither.slithir.operations import Call, HighLevelCall, InternalCall, LibraryCall
from slither.detectors.operations.unused_return_values import UnusedReturnValues

def lines(obj):
    return sorted(set(getattr(getattr(obj, "source_mapping", None), "lines", []) or []))

def inspect_source(path, compiler):
    start=time.monotonic()
    sl=Slither(str(path),solc=str(compiler))
    sl.register_detector(UnusedReturnValues)
    # Inventory all parsed function declarations, not just derived contracts.
    functions={}
    for c in sl.contracts:
        for f in c.functions_declared+c.modifiers_declared:
            functions[(f.canonical_name, tuple(lines(f)))]=f
    for f in (f for cu in sl.compilation_units for f in cu.functions_top_level):
        functions[(f.canonical_name, tuple(lines(f)))]=f
    rows=[]; candidates=[]; ambiguous=[]
    for _,f in sorted(functions.items()):
        ctr=Counter()
        for n in f.nodes:
            calls=[i for i in n.irs if isinstance(i,Call)]
            ctr.update(type(i).__name__ for i in calls)
            if n.type!=NodeType.EXPRESSION or not isinstance(n.expression,CallExpression):
                continue
            resolved=[i for i in calls if isinstance(i,(HighLevelCall,InternalCall,LibraryCall)) and isinstance(i.function,Function)]
            if len(calls)!=1 or len(resolved)!=1:
                if calls:ambiguous.append({"function":f.canonical_name,"lines":lines(n),"call_kinds":[type(i).__name__ for i in calls]})
                continue
            i=resolved[0]
            if i.function.returns:
                candidates.append({"function":f.canonical_name,"callee":i.function.canonical_name,"call_kind":type(i).__name__,"return_arity":len(i.function.returns),"lines":lines(n),"declaration_lines":lines(i.function),"observation":"DIRECT_DISCARDED_NONVOID_EXPRESSION"})
        rows.append({"function":f.canonical_name,"lines":lines(f),"implemented":f.is_implemented,"visibility":f.visibility,"nodes":len(f.nodes),"call_kinds":dict(ctr),"loop_markers":sum(n.type==NodeType.STARTLOOP for n in f.nodes)})
    findings=[]
    for result in sl.run_detectors():
        for f in result:
            findings.append({"detector":f["check"],"elements":[{"type":e.get("type"),"name":e.get("name"),"lines":e.get("source_mapping",{}).get("lines",[])} for e in f.get("elements",[])]})
    return {"status":"COMPLETED","components":[{"name":c.name,"lines":lines(c),"kind":str(c.contract_kind)} for c in sl.contracts],"functions":rows,"direct_candidates":candidates,"ambiguous_standalone_calls":ambiguous,"unused_return_detector_findings":findings,"seconds":round(time.monotonic()-start,4)}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--compiler",type=Path,required=True);a=p.parse_args()
    try:r=inspect_source(a.source,a.compiler)
    except Exception as e:r={"status":"FAILED","error_type":type(e).__name__,"error_sha256":hashlib.sha256(str(e).encode()).hexdigest()}
    r["source_sha256"]=hashlib.sha256(a.source.read_bytes()).hexdigest()
    r["compiler_sha256"]=hashlib.sha256(a.compiler.read_bytes()).hexdigest()
    r["slither_version"]=importlib.metadata.version("slither-analyzer")
    print(json.dumps(r))
