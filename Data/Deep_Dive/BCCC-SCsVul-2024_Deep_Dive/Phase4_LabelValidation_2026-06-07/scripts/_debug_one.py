"""Quick test: run slither on one contract and check actual detector output."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

src = "/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/IntegerUO/002082d0d435c79cd31a5daaf4d3d2af6aac79240ab4863c8b8c14236017f504.sol"
solc = "0.4.26"
print(f"Source: {src}")
print(f"File size: {Path(src).stat().st_size} bytes")
print(f"First 300 chars: {Path(src).read_text()[:300]}")
print()

with tempfile.TemporaryDirectory() as tmpdir:
    driver = Path(tmpdir) / "_t.py"
    driver.write_text(f'''
import json, sys
import subprocess
r = subprocess.run(["{Path(sys.executable).parent / 'solc-select'}", "use", "{solc}"], capture_output=True, text=True)
print("solc-select rc:", r.returncode, "stderr:", r.stderr[:100])
from slither import Slither
from slither.detectors import all_detectors
import slither.detectors.all_detectors as _ad
from slither.detectors.abstract_detector import AbstractDetector
_dets = []
for _n in dir(_ad):
    _o = getattr(_ad, _n)
    if isinstance(_o, type):
        try:
            if issubclass(_o, AbstractDetector) and _o is not AbstractDetector:
                _dets.append(_o)
        except Exception:
            pass
print(f"Detectors: {{len(_dets)}}")
slither = Slither("{src}")
for _c in _dets:
    slither.register_detector(_c)
dr = slither.run_detectors()
print(f"Detector results: {{len(dr)}}")
hits = []
for finding in dr:
    check = finding.get("check") if isinstance(finding, dict) else None
    if check:
        hits.append(check)
print(f"Hits: {{len(hits)}}")
if hits:
    print(f"First 10: {{hits[:10]}}")
print("RAW DR (first item):")
if dr:
    print(repr(dr[0])[:500])
''')
    proc = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True, timeout=60)
    print("RC:", proc.returncode)
    print("STDOUT:")
    print(proc.stdout)
    print("STDERR:", proc.stderr[:500])
