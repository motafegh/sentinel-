"""Quick slither test - print full error."""
import subprocess
import sys
import tempfile
from pathlib import Path

src_path = "/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/CallToUnknown/00039d86633c712f65f5a48ec47b00d000a3a11c8cdccb3ac2c7f6f45b1d7da4.sol"

with tempfile.TemporaryDirectory() as tmpdir:
    driver = Path(tmpdir) / "_test.py"
    driver.write_text(f'''
import subprocess
import sys
import traceback

src_path = {src_path!r}

try:
    r = subprocess.run(["solc-select", "use", "0.5.17"], check=False, capture_output=True, text=True, timeout=10)
    print("solc-select returncode:", r.returncode)
    print("solc-select stdout:", r.stdout[:200])
    print("solc-select stderr:", r.stderr[:200])
except Exception as e:
    print("solc-select EXC:", e)
    traceback.print_exc()

try:
    from slither import Slither
    print("Slither imported")
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
    print("Dets:", len(_dets))
    slither = Slither(src_path)
    for _c in _dets:
        slither.register_detector(_c)
    dr = slither.run_detectors()
    print("Det results:", len(dr))
except Exception as e:
    traceback.print_exc()
''')
    proc = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True, timeout=60)
    print(f"returncode: {proc.returncode}")
    print(f"STDOUT:\n{proc.stdout}\n")
    print(f"STDERR:\n{proc.stderr[:3000]}")
