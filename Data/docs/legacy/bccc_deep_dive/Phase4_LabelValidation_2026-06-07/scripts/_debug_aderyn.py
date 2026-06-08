"""Debug aderyn in script-style tempdir."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

src = Path("/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes/IntegerUO/002082d0d435c79cd31a5daaf4d3d2af6aac79240ab4863c8b8c14236017f504.sol")
aderyn_bin = Path.home() / ".cargo" / "bin" / "aderyn"
with tempfile.TemporaryDirectory(prefix="aderyn_test_") as td_str:
    td = Path(td_str)
    dest = td / "002082d0d435c79c.sol"
    shutil.copy2(src, dest)
    print(f"tmpdir: {td}")
    print(f"files: {list(td.iterdir())}")
    print(f"file size: {dest.stat().st_size}")
    print(f"file head: {dest.read_text()[:200]}")
    proc = subprocess.run(
        [str(aderyn_bin), str(td), "-o", str(td / "report.json")],
        capture_output=True, text=True, timeout=60,
    )
    print(f"rc: {proc.returncode}")
    print(f"files after: {list(td.iterdir())}")
    print(f"STDOUT (last 2000): {proc.stdout[-2000:]}")
    print(f"STDERR (last 2000): {proc.stderr[-2000:]}")
