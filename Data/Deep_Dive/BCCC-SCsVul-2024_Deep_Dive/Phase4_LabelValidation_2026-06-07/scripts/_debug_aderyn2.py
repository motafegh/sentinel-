"""Debug aderyn batch behavior."""
import csv, shutil, subprocess, tempfile, json
from pathlib import Path

SRC = Path("/home/motafeq/projects/sentinel/BCCC-SCsVul-2024/SourceCodes")
ADERYN = Path.home() / ".cargo" / "bin" / "aderyn"

with open("Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s1_sample.csv") as f:
    rows = [r for r in csv.DictReader(f) if r["in_stage1_sample"] == "1"][:3]

with tempfile.TemporaryDirectory(prefix="aderyn_batch_") as td:
    tmpdir = Path(td)
    for r in rows:
        rel = r["bccc_file_path"].replace("BCCC-SCsVul-2024/Source Codes/", "").replace("BCCC-SCsVul-2024/SourceCodes/", "")
        src = SRC / rel
        dest = tmpdir / (r["id"][:16] + ".sol")
        if src.exists():
            shutil.copy2(src, dest)
            print(f"Copied: {rel} -> {dest.name} ({src.stat().st_size} bytes)")
        else:
            print(f"MISSING: {src}")
    print(f"Files in tmpdir: {list(tmpdir.iterdir())}")
    report = tmpdir / "report.json"
    proc = subprocess.run([str(ADERYN), str(tmpdir), "-o", str(report)], capture_output=True, text=True, timeout=60)
    print(f"rc: {proc.returncode}, report exists: {report.exists()}")
    if report.exists():
        data = json.loads(report.read_text())
        low = data.get("low_issues", {}).get("issues", [])
        high = data.get("high_issues", {}).get("issues", [])
        print(f"High issues: {len(high)}, Low issues: {len(low)}")
        for i in low:
            print(f"  detector: {i.get('detector_name')} ({len(i.get('instances', []))} instances)")
    else:
        print(f"STDOUT tail: {proc.stdout[-500:]}")
        print(f"STDERR tail: {proc.stderr[-500:]}")
