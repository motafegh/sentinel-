import json
with open("/home/motafeq/projects/sentinel/Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s1_aderyn_checkpoint.jsonl") as f:
    lines = f.readlines()
print(f"Total: {len(lines)} lines")
for i, line in enumerate(lines[:3]):
    r = json.loads(line)
    print(f"id={r['id'][:16]}.. status={r['status']} n_hits={r.get('n_hits',0)} err={r.get('err','')[:120]}")
if len(lines) > 3:
    r = json.loads(lines[-1])
    print(f"...last: id={r['id'][:16]}.. status={r['status']} n_hits={r.get('n_hits',0)}")
