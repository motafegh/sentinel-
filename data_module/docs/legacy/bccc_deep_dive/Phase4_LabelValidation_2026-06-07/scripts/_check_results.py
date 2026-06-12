import csv
with open("/home/motafeq/projects/sentinel/Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase4_LabelValidation_2026-06-07/outputs/ws_p4_s1_aderyn_results.csv") as f:
    for row in csv.DictReader(f):
        print(f"id={row['id'][:16]}.. status={row['status']} n_hits={row['n_hits']} err={row['err'][:120]}")
