"""Download one scenario partition of GENCO's published Texas-2k N-1 test set, plus GENCO's predictions for
the same scenarios. Source: huggingface.co/gridfm/genco-pf-contingency (Apache-2.0)."""
import os, sys
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from huggingface_hub import HfFileSystem

PART = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "test_1"
OUT = os.path.join("data", SPLIT, f"part{PART}")
os.makedirs(OUT, exist_ok=True)
fs = HfFileSystem()
base = "gridfm/genco-pf-contingency"
raw = f"{base}/data/{SPLIT}/Texas2k_case1_2016summerpeak/raw"

for t in ["bus_data", "branch_data", "gen_data", "y_bus_data"]:
    dst = os.path.join(OUT, f"{t}.parquet")
    if os.path.exists(dst):
        continue
    parts = []
    for x in fs.ls(f"{raw}/{t}.parquet/scenario_partition={PART}", detail=True):
        with fs.open(x["name"], "rb") as f:
            parts.append(pq.read_table(f))
    tab = pa.concat_tables(parts)
    pq.write_table(tab, dst)
    print(t, tab.num_rows, flush=True)

scen = sorted(pd.read_parquet(os.path.join(OUT, "bus_data.parquet"), columns=["scenario"])["scenario"].unique())
print("scenarios in partition:", len(scen), flush=True)

dst = os.path.join(OUT, "genco_predictions.parquet")
if not os.path.exists(dst):
    cols = ["scenario", "bus", "vm_pu", "va", "pg_mw", "qg_mvar", "vm_pu_target", "va_target", "PBE",
            "active res. (MW)", "reactive res. (MVar)"]
    with fs.open(f"{base}/mlflow/eval/{SPLIT}/predictions.parquet", "rb") as f:
        pf = pq.ParquetFile(f)
        keep = []
        sset = set(int(s) for s in scen)
        for g in range(pf.metadata.num_row_groups):
            ids = pf.read_row_group(g, columns=["scenario"]).column(0).to_numpy()
            if not sset.intersection(set(ids.tolist())):
                continue
            t = pf.read_row_group(g, columns=cols).to_pandas()
            keep.append(t[t["scenario"].isin(sset)])
            print("row group", g, "kept", len(keep[-1]), flush=True)
    pd.concat(keep).to_parquet(dst)
print("done")
