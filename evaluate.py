"""Evaluate the certify-or-reject layer on GENCO's published N-1 predictions (Texas 2000-bus grid).

For every scenario in the downloaded partition:
  1. rebuild the network from the published bus, branch, generator and admittance tables
  2. sanity check: the published ground-truth solution must satisfy the AC power-flow equations here
  3. score GENCO's raw prediction (error vs ground truth, power-balance mismatch, limit violations)
  4. run the certify-or-reject layer warm-started from GENCO's prediction
  5. run the same Newton solve from a flat start, for the time and iteration comparison
  6. check every CERTIFIED verdict against the ground truth (false-certify count must be zero)
Usage: python evaluate.py [partition] [split] [max_scenarios]
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from certify import Network, Settings, certify, newton, limits

PART = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "test_1"
MAXS = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
D = os.path.join("data", SPLIT, f"part{PART}")
BASE = 100.0

bus = pd.read_parquet(os.path.join(D, "bus_data.parquet"))
br = pd.read_parquet(os.path.join(D, "branch_data.parquet"))
gen = pd.read_parquet(os.path.join(D, "gen_data.parquet"))
yb = pd.read_parquet(os.path.join(D, "y_bus_data.parquet"))
pred = pd.read_parquet(os.path.join(D, "genco_predictions.parquet"))

scen = sorted(set(bus.scenario.unique()) & set(pred.scenario.unique()))[:MAXS]
bus_g, br_g, gen_g, yb_g, pred_g = (x.groupby("scenario") for x in (bus, br, gen, yb, pred))
print(f"{SPLIT} partition {PART}: {len(scen)} scenarios", flush=True)


def build(s):
    b = bus_g.get_group(s).sort_values("bus")
    n = len(b)
    y = yb_g.get_group(s)
    Y = sp.csr_matrix((y.G.values + 1j * y.B.values, (y.index1.values, y.index2.values)), shape=(n, n))
    pq, pv, ref = (b[c].values > 0.5 for c in ("PQ", "PV", "REF"))
    g = gen_g.get_group(s)
    g = g[g.in_service > 0.5]
    qmin = np.full(n, -np.inf); qmax = np.full(n, np.inf)
    agg = g.groupby("bus")[["min_q_mvar", "max_q_mvar"]].sum()
    qmin[agg.index.values.astype(int)] = agg.min_q_mvar.values / BASE
    qmax[agg.index.values.astype(int)] = agg.max_q_mvar.values / BASE
    r = br_g.get_group(s)
    r = r[r.br_status > 0.5]
    c = lambda a, bb: r[a].values + 1j * r[bb].values
    net = Network(
        Y=Y, pq=pq, pv=pv, ref=ref,
        P_spec=(b.Pg.values - b.Pd.values) / BASE, Q_spec=(b.Qg.values - b.Qd.values) / BASE,
        Vm_set=b.Vm.values, Va_ref=float(np.radians(b.Va.values[ref][0])),
        vmin=b.min_vm_pu.values, vmax=b.max_vm_pu.values, Qd=b.Qd.values / BASE,
        qg_min=qmin, qg_max=qmax,
        br_f=r.from_bus.values.astype(int), br_t=r.to_bus.values.astype(int),
        Yff=c("Yff_r", "Yff_i"), Yft=c("Yft_r", "Yft_i"), Ytf=c("Ytf_r", "Ytf_i"), Ytt=c("Ytt_r", "Ytt_i"),
        rate=r.rate_a.values / BASE)
    return net, b


st = Settings()
rows = []
t0 = time.time()
for k, s in enumerate(scen):
    net, b = build(s)
    p = pred_g.get_group(s).sort_values("bus")
    Vm_t, Va_t = b.Vm.values, np.radians(b.Va.values)
    Vm_p, Va_p = p.vm_pu.values, p.va.values  # GENCO predictions store angles in radians

    # 2. ground truth satisfies the equations as rebuilt here
    Vt = Vm_t * np.exp(1j * Va_t)
    St = Vt * np.conj(net.Y @ Vt)
    gt_mis = max(np.max(np.abs((St.real - net.P_spec)[net.pv | net.pq])), np.max(np.abs((St.imag - net.Q_spec)[net.pq])))
    gt_lim = limits(net, Vm_t, Va_t, St, st)

    # 3. raw GENCO point
    Vp = Vm_p * np.exp(1j * Va_p)
    Sp = Vp * np.conj(net.Y @ Vp)
    raw_mis = max(np.max(np.abs((Sp.real - net.P_spec)[net.pv | net.pq])), np.max(np.abs((Sp.imag - net.Q_spec)[net.pq])))
    raw_lim = limits(net, Vm_p, Va_p, Sp, st)

    # 4. certify from GENCO
    tc = time.perf_counter()
    res, sol = certify(net, Vm_p, Va_p, st)
    t_cert = time.perf_counter() - tc

    # 5. flat start
    tf = time.perf_counter()
    conv_f, it_f, Vm_f, Va_f, S_f, _ = newton(net, np.ones_like(Vm_t), np.full_like(Va_t, net.Va_ref), st)
    t_flat = time.perf_counter() - tf

    # 6. ground-truth check of the verdict
    sol_vs_gt = None
    if sol is not None:
        sol_vs_gt = float(np.max(np.abs(sol[0] - Vm_t)))
    false_cert = bool(res["verdict"] == "CERTIFIED" and (not gt_lim["ok"] or (sol_vs_gt is not None and sol_vs_gt > 1e-4)))

    rows.append({
        "scenario": int(s), "verdict": res["verdict"], "reason": res["reason"],
        "nr_iter_from_genco": res["nr_iterations"], "nr_iter_flat": it_f, "flat_converged": conv_f,
        "t_certify_s": t_cert, "t_flat_s": t_flat,
        "gt_mismatch_pu": float(gt_mis), "gt_limits_ok": gt_lim["ok"], "gt_max_loading": gt_lim["max_loading"],
        "raw_mismatch_pu": float(raw_mis), "raw_limits_ok": raw_lim["ok"], "raw_max_loading": raw_lim["max_loading"],
        "raw_mae_vm": float(np.mean(np.abs(Vm_p - Vm_t))), "raw_max_err_vm": float(np.max(np.abs(Vm_p - Vm_t))),
        "raw_mae_va_deg": float(np.degrees(np.mean(np.abs(np.angle(np.exp(1j * (Va_p - Va_t))))))),
        "dist_vm": res.get("dist_vm"), "dist_va_deg": res.get("dist_va_deg"),
        "solution_vs_gt_vm": sol_vs_gt, "false_certify": false_cert,
    })
    if k % 10 == 0:
        print(k, s, res["verdict"], res["reason"], f"gt_mis={gt_mis:.1e} it={res['nr_iterations']}/{it_f} "
              f"{time.time() - t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_csv(f"results/per_scenario_{SPLIT}_part{PART}.csv", index=False)
valid = df[df.gt_mismatch_pu < 1e-4]
summ = {
    "source": "huggingface.co/gridfm/genco-pf-contingency, mlflow/eval/" + SPLIT + "/predictions.parquet (GENCO paper checkpoint)",
    "grid": "Texas2k_case1_2016summerpeak (ACTIVSg2000)", "split": SPLIT, "partition": PART,
    "n_scenarios": int(len(df)), "n_ground_truth_consistent": int(len(valid)),
    "certified": int((valid.verdict == "CERTIFIED").sum()),
    "rejected": int((valid.verdict == "REJECTED").sum()),
    "reject_reasons": valid[valid.verdict == "REJECTED"].reason.value_counts().to_dict(),
    "false_certifies": int(valid.false_certify.sum()),
    "ground_truth_limit_violating": int((~valid.gt_limits_ok).sum()),
    "raw_genco_limit_violating": int((~valid.raw_limits_ok).sum()),
    "raw_genco_mismatch_pu_median": float(valid.raw_mismatch_pu.median()),
    "raw_genco_mismatch_pu_max": float(valid.raw_mismatch_pu.max()),
    "raw_genco_mae_vm_pu": float(valid.raw_mae_vm.mean()),
    "raw_genco_max_err_vm_pu": float(valid.raw_max_err_vm.max()),
    "raw_genco_mae_va_deg": float(valid.raw_mae_va_deg.mean()),
    "nr_iterations_from_genco_median": float(valid.nr_iter_from_genco.median()),
    "nr_iterations_flat_median": float(valid.nr_iter_flat.median()),
    "flat_start_failures": int((~valid.flat_converged).sum()),
    "t_certify_s_median": float(valid.t_certify_s.median()),
    "t_flat_s_median": float(valid.t_flat_s.median()),
    "settings": st.__dict__,
}
json.dump(summ, open(f"results/summary_{SPLIT}_part{PART}.json", "w"), indent=2)
print(json.dumps(summ, indent=2))
