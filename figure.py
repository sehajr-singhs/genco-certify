"""Grayscale figures over every evaluated partition of the N-1 split."""
import glob, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.concat([pd.read_csv(f).assign(part=f.split("part")[-1].split(".")[0])
               for f in sorted(glob.glob("results/per_scenario_test_1_part*.csv"))], ignore_index=True)
plt.rcParams.update({"font.size": 11, "font.family": "serif", "axes.edgecolor": "#333", "axes.linewidth": 0.8})
DARK, MID, LIGHT = "#333333", "#7a7a7a", "#cfcfcf"


def clean(ax):
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)


n = len(d)
cert = int((d.verdict == "CERTIFIED").sum())
lim = int((d.reason == "limit_violated_at_solution").sum())
far = int((d.reason == "prediction_far_from_solution").sum())
other = n - cert - lim - far
fc = int(d.false_certify.sum())

# 1. verdicts
fig, ax = plt.subplots(figsize=(7.2, 3.9))
lab = ["Certified", "Rejected:\ntrue state violates\na limit", "Rejected:\nprediction too far\nfrom the solution"]
val = [cert, lim, far]
if other:
    lab.append("Rejected:\nother"); val.append(other)
r = ax.bar(lab, val, color=[DARK, MID, LIGHT, "white"][:len(val)], edgecolor="#222", linewidth=0.8, width=0.6)
for b, v in zip(r, val):
    ax.text(b.get_x() + b.get_width() / 2, v + n * 0.01, f"{v}  ({100 * v / n:.0f}%)", ha="center", fontsize=10)
ax.set_ylabel("N-1 scenarios"); ax.set_ylim(0, max(val) * 1.18)
ax.set_title(f"{n} N-1 scenarios, Texas 2000-bus. False certifies: {fc}", fontsize=11)
clean(ax); fig.tight_layout(); fig.savefig("figures/v2_verdicts.png", dpi=220); plt.close(fig)

# 2. distance tolerance sweep (certified share vs tolerance, false certifies audited)
conv = d.solution_vs_gt_vm.notna() & (d.solution_vs_gt_vm < 1e-6)
ks = np.array([0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8])
cs, fs = [], []
for k in ks:
    ok = conv & (d.dist_vm <= 0.01 * k) & (d.dist_va_deg <= 1.0 * k) & d.gt_limits_ok.astype(bool)
    cs.append(100 * ok.sum() / n)
    fs.append(int((ok & ~d.gt_limits_ok.astype(bool)).sum()))
feas = 100 * (conv & d.gt_limits_ok.astype(bool)).sum() / n
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot(ks, cs, "-o", color=DARK, ms=4.5, label="certified")
ax.axhline(feas, color=MID, ls="--", lw=1, label=f"ceiling: feasible and solved ({feas:.0f}%)")
ax.axvline(1, color=LIGHT, lw=1)
ax.text(1.05, 3, "setting used\n(0.01 pu, 1°)", fontsize=9, color="#555")
ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels([f"{k:g}×" for k in ks])
ax.set_xlabel("distance tolerance, multiple of (0.01 pu, 1°)"); ax.set_ylabel("share of scenarios (%)")
ax.set_ylim(0, 100); ax.legend(frameon=False, fontsize=9.5, loc="lower right")
ax.set_title("Certified share as the distance tolerance is loosened", fontsize=11)
clean(ax); fig.tight_layout(); fig.savefig("figures/v2_tolerance.png", dpi=220); plt.close(fig)

# 3. raw vs solved limit checks
raw_fail = int((~d.raw_limits_ok.astype(bool)).sum())
gt_fail = int((~d.gt_limits_ok.astype(bool)).sum())
fig, ax = plt.subplots(figsize=(6.4, 3.6))
r = ax.bar(["raw GENCO prediction\nchecked as-is", "true N-1 state"], [raw_fail, gt_fail],
           color=[LIGHT, DARK], edgecolor="#222", linewidth=0.8, width=0.5)
for b, v in zip(r, [raw_fail, gt_fail]):
    ax.text(b.get_x() + b.get_width() / 2, v + n * 0.01, f"{v} of {n}", ha="center", fontsize=10)
ax.set_ylabel("scenarios failing a limit check"); ax.set_ylim(0, n * 1.12)
ax.set_title("Why the check solves first, then checks limits", fontsize=11)
clean(ax); fig.tight_layout(); fig.savefig("figures/v2_raw_vs_true.png", dpi=220); plt.close(fig)

# 4. cost
fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.5))
bins = np.arange(0.5, max(d.nr_iter_flat.max(), d.nr_iter_from_genco.max()) + 1.5)
a.hist([d.nr_iter_from_genco, d.nr_iter_flat], bins=bins, color=[DARK, LIGHT], edgecolor="#222", linewidth=0.6,
       label=["from GENCO prediction", "from flat start"])
a.set_xlabel("Newton iterations"); a.set_ylabel("scenarios"); a.legend(frameon=False, fontsize=9)
a.set_title("Iterations to converge", fontsize=11)
tm = [1000 * d.t_certify_s.median(), 1000 * d.t_flat_s.median()]
r = b.bar(["from GENCO\nprediction", "from flat\nstart"], tm, color=[DARK, LIGHT], edgecolor="#222", linewidth=0.8, width=0.5)
for q, v in zip(r, tm):
    b.text(q.get_x() + q.get_width() / 2, v * 1.03, f"{v:.0f} ms", ha="center", fontsize=10)
b.set_ylabel("median wall time per scenario"); b.set_ylim(0, max(tm) * 1.25)
b.set_title("One CPU thread, scipy sparse solve", fontsize=11)
for ax in (a, b):
    clean(ax)
fig.tight_layout(); fig.savefig("figures/v2_cost.png", dpi=220); plt.close(fig)

s = dict(n=n, partitions=sorted(d.part.unique().tolist()), certified=cert, rejected_limit=lim, rejected_far=far,
         rejected_other=other, false_certifies=fc, gt_limit_violating=gt_fail, raw_limit_failing=raw_fail,
         feasible_ceiling_pct=round(feas, 1), it_genco_median=float(d.nr_iter_from_genco.median()),
         it_flat_median=float(d.nr_iter_flat.median()), t_genco_ms=round(tm[0], 1), t_flat_ms=round(tm[1], 1),
         sweep={f"{k:g}": round(c, 1) for k, c in zip(ks, cs)}, sweep_false=fs,
         flat_failures=int((~d.flat_converged.astype(bool)).sum()),
         mae_vm=float(d.raw_mae_vm.mean()), mae_va=float(d.raw_mae_va_deg.mean()),
         mismatch_median=float(d.raw_mismatch_pu.median()))
json.dump(s, open("results/summary_all.json", "w"), indent=1)
print(json.dumps(s, indent=1))
