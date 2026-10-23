import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

d = pd.read_csv("results/per_scenario_test_1_part0.csv")
s = json.load(open("results/summary_test_1_part0.json"))
plt.rcParams.update({"font.size": 11})
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))

cert = (d.verdict == "CERTIFIED").sum()
far = ((d.reason == "prediction_far_from_solution")).sum()
lim = ((d.reason == "limit_violated_at_solution")).sum()
bars = a.bar(["Certified", "Rejected:\ntrue state violates\na limit", "Rejected:\nprediction too far\nfrom true solution"],
             [cert, lim, far], color=["#2f5d8a", "#b3482f", "#8a8a85"])
for r, v in zip(bars, [cert, lim, far]):
    a.text(r.get_x() + r.get_width() / 2, v + 2, str(v), ha="center")
a.set_ylabel("N-1 scenarios")
a.set_title(f"GENCO N-1 predictions, Texas 2000-bus ({len(d)} scenarios)\nfalse certifies: {s['false_certifies']}", fontsize=10)
for k in ("top", "right"):
    a.spines[k].set_visible(False)

it = [d.nr_iter_from_genco.median(), d.nr_iter_flat.median()]
tt = [d.t_certify_s.median() * 1000, d.t_flat_s.median() * 1000]
x = [0, 1]
b.bar([0, 1], it, color=["#2f5d8a", "#8a8a85"], width=0.5)
b.set_xticks(x)
b.set_xticklabels(["Newton from\nGENCO prediction", "Newton from\nflat start"])
for i in x:
    b.text(i, it[i] + 0.1, f"{it[i]:.0f} iterations\n{tt[i]:.0f} ms", ha="center")
b.set_ylim(0, max(it) * 1.6)
b.set_ylabel("median Newton iterations")
b.set_title("Cost of the check", fontsize=10)
for k in ("top", "right"):
    b.spines[k].set_visible(False)
fig.tight_layout()
fig.savefig("figures/certify_genco_n1.png", dpi=200)
print("ok")
