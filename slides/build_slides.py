import json, subprocess
s = json.load(open("../results/summary_all.json"))
n = s["n"]; P = len(s["partitions"])
pct = lambda v: f"{100 * v / n:.0f}%"
html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Certify-or-reject for GENCO</title>
<style>
@page{{size:1280px 720px;margin:0}}*{{box-sizing:border-box;margin:0;padding:0}}
html{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
body{{background:#fff;color:#111;font-family:Georgia,"Times New Roman",serif;font-size:19px;line-height:1.45}}
.s{{width:1280px;height:720px;background:#fff;padding:52px 76px 44px;position:relative;overflow:hidden;break-after:page}}
.k{{font-family:Arial,Helvetica,sans-serif;font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#666;border-bottom:1px solid #bbb;padding-bottom:8px;margin-bottom:22px;display:flex;justify-content:space-between}}
h1{{font-weight:normal;font-size:31px;line-height:1.2;margin-bottom:16px}}
h1.t{{font-size:42px;margin-top:150px;max-width:1050px}}
ul{{margin:2px 0 0 22px}}li{{margin-bottom:10px;max-width:1110px}}
table{{border-collapse:collapse;font-family:Arial,Helvetica,sans-serif;font-size:16px;margin:4px 0 10px}}
th{{text-align:left;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:#555;border-bottom:1.5px solid #111;padding:6px 26px 6px 0}}
td{{border-bottom:1px solid #ccc;padding:7px 26px 7px 0}}
.eq{{font-family:Consolas,monospace;font-size:15.5px;border:1px solid #bbb;padding:12px 16px;margin:4px 0 14px;white-space:pre;line-height:1.55}}
.f{{position:absolute;left:76px;right:76px;bottom:20px;display:flex;justify-content:space-between;font-family:Arial,sans-serif;font-size:11.5px;color:#777}}
.two{{display:flex;gap:34px;align-items:flex-start}}.two>div{{flex:1}}
.m{{color:#555}}
</style></head><body>

<section class="s"><h1 class="t">A certify-or-reject check for GENCO's N-1 power-flow predictions</h1>
<p>First results on the published Texas 2000-bus contingency set. Work in progress.</p>
<p class="m" style="margin-top:28px">Sehaj Randhir Singh · Queens High School for the Sciences · October 2026</p>
<div class="f"><span>github.com/sehajr-singhs/genco-certify</span><span>1</span></div></section>

<section class="s"><div class="k"><span>The question</span><span>2</span></div>
<h1>When GENCO predicts the grid after an outage, which predictions can someone use without re-running a full study?</h1>
<ul>
<li>GENCO's predictions are accurate on average. Over these {n} scenarios the mean voltage error is {s['mae_vm']:.1e} pu, the mean angle error {s['mae_va']:.2f}°, and the median power-balance mismatch {s['mismatch_median']:.3f} pu.</li>
<li>In power-flow mode nothing in the model enforces voltage, thermal or generator reactive limits, so an accurate prediction can still describe an operating point that isn't allowed.</li>
<li>What a user needs per scenario is a yes or no they can rely on: is this an AC-feasible operating point that satisfies every limit, and if not, why.</li>
</ul>
<div class="f"><span></span><span>2</span></div></section>

<section class="s"><div class="k"><span>Data</span><span>3</span></div>
<h1>Everything comes from the public GENCO release, so nothing is retrained.</h1>
<ul>
<li><b>Predictions:</b> the paper checkpoint's N-1 predictions, <span style="font-family:Consolas">huggingface.co/gridfm/genco-pf-contingency</span>, file <span style="font-family:Consolas">mlflow/eval/test_1/predictions.parquet</span>.</li>
<li><b>Networks and ground truth:</b> the published bus, branch, generator and Y-bus tables for the same split, used to rebuild each scenario's network.</li>
<li><b>Scenarios so far:</b> {P} partitions of <span style="font-family:Consolas">test_1</span>, {n} N-1 cases on the ACTIVSg2000 (Texas) grid, out of 8,823 in the split.</li>
<li><b>Check before trusting the setup:</b> the published ground-truth solution satisfies the rebuilt power-flow equations on every scenario, with a worst mismatch around 1e-9 pu.</li>
</ul>
<div class="f"><span></span><span>3</span></div></section>

<section class="s"><div class="k"><span>Method</span><span>4</span></div>
<h1>Solve from the prediction first, then check limits at the solution.</h1>
<div class="eq">1  Newton-Raphson, warm-started from GENCO's (Vm, θ), with PV and reference set points fixed.
   Converged means a mismatch below 1e-8 pu within 15 iterations.
2  Distance: the converged point must be within 0.01 pu and 1° of the prediction at every bus.
3  Limits at the converged point: bus voltage bounds, branch MVA ratings at both ends,
   and generator reactive limits aggregated per bus.
4  CERTIFIED only if 1 to 3 pass. Otherwise REJECTED with the reason, and the case goes to a full study.</div>
<ul><li>Model-agnostic, plain numpy and scipy, about 300 lines.</li>
<li>Every certified verdict is audited against the published ground truth. A false certify is a certified case whose true solution differs from the converged point or violates a limit.</li></ul>
<div class="f"><span>certify.py</span><span>4</span></div></section>

<section class="s"><div class="k"><span>Results, {n} N-1 scenarios</span><span>5</span></div>
<div class="two"><div style="flex:1.25"><img src="../figures/v2_verdicts.png" style="width:100%"></div>
<div><table>
<tr><th>Outcome</th><th>Scenarios</th></tr>
<tr><td>Certified</td><td>{s['certified']} ({pct(s['certified'])})</td></tr>
<tr><td>Rejected, true state violates a limit</td><td>{s['rejected_limit']}</td></tr>
<tr><td>Rejected, prediction too far from solution</td><td>{s['rejected_far']}</td></tr>
<tr><td><b>False certifies</b></td><td><b>{s['false_certifies']}</b></td></tr>
</table>
<p style="font-size:17px">Of the {n} true N-1 states, {s['gt_limit_violating']} violate at least one limit, and the check rejected all of them.</p></div></div>
<div class="f"><span>results/summary_all.json</span><span>5</span></div></section>

<section class="s"><div class="k"><span>The distance tolerance</span><span>6</span></div>
<div class="two"><div style="flex:1.3"><img src="../figures/v2_tolerance.png" style="width:100%"></div>
<div><ul style="font-size:17.5px">
<li>Most of the "too far" rejections come from the 0.01 pu, 1° rule, which is a design choice.</li>
<li>Doubling it raises the certified share from {s['sweep']['1']:.0f}% to {s['sweep']['2']:.0f}%, and it levels off at {s['feasible_ceiling_pct']:.0f}%, the share of scenarios that are feasible.</li>
<li>Here, whenever Newton converged it converged to the true solution, so loosening the rule didn't add wrong answers. That's a property of this data and I wouldn't assume it carries to harder splits.</li>
</ul></div></div>
<div class="f"><span></span><span>6</span></div></section>

<section class="s"><div class="k"><span>Why solve first</span><span>7</span></div>
<div class="two"><div style="flex:1.1"><img src="../figures/v2_raw_vs_true.png" style="width:100%"></div>
<div><ul style="font-size:17.5px">
<li>Checked as-is, {s['raw_limit_failing']} of {n} raw predictions fail a limit, against {s['gt_limit_violating']} of {n} true states.</li>
<li>Most of the extra failures are generator reactive limits. Reactive power recovered from an approximate state is very sensitive to small voltage errors, so checking the raw prediction rejects cases that are actually fine.</li>
<li>Solving from the prediction removes that noise, and the prediction still pays off as a warm start.</li>
</ul></div></div>
<div class="f"><span></span><span>7</span></div></section>

<section class="s"><div class="k"><span>Cost</span><span>8</span></div>
<img src="../figures/v2_cost.png" style="width:92%;display:block;margin:0 auto">
<ul style="margin-top:10px"><li>Warm-starting from GENCO takes a median of {s['it_genco_median']:.0f} Newton iterations against {s['it_flat_median']:.0f} from a flat start, {s['t_genco_ms']:.0f} ms against {s['t_flat_ms']:.0f} ms per scenario. No flat-start solve failed on these partitions.</li></ul>
<div class="f"><span></span><span>8</span></div></section>

<section class="s"><div class="k"><span>What this doesn't show yet</span><span>9</span></div>
<h1>Limits of this first version</h1>
<ul>
<li><b>Coverage:</b> {n} of 8,823 N-1 scenarios, and none of the N-2 to N-20 splits yet. The pipeline is the same for all of them.</li>
<li><b>Limits checked:</b> only those in the published tables. Generator reactive limits are aggregated per bus, and angle-difference limits aren't checked.</li>
<li><b>Tolerances:</b> the distance rule is my choice, and the audit against ground truth is what keeps any setting honest.</li>
<li><b>Not integrated:</b> it runs on GENCO's saved predictions, not inside gridfm-graphkit.</li>
</ul>
<div class="f"><span></span><span>9</span></div></section>

<section class="s"><div class="k"><span>What I'm working on next (in progress)</span><span>10</span></div>
<h1>Next steps, roughly in this order</h1>
<ul>
<li>Run all partitions of the N-1 split, then the N-2 and higher splits, where I expect predictions and Newton convergence to get harder and the audit to matter more.</li>
<li>A thin adapter so <span style="font-family:Consolas">gridfm_graphkit predict</span> outputs go straight into the check and come back with a verdict, a reason and a margin per scenario.</li>
<li>Per-generator reactive limits and angle-difference limits.</li>
<li>A feasibility margin per scenario, so a user can see how close a certified case is to a limit, not only whether it passed.</li>
</ul>
<div class="f"><span></span><span>10</span></div></section>

<section class="s"><div class="k"><span>Questions for the team</span><span>11</span></div>
<h1>Where I'd value your input</h1>
<ul>
<li>Which limits and which tolerance would be most useful to OpenGridFM users?</li>
<li>Is the Texas contingency set the right first benchmark, or is there a case you'd rather see, for example one closer to Hydro-Québec's network?</li>
<li>Would a post-processing step after <span style="font-family:Consolas">predict</span> be the right place for this in the codebase?</li>
</ul>
<p class="m" style="margin-top:30px">Code and per-scenario results: github.com/sehajr-singhs/genco-certify</p>
<div class="f"><span></span><span>11</span></div></section>
</body></html>"""
open("slides_v2.html", "w", encoding="utf-8").write(html)
edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
import os
here = os.path.abspath(".").replace("\\", "/")
subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                "--print-to-pdf=" + os.path.abspath("GENCO-certify-or-reject.pdf"), "file:///" + here + "/slides_v2.html"],
               stderr=subprocess.DEVNULL)
print("built", n, P)
