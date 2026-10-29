"""Stage 3 -- Testing the baseline: Tables 3 & 4, Figures 3, 4, 5 (+ GIFs) and a speed breakdown.  Model A only.

!! Run only AFTER predictions/Q2.5.md has been committed (the handout requires it). !!
"""
import sys
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, str(Path(__file__).parent))
import mmdata as md, models as M, evalkit as ek
from common import *
from figkit import *
from train import load_model

allv = md.load_all("data", "cpu"); test_f = md.to_float(allv[md.TEST])
Mi, Ro = md.mirror, md.rotate
models = {s: load_model("A", s) for s in (0, 1)}
preds = {"Blank": ek.blank, "Copy-last": ek.copy_last, "A_s0": models[0], "A_s1": models[1]}

# ------------------------------------------------------------------ Table 3 : E5, C_M, E_M, C_R, E_R
t3 = {}
store = {}
for name, f in preds.items():
    X, Y, V, P = ek.predict_all(f, test_f)
    e5 = ek.e_metric(P, Y, V)[0]
    cm, em, cm_v, em_v, _ = ek.consistency_and_error(f, test_f, Mi)
    cr, er, cr_v, er_v, _ = ek.consistency_and_error(f, test_f, Ro)
    t3[name] = dict(E5=e5, C_M=cm, E_M=em, C_R=cr, E_R=er)
    store[name] = dict(cm_v=cm_v, cr_v=cr_v)
t3["A_mean"] = {k: float(np.mean([t3["A_s0"][k], t3["A_s1"][k]])) for k in t3["A_s0"]}
save_json(t3, "results/table3.json")
rows = []
for lab, key in (("Model A (mean of seeds 0, 1)", "A_mean"), ("\\quad seed 0", "A_s0"), ("\\quad seed 1", "A_s1"), ("Blank", "Blank"), ("Copy-last", "Copy-last")):
    r = t3[key]; rows.append([lab] + [fmt(r[k]) for k in ("E5", "C_M", "E_M", "C_R", "E_R")])
latex_table(rows, ["Predictor", "$E^{(5)}$", "$C_M$", "$E_M$", "$C_R$", "$E_R$"], "lrrrrr",
            "Consistency gaps $C_T$ and transformed errors $E_T$ at $1\\times$, $t=1$ on the 1000 test videos.", "tab:t3", REPORT_TABLES / "table3.tex")
print(json.dumps(t3, indent=1))

# ------------------------------------------------------------------ Figure 3 / Figure 4 (seed-0 model A)
cm_v, cr_v = store["A_s0"]["cm_v"], store["A_s0"]["cr_v"]
top_m = torch.argsort(cm_v, descending=True)[:2].tolist(); top_r = torch.argsort(cr_v, descending=True)[:2].tolist()
save_json(dict(top_C_M=[9000 + i for i in top_m], top_C_R=[9000 + i for i in top_r],
               C_M_values=[float(cm_v[i]) for i in top_m], C_R_values=[float(cr_v[i]) for i in top_r]), "results/fig3_videos.json")
for T, tag, tops, cv in ((Mi, "M", top_m, cm_v), (Ro, "R", top_r, cr_v)):
    panels = fig3_panels(models[0], test_f, T, tops)
    titles = [f"$T={tag}$, rank {r+1}: test video {9000+i}, per-video $C_{tag}$ = {fmt(float(cv[i]))}" for r, i in enumerate(tops)]
    draw_fig3(panels, titles, f"figures/fig3_{tag}")
    for r, (i, rows_) in enumerate(zip(tops, panels)):
        make_gif(rows_, f"T={tag}, rank {r+1}, video {9000+i}", f"figures/gifs/fig3_{tag}_rank{r+1}_video{9000+i}.gif")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for a, v, tops, tag in ((ax[0], cm_v, top_m, "M"), (ax[1], cr_v, top_r, "R")):
    vv = v.numpy(); bins = np.logspace(np.log10(max(vv.min(), 1e-6)), np.log10(vv.max()), 40)
    a.hist(vv, bins=bins, color="#3b6a8f", edgecolor="white"); a.set_xscale("log")
    a.axvline(vv.mean(), color="k", ls=":", label=f"mean = {fmt(float(vv.mean()))}")
    for r, i in enumerate(tops): a.axvline(vv[i], color="crimson", lw=1.5, label="Figure 3 videos" if r == 0 else None)
    a.set_xlabel(f"per-video $C_{tag}$"); a.set_ylabel("number of test videos"); a.legend(frameon=False, fontsize=8)
    a.set_title(f"$C_{tag}$ of model A (seed 0)"); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/fig4_hist.png", dpi=200); fig.savefig("figures/fig4_hist.pdf"); plt.close(fig)

# ------------------------------------------------------------------ Table 4 : speeds
t4 = {}
for name, f in preds.items():
    r = {}
    for k in (1, 2, 3):
        X, Y, V, P = ek.predict_all(f, test_f, k=k)
        r[f"E2_{k}x"] = ek.e_metric(P, Y, V, horizons=2)[0]
        if k <= 2: r[f"E5_{k}x"] = ek.e_metric(P, Y, V, horizons=5)[0]
    t4[name] = r
t4["A_mean"] = {k: float(np.mean([t4["A_s0"][k], t4["A_s1"][k]])) for k in t4["A_s0"]}
save_json(t4, "results/table4.json")
rows = []
for lab, key in (("Model A (mean of seeds 0, 1)", "A_mean"), ("\\quad seed 0", "A_s0"), ("\\quad seed 1", "A_s1"), ("Copy-last", "Copy-last"), ("Blank", "Blank")):
    r = t4[key]; rows.append([lab] + [fmt(r[k]) for k in ("E2_1x", "E2_2x", "E2_3x", "E5_1x", "E5_2x")])
latex_table(rows, ["Predictor", "$E^{(2)}$ 1$\\times$", "$E^{(2)}$ 2$\\times$", "$E^{(2)}$ 3$\\times$", "$E^{(5)}$ 1$\\times$", "$E^{(5)}$ 2$\\times$"], "lrrrrr",
            "Speed test on the 1000 test videos ($t=1$). $E^{(2)}$ uses only the first two horizons (the only ones that exist at $3\\times$).", "tab:t4", REPORT_TABLES / "table4.tex")
print(json.dumps(t4, indent=1))

# ------------------------------------------------------------------ Figure 5
vi = pick_nonoverlap_video(test_f)
draw_fig5(models[0], test_f, vi, "figures/fig5_speed", "model A")
save_json(dict(video=9000 + vi), "results/fig5_video.json"); print("fig5 video", 9000 + vi)

# ------------------------------------------------------------------ extra: error vs apparent displacement k*v (supports Q2.9 / Q2.10)
def video_speed(v):                       # median Farneback speed of the digits in frames 1..10 of the video
    fr = (v.squeeze(1).numpy() * 255).round().astype(np.uint8); sp = []
    for s in range(9):
        fl = cv2.calcOpticalFlowFarneback(fr[s], fr[s + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.hypot(fl[..., 0], fl[..., 1]); keep = (fr[s] > 51) & (fr[s + 1] > 51) & (mag > 0.5)
        if keep.any(): sp.append(mag[keep])
    sp = np.concatenate(sp); return float(np.quantile(sp, 0.9))
vspeed = np.array([video_speed(test_f[i]) for i in range(len(test_f))])
bins = [(0, 2.5), (2.5, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 10)]
rows_s1 = {}
for k in (1, 2, 3):
    X, Y, V, P = ek.predict_all(models[0], test_f, k=k); _, _, _, Pc = ek.predict_all(ek.copy_last, test_f, k=k)
    e = ek.per_video_mse_h(P[:, :2], Y[:, :2]).mean(1).numpy(); ec = ek.per_video_mse_h(Pc[:, :2], Y[:, :2]).mean(1).numpy()
    rows_s1[k] = [(lo, hi, int(((vspeed >= lo) & (vspeed < hi)).sum()), float(e[(vspeed >= lo) & (vspeed < hi)].mean() / ec[(vspeed >= lo) & (vspeed < hi)].mean()))
                  for lo, hi in bins if ((vspeed >= lo) & (vspeed < hi)).any()]
save_json(dict(video_speed_q90=vspeed.tolist(), ratio_to_copylast=rows_s1), "results/speed_breakdown.json")
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for k, c in zip((1, 2, 3), ("#2c7fb8", "#f39c12", "#c0392b")):
    xs = [k * 0.5 * (lo + min(hi, 5.0)) for lo, hi, n, _ in rows_s1[k]]; ys = [r for *_, r in rows_s1[k]]
    ax.plot(xs, ys, "o-", color=c, label=f"{k}$\\times$")
ax.axvline(8, color="k", ls="--", lw=1); ax.text(8.1, ax.get_ylim()[1] * 0.9, "reach = 8 px", fontsize=8)
ax.axhline(1, color="gray", ls=":"); ax.set_xlabel("apparent displacement between seen frames (px) = speed $\\times k$")
ax.set_ylabel("$E^{(2)}$(A) / $E^{(2)}$(Copy-last)"); ax.legend(frameon=False, title="speed-up"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/figS1_speed_breakdown.png", dpi=200); fig.savefig("figures/figS1_speed_breakdown.pdf")
print(json.dumps(rows_s1))
