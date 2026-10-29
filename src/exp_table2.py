"""Stage 2 -- Table 2 (MSE_1..5, E^(5), #params for A / Blank / Copy-last) and Figure 2."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import mmdata as md, models as M, evalkit as ek
from common import *
from train import load_model

allv = md.load_all("data", "cpu")
test_f = md.to_float(allv[md.TEST])
res = {}
for name, f, npar in (("Blank", ek.blank, 0), ("Copy-last", ek.copy_last, 0)):
    X, Y, V, P = ek.predict_all(f, test_f)
    res[name] = dict(mse_h=ek.mse_per_horizon(P, Y), E5=ek.e_metric(P, Y, V)[0], params=npar)
models = {}
for s in (0, 1):
    m = load_model("A", s); models[s] = m
    X, Y, V, P = ek.predict_all(m, test_f)
    res[f"A_s{s}"] = dict(mse_h=ek.mse_per_horizon(P, Y), E5=ek.e_metric(P, Y, V)[0], params=M.n_params(m))
res["A_mean"] = dict(mse_h=[float(np.mean([res[f"A_s{s}"]["mse_h"][h] for s in (0, 1)])) for h in range(5)],
                     E5=float(np.mean([res[f"A_s{s}"]["E5"] for s in (0, 1)])), params=res["A_s0"]["params"])
save_json(res, "results/table2.json")
rows = []
for lab, key in (("Model A (mean of seeds 0, 1)", "A_mean"), ("\\quad seed 0", "A_s0"), ("\\quad seed 1", "A_s1"),
                 ("Blank", "Blank"), ("Copy-last", "Copy-last")):
    r = res[key]
    rows.append([lab] + [fmt(v) for v in r["mse_h"]] + [fmt(r["E5"]), f"{r['params']:,}".replace(",", "{,}")])
latex_table(rows, ["Predictor", "MSE$_1$", "MSE$_2$", "MSE$_3$", "MSE$_4$", "MSE$_5$", "$E^{(5)}$", "Params"], "lrrrrrrr",
            "Test-set forecast error at $1\\times$, $t=1$ (1000 test videos): per-horizon MSE, $E^{(5)}$ and trainable parameters.",
            "tab:t2", REPORT_TABLES / "table2.tex")
print(json.dumps({k: (v["E5"], v["mse_h"]) for k, v in res.items()}, indent=1))

# ---------------------------------------------------------------- Figure 2 (seed-0 model A, three test videos)
ov = [overlapping(test_f[i]) for i in range(len(test_f))]          # frames 0..19 (0-based)
future_ov = [o[5:10].any() for o in ov]                            # any overlap among frames 6..10
clean = [i for i in range(len(test_f)) if not future_ov[i]][:2]
over = [i for i in range(len(test_f)) if future_ov[i]][:1]
idxs = clean + over
fig, axes = plt.subplots(3 * len(idxs), 10, figsize=(7.6, 0.76 * 3 * len(idxs) + 0.5))
m = models[0]
with torch.no_grad():
    for b, vi in enumerate(idxs):
        X, Y, V = md.gather_windows(test_f[vi:vi + 1], torch.tensor([1]), torch.tensor([1]))
        P = m(X.to(ek.DEV)).cpu()
        full = torch.cat([X, Y], 1)[0]; pred = torch.cat([X, P], 1)[0]
        err = (P - Y).abs()[0]
        for c in range(10):
            show_frame(axes[3 * b, c], full[c], f"{c+1}" if b == 0 else None)
            show_frame(axes[3 * b + 1, c], pred[c])
            if c >= 5: show_frame(axes[3 * b + 2, c], err[c - 5], cmap="magma")
            else: axes[3 * b + 2, c].axis("off")
        for r, lab in enumerate(["truth", "model A", "|error|"]):
            axes[3 * b + r, 0].set_ylabel(lab, fontsize=7)
        axes[3 * b, 0].annotate(f"test video {9000+vi}" + (" (digits overlap)" if vi in over else ""),
                                (0, 1.35), xycoords="axes fraction", fontsize=8, fontweight="bold")
        for c in range(5, 10): axes[3 * b + 1, c].spines[:].set_color("#c0392b")
fig.subplots_adjust(left=0.05, right=0.995, top=0.965, bottom=0.01, hspace=0.35, wspace=0.05)
fig.savefig("figures/fig2_predictions.png", dpi=200, bbox_inches="tight"); fig.savefig("figures/fig2_predictions.pdf", bbox_inches="tight")
save_json(dict(videos=[9000 + i for i in idxs], overlap_video=9000 + over[0]), "results/fig2_videos.json")
print("fig2 videos", [9000 + i for i in idxs])
