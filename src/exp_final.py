"""Stage 5 -- Table 5 (A-D, both seeds), Table 6 (symmetrised B, D, final), Figure 6, final-model choice (validation only)."""
import sys, statistics
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import mmdata as md, models as M, evalkit as ek
from common import *
from figkit import *
from train import load_model

allv = md.load_all("data", "cpu"); test_f = md.to_float(allv[md.TEST]); val_f = md.to_float(allv[md.VAL])
Mi, Ro = md.mirror, md.rotate
OPTIONAL_DB = Path("checkpoints/DB_s0.pt").exists()


def full_metrics(f, videos, with_time=True):
    r = {}
    X, Y, V, P = ek.predict_all(f, videos); r["E5"] = ek.e_metric(P, Y, V)[0]
    r["C_M"], r["E_M"], *_ = ek.consistency_and_error(f, videos, Mi)
    r["C_R"], r["E_R"], *_ = ek.consistency_and_error(f, videos, Ro)
    for k in (1, 2, 3):
        X, Y, V, P = ek.predict_all(f, videos, k=k); r[f"E2_{k}x"] = ek.e_metric(P, Y, V, horizons=2)[0]
    if with_time:
        ek.timed_inference(f, videos)                                   # warm-up
        r["inference_s"] = statistics.median(ek.timed_inference(f, videos) for _ in range(5))
    return r


def train_info(tag, s):
    j = json.load(open(f"results/train/{tag}_s{s}.json")); return j["train_sec_per_epoch"], j["n_params"]


names = ["A", "B", "C", "D"] + (["DB"] if OPTIONAL_DB else [])
t5 = {}
for n in names:
    for s in (0, 1):
        m = load_model(n, s); r = full_metrics(m, test_f); r["train_s_per_epoch"], r["params"] = train_info(n, s); t5[f"{n}_s{s}"] = r
        print(n, s, {k: round(v, 5) for k, v in r.items()}, flush=True)
for n in names:
    t5[f"{n}_mean"] = {k: float(np.mean([t5[f"{n}_s{s}"][k] for s in (0, 1)])) for k in t5[f"{n}_s0"]}
r = full_metrics(ek.copy_last, test_f); r["params"] = 0; t5["Copy-last"] = r
save_json(t5, "results/table5.json")

COLS = ["E5", "C_M", "E_M", "C_R", "E_R", "E2_1x", "E2_2x", "E2_3x", "params", "train_s_per_epoch", "inference_s"]
HDR = ["Model", "$E^{(5)}$", "$C_M$", "$E_M$", "$C_R$", "$E_R$", "$E^{(2)}_{1\\times}$", "$E^{(2)}_{2\\times}$", "$E^{(2)}_{3\\times}$", "Params", "s/epoch", "Infer. (s)"]
def row(lab, r): return [lab] + [(f"{int(r[k]):,}".replace(",", "{,}") if k == "params" else (fmt(r[k], 3) if k != "train_s_per_epoch" else f"{r[k]:.1f}") if k in r else "--") for k in COLS]
LAB = {"A": "A baseline", "B": "B +sym.\\ aug.", "C": "C +speed aug.", "D": "D larger reach", "DB": "D+B"}
rows = []
for n in names:
    rows += [row(f"{LAB[n]} (mean)", t5[f"{n}_mean"]), row("\\quad seed 0", t5[f"{n}_s0"]), row("\\quad seed 1", t5[f"{n}_s1"]), "MID"]
rows.append(row("Copy-last", t5["Copy-last"]))
latex_table(rows, HDR, "l" + "r" * 11, "Models A--D (and optional D+B) on the 1000 test videos, both seeds. $E^{(5)}$, $C_T$, $E_T$ at $1\\times$, $t=1$; "
            "$E^{(2)}$ at $1\\times$, $2\\times$, $3\\times$. Times: see hardware in the text.", "tab:t5", REPORT_TABLES / "table5.tex", size=r"\scriptsize\setlength{\tabcolsep}{2.5pt}")

# ----------------------------------------------------- final-model choice: validation split only
def val_e5(f):
    X, Y, V, P = ek.predict_all(f, val_f); return ek.e_metric(P, Y, V)[0]
val_sym = {}
for n in ("D",) + (("DB",) if OPTIONAL_DB else ()):
    val_sym[n] = float(np.mean([val_e5(ek.symmetrised(load_model(n, s))) for s in (0, 1)]))
final = min(val_sym, key=val_sym.get)
save_json(dict(val_E5_symmetrised=val_sym, final=final), "results/final_choice.json"); print("final choice", final, val_sym)

# ----------------------------------------------------- Table 6 : symmetrised wrappers
t6 = {}
for n in ("B", "D") + (("DB",) if OPTIONAL_DB else ()):
    for s in (0, 1):
        f = ek.symmetrised(load_model(n, s)); r = full_metrics(f, test_f); r["train_s_per_epoch"], r["params"] = train_info(n, s); t6[f"{n}_s{s}"] = r
        print("sym", n, s, {k: (f"{v:.3e}") for k, v in r.items()}, flush=True)
    t6[f"{n}_mean"] = {k: float(np.mean([t6[f"{n}_s{s}"][k] for s in (0, 1)])) for k in t6[f"{n}_s0"]}
save_json(t6, "results/table6.json")
def row6(lab, r):
    out = [lab]
    for k in COLS:
        out.append(f"{int(r[k]):,}".replace(",", "{,}") if k == "params" else (f"{r[k]:.1f}" if k == "train_s_per_epoch" else
                   (f"{r[k]:.2e}" if k in ("C_M", "C_R") else fmt(r[k]))))
    return out
L6 = {"B": "sym.\\ B", "D": "sym.\\ D", "DB": "sym.\\ D+B"}
rows = []
for n in ("B", "D") + (("DB",) if OPTIONAL_DB else ()):
    rows += [row6(f"{L6[n]} (mean)", t6[f"{n}_mean"]), row6("\\quad seed 0", t6[f"{n}_s0"]), row6("\\quad seed 1", t6[f"{n}_s1"]), "MID"]
rows.pop()
latex_table(rows, HDR, "l" + "r" * 11, "Symmetrised forecasters $\\bar f$ (eq.~7) on the test set. Training time is that of the underlying learned model; "
            "inference includes all 8 transformations, forward passes, inverse transformations and averaging.", "tab:t6", REPORT_TABLES / "table6.tex", size=r"\scriptsize\setlength{\tabcolsep}{2.5pt}")

# ----------------------------------------------------- Figure 6 (final model, seed 0, identical videos to Figures 3 and 5)
fm = ek.symmetrised(load_model(final, 0))
v3 = json.load(open("results/fig3_videos.json")); v5 = json.load(open("results/fig5_video.json"))["video"] - 9000
for T, tag, ids in ((Mi, "M", v3["top_C_M"]), (Ro, "R", v3["top_C_R"])):
    ids = [i - 9000 for i in ids]
    panels = fig3_panels(fm, test_f, T, ids)
    draw_fig3(panels, [f"final model, $T={tag}$, test video {9000+i}" for i in ids], f"figures/fig6a_{tag}")
    for r, (i, rows_) in enumerate(zip(ids, panels)):
        make_gif(rows_, f"FINAL T={tag}, video {9000+i}", f"figures/gifs/fig6a_{tag}_rank{r+1}_video{9000+i}.gif")
draw_fig5(fm, test_f, v5, "figures/fig6b_speed", "final")
print("done")
