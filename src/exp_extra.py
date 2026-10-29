"""Extra evidence for Q2.9-Q2.17 (not graded tables; supplementary): for every model (and symmetrised wrappers)
  * digit displacement last-input -> first forecast at 1x/2x/3x (Table S1),
  * E^(5) on test videos whose digits overlap vs do not overlap in the target frames (1x),
  * E^(5) on time-reversed videos (the dynamics are time-reversible, so a motion rule should not care),
  * Q2.7-style centroid analysis of the Figure 3 videos.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import mmdata as md, models as M, evalkit as ek
from common import *
from diag import displacement_stats
from figkit import fig3_panels
from analysis_q27 import stats as blob_stats
from train import load_model

allv = md.load_all("data", "cpu"); test_f = md.to_float(allv[md.TEST])
ov_future = np.array([overlapping(test_f[i])[5:10].any() for i in range(1000)])      # frames 6..10 (1x, t=1)
rev = torch.flip(test_f, dims=(1,))                                                   # time-reversed videos


def eval_model(f, label):
    r = {}
    X, Y, V, P = ek.predict_all(f, test_f)
    e_vid = ek.per_video_mse_h(P, Y).mean(1).numpy()
    r["E5_overlap"], r["E5_no_overlap"] = float(e_vid[ov_future].mean()), float(e_vid[~ov_future].mean())
    r["n_overlap"] = int(ov_future.sum())
    Xr, Yr, Vr, Pr = ek.predict_all(f, rev)
    r["E5_time_reversed"] = float(ek.e_metric(Pr, Yr, Vr)[0])
    r["E5_forward"] = float(e_vid.mean())
    r["displacement"] = {f"{k}x": displacement_stats(f, test_f, k)[0] for k in (1, 2, 3)}
    # Q2.7 style centroid analysis on the Figure 3 videos (T = M and T = R)
    v3 = json.load(open("results/fig3_videos.json")); q = {}
    for T, tag, ids in ((md.mirror, "M", v3["top_C_M"]), (md.rotate, "R", v3["top_C_R"])):
        for vid, rows in zip(ids, fig3_panels(f, test_f, T, [i - 9000 for i in ids])):
            path = lambda i: np.array([[s["cy"], s["cx"]] for s in blob_stats(rows[i][1])])
            sp = lambda p: float(np.linalg.norm(p[-1] - p[0]) / 4)
            q[f"{tag}_{vid}"] = dict(speed_truth=sp(path(1)), speed_row3=sp(path(2)), speed_row4=sp(path(3)),
                                     gap_row3_row4=float(np.linalg.norm(path(2) - path(3), axis=1).mean()),
                                     gap_row3_truth=float(np.linalg.norm(path(2) - path(1), axis=1).mean()),
                                     mass_row3=[blob_stats(rows[2][1])[0]["mass"], blob_stats(rows[2][1])[-1]["mass"]],
                                     mass_truth=[blob_stats(rows[1][1])[0]["mass"], blob_stats(rows[1][1])[-1]["mass"]])
    r["fig3_videos"] = q
    # does the consistency gap concentrate where the digits overlap at the end of the input (frames 4-5)?
    ov_in = np.array([overlapping(test_f[i])[3:5].any() for i in range(1000)])
    for T, tag in ((md.mirror, "M"), (md.rotate, "R")):
        c = ek.consistency_and_error(f, test_f, T)[2].numpy()
        r[f"C_{tag}_by_input_overlap"] = dict(overlap=float(c[ov_in].mean()), no_overlap=float(c[~ov_in].mean()), n_overlap=int(ov_in.sum()))
    print(label, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k not in ("displacement", "fig3_videos")}, flush=True)
    return r


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["A"]
    out_path = Path("results/extra.json"); out = json.load(open(out_path)) if out_path.exists() else {}
    for spec in names:                       # e.g. "A", "D", "sym:D"
        sym = spec.startswith("sym:"); name = spec.split(":")[-1]
        for s in (0, 1):
            f = load_model(name, s); f = ek.symmetrised(f) if sym else f
            out[f"{spec}_s{s}"] = eval_model(f, f"{spec} s{s}")
    json.dump(out, open(out_path, "w"), indent=1)
