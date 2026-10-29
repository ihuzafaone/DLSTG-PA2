"""Q2.7 evidence: centroid path, spread and ink mass of rows (2) T(Y), (3) f(T(X)), (4) T(f(X)) for the Figure 3 videos."""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import mmdata as md, evalkit as ek
from figkit import fig3_panels
from train import load_model

def stats(frames):                                # frames [5,1,64,64]
    out = []
    for fr in frames:
        im = fr.squeeze().numpy().astype(np.float64); m = im.sum()
        yy, xx = np.mgrid[0:64, 0:64]
        cy, cx = (yy * im).sum() / m, (xx * im).sum() / m
        sd = np.sqrt(((yy - cy) ** 2 * im).sum() / m + ((xx - cx) ** 2 * im).sum() / m)
        out.append(dict(cy=cy, cx=cx, spread=sd, mass=m))
    return out

if __name__ == "__main__":
    allv = md.load_all("data", "cpu"); test_f = md.to_float(allv[md.TEST]); m0 = load_model("A", 0)
    v3 = json.load(open("results/fig3_videos.json")); res = {}
    for T, tag, ids in ((md.mirror, "M", v3["top_C_M"]), (md.rotate, "R", v3["top_C_R"])):
        panels = fig3_panels(m0, test_f, T, [i - 9000 for i in ids])
        for vid, rows in zip(ids, panels):
            r = {n: stats(rows[i][1]) for n, i in (("truth_TY", 1), ("f_of_TX", 2), ("T_of_fX", 3))}
            # last input frame centroid for reference (row 2 inputs = T(X))
            r["last_input"] = stats(rows[1][0][4:5])[0]
            def path(n): return np.array([[s["cy"], s["cx"]] for s in r[n]])
            d = {n: float(np.linalg.norm(path(n)[-1] - path(n)[0]) / 4) for n in ("truth_TY", "f_of_TX", "T_of_fX")}
            gap = float(np.linalg.norm(path("f_of_TX") - path("T_of_fX"), axis=1).mean())
            res[f"{tag}_{vid}"] = dict(speed_px_per_frame=d, mean_centroid_gap_px=gap,
                                       spread_h1_h5={n: [r[n][0]["spread"], r[n][-1]["spread"]] for n in ("truth_TY", "f_of_TX", "T_of_fX")},
                                       mass_h1_h5={n: [r[n][0]["mass"], r[n][-1]["mass"]] for n in ("truth_TY", "f_of_TX", "T_of_fX")},
                                       centroid_h1={n: [round(r[n][0]["cy"], 1), round(r[n][0]["cx"], 1)] for n in ("truth_TY", "f_of_TX", "T_of_fX")},
                                       centroid_h5={n: [round(r[n][-1]["cy"], 1), round(r[n][-1]["cx"], 1)] for n in ("truth_TY", "f_of_TX", "T_of_fX")})
            print(tag, vid, json.dumps(res[f"{tag}_{vid}"]))
    json.dump(res, open("results/q27_analysis.json", "w"), indent=1)
