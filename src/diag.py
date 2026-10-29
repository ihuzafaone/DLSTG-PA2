"""Diagnostics for the speed failure: how far does each digit move between the last seen frame and the first forecast?

For test videos in which the two digits are separate blobs both in the last input frame and in the true first target
frame, track each blob's centroid.  true displacement = |c(y_1) - c(x_5)| ;  predicted displacement = |c(y_hat_1) - c(x_5)|.
"""
import sys
from pathlib import Path
import numpy as np, torch
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).parent))
import mmdata as md, evalkit as ek


def blobs(frame, thr, min_area=12):
    img = np.asarray(frame).squeeze()
    lab, n = ndimage.label(img > thr)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() >= min_area:
            out.append(np.array(ndimage.center_of_mass(img * m)))
    return out


def displacement_stats(f, test_f, k, max_videos=1000):
    X, Y, V, P = ek.predict_all(f, test_f[:max_videos], k=k)
    true_d, pred_d, ratio_vid = [], [], []
    for b in range(len(X)):
        c0 = blobs(X[b, -1], 0.2); c1 = blobs(Y[b, 0], 0.2)
        if len(c0) != 2 or len(c1) != 2: continue
        cp = blobs(P[b, 0], 0.5 * float(P[b, 0].max()) if float(P[b, 0].max()) > 0.1 else 1.1)
        if len(cp) < 2: continue
        for a in c0:
            j = int(np.argmin([np.linalg.norm(a - c) for c in c1]))          # true successor of blob a
            m = int(np.argmin([np.linalg.norm(a - c) for c in cp]))          # prediction blob nearest to the old position
            true_d.append(np.linalg.norm(c1[j] - a)); pred_d.append(np.linalg.norm(cp[m] - a))
    true_d, pred_d = np.array(true_d), np.array(pred_d)
    return dict(n_digits=int(len(true_d)), true_median=float(np.median(true_d)), pred_median=float(np.median(pred_d)),
                ratio_median=float(np.median(pred_d / np.maximum(true_d, 1e-6))),
                true_q=[float(x) for x in np.quantile(true_d, [.1, .5, .9])], pred_q=[float(x) for x in np.quantile(pred_d, [.1, .5, .9])]), (true_d, pred_d)


if __name__ == "__main__":
    from train import load_model
    import json
    allv = md.load_all("data", "cpu"); test_f = md.to_float(allv[md.TEST])
    out = {}
    for name, s in (("A", 0), ("A", 1)):
        m = load_model(name, s)
        out[f"{name}_s{s}"] = {}
        for k in (1, 2, 3):
            st, _ = displacement_stats(m, test_f, k)
            out[f"{name}_s{s}"][f"{k}x"] = st
            print(name, s, k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in st.items() if a not in ("true_q", "pred_q")}, flush=True)
    Path("results").mkdir(exist_ok=True); json.dump(out, open("results/displacement_A.json", "w"), indent=1)
