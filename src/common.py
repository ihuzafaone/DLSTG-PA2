"""Shared helpers for experiment scripts: number formatting, image grids, overlap detection."""
import json, math
from pathlib import Path
import numpy as np, torch
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mmdata as md

REPORT_TABLES = Path("report/tables"); REPORT_TABLES.mkdir(parents=True, exist_ok=True)
Path("results").mkdir(exist_ok=True); Path("figures").mkdir(exist_ok=True)


def fmt(x, sig=3):
    """Three significant figures; scientific notation below 1e-3 (as the handout asks)."""
    if x is None: return "--"
    if x == 0: return "0"
    if abs(x) < 1e-3: return f"{x:.{sig-1}e}"
    dec = max(sig - 1 - math.floor(math.log10(abs(x))), 0)
    return f"{x:.{dec}f}"


def mean_seed(vals): return float(np.mean(vals))


def n_components(frame):                      # frame: 2-D float in [0,1]
    return ndimage.label(np.asarray(frame) > 0.2)[1]


def overlapping(video):
    """True per frame where the two digits touch / overlap (fewer than two ink components)."""
    v = video.squeeze(1).cpu().numpy() if torch.is_tensor(video) else video
    return np.array([n_components(f) < 2 for f in v])


def save_json(obj, path):
    json.dump(obj, open(path, "w"), indent=1)


def show_frame(ax, img, title=None, cmap="gray"):
    ax.imshow(np.asarray(img).squeeze(), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=7, pad=2)


def latex_table(rows, header, colspec, caption, label, path, note=None, size=r"\small"):
    """Write a booktabs LaTeX table."""
    lines = [r"\begin{table}[H]\centering" + size, rf"\caption{{{caption}}}\label{{{label}}}",
             rf"\begin{{tabular}}{{{colspec}}}\toprule", " & ".join(header) + r" \\ \midrule"]
    for r in rows:
        lines.append(r"\midrule" if r == "MID" else " & ".join(str(c) for c in r) + r" \\")
    lines += [r"\bottomrule\end{tabular}"]
    if note: lines.append(rf"\\[2pt]\parbox{{0.95\linewidth}}{{\footnotesize {note}}}")
    lines.append(r"\end{table}")
    Path(path).write_text("\n".join(lines))
