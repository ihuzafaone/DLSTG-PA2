"""Tables S1 (digit displacement) and S2 (time reversal / overlap), from results/extra.json."""
import json
from pathlib import Path
import numpy as np
ex = json.load(open("results/extra.json"))
M = [("A", "A (baseline)"), ("B", "B (+sym.\\ aug.)"), ("C", "C (+speed aug.)"), ("D", "D (reach 12 px)"), ("sym:D", "final: sym.\\ D")]
mean = lambda k, f: float(np.mean([f(ex[f"{k}_s{s}"]) for s in (0, 1)]))
rows = []
for k, lab in M:
    cells = []
    for sp in ("1x", "2x", "3x"):
        p = mean(k, lambda r: r["displacement"][sp]["pred_median"]); q = mean(k, lambda r: r["displacement"][sp]["ratio_median"])
        cells.append(f"{p:.2f} ({100*q:.0f}\\%)")
    rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
true = [mean("A", lambda r, sp=sp: r["displacement"][sp]["true_median"]) for sp in ("1x", "2x", "3x")]
tex = r"""\begin{table}[H]\centering\small
\caption*{\textbf{Table S1} (supplementary). Median centroid displacement (px) of a digit from the last input frame to the first forecast frame at $1\times$, $2\times$, $3\times$ (digits that form separate blobs in both frames; mean over seeds 0, 1 of each model's median; in brackets: median predicted/true per digit). Truth: """ + f"{true[0]:.2f}, {true[1]:.2f}, {true[2]:.2f}" + r""" px.}
\begin{tabular}{lrrr}\toprule
Model & $1\times$ & $2\times$ & $3\times$ \\ \midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}"""
Path("report/tables/tableS1.tex").write_text(tex)
rows = []
for k, lab in M:
    rows.append(f"{lab} & " + " & ".join(f"{mean(k, lambda r, n=n: r[n]):.4f}" for n in ("E5_forward", "E5_time_reversed", "E5_overlap", "E5_no_overlap")) + r" \\")
tex = r"""\begin{table}[H]\centering\small
\caption*{\textbf{Table S2} (supplementary). $E^{(5)}$ (mean over seeds, $1\times$, $t=1$) on forward and on \emph{time-reversed} test videos, and on the 586 test videos whose digits overlap in frames 6--10 versus the other 414.}
\begin{tabular}{lrrrr}\toprule
Model & forward & time-reversed & overlap in 6--10 & no overlap \\ \midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}"""
Path("report/tables/tableS2.tex").write_text(tex)
