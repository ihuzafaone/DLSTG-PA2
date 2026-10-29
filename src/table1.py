"""Table 1: sanity checks of the transformation and window functions (synthetic tensors)."""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import mmdata as md

torch.manual_seed(0)
x = torch.rand(20, 1, 64, 64)                      # random video
p = torch.zeros(20, 1, 64, 64); p[7, 0, 10, 20] = 1  # marker video
M, R = md.mirror, md.rotate
MR = lambda v: md.apply_g(v, 4)                    # M R  (rotate, then mirror)

rows = []
def diff(a, b): return float((a - b).abs().max())
a = diff(M(M(x)), x);            rows.append(("(a)", "max|M(M(x)) - x|", "0", f"{a:.1f}", torch.equal(M(M(x)), x)))
r4 = R(R(R(R(x))));  b = diff(r4, x); rows.append(("(b)", "max|R^4(x) - x|", "0", f"{b:.1f}", torch.equal(r4, x)))
lhs = M(R(M(x)))     # M R M
rhs = md.rotate(x, -1)
c = diff(lhs, rhs);               rows.append(("(c)", "max|M R M(x) - R^-1(x)|", "0", f"{c:.1f}", torch.equal(lhs, rhs)))
loc = lambda v: tuple(int(i) for i in (v.flatten().argmax().item() // (64 * 64),
                                       (v.flatten().argmax().item() // 64) % 64,
                                       v.flatten().argmax().item() % 64))
rows.append(("(d)", "bright pixel of M(p)", "(7, 10, 43)", str(loc(M(p))), loc(M(p)) == (7, 10, 43)))
rows.append(("(e)", "bright pixel of R(p)", "(7, 43, 10)", str(loc(R(p))), loc(R(p)) == (7, 43, 10)))

v = torch.arange(1, 21, dtype=torch.float32).view(20, 1, 1, 1).expand(20, 1, 64, 64).contiguous()  # frame f has value f
def frames(k):
    X, Y, in_f, tg_f = md.make_window(v, 1, k)
    xin = [int(X[i, 0, 0, 0]) for i in range(X.shape[0])]
    yout = [int(Y[i, 0, 0, 0]) for i in range(Y.shape[0])]
    assert xin == in_f and yout == tg_f
    return xin, yout
exp = {1: ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]), 2: ([1, 3, 5, 7, 9], [11, 13, 15, 17, 19]), 3: ([1, 4, 7, 10, 13], [16, 19])}
for lab, k in (("(f)", 1), ("(g)", 2), ("(h)", 3)):
    xin, yout = frames(k)
    fmt = lambda a, b: ", ".join(map(str, a)) + " / " + ", ".join(map(str, b))
    rows.append((lab, f"{k}x input / target frames", fmt(*exp[k]), fmt(xin, yout), (xin, yout) == exp[k]))

# extra: inverse maps & the allowed start ranges of the speed-augmented windows
for g in range(8):
    assert torch.equal(md.inverse_g(md.apply_g(x, g), g), x), g
starts = {k: len(md.allowed_starts(k, True)) for k in (1, 2, 3)}
assert starts == {1: 14, 2: 8, 3: 2}, starts
assert md.allowed_starts(1, False) == list(range(1, 12))

for r in rows: print(r)
assert all(r[4] for r in rows), "Table 1 mismatch!"
Path("results").mkdir(exist_ok=True)
json.dump([dict(check=r[0], what=r[1], expected=r[2], yours=r[3], ok=bool(r[4])) for r in rows], open("results/table1.json", "w"), indent=1)
print("all Table 1 rows match; inverse_g correct for all 8 elements; start ranges 14/8/2 OK")

# ---- LaTeX version of Table 1 (handout format; every "Yours" value is what the asserts above verified) ----------
tex = r"""\begin{table}[H]\centering\small
\caption{Sanity checks of the transformation and window functions (the same functions every experiment calls). (d), (e): (frame, row, column), 0-based, for $p[7,0,10,20]=1$. For (a)--(c) \texttt{torch.equal} returned \texttt{True}.}\label{tab:t1}
\begin{tabular}{llll}\toprule
 & Check & Expected & Yours \\ \midrule
(a) & $\max|M(M(x))-x|$ & 0 & %s \\
(b) & $\max|R^{4}(x)-x|$ & 0 & %s \\
(c) & $\max|MRM(x)-R^{-1}(x)|$ & 0 & %s \\
(d) & bright pixel of $M(p)$ (frame, row, column) & $(7,10,43)$ & $%s$ \\
(e) & bright pixel of $R(p)$ (frame, row, column) & $(7,43,10)$ & $%s$ \\ \midrule
(f) & $1\times$ input / target frames & 1--5 / 6--10 & %s \\
(g) & $2\times$ input / target frames & 1, 3, 5, 7, 9 / 11, 13, 15, 17, 19 & %s \\
(h) & $3\times$ input / target frames & 1, 4, 7, 10, 13 / 16, 19 & %s \\
\bottomrule\end{tabular}
\end{table}""" % (rows[0][3], rows[1][3], rows[2][3], rows[3][3], rows[4][3],
                  rows[5][3].replace(", ", ", "), rows[6][3], rows[7][3])
Path("report/tables").mkdir(parents=True, exist_ok=True)
Path("report/tables/table1.tex").write_text(tex)
