"""Figure 3 / Figure 5 builders shared by the baseline experiments (Stage 3) and the final model (Stage 5)."""
import sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import mmdata as md, evalkit as ek
from common import *


def fig3_panels(f, test_f, T, vid_ids):
    """Rows (1) X | f(X)  (2) T(X) | T(Y)  (3) T(X) | f(T(X))  (4) T(X) | T(f(X)) for each chosen test-set index."""
    out = []
    with torch.no_grad():
        for vi in vid_ids:
            X, Y, _ = md.gather_windows(test_f[vi:vi + 1].to(ek.DEV), torch.tensor([1]), torch.tensor([1]))
            fX, fTX = f(X), f(T(X))
            rows = [(X, fX), (T(X), T(Y)), (T(X), fTX), (T(X), T(fX))]
            out.append([(a[0].cpu(), b[0].cpu()) for a, b in rows])
    return out


ROW_LABELS = ["(1) $X$ | $f(X)$", "(2) $T(X)$ | $T(Y)$", "(3) $T(X)$ | $f(T(X))$", "(4) $T(X)$ | $T(f(X))$"]


def draw_fig3(panels, titles, path):
    n = len(panels)
    fig = plt.figure(figsize=(8.6, 3.35 * n))
    subs = fig.subfigures(n, 1, hspace=0.02)
    subs = [subs] if n == 1 else list(subs)
    for p, (sub, rows) in enumerate(zip(subs, panels)):
        axes = sub.subplots(4, 10, gridspec_kw=dict(left=0.13, right=0.995, top=0.86, bottom=0.01, hspace=0.06, wspace=0.05))
        sub.suptitle(titles[p], fontsize=8, fontweight="bold", y=0.985)
        for r, (inp, fut) in enumerate(rows):
            for c in range(10):
                ax = axes[r, c]
                show_frame(ax, inp[c] if c < 5 else fut[c - 5], f"{c+1}" if r == 0 else None)
                if c == 5: ax.spines["left"].set_linewidth(1.8)
            axes[r, 0].set_ylabel(ROW_LABELS[r].replace("|", "$\\vert$"), fontsize=6, rotation=0, ha="right", va="center", labelpad=2)
    fig.savefig(path + ".png", dpi=200, bbox_inches="tight"); fig.savefig(path + ".pdf", bbox_inches="tight"); plt.close(fig)


def make_gif(rows, title, path, scale=3):
    """GIF stepping through the five forecast horizons; the four rows of Figure 3, last input | horizon-h frame."""
    frames = []
    for h in range(5):
        fig, axes = plt.subplots(4, 2, figsize=(2.6, 5.2))
        for r, (inp, fut) in enumerate(rows):
            show_frame(axes[r, 0], inp[4], "last input" if r == 0 else None)
            show_frame(axes[r, 1], fut[h], f"horizon {h+1}" if r == 0 else None)
            axes[r, 0].set_ylabel(ROW_LABELS[r].replace("|", "$\\vert$"), fontsize=5, rotation=0, ha="right", va="center")
        fig.suptitle(title, fontsize=7); fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())); plt.close(fig)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=700, loop=0)


def pick_nonoverlap_video(test_f):
    """First test video whose digits never overlap in any frame shown in Figure 5 (frames used at 1x, 2x, 3x)."""
    need = sorted(set(range(1, 8)) | {1, 3, 5, 7, 9, 11, 13} | {1, 4, 7, 10, 13, 16, 19})
    for i in range(len(test_f)):
        o = overlapping(test_f[i])
        if not any(o[f - 1] for f in need):
            return i
    raise RuntimeError("no suitable video")


def draw_fig5(f, test_f, vi, path, label="model A"):
    fig, axes = plt.subplots(6, 7, figsize=(7.6, 6.3))
    with torch.no_grad():
        for s, k in enumerate((1, 2, 3)):
            X, Y, valid = md.gather_windows(test_f[vi:vi + 1].to(ek.DEV), torch.tensor([1]), torch.tensor([k]))
            P = f(X).cpu(); X, Y = X.cpu(), Y.cpu()
            inf = [1 + i * k for i in range(5)]; tgf = [1 + (5 + i) * k for i in range(2)]
            for c in range(7):
                img = X[0, c] if c < 5 else Y[0, c - 5]
                show_frame(axes[2 * s, c], img, f"frame {(inf + tgf)[c]}")
                if c >= 5: show_frame(axes[2 * s + 1, c], P[0, c - 5]); axes[2 * s + 1, c].spines[:].set_color("#c0392b")
                else: axes[2 * s + 1, c].axis("off")
            axes[2 * s, 0].set_ylabel(f"{k}$\\times$ truth", fontsize=8); axes[2 * s + 1, 0].set_ylabel(f"{k}$\\times$ {label}", fontsize=8)
            axes[2 * s + 1, 0].axis("on"); axes[2 * s + 1, 0].set_xticks([]); axes[2 * s + 1, 0].set_yticks([])
            axes[2 * s + 1, 0].imshow(np.zeros((64, 64)), cmap="gray", alpha=0); [sp.set_visible(False) for sp in axes[2 * s + 1, 0].spines.values()]
    fig.suptitle(f"Test video {9000 + vi}: inputs (first 5 columns of each truth row), first two true and predicted future frames", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.3, w_pad=0.1)
    fig.savefig(path + ".png", dpi=200); fig.savefig(path + ".pdf"); plt.close(fig)
