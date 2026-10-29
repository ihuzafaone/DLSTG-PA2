"""Figure 1: dense optical flow (Farneback) on the first 50 training videos."""
import json, sys
from pathlib import Path
import numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).parent))
import mmdata as md

data = md.load_all("data")[:50, :, 0].numpy()          # [50, 20, 64, 64] uint8 (training videos 0..49)
frames = (md.to_float(torch.from_numpy(data)).numpy() * 255.0).round().astype(np.uint8)  # via [0,1] -> uint8 as specified

def n_components(img):
    """Connected blobs of ink in a frame (threshold 0.2): fewer than 2 => the digits touch/overlap."""
    return ndimage.label(img > 51)[1]

def collect(exclude_overlap=False):
    th, sp, kept_pairs, n_pairs, skipped = [], [], 0, 0, 0
    for v in range(50):
        for s in range(19):
            n_pairs += 1
            prev, nxt = frames[v, s], frames[v, s + 1]
            if exclude_overlap and (n_components(prev) < 2 or n_components(nxt) < 2):
                skipped += 1
                continue
            flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            dx, dy = flow[..., 0], flow[..., 1]
            mag = np.sqrt(dx ** 2 + dy ** 2)
            keep = (prev > 51) & (nxt > 51) & (mag > 0.5)     # >0.2 on the [0,1] scale in both frames
            th.append(np.degrees(np.arctan2(-dy[keep], dx[keep])) % 360.0)
            sp.append(mag[keep])
    return np.concatenate(th), np.concatenate(sp), n_pairs, skipped

theta, speed, n_pairs, _ = collect()
assert n_pairs == 950
edges = np.linspace(0, 360, 19)
frac_dir = np.histogram(theta, bins=edges)[0] / len(theta)
med = float(np.median(speed))
print("pixels kept", len(theta), "median speed", med)

# sensitivity: drop every pair in which the digits touch in either frame (flow unreliable there)
theta2, speed2, _, skipped = collect(exclude_overlap=True)
frac_dir2 = np.histogram(theta2, bins=edges)[0] / len(theta2)

stats = dict(n_pairs=n_pairs, n_kept=int(len(theta)), median_speed=med,
             speed_quantiles={q: float(np.quantile(speed, q / 100)) for q in (5, 25, 50, 75, 95)},
             frac_speed_gt_8=float((speed > 8).mean()), frac_speed_gt_4=float((speed > 4).mean()),
             direction_fraction=frac_dir.tolist(), uniform_level=1 / 18,
             max_abs_dev_from_uniform=float(np.abs(frac_dir - 1 / 18).max()),
             max_rel_dev_from_uniform=float((np.abs(frac_dir - 1 / 18) / (1 / 18)).max()),
             axis_aligned_fraction=float(sum(frac_dir[i] for i in (0, 4, 8, 9, 13, 17))),
             no_overlap=dict(pairs_kept=n_pairs - skipped, pairs_dropped=skipped, n_kept=int(len(theta2)),
                             median_speed=float(np.median(speed2)),
                             direction_fraction=frac_dir2.tolist(),
                             max_rel_dev_from_uniform=float((np.abs(frac_dir2 - 1 / 18) / (1 / 18)).max()),
                             frac_speed_gt_8=float((speed2 > 8).mean())))
Path("results").mkdir(exist_ok=True)
json.dump(stats, open("results/flow_stats.json", "w"), indent=1)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
ax[0].bar(edges[:-1], frac_dir, width=20, align="edge", color="#3b6a8f", edgecolor="white")
ax[0].axhline(1 / 18, color="crimson", ls="--", lw=1.2, label="uniform = 1/18")
ax[0].set_xticks(range(0, 361, 60)); ax[0].set_xlim(0, 360)
ax[0].set_xlabel(r"flow direction $\theta$ (degrees; 0$^\circ$ = right, 90$^\circ$ = up)")
ax[0].set_ylabel("fraction of kept pixels"); ax[0].set_title("(a) direction"); ax[0].legend(frameon=False)
bins = np.linspace(0, max(12, np.ceil(speed.max())), 49)
ax[1].hist(speed, bins=bins, weights=np.full(len(speed), 1 / len(speed)), color="#3b6a8f", edgecolor="white")
ax[1].axvline(med, color="crimson", lw=1.5, label=f"median = {med:.2f} px/frame")
ax[1].set_xlabel("flow speed (pixels per frame)"); ax[1].set_ylabel("fraction of kept pixels")
ax[1].set_title("(b) speed"); ax[1].legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/fig1_flow.pdf"); fig.savefig("figures/fig1_flow.png", dpi=200)
print(json.dumps({k: v for k, v in stats.items() if k not in ("direction_fraction",)}, indent=1))
print("dir frac", np.round(frac_dir, 4).tolist())

# ---- calibration: how non-uniform would 100 independent trajectories look under a uniform direction law?
rng = np.random.default_rng(0)
def tv_mirror(h):   # mirror x -> -x sends theta -> 180 - theta : bin i -> bin (8 - i) mod 18 (bins are 20 deg wide)
    return 0.5 * np.abs(h - h[[(8 - i) % 18 for i in range(18)]]).sum()
def tv_rot(h):      # 90 degree rotation sends theta -> theta + 90 : shift by 4.5 bins -> average of shifts by 4 and 5
    return 0.5 * np.abs(h - 0.5 * (np.roll(h, 4) + np.roll(h, 5))).sum()
n_traj = 100
null_max, null_m, null_r = [], [], []
for _ in range(5000):
    h = np.bincount(rng.integers(0, 18, n_traj), minlength=18) / n_traj
    null_max.append((np.abs(h - 1 / 18) / (1 / 18)).max()); null_m.append(tv_mirror(h)); null_r.append(tv_rot(h))
cal = dict(n_independent_trajectories=n_traj,
           observed=dict(max_rel_dev=stats["max_rel_dev_from_uniform"], tv_mirror=float(tv_mirror(frac_dir)), tv_rot=float(tv_rot(frac_dir))),
           null_mean=dict(max_rel_dev=float(np.mean(null_max)), tv_mirror=float(np.mean(null_m)), tv_rot=float(np.mean(null_r))),
           null_p95=dict(max_rel_dev=float(np.quantile(null_max, .95)), tv_mirror=float(np.quantile(null_m, .95)), tv_rot=float(np.quantile(null_r, .95))),
           p_value_max_rel_dev=float(np.mean(np.array(null_max) >= stats["max_rel_dev_from_uniform"])))
stats["calibration"] = cal
json.dump(stats, open("results/flow_stats.json", "w"), indent=1)
print(json.dumps(cal, indent=1))
