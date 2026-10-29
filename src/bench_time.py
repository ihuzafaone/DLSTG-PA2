"""Controlled re-timing of the training cost per epoch for the five architectures (4 epochs each, seed 0, idle GPU, one after another)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import train
res = {}
for n in ("A", "B", "C", "D", "DB"):
    _, h = train.run(n, 0, epochs=4, out="bench", tag=f"bench_{n}", log=False)
    s = [e["seconds"] for e in h["epochs"]][1:]            # drop the first (cuDNN autotune) epoch
    res[n] = sum(s) / len(s); print(n, [round(e["seconds"], 1) for e in h["epochs"]], flush=True)
json.dump(res, open("results/bench_time.json", "w"), indent=1)
