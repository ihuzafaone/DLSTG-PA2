"""Validation-only screening of candidate designs for model D (speed-augmented training, no symmetry augmentation).

Every candidate must (i) have at most 721,176 trainable parameters and (ii) a strictly larger one-step reach than A (8 px).
Each is trained for 12 epochs with seed 0 under the protocol of model C (Adam 1e-3, batch 32, k in {1,2,3}) and scored on the
VALIDATION split only:   score = mean(E5@1x, E2@2x, E2@3x)  (final epoch; lower is better).  The test split is never touched.
"""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import models as M

CANDS = {
    "d1_dil13_h64":     M.Config(4, 64, (3, 3), (1, 3)),
    "d2_dil13_h78":     M.Config(4, 78, (3, 3), (1, 3)),
    "d3_k53_dil12_h56": M.Config(4, 56, (5, 3), (1, 2)),
    "d4_k53_dil13_h56": M.Config(4, 56, (5, 3), (1, 3)),
    "d5_patch8_h64":    M.Config(8, 64, (3, 3), (1, 1)),
    "d6_k33_dil12_h72": M.Config(4, 72, (3, 3), (1, 2)),
}
EPOCHS, PAR = 12, 3
LIMIT, REACH_A = 721_176, 8

if __name__ == "__main__":
    names = list(CANDS)
    for i in range(0, len(names), PAR):                       # three concurrent trainings at a time (GPU memory)
        procs = []
        for n in names[i:i + PAR]:
            c = CANDS[n]
            assert M.n_params(M.ConvLSTMForecaster(c)) <= LIMIT and M.reach_in_pixels(c) > REACH_A, n
            procs.append(subprocess.Popen([sys.executable, "src/train.py", "--model", "D", "--seed", "0", "--epochs", str(EPOCHS),
                                           "--cfg", json.dumps(c.dict()), "--tag", n, "--out", "screening"],
                                          stdout=open(f"logs/screen_{n}.log", "w"), stderr=subprocess.STDOUT))
        for p in procs: p.wait()
    res = {}
    for n, c in CANDS.items():
        h = json.load(open(f"results/train/{n}_s0.json")); e = h["epochs"][-1]
        res[n] = dict(cfg=c.dict(), params=h["n_params"], reach_px=M.reach_in_pixels(c), val_E5_1x=e["val_E5_1x"], val_E2_2x=e["val_E2_2x"],
                      val_E2_3x=e["val_E2_3x"], score=(e["val_E5_1x"] + e["val_E2_2x"] + e["val_E2_3x"]) / 3)
    best = min(res, key=lambda n: (res[n]["score"], res[n]["params"]))
    json.dump(dict(candidates=res, best=best, best_cfg=res[best]["cfg"], rule="min mean(E5@1x, E2@2x, E2@3x) on validation, epoch 12, seed 0"),
              open("results/d_design.json", "w"), indent=1)
    for n, r in res.items(): print(f"{n:20s} params {r['params']:7d} reach {r['reach_px']:2d}px  val E5@1x {r['val_E5_1x']:.5f}  E2@2x {r['val_E2_2x']:.5f}  E2@3x {r['val_E2_3x']:.5f}  score {r['score']:.5f}")
    print("BEST", best)
