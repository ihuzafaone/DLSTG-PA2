"""Train one model under the shared protocol.

  python src/train.py --model A --seed 0
Models: A (baseline, 1x, no sym) | B (1x + symmetry aug) | C (baseline, speed aug) |
        D (larger-reach arch, speed aug) | DB (D + symmetry aug)
"""
import argparse, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import mmdata as md, models as M, evalkit as ek

SPECS = {  # speed_aug, sym_aug
    "A": (False, False), "B": (False, True), "C": (True, False), "D": (True, False), "DB": (True, True)}
# Architecture of D (chosen on the validation split, see results/d_design.json / report)
_dd = Path("results/d_design.json")
D_CFG = (M.Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in json.load(open(_dd))["best_cfg"].items()})
         if _dd.exists() else M.Config(patch=4, hidden=64, kernels=(3, 3), dilations=(1, 3)))
CFGS = {"A": M.Config(), "B": M.Config(), "C": M.Config(), "D": D_CFG, "DB": D_CFG}


def sample_windows(videos_u8, idx, gen, speed_aug, sym_aug):
    n = len(idx)
    if speed_aug:
        k = torch.randint(1, 4, (n,), generator=gen)
        hi = 20 - 6 * k                                              # allowed t = 1..hi  (14 / 8 / 2)
        t = 1 + (torch.rand(n, generator=gen) * hi).long()
    else:
        k = torch.ones(n, dtype=torch.long)
        t = torch.randint(1, 12, (n,), generator=gen)                # t in 1..11
    gs = torch.randint(0, 8, (n,), generator=gen) if sym_aug else None
    v = md.to_float(videos_u8[idx.to(videos_u8.device)])
    X, Y, valid = md.gather_windows(v, t, k)
    if gs is not None:
        gs_d = gs.to(X.device)
        X, Y = md.apply_g_per_sample(X, gs_d), md.apply_g_per_sample(Y, gs_d)
    return X, Y, valid


def run(name, seed, epochs=30, out="checkpoints", cfg=None, log=True, tag=None):
    speed_aug, sym_aug = SPECS[name]
    cfg = cfg or CFGS[name]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)                        # shuffling + window / augmentation sampling
    allv = md.load_all("data", ek.DEV)
    train_u8, val_f = allv[md.TRAIN], md.to_float(allv[md.VAL])
    model = M.ConvLSTMForecaster(cfg).to(ek.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = dict(model=name, seed=seed, cfg=cfg.dict(), n_params=M.n_params(model), epochs=[], device=torch.cuda.get_device_name(0))
    for ep in range(1, epochs + 1):
        model.train()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        perm = torch.randperm(8000, generator=gen)
        tot = 0.0
        for i in range(0, 8000, 32):
            X, Y, valid = sample_windows(train_u8, perm[i:i + 32], gen, speed_aug, sym_aug)
            P = model(X)
            per_h = ek.per_video_mse_h(P, Y)                          # [B, 5]
            vf = valid.float()
            loss = ((per_h * vf).sum(1) / vf.sum(1)).mean()           # mean over *available* targets, then videos
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item()
        torch.cuda.synchronize(); dt = time.perf_counter() - t0       # epoch time excludes validation below
        model.eval()
        with torch.no_grad():
            _, Yv, Vv, Pv = ek.predict_all(model, val_f)
            e5 = ek.e_metric(Pv, Yv, Vv)[0]
            ev = {}
            for k in (2, 3):
                _, Yk, Vk, Pk = ek.predict_all(model, val_f, k=k)
                ev[f"val_E2_{k}x"] = ek.e_metric(Pk, Yk, Vk, horizons=2)[0]
        hist["epochs"].append(dict(epoch=ep, train_loss=tot / 250, seconds=dt, val_E5_1x=e5, **ev))
        if log:
            print(f"[{name} s{seed}] ep {ep:2d} loss {tot/250:.5f} val E5 {e5:.5f} E2@2x {ev['val_E2_2x']:.5f} E2@3x {ev['val_E2_3x']:.5f}  {dt:.1f}s", flush=True)
    hist["train_sec_per_epoch"] = sum(e["seconds"] for e in hist["epochs"]) / epochs
    Path(out).mkdir(exist_ok=True)
    stem = f"{tag or name}_s{seed}"
    torch.save(dict(state_dict=model.state_dict(), cfg=cfg.dict(), name=name, seed=seed), f"{out}/{stem}.pt")
    Path("results/train").mkdir(parents=True, exist_ok=True)
    json.dump(hist, open(f"results/train/{stem}.json", "w"), indent=1)
    return model, hist


def load_model(name, seed, out="checkpoints", tag=None):
    ck = torch.load(f"{out}/{tag or name}_s{seed}.pt", map_location=ek.DEV)
    model = M.ConvLSTMForecaster(M.Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cfg"].items()})).to(ek.DEV)
    model.load_state_dict(ck["state_dict"]); return model.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SPECS)); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--cfg", type=str, default=None, help="JSON Config override (used only for validation screening of D)")
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--out", type=str, default="checkpoints")
    a = ap.parse_args()
    import json as _j
    cfg = None
    if a.cfg:
        d = _j.loads(a.cfg); cfg = M.Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})
    run(a.model, a.seed, a.epochs, out=a.out, cfg=cfg, tag=a.tag)
