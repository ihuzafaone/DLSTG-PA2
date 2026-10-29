"""Metrics, prediction helpers and the symmetrised forecaster."""
import time
import torch
import mmdata as md

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def per_video_mse_h(pred, target):
    """MSE_h per video: mean over the 64x64 pixels. pred/target [B, n, 1, 64, 64] -> [B, n]."""
    return ((pred - target) ** 2).mean(dim=(2, 3, 4))


def blank(X):      return torch.zeros_like(X)                       # all-zero frames
def copy_last(X):  return X[:, -1:].expand(-1, 5, -1, -1, -1)       # repeat the last input frame


def symmetrised(f):
    """f_bar(X) = 1/8 sum_g g^{-1}(f(g(X))); the sum is accumulated in float64."""
    def fbar(X):
        acc = None
        for g in range(8):
            y = md.inverse_g(f(md.apply_g(X, g)), g).double()
            acc = y if acc is None else acc + y
        return (acc / 8).float()
    return fbar


@torch.no_grad()
def predict_all(f, videos_f, t=1, k=1, bs=32, transform=None):
    """Run forecaster f over videos [N,20,1,64,64] (float).  Returns X, Y, valid, Yhat (all on CPU).
    `transform(x)` (e.g. md.mirror) is applied to every frame of X and Y *after* windowing; f sees T(X)."""
    Xs, Ys, Vs, Ps = [], [], [], []
    for i in range(0, len(videos_f), bs):
        v = videos_f[i:i + bs].to(DEV)
        n = len(v)
        X, Y, valid = md.gather_windows(v, torch.full((n,), t), torch.full((n,), k))
        if transform is not None:
            X, Y = transform(X), transform(Y)
        P = f(X)
        Xs.append(X.cpu()); Ys.append(Y.cpu()); Vs.append(valid.cpu()); Ps.append(P.cpu())
    return torch.cat(Xs), torch.cat(Ys), torch.cat(Vs), torch.cat(Ps)


def e_metric(P, Y, valid, horizons=5):
    """E^(h) = mean over videos of the mean over the first `horizons` horizons of MSE_h."""
    m = per_video_mse_h(P[:, :horizons], Y[:, :horizons])
    assert bool(valid[:, :horizons].all()), "horizons beyond the available targets are not scored"
    return m.mean(dim=1).double().mean().item(), m


def mse_per_horizon(P, Y):
    return per_video_mse_h(P, Y).double().mean(dim=0).tolist()


def consistency_and_error(f, videos_f, T, bs=32):
    """C_T = MSE(f(T X), T f(X)), E_T = MSE(f(T X), T Y) at 1x, t=1 (averaged over pixels, 5 horizons, videos)."""
    X, Y, V, P = predict_all(f, videos_f, bs=bs)                      # f(X)
    _, TY, _, PT = predict_all(f, videos_f, bs=bs, transform=T)       # f(T X)  (TY = T Y)
    c_vid = per_video_mse_h(PT, T(P)).mean(1)
    e_vid = per_video_mse_h(PT, TY).mean(1)
    return c_vid.double().mean().item(), e_vid.double().mean().item(), c_vid, e_vid, (X, Y, P, PT)


def timed_inference(f, videos_f, bs=32):
    """Wall-clock time to predict all test videos (1x, t=1) at batch size 32; each batch is already on device."""
    batches = []
    for i in range(0, len(videos_f), bs):
        v = videos_f[i:i + bs].to(DEV)
        X, _, _ = md.gather_windows(v, torch.full((len(v),), 1), torch.full((len(v),), 1))
        batches.append(X)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for X in batches:
            f(X)
    torch.cuda.synchronize()
    return time.perf_counter() - t0
