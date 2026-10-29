"""Moving MNIST loading, symmetries (mirror / rotation) and speed windows.

Everything in the experiments goes through the functions in this file, so the
checks of Table 1 test exactly the code that the later experiments use.

Conventions
-----------
* A video is a float32 tensor of shape [20, 1, 64, 64] with values in [0, 1].
* Frames are numbered 1..20 in the text of the assignment; tensor indices are
  0-based (frame f is index f - 1).  Table 1 (d)/(e) use raw tensor indices.
* A symmetry g in G is an integer 0..7: g = 4*m + a means  R^a  (m = 0)  or
  M R^a (m = 1), i.e. rotate first, then mirror.
"""
from pathlib import Path

import numpy as np
import torch
import torchvision

N_FRAMES = 20
N_IN = 5
N_OUT = 5
TRAIN = slice(0, 8000)
VAL = slice(8000, 9000)
TEST = slice(9000, 10000)


# --------------------------------------------------------------------------- data
def load_all(root="data", device="cpu"):
    """All 10,000 videos as a uint8 tensor [10000, 20, 1, 64, 64] (no split option)."""
    ds = torchvision.datasets.MovingMNIST(root=str(root), split=None, download=True)
    if hasattr(ds, "data"):
        data = ds.data
    else:  # pragma: no cover - fall back to indexing
        data = torch.stack([ds[i] for i in range(len(ds))])
    data = torch.as_tensor(data)
    if data.dim() == 4:  # [N, 20, 64, 64]
        data = data.unsqueeze(2)
    assert data.shape == (10000, 20, 1, 64, 64), data.shape
    return data.to(torch.uint8).to(device)


def to_float(u8):
    """uint8 -> float32 in [0, 1]."""
    return u8.to(torch.float32) / 255.0


# ----------------------------------------------------------------- symmetries (G)
def mirror(x):
    """(Mx)[i, j] = x[i, 63 - j]."""
    return torch.flip(x, dims=(-1,))


def rotate(x, a=1):
    """R^a: a quarter turns anticlockwise (torch.rot90 with dims=(-2,-1))."""
    return torch.rot90(x, k=a, dims=(-2, -1))


def apply_g(x, g):
    """(M^m R^a)(x) for g = 4*m + a: rotate first, then mirror."""
    m, a = divmod(int(g), 4)
    x = rotate(x, a) if a else x
    return mirror(x) if m else x


def inverse_g(x, g):
    """g^{-1}(x).  R^a has inverse R^{-a}; every M R^a is its own inverse."""
    m, a = divmod(int(g), 4)
    if m:
        # (M R^a)^{-1} = R^{-a} M, which equals M R^a as a map
        x = mirror(x)
    return rotate(x, -a) if a else x


def apply_g_per_sample(x, gs):
    """Apply a possibly different g to every sample of x ([B, ...]); gs is a LongTensor [B]."""
    out = x.clone()
    for g in torch.unique(gs).tolist():
        sel = gs == g
        out[sel] = apply_g(x[sel], g)
    return out


# ---------------------------------------------------------------------- windows
def allowed_starts(k, speed_augmented):
    """Allowed start frames t (1-based).

    normal-speed training (A, B): t in 1..11 (all five targets exist at 1x);
    speed-augmented training (C, D): all five inputs and >= 2 targets exist.
    """
    if not speed_augmented:
        assert k == 1
        return list(range(1, 12))
    last_needed = 20 - 6 * k  # t + 5k + k <= 20
    return list(range(1, last_needed + 1))


def window_frames(t, k):
    """1-based frame numbers of the inputs and the *available* targets (eq. 2)."""
    inputs = [t + i * k for i in range(N_IN)]
    targets = [t + (N_IN + i) * k for i in range(N_OUT)]
    return inputs, [f for f in targets if f <= N_FRAMES]


def gather_windows(videos, t, k):
    """Batched windows following eq. (2).

    videos : [B, 20, 1, 64, 64] (float or uint8)
    t, k   : LongTensors [B] (start frame, 1-based; speed)
    returns X [B, 5, 1, 64, 64], Y [B, 5, 1, 64, 64], valid [B, 5] bool.
    Targets that do not exist (frame > 20) are padded by repeating the last frame and
    flagged False in `valid`; they must be excluded from loss and metrics.
    """
    B = videos.shape[0]
    steps = torch.arange(N_OUT + N_IN, device=videos.device)  # 0..9
    frame_no = t[:, None].to(videos.device) + steps[None, :] * k[:, None].to(videos.device)  # 1-based
    valid_all = frame_no <= N_FRAMES
    idx = (frame_no.clamp(max=N_FRAMES) - 1)
    b = torch.arange(B, device=videos.device)[:, None]
    seq = videos[b, idx]  # [B, 10, 1, 64, 64]
    return seq[:, :N_IN], seq[:, N_IN:], valid_all[:, N_IN:]


def make_window(video, t, k=1):
    """Single-video window; returns (X, Y_available) plus the 1-based frame numbers.

    Calls `gather_windows`, so tests of this function test the training/eval code path.
    """
    X, Y, valid = gather_windows(video[None], torch.tensor([t]), torch.tensor([k]))
    n = int(valid[0].sum())
    in_f, tg_f = window_frames(t, k)
    return X[0], Y[0, :n], in_f, tg_f
