"""ConvLSTM forecaster (Shi et al., 2015, without peephole terms).

Model A is `ConvLSTMForecaster()` with its defaults.  Model D uses the very same class with a
different spatial configuration (kernel sizes / dilations / patch size); the 9-step rollout
without teacher forcing, the input/output shapes and the training protocol are identical.
"""
from dataclasses import dataclass, asdict

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    patch: int = 4              # pixel_unshuffle factor
    hidden: int = 64            # hidden channels of both ConvLSTM layers
    kernels: tuple = (3, 3)     # spatial kernel size of ConvLSTM layer 1, 2
    dilations: tuple = (1, 1)   # dilation of the gate convolution of layer 1, 2

    def dict(self):
        return asdict(self)


class ConvLSTMCell(nn.Module):
    """[i, f, o, g] = Conv([u, h_prev]);  c = s(f)*c_prev + s(i)*tanh(g);  h = s(o)*tanh(c)."""

    def __init__(self, in_ch, hidden, kernel=3, dilation=1):
        super().__init__()
        self.hidden = hidden
        pad = dilation * (kernel - 1) // 2  # keeps the spatial size ("same" padding, zero pad)
        self.conv = nn.Conv2d(in_ch + hidden, 4 * hidden, kernel, stride=1, padding=pad,
                              dilation=dilation, bias=True)

    def forward(self, u, h, c):
        i, f, o, g = torch.chunk(self.conv(torch.cat([u, h], dim=1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMForecaster(nn.Module):
    """5 input frames -> 5 forecast frames, by a 9-step free-running rollout."""

    def __init__(self, cfg: Config = Config()):
        super().__init__()
        self.cfg = cfg
        p, H = cfg.patch, cfg.hidden
        self.cell1 = ConvLSTMCell(p * p, H, cfg.kernels[0], cfg.dilations[0])
        self.cell2 = ConvLSTMCell(H, H, cfg.kernels[1], cfg.dilations[1])
        self.out = nn.Conv2d(H, p * p, kernel_size=1, bias=True)

    def forward(self, X):
        """X: [B, 5, 1, 64, 64] -> Y_hat: [B, 5, 1, 64, 64] (predictions of steps 5..9)."""
        B, T = X.shape[:2]
        p, H = self.cfg.patch, self.cfg.hidden
        side = X.shape[-1] // p
        z = X.new_zeros(B, H, side, side)  # states start at zero for every video
        h1, c1, h2, c2 = z, z, z, z
        preds = []
        u = None
        for tau in range(1, T + N_OUT_STEPS):  # tau = 1..9
            frame = X[:, tau - 1] if tau <= T else u          # own previous prediction from tau = 6 on
            folded = F.pixel_unshuffle(frame, p)               # [B, p*p, 16, 16]
            h1, c1 = self.cell1(folded, h1, c1)
            h2, c2 = self.cell2(h1, h2, c2)                    # layer 2 reads layer 1's *new* hidden state
            u = torch.sigmoid(F.pixel_shuffle(self.out(h2), p))  # x_hat_{tau+1}
            if tau >= T:
                preds.append(u)                                # keep the outputs of steps 5..9
        return torch.stack(preds, dim=1)


N_OUT_STEPS = 5  # forecast horizons; the rollout has T + 4 = 9 steps


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------------------ reach
def reach_in_pixels(cfg: Config):
    """Largest centre-to-centre distance (original pixels) information can travel in ONE step.

    Following the conventions of the handout: only spatial convolutions with a neighbourhood
    larger than 1x1 move information; one cell of the folded grid is `patch` original pixels
    (the patch width itself is not added); consider every path inside one step tau from the new
    input u_tau or from the previous states h1_{tau-1}, h2_{tau-1} to the prediction.

      * u_tau      -> conv1 (radius r1) -> conv2 (radius r2) -> 1x1 -> prediction : r1 + r2
      * h1_{tau-1} -> conv1 (recurrent half, r1) -> conv2 (r2)  -> 1x1            : r1 + r2
      * h2_{tau-1} -> conv2 (recurrent half, r2)                -> 1x1            : r2
    with radius r = dilation * (kernel - 1) / 2 cells.
    """
    r1 = cfg.dilations[0] * (cfg.kernels[0] - 1) // 2
    r2 = cfg.dilations[1] * (cfg.kernels[1] - 1) // 2
    cells = max(r1 + r2, r1 + r2, r2)
    return cells * cfg.patch
