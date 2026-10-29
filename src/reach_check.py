"""Numerical check of the one-step reach (Q2.3, Q2.12): which folded cells can change prediction x_hat_{tau+1}?

Uses autograd on ONE step of the network, from its three inputs (new input u_tau and the previous hidden
states h1_{tau-1}, h2_{tau-1}) to the prediction; the cell states enter only point-wise and add no spatial mixing.
"""
import sys, json
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import models as M

def measure(cfg):
    torch.manual_seed(0)
    net = M.ConvLSTMForecaster(cfg).double()
    p, H = cfg.patch, cfg.hidden
    side = 64 // p
    u = torch.rand(1, 1, 64, 64, dtype=torch.double, requires_grad=True)
    h1, h2 = (torch.randn(1, H, side, side, dtype=torch.double, requires_grad=True) for _ in range(2))
    c1, c2 = torch.randn(1, H, side, side, dtype=torch.double), torch.randn(1, H, side, side, dtype=torch.double)
    g1, _ = net.cell1(F.pixel_unshuffle(u, p), h1, c1)
    g2, _ = net.cell2(g1, h2, c2)
    pred = torch.sigmoid(F.pixel_shuffle(net.out(g2), p))
    cell = side // 2                                    # output pixel in the middle cell (row cell, col cell)
    r = cell * p + p // 2                               # a pixel inside that cell
    res = {}
    for name, var, scale in (("u_tau", u, p), ("h1_prev", h1, 1), ("h2_prev", h2, 1)):
        g, = torch.autograd.grad(pred[0, 0, r, r], var, retain_graph=True)
        nz = (g.abs() > 0)
        if name == "u_tau":                             # pixel -> cell index
            nz = F.max_pool2d(nz.double(), p) > 0
            nz = nz[0, 0]
        else:
            nz = nz[0].any(0)
        ys, xs = torch.nonzero(nz, as_tuple=True)
        res[name] = int(max((ys - cell).abs().max(), (xs - cell).abs().max()))
    cells = max(res.values())
    return dict(cells=res, max_cells=cells, reach_px=cells * p, analytic_px=M.reach_in_pixels(cfg))

if __name__ == "__main__":
    out = {}
    for name, cfg in (("A", M.Config()), ("D", M.Config(patch=4, hidden=64, kernels=(3, 3), dilations=(1, 3)))):
        out[name] = measure(cfg); print(name, out[name])
        assert out[name]["reach_px"] == out[name]["analytic_px"]
    Path("results").mkdir(exist_ok=True); json.dump(out, open("results/reach_check.json", "w"), indent=1)
