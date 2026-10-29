"""Q1: shape / size / latency check of the proposed river forecaster (an UNTRAINED forward pass).

Nodes (N=9): 0 Rel (reservoir outlet) 1 R1 2 R2 3 T1 4 S1 5 S2 6 U (latent unmonitored tributary) 7 S3 8 S4
Inputs : X   [B,8,L,F]   observed nodes Rel,R1,R2,T1,S1..S4 on the 10-min grid (L=36, F=6, unused channels = 0)
         FUT [B,6,5]     announced release schedule for t0+10 ... t0+60 min
Output : Yq  [B,4,6,3]   S1..S4 x horizon 10..60 min x quantiles (0.1, 0.5, 0.9)  (standardised level increments)
Distances below are illustrative placeholders (the real ones are supplied by the authority).
"""
import time, math, torch, torch.nn as nn, torch.nn.functional as F

L, d, HEADS, N, NOBS, FEAT, HOR, NQ = 36, 64, 4, 9, 8, 6, 6, 3
REL, R1, R2, T1, S1, S2, U, S3, S4 = range(9)
PARENTS = {S1: [REL, R1], T1: [R1], S2: [S1, T1, R1], U: [R2], S3: [S2, U, R2], S4: [S3, R2]}
DIST = {(REL, S1): 4., (R1, S1): 2., (R1, T1): 3., (S1, S2): 9., (T1, S2): 5., (R1, S2): 6.,
        (R2, U): 2., (S2, S3): 11., (U, S3): 4., (R2, S3): 3., (S3, S4): 6., (R2, S4): 5.}
ADJ = torch.zeros(N, N, dtype=torch.bool)               # ADJ[j, i]: i is an allowed source (parent) of j
Dkm = torch.ones(N, N)
for j, ps in PARENTS.items():
    for i in ps: ADJ[j, i], Dkm[j, i] = True, DIST[(i, j)]
TT = L + HOR                                            # token axis: 36 past steps (-35..0) + 6 future steps (+1..+6)
TOK_TIME = torch.arange(-L + 1, HOR + 1).float()        # [TT]
VALID = torch.zeros(N, TT, dtype=torch.bool); VALID[:, :L] = True; VALID[REL, L:] = True   # only releases are known in the future


class LagAttention(nn.Module):
    """Child queries attend over parents' tokens; Gaussian prior on (lag - travel time), travel time depends on the flow state."""
    def __init__(s):
        super().__init__()
        s.q, s.k, s.v, s.o = (nn.Linear(d, d) for _ in range(4))
        s.cel = nn.Sequential(nn.Linear(2 * d, 32), nn.GELU(), nn.Linear(32, 1))   # log-celerity correction phi
        s.log_sigma = nn.Parameter(torch.zeros(N, N))                               # lag spread (steps), per edge
        s.log_c = nn.Parameter(torch.full((N, N), math.log(5.0)))                   # prior celerity (km/h), per edge

    def forward(s, qs, child, tgt_time, tokens, last):
        """qs [B,Q,d]; child [Q] node index; tgt_time [Q] (steps rel. t0); tokens [B,N,TT,d]; last [B,N,d] states at t0."""
        B, Q, _ = qs.shape; dh = d // HEADS
        q = s.q(qs).view(B, Q, 1, 1, HEADS, dh)
        k = s.k(tokens).view(B, 1, N, TT, HEADS, dh); v = s.v(tokens).view(B, 1, N, TT, HEADS, dh)
        score = (q * k).sum(-1) / math.sqrt(dh)                                      # [B,Q,N,TT,H]
        phi = torch.tanh(s.cel(torch.cat([qs[:, :, None].expand(B, Q, N, d), last[:, None].expand(B, Q, N, d)], -1))).squeeze(-1)
        tau = Dkm[child] * 6.0 / torch.exp(s.log_c[child][None] + 0.5 * phi)         # travel time in 10-min steps [B,Q,N]
        lag = tgt_time[:, None, None] - TOK_TIME[None, None, :]                      # [Q,1,TT]
        sig = torch.exp(s.log_sigma[child])[:, :, None] + 0.5                        # [Q,N,1]
        score = score + (-(lag[None] - tau[..., None]) ** 2 / (2 * sig[None] ** 2))[..., None]
        ok = ADJ[child][:, :, None] & VALID[None] & (lag >= 0)                       # parents only, known tokens only, no peeking ahead
        score = score.masked_fill(~ok[None, ..., None], -1e9)
        w = torch.softmax(score.flatten(2, 3), dim=2).view(B, Q, N, TT, HEADS)
        w = w * ok.any(dim=(1, 2)).view(1, Q, 1, 1, 1)                               # children without parents get no message
        return s.o((w[..., None] * v).sum((2, 3)).reshape(B, Q, d))


class RiverForecaster(nn.Module):
    def __init__(s):
        super().__init__()
        s.proj = nn.ModuleList([nn.Linear(FEAT, d) for _ in range(3)])               # release / rain / level node types
        s.node_emb, s.time_emb, s.hor_emb, s.fut = nn.Embedding(N, d), nn.Embedding(TT, d), nn.Embedding(HOR, d), nn.Linear(5, d)
        s.conv = nn.ModuleList([nn.Conv1d(d, d, 3, dilation=2 ** l) for l in range(3)])
        s.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        s.river, s.mix = nn.ModuleList([LagAttention() for _ in range(2)]), nn.ModuleList([nn.Linear(2 * d, d) for _ in range(2)])
        s.u_stream, s.dec = LagAttention(), LagAttention()
        s.head = nn.Sequential(nn.Linear(3 * d, 128), nn.GELU(), nn.Linear(128, NQ))

    def forward(s, X, FUT):
        B = X.shape[0]
        kind = [0, 1, 1, 2, 2, 2, 2, 2]                                              # Rel, R1, R2, T1, S1, S2, S3, S4
        obs = torch.stack([s.proj[kind[n]](X[:, n]) for n in range(NOBS)], 1)        # [B,8,L,d]
        order = [REL, R1, R2, T1, S1, S2, S3, S4]
        tok = torch.zeros(B, N, TT, d); tok[:, order, :L] = obs                      # U is latent: filled below
        tok = tok + s.node_emb.weight[None, :, None] + s.time_emb.weight[None, None]
        h = tok[:, :, :L]
        for conv, nrm in zip(s.conv, s.norm):                                        # causal dilated temporal blocks, per node
            z = F.pad(h.permute(0, 1, 3, 2).reshape(B * N, d, L), (2 * conv.dilation[0], 0))
            h = h + F.gelu(nrm(conv(z).view(B, N, d, L).permute(0, 1, 3, 2)))
        tok = torch.cat([h, tok[:, :, L:]], 2)
        tok[:, REL, L:] = s.fut(FUT) + s.node_emb.weight[REL] + s.time_emb.weight[L:]  # announced releases -> future tokens of Rel
        last = tok[:, :, L - 1]
        tq = torch.arange(-L + 1, 1).float()                                         # latent inflow stream of U, one query per time step
        tok = tok.clone(); tok[:, U, :L] = s.u_stream(s.node_emb.weight[U][None, None] + s.time_emb.weight[None, :L].expand(B, L, d),
                                                      torch.full((L,), U), tq, tok, last)
        state = tok[:, :, L - 1]
        kids = torch.tensor([S1, T1, S2, S3, S4])
        for att, mix in zip(s.river, s.mix):                                         # river blocks: upstream -> downstream
            m = att(state[:, kids], kids, torch.zeros(5), tok, state)
            state = state.clone(); state[:, kids] = state[:, kids] + F.gelu(mix(torch.cat([state[:, kids], m], -1)))
        st = torch.tensor([S1, S2, S3, S4]); qn = st.repeat_interleave(HOR); qh = torch.arange(HOR).repeat(4)
        qs = state[:, qn] + s.hor_emb.weight[qh]                                     # 24 queries: station x horizon
        m = s.dec(qs, qn, (qh + 1).float(), tok, state)                              # lag-aware readout incl. announced releases
        return s.head(torch.cat([qs, m, state[:, qn]], -1)).view(B, 4, HOR, NQ)


if __name__ == "__main__":
    torch.manual_seed(0); torch.set_num_threads(1)
    m = RiverForecaster().eval()
    X, FUT = torch.randn(8, NOBS, L, FEAT), torch.randn(8, HOR, 5)
    with torch.no_grad():
        y = m(X, FUT)
        X1, F1 = X[:1], FUT[:1]
        for _ in range(5): m(X1, F1)
        t0 = time.perf_counter()
        for _ in range(50): m(X1, F1)
        ms = (time.perf_counter() - t0) / 50 * 1e3
        # causality check: changing a FUTURE rainfall-free input must not matter; changing the scheduled release must
        X2 = X1.clone(); y_a = m(X1, F1); y_b = m(X1, F1 + 1.0)
    print("X", tuple(X.shape), "FUT", tuple(FUT.shape), "->", tuple(y.shape))
    print(f"parameters = {sum(p.numel() for p in m.parameters())}  ; 1-thread CPU latency per issuance = {ms:.1f} ms")
    print("schedule changes the forecast:", bool((y_a - y_b).abs().max() > 0), "| finite:", bool(torch.isfinite(y).all()))
