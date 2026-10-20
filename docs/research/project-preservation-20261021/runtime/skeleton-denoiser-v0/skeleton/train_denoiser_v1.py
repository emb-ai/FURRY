"""
Обучение v1: шумный depth-фьюжн скелет (окно T=9) -> чистые суставы ног.

Отличия от v0 (train_denoiser_v0.py):
  - вход: окно из 9 кадров по 153 фичи (фьюжн xyz root-relative + presence +
    src-флаги + 2D + conf), а не один вектор 2D-кейпоинтов;
  - энкодер: per-frame MLP -> causal TCN (dilation 1,4);
  - + conf-голова: per-joint log-сигма, гетероскедастичный NLL на прямой голове
    (лечит «conf всегда 0.9» из v0);
  - + штраф длин костей (антомический приор);
  - аугментация входа: измеренный шум фьюжна (sigma из v1_pairs.npz).

Головы те же: прямая (x0_direct) + диффузионная (x0, FiLM от t, DDIM-5,
старты obs/prev как в v0). Чекпоинт: models/denoiser_v1.pt.

Запуск: .venv/Scripts/python train_denoiser_v1.py --epochs 300
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

T_WIN, FEAT_DIM, X_DIM = 9, 153, 18
BONES = [(0, 2, 0.42), (1, 3, 0.41), (2, 4, 0.32), (3, 5, 0.31)]  # idx_a, idx_b, дл. м


def cosine_schedule(T, s=0.008):
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return torch.tensor(np.clip(f / f[0], 1e-5, 1.0), dtype=torch.float32)


class CausalConv(nn.Module):
    def __init__(self, ch, dil):
        super().__init__()
        self.pad = dil * 4                 # kernel 5: левый пад = (k-1)*dil
        self.conv = nn.Conv1d(ch, ch, 5, dilation=dil)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):                       # (B,T,ch)
        y = self.conv(nn.functional.pad(x, (self.pad, 0)))
        return torch.relu(x + y)


class DenoiserV1(nn.Module):
    def __init__(self, hidden=512, t_dim=128):
        super().__init__()
        self.frame_enc = nn.Sequential(nn.Linear(FEAT_DIM, 256), nn.ReLU(),
                                       nn.Linear(256, 256), nn.ReLU())
        self.x_enc = nn.Sequential(nn.Linear(X_DIM, 256), nn.ReLU())
        self.tcn = nn.Sequential(CausalConv(256, 1), CausalConv(256, 4))
        self.t_mlp = nn.Sequential(nn.Linear(64, t_dim), nn.ReLU(), nn.Linear(t_dim, t_dim))
        self.inp = nn.Linear(256 * 3, hidden)        # окно-контекст + кадр + x_t
        self.film = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for _ in range(4):
            self.blocks.append(nn.Sequential(nn.Linear(hidden, hidden),
                                             nn.ReLU(), nn.Linear(hidden, hidden)))
            self.film.append(nn.Linear(t_dim, 2 * hidden))
        self.head = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, X_DIM))
        self.direct = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, X_DIM))
        self.conf = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, X_DIM))

    @staticmethod
    def t_embed(t):
        half = 32
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        ang = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], -1)

    def forward(self, x, t, win):
        """x (B,18), t (B,), win (B,T,153) -> (x0, x0_direct, log_sigma)."""
        h = self.frame_enc(win)                       # (B,T,256)
        ctx = self.tcn(h.transpose(1, 2)).transpose(1, 2)
        g = torch.cat([ctx[:, -1], h[:, -1], self.x_enc(x)], dim=-1)
        e = torch.relu(self.inp(g))
        te = self.t_mlp(self.t_embed(t))
        for blk, fl in zip(self.blocks, self.film):
            sc, sh = fl(te).chunk(2, -1)
            e = torch.relu(blk(e) + e) * (1 + torch.tanh(sc)) + sh
        return self.head(e), self.direct(h[:, -1]), self.conf(h[:, -1]).clamp(-6, 4)

    def x0_direct(self, win):
        h = self.frame_enc(win)
        return self.direct(h[:, -1])


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def copy_to(self, model):
        model.load_state_dict({k: v.clone() for k, v in self.shadow.items()})


def bone_loss(x0):
    """Штраф отклонения длин костей (в единицах scale=0.153 м)."""
    pen = 0.0
    for a, b, L in BONES:
        d = torch.linalg.norm(x0[:, 3*a:3*a+3] - x0[:, 3*b:3*b+3], dim=1)
        pen = pen + (d - L / 0.153).abs().mean()
    return pen / len(BONES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="l2_dataset/v1_pairs.npz")
    ap.add_argument("--out", default="models/denoiser_v1.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--flex-weight", type=float, default=1.0,
                    help="доп. вес кадрам с GT-сгибом колена >45° (v1.2)")
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() - 1))
    torch.manual_seed(0)
    z = np.load(args.data, allow_pickle=True)
    X = torch.tensor(z["X"])                        # (N,9,153)
    Y = torch.tensor(z["Y"])                        # (N,18)
    val = torch.tensor(z["val"] > 0.5)
    sigma = torch.tensor(z["noise_sigma"], dtype=torch.float32)     # (3,)
    idx_tr = torch.tensor(np.flatnonzero(~val.numpy()))
    print(f"данные: {len(Y)} окон (train {len(idx_tr)}, val {int(val.sum())}), "
          f"шум фьюжна {np.round(z['noise_sigma']*100,1)} см")

    model = DenoiserV1()
    print(f"модель: {sum(p.numel() for p in model.parameters())/1e6:.2f}M параметров, T={args.T}")
    abar = cosine_schedule(args.T)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = EMA(model)
    huber = nn.SmoothL1Loss(beta=0.5)

    hist, t0 = [], time.perf_counter()
    for ep in range(args.epochs):
        perm = idx_tr[torch.randperm(len(idx_tr))]
        tot = 0.0
        for a in range(0, len(perm), args.batch):
            b = perm[a:a + args.batch]
            win, y0 = X[b], Y[b]
            # аугментация: шум фьюжна на xyz-часть фич (первые 51 из 153)
            if torch.rand(()) < 0.5:
                w = win.clone()
                amp = 0.5 + torch.rand(())          # 0.5x..1.5x измеренного шума
                n = torch.randn(w.shape[0], T_WIN, 17, 3) * sigma[None, None, None, :] * amp
                w[..., :51] = w[..., :51] + n.reshape(w.shape[0], T_WIN, 51)
                win = w
            t = torch.randint(1, args.T + 1, (len(b),))
            e = torch.randn_like(y0)
            at = abar[t][:, None]
            x_t = at.sqrt() * y0 + (1 - at).clamp_min(1e-6).sqrt() * e
            x0, x0d, log_s = model(x_t, t, win)
            nll = (log_s + 0.5 * ((x0d - y0) / log_s.exp()).pow(2)).mean()
            if args.flex_weight > 1.0:
                # v1.2: вес кадрам, где GT-сгиб колена >45° (по любой ноге)
                with torch.no_grad():
                    j3 = y0.reshape(-1, 6, 3)
                    deep = torch.zeros(len(y0), device=y0.device)
                    for h, k, a in ((0, 2, 4), (1, 3, 5)):
                        u = j3[:, h] - j3[:, k]
                        v = j3[:, a] - j3[:, k]
                        cos = (u * v).sum(1) / (u.norm(dim=1) * v.norm(dim=1) + 1e-9)
                        flex = torch.pi - torch.acos(cos.clamp(-1, 1))   # 0 = прямая
                        deep = torch.maximum(deep, flex)
                    w = 1.0 + (args.flex_weight - 1.0) * (deep > torch.deg2rad(torch.tensor(45.0))).float()
                per = (x0 - y0).abs().sum(1) / X_DIM                      # робастный пер-сэмпл
                l_huber = (per * w).mean()
            else:
                l_huber = huber(x0, y0)
            loss = (l_huber + 0.3 * huber(x0d, y0)
                    + 0.1 * nll + 0.05 * bone_loss(x0))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            tot += float(loss) * len(b)
        hist.append(tot / len(perm))
        if (ep + 1) % 25 == 0 or ep == 0:
            rate = (ep + 1) / (time.perf_counter() - t0)
            print(f"эпоха {ep+1}/{args.epochs}, лосс {hist[-1]:.4f}, {rate:.2f} эп/с, "
                  f"ETA {(args.epochs-ep-1)/rate/60:.0f} мин", flush=True)

    ema.copy_to(model)
    torch.save(dict(model=model.state_dict(), config=vars(args), abar=abar,
                    loss_hist=hist, t_win=T_WIN, feat_dim=FEAT_DIM,
                    bone_med=z["bone_med"], noise_sigma=z["noise_sigma"]), args.out)
    print(f"сохранено: {args.out} (лосс {hist[0]:.4f} -> {hist[-1]:.4f})")


if __name__ == "__main__":
    main()
