"""
Eval v1 на val-сплите (l2_dataset/v1_pairs.npz + models/denoiser_v1.pt).

Главные метрики:
  - MPJPE по 6 суставам (см) и по суставам
  - СОХРАНЕНИЕ АМПЛИТУДЫ: на кадрах с GT-сгибом колена >60° — медиана
    предсказанного сгиба (v0 там выдавал 12°; вот где v1 должен доказать себя)
  - std длин костей, джиттер между соседними val-кадрами
  - baseline: сырой фьюжн-вход (его ошибка = то, что модель должна чистить)

Методы: прямая голова, DDIM obs (старт с прямой, 10 шагов), DDIM prev.
Запуск: .venv/Scripts/python eval_denoiser_v1.py [--data ...] [--ckpt ...]
"""
import argparse

import numpy as np
import torch

from train_denoiser_v1 import DenoiserV1, T_WIN, cosine_schedule

SCALE = 0.153
T0_FRAC, STEPS = 0.3, 10


def knee_flex(j3, side):
    """j3 (...,6,3) м -> сгиб колена (0 = прямая)."""
    hip, knee, ank = (0, 2, 4) if side == "L" else (1, 3, 5)
    u = j3[..., hip, :] - j3[..., knee, :]
    v = j3[..., ank, :] - j3[..., knee, :]
    n = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
    a = np.degrees(np.arccos(np.clip((u * v).sum(-1) / n, -1, 1)))
    return 180 - a


@torch.no_grad()
def ddim_obs(model, win, abar, steps=STEPS, t0_frac=T0_FRAC):
    T = len(abar) - 1
    t0 = int(t0_frac * T)
    x0 = model.x0_direct(win)
    rng = np.random.default_rng(0)
    a = abar[t0]
    e = torch.tensor(rng.normal(size=x0.shape), dtype=torch.float32)
    x = a.sqrt() * x0 + (1 - a).clamp_min(1e-6).sqrt() * e
    ts = np.linspace(t0, 1, steps).astype(int)
    for i, t in enumerate(ts):
        ab_t = abar[t]
        x0 = model(x, torch.full((len(win),), t, dtype=torch.float32), win)[0].clamp(-8, 8)
        e_hat = (x - ab_t.sqrt() * x0) / (1 - ab_t).clamp_min(1e-6).sqrt()
        ab_next = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        x = ab_next.sqrt() * x0 + (1 - ab_next).clamp_min(1e-6).sqrt() * e_hat
    return x0


def metrics(pred, gt, frames):
    p = pred.reshape(-1, 6, 3) * SCALE
    g = gt.reshape(-1, 6, 3) * SCALE
    err = np.linalg.norm(p - g, axis=2)
    out = {"mpjpe": 100 * err.mean()}
    # амплитуда: кадры с GT-сгибом >60
    for side in ("L", "R"):
        kg = knee_flex(g, side)
        kp = knee_flex(p, side)
        m = kg > 60
        out[f"amp_{side}"] = (int(m.sum()), float(np.median(kp[m])) if m.any() else np.nan,
                              float(np.median(kg[m])) if m.any() else np.nan)
    bones = np.stack([np.linalg.norm(p[:, a] - p[:, b], axis=1)
                      for a, b in ((0, 2), (1, 3), (2, 4), (3, 5))], 1)
    out["bone_std"] = 100 * bones.std(0).mean()
    d = np.linalg.norm(p[1:] - p[:-1], axis=2)
    out["jitter"] = 100 * np.median(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="l2_dataset/v1_pairs.npz")
    ap.add_argument("--ckpt", default="models/denoiser_v1.pt")
    a = ap.parse_args()
    z = np.load(a.data, allow_pickle=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = DenoiserV1()
    model.load_state_dict(ck["model"])
    model.eval()
    abar = ck["abar"]

    val = z["val"] > 0.5
    X = torch.tensor(z["X"][val])
    Y = z["Y"][val]
    fidx = z["frame_idx"][val]

    with torch.no_grad():
        direct = model.x0_direct(X).numpy()
    obs = ddim_obs(model, X, abar).numpy()

    # baseline: сырой фьюжн (суставы 11-16 из фич окна, текущий кадр)
    raw = np.stack([z["X"][k][:, :51].reshape(T_WIN, 17, 3)[-1][[11, 12, 13, 14, 15, 16]].reshape(-1)
                    for k in np.flatnonzero(val)])
    raw = raw.reshape(len(raw), 6, 3)
    have = np.isfinite(raw).all(-1).all(-1)
    print(f"val: {len(Y)} окон; фьюжн-базлайн доступен на {have.sum()} из них\n")

    rows = []
    for name, p in (("прямая", direct), ("diffusion obs/10", obs)):
        rows.append((name, metrics(p, Y, fidx)))
    g = metrics(Y, Y, fidx)
    print(f"{'метод':>18} {'MPJPE см':>9} {'std костей см':>13} {'джиттер см':>11}")
    print(f"{'GT (референс)':>18} {0.0:>9.1f} {g['bone_std']:>13.2f} {g['jitter']:>11.2f}")
    for name, m in rows:
        print(f"{name:>18} {m['mpjpe']:>9.2f} {m['bone_std']:>13.2f} {m['jitter']:>11.2f}")
    if have.any():
        mb = metrics(raw[have].reshape(len(raw[have]), 18), Y[have], fidx[have])
        print(f"{'сырой фьюжн (до)':>18} {mb['mpjpe']:>9.2f} {mb['bone_std']:>13.2f} {mb['jitter']:>11.2f}")

    print("\nСОХРАНЕНИЕ АМПЛИТУДЫ (кадры с GT-сгибом >60°):")
    for side in ("L", "R"):
        n, pd, gd = rows[0][1][f"amp_{side}"], None, None
        _, pd_dir, _ = rows[0][1][f"amp_{side}"]
        _, pd_obs, _ = rows[1][1][f"amp_{side}"]
        print(f"  колено {side}: кадров {n}; GT мед {gd if gd else rows[0][1][f'amp_{side}'][2]:.0f}°; "
              f"прямая {pd_dir:.0f}°; obs/10 {pd_obs:.0f}°")


if __name__ == "__main__":
    main()
