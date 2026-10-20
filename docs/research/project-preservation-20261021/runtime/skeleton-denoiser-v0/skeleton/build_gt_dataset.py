"""
Сборка датасета денойзера скелета v0: AVI + CSV тегов -> denoiser_v0_data.npz.

Пайплайн на сессию (стем, например tags_20261009_172221):
  1) YOLO 2D-кейпоинты по всем кадрам AVI (OV-экспорт, кэш <стем>_yolo.npz)
  2) 6DoF-позы тегов из CSV live-записи (subpix; кадры CSV и AVI идут 1:1,
     оба пишутся из одного цикла обработки — проверяем счётчики)
  3) спайк-фильтр тегов: изолированные скачки >0.3 м между соседними кадрами
  4) оффсеты тег->сустав: L1-робастный Гаусс-Ньютон по невязке проекции
     proj(tvec + R@off) ~= YOLO-кейпоинт   (per-session; рулетка не нужна)
  5) GT: 6 суставов ног (L/R hip, knee, ankle) в 3D, root-relative к тазу,
     нормировано на медианную дистанцию таз-бедро сессии -> 18 чисел
  6) eligibility: теги {0,2,4,7,11} видны и не спайк, длины костей в норме,
     YOLO-бёдра conf > 0.5

Выход: denoiser_v0_data.npz {
  cond (N,51)   вход денойзера: u/1280, v/720, conf x 17 кейпоинтов
  gt   (N,18)   6 суставов x xyz (root-relative, в единицах scale)
  t_s, frame, session_ids, val_mask, scales (м), joints (имена) }
Оффсеты и статы сессий -> denoiser_v0_offsets.json.

Запуск:
  .venv/Scripts/python build_gt_dataset.py tags_20261009_165738          # одна сессия
  .venv/Scripts/python build_gt_dataset.py tags_20261009_172221 tags_20261009_171539
  --force   перегнать YOLO-кэш   --val-frac 0.12
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from extract_pairs import CX, CY, FX, FY   # интринсики D435i color 1280x720

W, H = 1280, 720
TAG_IDS = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
JOINTS = ["L-hip", "R-hip", "L-knee", "R-knee", "L-ankle", "R-ankle"]
# COCO-17: 11/12 hips, 13/14 knees, 15/16 ankles
# (тег, имя сустава, кейпоинт YOLO для фита оффсета, конф-порог фита)
TAG_FIT = [(0, "L-ankle", 15), (2, "R-ankle", 16),
           (4, "L-knee", 13), (7, "R-knee", 14),
           (11, "L-hip", 11), (11, "R-hip", 12)]
# корень (таз) — тот же тег 11, фит к середине YOLO-бёдер
SPIKE_M = 0.30      # межкадровый скачок позиции тега = артефакт (м)
BONE_TOL_M = 0.12   # допуск длины кости от медианы сессии
HIP_CONF = 0.5      # минимум conf YOLO-бёдер для кадра
FIT_CONF = 0.6      # conf кейпоинта для участия в фите оффсета


# ---- YOLO-проход по AVI (кэшируется) ---------------------------------------------
def yolo_pass(avi: str, cache: str, force: bool = False, limit: int = 0):
    """-> xy (N,17,2) nan если нет, conf (N,17); кэш <стем>_yolo.npz."""
    if os.path.exists(cache) and not force:
        z = np.load(cache)
        return z["xy"], z["conf"]
    from ultralytics import YOLO
    model = None
    for cand in ("models/yolo11s-pose_openvino_model",
                 "models/yolo11s-pose.onnx", "models/yolo11s-pose.pt"):
        if os.path.exists(cand):
            model = YOLO(cand)
            print("YOLO бекенд:", cand)
            break
    if model is None:
        raise SystemExit("нет models/yolo11s-pose* — см. PIPELINE.md §3")

    cap = cv2.VideoCapture(avi)
    xy_all, cf_all = [], []
    t0, n = time.perf_counter(), 0
    while True:
        ok, frame = cap.read()
        if not ok or (limit and n >= limit):
            break
        xy = np.full((17, 2), np.nan, np.float32)
        cf = np.zeros(17, np.float32)
        res = model(frame, verbose=False, conf=0.25)
        if res and res[0].keypoints is not None and len(res[0].keypoints.xy):
            k = res[0].keypoints
            xy[:len(k.xy[0])] = k.xy[0].cpu().numpy()
            if k.conf is not None:
                cf[:len(k.conf[0])] = k.conf[0].cpu().numpy()
        xy_all.append(xy)
        cf_all.append(cf)
        n += 1
        if n % 1000 == 0:
            rate = n / (time.perf_counter() - t0)
            print(f"  yolo {n} кадров, {rate:.1f} fps", flush=True)
    cap.release()
    xy, cf = np.stack(xy_all), np.stack(cf_all)
    np.savez_compressed(cache, xy=xy, conf=cf)
    rate = n / max(time.perf_counter() - t0, 1e-9)
    print(f"  yolo: {n} кадров за {n/rate:.0f} с ({rate:.1f} fps), кэш {cache}")
    return xy, cf


# ---- CSV тегов -> матрицы по кадрам ----------------------------------------------
def load_tags(csv: str):
    """-> t (n,), tv/rv (n,K,3), seen (n,K) в порядке TAG_IDS."""
    d = np.atleast_1d(np.genfromtxt(csv, delimiter=",", names=True))
    tvals = np.unique(np.round(d["t_s"], 3))
    idx = {v: k for k, v in enumerate(tvals)}
    n = len(tvals)
    tv = np.full((n, len(TAG_IDS), 3), np.nan)
    rv = np.full((n, len(TAG_IDS), 3), np.nan)
    seen = np.zeros((n, len(TAG_IDS)), bool)
    for row in d:
        try:
            j = TAG_IDS.index(int(row["id"]))
        except ValueError:
            continue
        if row["seen"]:
            k = idx[np.round(float(row["t_s"]), 3)]
            tv[k, j] = [row["tx"], row["ty"], row["tz"]]
            rv[k, j] = [row["rx"], row["ry"], row["rz"]]
            seen[k, j] = True
    return tvals, tv, rv, seen


# ---- спайк-фильтр -----------------------------------------------------------------
def spike_bad(seen: np.ndarray, tv: np.ndarray) -> np.ndarray:
    """bad (n,K): изолированный скачок >SPIKE_M между соседними кадрами (dt=1)."""
    bad = np.zeros_like(seen)
    for j in range(seen.shape[1]):
        f = np.flatnonzero(seen[:, j])
        for a in range(len(f) - 2):
            i0, i1, i2 = f[a], f[a + 1], f[a + 2]
            if i1 - i0 == 1 and i2 - i1 == 1:
                d1 = np.linalg.norm(tv[i1, j] - tv[i0, j])
                d2 = np.linalg.norm(tv[i2, j] - tv[i1, j])
                if d1 > SPIKE_M and d2 > SPIKE_M:   # туда-обратно = артефакт
                    bad[i0:i2 + 1, j] = True
    return bad


# ---- L1-робастный фит оффсета тег->сустав (Гаусс-Ньютон, IRLS) --------------------
def proj_px(p3: np.ndarray) -> np.ndarray:
    """(N,3) -> (N,2) проекция пикселей."""
    return np.stack([FX * p3[:, 0] / p3[:, 2] + CX,
                     FY * p3[:, 1] / p3[:, 2] + CY], axis=1)


def fit_offset(tv, rv, px):
    """off (3,): argmin sum |proj(tvec + R@off) - px|_2, L1-робастно.

    tv (N,3), rv (N,3), px (N,2). Инициализация — линейный МНК по
    депроекциям кейпоинтов на глубину тега; дальше 8 итераций Гаусса-Ньютона
    с весами IRLS w=1/|r| (пол 2 px).
    """
    R = np.stack([cv2.Rodrigues(r)[0] for r in rv])              # (N,3,3)
    z = np.clip(tv[:, 2], 0.5, 6.0)
    p0 = np.stack([(px[:, 0] - CX) / FX * z, (px[:, 1] - CY) / FY * z, z], 1)
    d = p0 - tv                                                  # ~R@off
    off = np.linalg.solve(np.einsum("nij,nik->jk", R, R),
                          np.einsum("nij,ni->j", R, d))
    for _ in range(8):
        p3 = tv + np.einsum("nij,j->ni", R, off)
        r = proj_px(p3) - px                                      # (N,2)
        w = 1.0 / np.maximum(np.linalg.norm(r, axis=1), 2.0)      # L1-IRLS
        # якобиан d proj/d off = dproj/dp3 @ R   (2x3 на кадр)
        J = np.empty((len(tv), 2, 3))
        J[:, 0, 0], J[:, 0, 1], J[:, 0, 2] = FX / p3[:, 2], 0.0, -FX * p3[:, 0] / p3[:, 2] ** 2
        J[:, 1, 0], J[:, 1, 1], J[:, 1, 2] = 0.0, FY / p3[:, 2], -FY * p3[:, 1] / p3[:, 2] ** 2
        J = J @ R
        A = np.einsum("nij,n,nik->jk", J, w, J) + 1e-6 * np.eye(3)  # sum J^T W J
        g = np.einsum("nij,ni->j", J, r * w[:, None])               # sum J^T W r
        off = off - np.linalg.solve(A, g)
    p3 = tv + np.einsum("nij,j->ni", R, off)
    res = np.linalg.norm(proj_px(p3) - px, axis=1)
    return off, res


# ---- сборка одной сессии -----------------------------------------------------------
def build_session(stem: str, val_frac: float, force: bool, limit: int = 0):
    avi, csv = stem + ".avi", stem + ".csv"
    cache = stem + ("_yolo.npz" if not limit else f"_yolo{limit}.npz")
    xy, cf = yolo_pass(avi, cache, force, limit)
    t, tv, rv, seen = load_tags(csv)
    if limit:
        t, tv, rv, seen = (a[:limit] for a in (t, tv, rv, seen))
    n_vid = len(xy)
    if n_vid != len(t):
        raise SystemExit(f"{stem}: CSV-кадров {len(t)} != AVI-кадров {n_vid} — "
                         "тайминг не сшивается, сессию пропускаю (см. обзор 171539)")
    bad = spike_bad(seen, tv)
    print(f"{stem}: {n_vid} кадров, спайк-пометок {bad.sum()}")

    # ---- оффсеты: тег -> сустав (в системе маркера) ----
    ok_hips = (cf[:, 11] > FIT_CONF) & (cf[:, 12] > FIT_CONF)
    offsets, stats = {}, {}
    for tid, joint, kpt in TAG_FIT:
        j = TAG_IDS.index(tid)
        m = seen[:, j] & ~bad[:, j] & (cf[:, kpt] > FIT_CONF) & np.isfinite(xy[:, kpt, 0])
        if m.sum() < 200:
            raise SystemExit(f"{stem}: тег {tid} всего {m.sum()} пригодных кадров для фита — мало")
        off, res = fit_offset(tv[m, j], rv[m, j], xy[m, kpt])
        offsets[f"tag{tid}->{joint}"] = off.tolist()
        stats[f"tag{tid}->{joint}"] = dict(n=int(m.sum()),
                                           med_px=float(np.median(res)),
                                           p90_px=float(np.percentile(res, 90)))
        print(f"  off tag{tid}->{joint}: n={m.sum()}, |r| мед {np.median(res):.1f} px, "
              f"p90 {np.percentile(res, 90):.1f} px, off={np.round(off, 3).tolist()}")
    # таз: тег 11 к середине YOLO-бёдер
    j11 = TAG_IDS.index(11)
    m = seen[:, j11] & ~bad[:, j11] & ok_hips
    mid = (xy[:, 11] + xy[:, 12]) / 2
    off, res = fit_offset(tv[m, j11], rv[m, j11], mid[m])
    offsets["tag11->pelvis"] = off.tolist()
    stats["tag11->pelvis"] = dict(n=int(m.sum()), med_px=float(np.median(res)),
                                  p90_px=float(np.percentile(res, 90)))
    print(f"  off tag11->pelvis: n={m.sum()}, |r| мед {np.median(res):.1f} px")

    # ---- GT-суставы 3D (фрейм камеры) ----
    def joint3d(k_frame, tid_key):
        tid = int(tid_key.split("->")[0][3:])
        j = TAG_IDS.index(tid)
        R = cv2.Rodrigues(rv[k_frame, j])[0]
        return tv[k_frame, j] + R @ np.array(offsets[tid_key])

    need = [TAG_IDS.index(i) for i in (0, 2, 4, 7, 11)]
    elig = ~bad[:, need].any(1) & seen[:, need].all(1) & ok_hips
    f_idx = np.flatnonzero(elig)
    j3 = np.stack([[joint3d(k, key) for key in
                    ("tag11->L-hip", "tag11->R-hip", "tag4->L-knee", "tag7->R-knee",
                     "tag0->L-ankle", "tag2->R-ankle")] for k in f_idx])   # (n,6,3)
    root = np.stack([joint3d(k, "tag11->pelvis") for k in f_idx])
    scale = float(np.median(np.linalg.norm(j3[:, :2] - root[:, None], axis=2)))
    # гейт длин костей по медиане сессии
    bones = np.stack([np.linalg.norm(j3[:, 0] - j3[:, 2], axis=1),   # бедро L
                      np.linalg.norm(j3[:, 1] - j3[:, 3], axis=1),   # бедро R
                      np.linalg.norm(j3[:, 2] - j3[:, 4], axis=1),   # голень L
                      np.linalg.norm(j3[:, 3] - j3[:, 5], axis=1)], 1)
    bone_med = np.median(bones, 0)
    ok_bone = (np.abs(bones - bone_med) < BONE_TOL_M).all(1)
    keep = ok_bone
    f_idx, j3, root = f_idx[keep], j3[keep], root[keep]
    gt = ((j3 - root[:, None, :]) / scale).reshape(len(f_idx), -1)   # (n,18)

    cond = np.concatenate([np.nan_to_num(xy[f_idx, :, 0]) / W,
                           np.nan_to_num(xy[f_idx, :, 1]) / H,
                           cf[f_idx]], axis=1).astype(np.float32)   # (n,51)
    val = np.zeros(len(f_idx), bool)
    val[int(len(f_idx) * (1 - val_frac)):] = True
    print(f"  GT-кадров: {len(f_idx)} (из {elig.sum()} eligibility, "
          f"гейт костей снял {int((~ok_bone).sum())}), scale={scale:.3f} м, "
          f"кости мед [бедро {bone_med[0]:.2f}/{bone_med[1]:.2f}, "
          f"голень {bone_med[2]:.2f}/{bone_med[3]:.2f}] м, val {val.sum()}")
    return dict(cond=cond, gt=gt.astype(np.float32), t_s=t[f_idx], frame=f_idx,
                val=val, scale=scale, stats=stats, bone_med=bone_med.tolist(),
                offsets=offsets)


def main():
    ap = argparse.ArgumentParser(description="AVI+CSV -> denoiser_v0_data.npz")
    ap.add_argument("stems", nargs="+", help="стемы сессий (файлы <стем>.avi/.csv)")
    ap.add_argument("--out", default="denoiser_v0_data.npz")
    ap.add_argument("--force", action="store_true", help="перегнать YOLO-кэш")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--limit", type=int, default=0,
                    help="только первые N кадров (отладка; отдельный YOLO-кэш)")
    args = ap.parse_args()

    all_parts, meta = [], {}
    for s_id, stem in enumerate(args.stems):
        try:
            part = build_session(stem, args.val_frac, args.force, args.limit)
        except SystemExit as e:
            print(f"[!] {stem}: пропущена ({e})")
            continue
        n = len(part["gt"])
        all_parts.append(np.concatenate([part["cond"], part["gt"],
                                         part["t_s"][:, None],
                                         part["frame"][:, None],
                                         np.full((n, 1), s_id, np.float32),
                                         part["val"][:, None].astype(np.float32),
                                         np.full((n, 1), part["scale"], np.float32)],
                                        axis=1))
        meta[stem] = dict(scale=part["scale"], bone_med=part["bone_med"],
                          offsets=part["offsets"], stats=part["stats"],
                          n_frames=int(n), val=int(part["val"].sum()))
    data = np.concatenate(all_parts)
    np.savez_compressed(args.out, data=data.astype(np.float32),
                        cond_dim=51, gt_dim=18, joints=np.array(JOINTS))
    with open(args.out.rsplit(".", 1)[0] + "_offsets.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    n = len(data)
    print(f"\n{args.out}: {n} кадров (val {int(data[:, -1].sum())}), "
          f"сессий {len(args.stems)}: " + ", ".join(f"{k.split('_')[-1]}={v['n_frames']}"
                                                    for k, v in meta.items()))


if __name__ == "__main__":
    main()
