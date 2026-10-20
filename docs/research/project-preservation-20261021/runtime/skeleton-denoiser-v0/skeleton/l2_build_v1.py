"""
Сборка v1-пар из l2_extract-датасета: (окно фьюжн-входа) -> (GT-18).

Шаги:
  1) спайк-фильтр тегов (как build_gt_dataset)
  2) оффсеты тег->сустав: L1-Гаусс-Ньютон по проекции vs YOLO-2D
     (0->L-ankle, 2|9->R-ankle — пятка страхует стопу на приседах,
      4->L-knee, 7->R-knee, 11->hips/pelvis)
  3) GT: 6 суставов root-relative / scale, eligibility + гейт длин костей
  4) вход: фьюжн-скелет (root-relative/scale + presence + src-flags)
     + 2D-кейпоинты + conf, окно T=9 (паддинг первым валидным)
  5) split: последние 12% времени -> val
  6) статистика шума фьюжна vs GT (для аугментаций) -> в npz

Запуск: .venv/Scripts/python l2_build_v1.py [l2_dataset/l2_session2.npz]
Выход: l2_dataset/v1_pairs.npz
"""
import sys

import cv2
import numpy as np

from build_gt_dataset import fit_offset, proj_px  # L1-Гаусс-Ньютон, проекция

T_WIN = 9
VAL_FRAC = 0.12
VAL_MODE = "tail"        # "tail" | "stride" (перекрыть --val-mode)
BONE_TOL_M = 0.12
SPIKE_M = 0.30
COCO_LEG = [11, 12, 13, 14, 15, 16]          # L-hip..R-ankle (как в v0)
JOINTS = ["L-hip", "R-hip", "L-knee", "R-knee", "L-ankle", "R-ankle"]
TAG_IDS = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# (тег, сустав, кейпоинт YOLO для фита)
TAG_FIT = [(0, "L-ankle", 15), (2, "R-ankle", 16), (9, "R-ankle", 16),
           (4, "L-knee", 13), (7, "R-knee", 14), (11, "L-hip", 11),
           (11, "R-hip", 12)]
FIT_CONF = 0.6


def spike_bad(seen, tv):
    bad = np.zeros_like(seen)
    for j in range(seen.shape[1]):
        f = np.flatnonzero(seen[:, j])
        for a in range(len(f) - 2):
            i0, i1, i2 = f[a], f[a + 1], f[a + 2]
            if i1 - i0 == 1 and i2 - i1 == 1:
                d1 = np.linalg.norm(tv[i1, j] - tv[i0, j])
                d2 = np.linalg.norm(tv[i2, j] - tv[i1, j])
                if d1 > SPIKE_M and d2 > SPIKE_M:
                    bad[i0:i2 + 1, j] = True
    return bad


def main(path="l2_dataset/l2_session2.npz", val_mode=None, out="l2_dataset/v1_pairs.npz"):
    global VAL_MODE
    if val_mode:
        VAL_MODE = val_mode
    z = np.load(path, allow_pickle=True)
    n = len(z["t"])
    fx, fy, cx, cy = (float(z[k]) for k in ("fx", "fy", "cx", "cy"))
    k2d, c2d = z["kpts2d"], z["conf2d"]
    fus, fsrc = z["fusion_xyz"], z["fusion_src"]
    tv, rv, seen, px = z["tag_tvec"], z["tag_rvec"], z["tag_seen"], z["tag_px"]
    print(f"{path}: {n} кадров")

    bad = spike_bad(seen, tv)
    print(f"спайк-пометок: {bad.sum()}")

    # ---- оффсеты ----
    ok_hips = (c2d[:, 11] > FIT_CONF) & (c2d[:, 12] > FIT_CONF)
    mid = np.nan_to_num((k2d[:, 11] + k2d[:, 12]) / 2)
    offsets = {}
    for tid, joint, kpt in TAG_FIT:
        j = TAG_IDS.index(tid)
        m = seen[:, j] & ~bad[:, j] & (c2d[:, kpt] > FIT_CONF) & np.isfinite(k2d[:, kpt, 0])
        if m.sum() < 200:
            print(f"  [!] tag{tid}->{joint}: всего {m.sum()} кадров — пропуск")
            continue
        off, res = fit_offset(tv[m, j], rv[m, j], k2d[m, kpt])
        key = f"tag{tid}->{joint}"
        offsets[key] = off
        print(f"  off {key}: n={m.sum()}, |r| мед {np.median(res):.1f} px, "
              f"p90 {np.percentile(res, 90):.1f} px")
    j11 = TAG_IDS.index(11)
    m = seen[:, j11] & ~bad[:, j11] & ok_hips
    off, res = fit_offset(tv[m, j11], rv[m, j11], mid[m])
    offsets["tag11->pelvis"] = off
    print(f"  off tag11->pelvis: n={m.sum()}, |r| мед {np.median(res):.1f} px")

    def joint3d(k, key):
        j = TAG_IDS.index(int(key.split("->")[0][3:]))
        R = cv2.Rodrigues(rv[k, j])[0]
        return tv[k, j] + R @ offsets[key]

    # калибровка длин бёдер по кадрам с tag11: hip11 против колена бедренного тега
    j4, j7, j11 = TAG_IDS.index(4), TAG_IDS.index(7), TAG_IDS.index(11)
    cal = (seen[:, j11] & ~bad[:, j11] & seen[:, j4] & ~bad[:, j4]
           & seen[:, j7] & ~bad[:, j7])
    thigh_len = {}
    for side, jt in (("L", j4), ("R", j7)):
        d = np.linalg.norm(
            np.stack([joint3d(k, f"tag11->{side}-hip") - joint3d(k, f"tag{4 if side=='L' else 7}->{side}-knee")
                      for k in np.flatnonzero(cal)]), axis=1)
        thigh_len[side] = float(np.median(d))
    print(f"длины бёдер (калибровка tag11): L {thigh_len['L']:.3f} м, R {thigh_len['R']:.3f} м")

    # ---- GT-суставы и eligibility ----
    # бёдра: от tag11 (точнее), иначе реконструкция из бедренных тегов:
    # hip = knee + ось_бедра * длина_бедра; root = середина бёдер (одна
    # дефиниция для всех кадров, tag11-pelvis откалиброван к той же середине)
    elig = np.zeros(n, bool)
    j3_all = np.full((n, 6, 3), np.nan)
    root_all = np.full((n, 3), np.nan)
    hip_src = np.zeros(n, np.int8)          # 0 = tag11, 1 = реконструкция
    for k in range(n):
        need = [TAG_IDS.index(i) for i in (4, 7)]
        if not (seen[k, need].all() and not bad[k, need].any()):
            continue
        if not ok_hips[k]:
            continue
        ank_l = ank_r = None
        j0 = TAG_IDS.index(0)
        if seen[k, j0] and not bad[k, j0]:
            ank_l = joint3d(k, "tag0->L-ankle")
        for tid in (2, 9):
            jt = TAG_IDS.index(tid)
            if seen[k, jt] and not bad[k, jt] and f"tag{tid}->R-ankle" in offsets:
                ank_r = joint3d(k, f"tag{tid}->R-ankle")
                break
        if ank_l is None or ank_r is None:
            continue
        knees = [joint3d(k, "tag4->L-knee"), joint3d(k, "tag7->R-knee")]
        if seen[k, j11] and not bad[k, j11]:
            hips = [joint3d(k, "tag11->L-hip"), joint3d(k, "tag11->R-hip")]
            root = joint3d(k, "tag11->pelvis")
            hip_src[k] = 0
        else:
            def hip_from_thigh(jt, knee, L):
                Rd = cv2.Rodrigues(rv[k, jt])[0]
                return knee + Rd @ np.array([0.0, 1.0, 0.0]) * L
            hips = [hip_from_thigh(j4, knees[0], thigh_len["L"]),
                    hip_from_thigh(j7, knees[1], thigh_len["R"])]
            root = (hips[0] + hips[1]) / 2.0
            hip_src[k] = 1
        j3_all[k] = [hips[0], hips[1], knees[0], knees[1], ank_l, ank_r]
        root_all[k] = root
        elig[k] = True
    cand = np.flatnonzero(elig)
    scale = float(np.median(np.linalg.norm(
        j3_all[cand][:, :2] - root_all[cand][:, None], axis=2)))
    bones = np.stack([np.linalg.norm(j3_all[cand][:, 0] - j3_all[cand][:, 2], axis=1),
                      np.linalg.norm(j3_all[cand][:, 1] - j3_all[cand][:, 3], axis=1),
                      np.linalg.norm(j3_all[cand][:, 2] - j3_all[cand][:, 4], axis=1),
                      np.linalg.norm(j3_all[cand][:, 3] - j3_all[cand][:, 5], axis=1)], 1)
    bone_med = np.median(bones, 0)
    keep = (np.abs(bones - bone_med) < BONE_TOL_M).all(1)
    elig[cand[~keep]] = False
    print(f"GT-кадров: {elig.sum()} (из {len(cand)} eligibility, гейт костей снял "
          f"{int((~keep).sum())}), scale={scale:.3f} м, кости мед "
          f"[{bone_med[0]:.2f}/{bone_med[1]:.2f}, {bone_med[2]:.2f}/{bone_med[3]:.2f}]")

    gt = ((j3_all - root_all[:, None, :]) / scale).reshape(n, -1)
    gt[~elig] = np.nan

    # ---- вход: фьюжн root-relative + 2D + src ----
    fp = fus.copy()
    root_in = np.full((n, 3), np.nan)
    mh = np.isfinite(fp[:, 11, 0]) & np.isfinite(fp[:, 12, 0])
    root_in[mh] = (fp[mh, 11] + fp[mh, 12]) / 2
    has_any = np.isfinite(fp).any(axis=(1, 2))
    root_in[~mh & has_any] = np.nanmean(fp[~mh & has_any], axis=1)
    rel = (fp - root_in[:, None, :]) / scale
    present = np.isfinite(rel[..., 0])
    feat = np.concatenate([np.nan_to_num(rel).reshape(n, -1), present.astype(np.float32),
                           (fsrc == 0).astype(np.float32), (fsrc == 1).astype(np.float32),
                           np.nan_to_num(k2d[..., 0]) / 1280.0,
                           np.nan_to_num(k2d[..., 1]) / 720.0,
                           c2d], axis=1).astype(np.float32)     # (n, 17*9=153)
    feat_valid = present[:, 11] | present[:, 12] | (c2d[:, 11] > 0.3) | (c2d[:, 12] > 0.3)

    # ---- окна ----
    idx = np.flatnonzero(elig & feat_valid)
    half = T_WIN // 2
    X = np.zeros((len(idx), T_WIN, feat.shape[1]), np.float32)
    for a, k in enumerate(idx):
        w = np.clip(np.arange(k - half, k + half + 1), idx[0], idx[-1])
        X[a] = feat[w]
    Y = gt[idx].astype(np.float32)
    val = np.zeros(len(idx), bool)
    if VAL_MODE == "stride":
        # каждый 8-й eligible кадр: val покрывает все блоки (в т.ч. амплитудные);
        # соседние кадры (утечка) отстоят на 8 позиций ~0.3-0.5 с
        val[::8] = True
    elif VAL_MODE == "block":
        # амплитудный блок (марш 90-180 с сессии) целиком в val: окна не
        # пересекаются с train => нулевая утечка, но много глубоких сгибов
        tt = (z["t"][idx] - z["t"][0]) / 1000.0
        val = (tt >= 90) & (tt < 180)
    else:
        val[int(len(idx) * (1 - VAL_FRAC)):] = True

    # ---- шум фьюжна vs GT (для аугментаций) ----
    e = []
    for k in idx[::10]:
        r = root_in[k]
        for jj, ci in enumerate(COCO_LEG):
            if np.isfinite(fp[k, ci, 0]) and np.isfinite(gt[k]).all():
                e.append(fp[k, ci] - r - gt[k].reshape(6, 3)[jj] * scale)
    e = np.array(e) if e else np.zeros((1, 3))
    noise_sigma = e.std(0)

    np.savez_compressed(out, X=X, Y=Y, val=val,
                        frame_idx=idx, t=z["t"][idx], scale=scale,
                        noise_sigma=noise_sigma,
                        bone_med=bone_med, offsets=np.array(offsets, dtype=object),
                        feat_dim=feat.shape[1], t_win=T_WIN)
    print(f"{out}: {len(idx)} пар (val {val.sum()}), окно {T_WIN}, "
          f"фич {feat.shape[1]}; шум фьюжна vs GT: sigma {np.round(noise_sigma*100, 1)} см")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-mode", default="tail", choices=["tail", "stride", "block"])
    ap.add_argument("--out", default="l2_dataset/v1_pairs.npz")
    a = ap.parse_args()
    main(val_mode=a.val_mode, out=a.out)
