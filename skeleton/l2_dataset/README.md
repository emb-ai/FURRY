# Denoiser v1 processed data

Run `git lfs pull` from the repository root to download the data and v1 weights.

- `l2_session2.npz`: extracted L2 session with timestamps, 2D keypoints, depth fusion and tag observations.
- `v1_pairs.npz`: default training set, 4,854 windows (4,271 train / 583 validation); validation uses the final 12% of eligible frames.
- `v1_pairs_stride.npz`, `v1_pairs_block.npz`: alternative validation splits; the published v1 checkpoint uses the default set.
- `train_v1_log.txt`: original v1 training log.

Pair arrays: `X` is float32 `(N, 9, 153)`, `Y` is float32 `(N, 18)` for six leg joints, and `val` is a boolean validation mask. Joint order: left hip, right hip, left knee, right knee, left ankle, right ankle. Targets are relative to the pelvis and divided by `scale` (metres). Additional fields store frame indices, timestamps, noise statistics, bone medians, fitted tag offsets and input dimensions. Reading the object-valued `offsets` field requires `allow_pickle=True`; only enable this for trusted files.

From `skeleton/`, with NumPy, PyTorch and OpenCV installed:

```sh
python eval_denoiser_v1.py
python train_denoiser_v1.py --epochs 300 --out models/denoiser_v1_retrained.pt
python l2_build_v1.py --out l2_dataset/v1_pairs_rebuilt.npz
```

ONNX export additionally needs `onnx`. The existing live runtime uses OpenVINO:

```sh
python -m unittest test_denoiser_v1_ov
```

The published checkpoint and ONNX model are in `../models/`, alongside a PyTorch reference fixture for runtime parity checks. See `../DENOISER_V1.md` for the historical experiment notes. Raw camera recordings are not part of this processed dataset.

Packaging verification on 2026-10-13: the bundled checkpoint evaluated on the default 583 validation windows gives MPJPE 2.84 cm (direct), 1.36 cm (diffusion obs/10), and 7.87 cm (raw fusion). These differ from the historical results in `DENOISER_V1.md` and the original training log; the checkpoint's exact training provenance has not been re-established. Four OpenVINO runtime tests, including fixture parity, pass. The data cover one operator/session and contain only five left-knee and one right-knee validation examples above 60 degrees, so they do not establish generalization or reliable deep-flexion performance.
