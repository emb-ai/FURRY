# Quest WB-Dex: 15 reviewed cup-to-box trajectories

Recorded on Meta Quest 3 on 2026-10-15, scene `wb_dex`. This is a separate
reviewed subset, not an addition to the parent dataset's training split or manifest.

- 15 original episode ZIP archives; 259.15 accepted seconds in total.
- Task: transfer the cup into the floor box. Upright placement is not required.
- Six episodes have original `outcome=1`; nine have original `outcome=0`.
  Original labels and archive contents are preserved. The independent review result
  is recorded in `index.json`, not substituted for the application's outcome.
- Selection: cup origin inside the box throughout the final 0.5 seconds,
  translational speed below 0.03 m/s throughout that interval, grip released,
  no simulation fault, and no invalid input samples. Exact spatial bounds,
  per-archive SHA256, original labels and durations are in `index.json`.
- Final states were inspected by rendering the archived MuJoCo model and qpos.
  This is an outcome filter, not a guarantee of smooth motion or sim-to-real quality.
  Some trajectories drop the cup, and some end with it on its side or upside down.
- No train/validation split or RGB export is included in this subset.

## Download

From the repository root, with Git LFS installed:

```sh
git lfs pull --include="datasets/quest-cup/2026-10-15/wb-dex-reviewed-15/*.zip" --exclude=""
python datasets/quest-cup/2026-10-15/wb-dex-reviewed-15/verify.py
```

`verify.py` checks all 15 SHA256 hashes and ZIP CRCs. Extract archives to access
frames, inputs, scene state, camera poses, metadata and archived models.
MuJoCo 3.3.7 is required to load `learning-model.mjb`. Camera images must be
rendered offline; these archives are not headset screen recordings.
