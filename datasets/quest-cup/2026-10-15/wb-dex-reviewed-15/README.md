# Quest WB-Dex: 29 reviewed cup-to-box trajectories

Recorded on Meta Quest 3 on 2026-10-15, scene `wb_dex`. This is a separate
reviewed subset, not an addition to the parent dataset's training split or manifest.

- 29 original episode ZIP archives; 526.47 accepted seconds in total.
- Task: transfer the cup into the floor box. Upright placement is not required.
- Twelve episodes have original `outcome=1`; seventeen have original `outcome=0`.
  Original labels and archive contents are preserved. The independent review result
  is recorded in `index.json`, not substituted for the application's outcome.
- Selection: cup origin inside the box throughout the final 0.5 seconds,
  translational speed below 0.03 m/s throughout that interval, grip released,
  no simulation fault, and no invalid input samples. Exact spatial bounds,
  per-archive SHA256, original labels and durations are in `index.json`.
- The initial 15 final states were inspected by rendering the archived MuJoCo model and qpos.
  The 14 additional episodes were selected using recorded-state and input checks.
  This is an outcome filter, not a guarantee of smooth motion or sim-to-real quality.
  Some trajectories drop the cup, and some end with it on its side or upside down.
- No train/validation split or RGB export is included in this subset.

The directory retains its original `wb-dex-reviewed-15` name for stable links;
`index.json` is authoritative and now lists 29 episodes (15 initial + 14 added).

## Download

From the repository root, with Git LFS installed:

```sh
git lfs pull --include="datasets/quest-cup/2026-10-15/wb-dex-reviewed-15/*.zip" --exclude=""
python datasets/quest-cup/2026-10-15/wb-dex-reviewed-15/verify.py
```

`verify.py` checks all 29 SHA256 hashes and ZIP CRCs. Extract archives to access
frames, inputs, scene state, camera poses, metadata and archived models.
MuJoCo 3.3.7 is required to load `learning-model.mjb`. Camera images must be
rendered offline; these archives are not headset screen recordings.
