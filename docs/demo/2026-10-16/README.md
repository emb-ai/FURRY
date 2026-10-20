# FURRY — final presentation materials, 16 October 2026

- [Fresh presentation export — PPTX, 23 slides](furry-presentation.pptx)
- [Current presentation — Google Slides](https://docs.google.com/presentation/d/1hyqIzdS8b-cYbUaeu4GaNDoR_Sctr1FrsqFfs4joNZQ/edit)
- [Final montage — MP4, 1:29](furry-final-montage.mp4)
- [Eight-policy comparison — MP4, 0:26](policies-comparison.mp4)

The PPTX is a complete, unchanged Google Slides export retrieved on 21 October
2026. All 23 slides, ZIP checksums, XML parts and internal relationships were
checked. It includes later changes missing from the 16 October 12:30 export.

The export contains seven GIF assets and no embedded video parts. Video
playback from Google Slides is not guaranteed in PowerPoint. The two standalone
MP4 files below preserve the final montage and policy comparison.

The final 1280×720, 30 fps montage combines tracking, cup, Kick-T and tennis,
followed by the Rubik “coming soon” ending. It is an edited presentation reel
combining operator footage and simulation views, not a continuous benchmark.
The Rubik ending does not claim a completed autonomous solve.

The 3840×2160, 25 fps policy comparison contains 20 seconds of native offline
simulation followed by six seconds of error summaries. It is not a headset
screen capture. All eight controllers receive the same recorded right-turn
fragment, with the respective deployed input adapters. Position error is mean
Euclidean pelvis XY distance; angle error is mean absolute wrapped pelvis yaw
error. These are root-tracking metrics for one diagnostic clip, not measures of
foot sliding or general policy quality. See [comparison details](policy-comparison.md).

## Downloading

The videos use Git LFS. Use GitHub's download button, or from a clone:

```sh
git lfs install
git lfs pull --include='docs/demo/2026-10-16/*' --exclude=''
```

[manifest.json](manifest.json) records file sizes, SHA256 hashes and source
versions. Draft edits, intermediate renders, APK backups and unfinished training
experiments are not part of this package.

## Code and data

- [Standalone runtime](../../../prototypes/quest_g1/README.md)
- [Operator video and capture FOV](../../../prototypes/quest_g1/android/OPERATOR_VIDEO.md)
- [Quest motion dataset](../../../datasets/quest-virtual-legs-v2/2026-10-08/README.md)
- [Cup demonstrations, HOMIE Standing](../../../datasets/quest-cup/2026-10-15/README.md)
- [59 cup demonstrations, HOMIE v2](../../../datasets/quest-cup/2026-10-15-homie-v2/README.md)
- [Bimanual demonstrations](../../../datasets/quest-bimanual/2026-10-15/README.md)

These datasets and the visual-policy training code were already in main.
This publication additionally integrates the operator recorder and the capture
projection correction used in the live demonstration. A newly built integrated
APK is not evidence of a new headset installation or hardware acceptance.
