# FURRY

Full-body under-controlled robot retargeting, yo.

FURRY is a student research project on Meta Quest 3 teleoperation and trajectory
collection for humanoids in MuJoCo. We start from TWIST2, target Unitree G1 with
Dex3-1 hands, and investigate how far sparse human observations can support
whole-body motion and dexterous loco-manipulation.

**Simulation only.** Controller-based and controller-free Quest input are equal
research tracks. An external RealSense RGB-D camera is optional. Physical-robot
deployment is outside this project's scope.

## Start Here

- [Final demo, presentation and policy comparison — 16 October 2026](docs/demo/2026-10-16/README.md)
- [Knowledge base and reading order](docs/Home.md)
- [Getting started](docs/Getting-Started.md)
- [Team setup and Quest updates over Wi-Fi](docs/Quest-WiFi-Deploy.md)
- [Compressed literature review](docs/Literature-Review.md)
- [TWIST2 baseline audit](docs/TWIST2-Baseline.md)
- [GMR, TWIST2 and DexWeave for VR teleoperation](docs/DexWeave.md)
- [Quest training dataset: source, curated clips and mirrors](datasets/quest-virtual-legs-v2/2026-10-08/README.md)
- [Research roadmap and umbrella issues](https://github.com/emb-ai/FURRY/issues/1)
- [Contribution workflow](docs/Contributing.md)
- [GitHub Wiki mirror](https://github.com/emb-ai/FURRY/wiki)

## Status

As of 16 October 2026, the
[standalone Quest 3 / macOS G1 prototype](prototypes/quest_g1/README.md)
runs native OpenXR, MuJoCo and ONNX control on the headset. Meta body tracking
passes through C++ GMR; controller wrist targets, passthrough, calibration,
policy selection, trajectory capture and operator MP4 recording are available.
TWIST2, HOMIE and SONIC modes and experimental fine-tuned policies are included.

The [final demo package](docs/demo/2026-10-16/README.md) shows tracking, cup,
Kick-T and tennis, with a Rubik teaser. Published Quest datasets and autonomous
visual-policy results are available in `datasets/` and the prototype's training
reports. These demonstrations do not establish general task reliability:
walking, external-camera leg tracking and arbitrary Rubik sequences remain
research limitations. Physical-robot deployment remains outside scope.

`docs/` is authoritative; the Wiki is a manually published mirror. See the
[publishing procedure](docs/Contributing.md#wiki-publication).

## Foundation

- [TWIST2 paper](https://arxiv.org/abs/2511.02832)
- [TWIST2 implementation](https://github.com/amazon-far/TWIST2)
- [FURRY license: Apache-2.0](LICENSE)

Upstream software, models and datasets have their own licenses. FURRY's license
does not relicense them. The Quest training snapshot includes four TWIST2
example-motion replay fixtures with their original MIT notice; see its dataset
README for provenance.
