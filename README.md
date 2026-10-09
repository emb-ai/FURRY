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

- [Knowledge base and reading order](docs/Home.md)
- [Getting started](docs/Getting-Started.md)
- [Compressed literature review](docs/Literature-Review.md)
- [TWIST2 baseline audit](docs/TWIST2-Baseline.md)
- [Quest training dataset: source, curated clips and mirrors](datasets/quest-virtual-legs-v2/2026-10-08/README.md)
- [Research roadmap and umbrella issues](https://github.com/emb-ai/FURRY/issues/1)
- [Contribution workflow](docs/Contributing.md)
- [GitHub Wiki mirror](https://github.com/emb-ai/FURRY/wiki)

## Status

The repository includes the research foundation and an experimental
[standalone Quest 3 / macOS G1 prototype](prototypes/quest_g1/README.md).
The prototype runs MuJoCo and a TWIST2 policy locally on the headset, with
passthrough, controller wrist IK, simulation reset and local motion recording.
It is not a complete reproduction of TWIST2: operator-reported arm correspondence
remains poor, and cup lifting, Kick-T, walking and optical-hand tracking are not
validated. See [implementation task #11](https://github.com/emb-ai/FURRY/issues/11).
Architecture choices remain provisional research decisions.

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
