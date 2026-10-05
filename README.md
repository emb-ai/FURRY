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
- [Research roadmap and umbrella issues](https://github.com/emb-ai/FURRY/issues/1)
- [Contribution workflow](docs/Contributing.md)
- [GitHub Wiki mirror](https://github.com/emb-ai/FURRY/wiki)

## Status

This repository currently contains the research foundation and documentation
tooling, not a runnable teleoperation application. No TWIST2, Quest, or MuJoCo
runtime has been reproduced by this bootstrap. Architecture choices remain
explicit research decisions, not implied by the documentation tools.

`docs/` is authoritative; the Wiki is a manually published mirror. See the
[publishing procedure](docs/Contributing.md#wiki-publication).

## Foundation

- [TWIST2 paper](https://arxiv.org/abs/2511.02832)
- [TWIST2 implementation](https://github.com/amazon-far/TWIST2)
- [FURRY license: Apache-2.0](LICENSE)

Upstream software, models and datasets have their own licenses. FURRY's license
does not relicense them, and they are not bundled here.
