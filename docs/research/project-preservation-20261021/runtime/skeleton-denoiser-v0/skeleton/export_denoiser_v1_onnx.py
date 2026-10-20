"""
Экспорт v1 в ONNX для live-пайплайна: (x, t, win) -> (x0, x0_direct, log_sigma).

win (B,9,153) — окно фич (фьюжн root-relative + presence + src + 2D + conf);
DDIM-цикл остаётся на стороне рантайма (denoiser_v1_ov.py в FURRY).

Рядом fixture: <out>.fixture.npz с эталонами torch для parity-теста.

Запуск: .venv/Scripts/python export_denoiser_v1_onnx.py
"""
import numpy as np
import torch

from train_denoiser_v1 import FEAT_DIM, T_WIN, DenoiserV1


class Export(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, t, win):
        return self.m(x, t, win)


def main(ckpt="models/denoiser_v1.pt", out="models/denoiser_v1.onnx"):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = DenoiserV1()
    m.load_state_dict(ck["model"])
    m.eval()
    expo = Export(m).eval()

    x = torch.randn(2, 18)
    t = torch.tensor([120.0, 400.0])
    win = torch.randn(2, T_WIN, FEAT_DIM)
    with torch.no_grad():
        torch.onnx.export(expo, (x, t, win), out,
                          input_names=["x", "t", "win"],
                          output_names=["x0", "x0_direct", "log_sigma"],
                          dynamic_axes={k: {0: "B"} for k in
                                        ("x", "t", "win", "x0", "x0_direct", "log_sigma")},
                          opset_version=17, dynamo=False)
    rng = np.random.default_rng(0)
    fw = rng.normal(size=(3, T_WIN, FEAT_DIM)).astype(np.float32)
    fx = rng.normal(size=(3, 18)).astype(np.float32)
    ft = np.array([1.0, 120.0, 400.0], np.float32)
    with torch.no_grad():
        r = expo(torch.tensor(fx), torch.tensor(ft), torch.tensor(fw))
    np.savez(out + ".fixture.npz", x=fx, t=ft, win=fw,
             x0=r[0].numpy(), x0_direct=r[1].numpy(), log_sigma=r[2].numpy())
    print(f"ONNX: {out}\nfixture: {out}.fixture.npz (|x0| мед "
          f"{np.median(np.abs(r[0].numpy())):.3f})")


if __name__ == "__main__":
    main()
