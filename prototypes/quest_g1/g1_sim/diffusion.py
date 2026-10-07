"""Small low-dimensional Diffusion Policy for the scripted reach.

The denoiser is a conditional 1D convolution. It predicts the clean command
chunk, conditioned on a short ``state-v1`` history. This is the
low-dimensional setting from Chi et al., Diffusion Policy (RSS 2023). It is
not the unreleased TWIST2 high-level repository, and it does not see images.

Actions are ``p_cmd`` plus ``grip`` at the 20 Hz export rate. The closed loop
holds each sampled command for one export period and passes it to
``Controller.step``. Consecutive chunks are not blended.
"""
import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # The desktop viewer does not need this dependency.
    torch = None
    nn = None
    _TORCH_ERROR = exc
else:
    _TORCH_ERROR = None

from .state import ACTION_HZ, COMMAND_NAMES, RECORD_HZ, SCHEMA as STATE_SCHEMA

POLICY_SCHEMA = "lowdim-dp-v1"
OBS_HORIZON = 2
PRED_HORIZON = 8
ACTION_HORIZON = 2
TRAIN_STEPS = 50
INFERENCE_STEPS = 16
WIDTH = 64
ACTION_STD_FLOOR = 1e-2
STATE_STD_FLOOR = 1e-3


def require_torch():
    if torch is None:
        raise ImportError("Training and rollout need torch. Install the CPU wheel; the desktop viewer does not use it.") from _TORCH_ERROR


def cosine_alpha_bar(steps, s=0.008):
    """Squared-cosine cumulative product used by Diffusion Policy."""
    require_torch()
    grid = torch.arange(steps + 1, dtype=torch.float64)
    scale = torch.cos((grid / steps + s) / (1 + s) * np.pi * 0.5) ** 2
    scale = scale / scale[0]
    # The final cumulative product underflows, and the x0 reconstruction divides by it.
    betas = torch.clamp(1 - scale[1:] / scale[:-1], min=1e-4, max=0.999)
    return torch.cumprod(1 - betas, dim=0).clamp(min=1e-4).float()


def _groups(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(inputs, outputs, kernel, padding=kernel // 2),
            nn.GroupNorm(_groups(outputs), outputs),
            nn.Mish(),
        )

    def forward(self, values):
        return self.net(values)


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, condition_dim):
        super().__init__()
        self.blocks = nn.ModuleList((ConvBlock(inputs, outputs), ConvBlock(outputs, outputs)))
        self.condition = nn.Sequential(nn.Mish(), nn.Linear(condition_dim, outputs * 2))
        self.skip = nn.Identity() if inputs == outputs else nn.Conv1d(inputs, outputs, 1)

    def forward(self, values, condition):
        scale, shift = self.condition(condition).unsqueeze(-1).chunk(2, dim=1)
        hidden = self.blocks[0](values) * (scale + 1) + shift
        return self.blocks[1](hidden) + self.skip(values)


class Denoiser(nn.Module):
    """Noise predictor. Action chunks are shaped ``(batch, time, action)``."""

    def __init__(self, action_dim, obs_dim, obs_horizon, width=WIDTH):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(width, width * 2), nn.Mish(), nn.Linear(width * 2, width))
        self.obs_mlp = nn.Sequential(
            nn.Linear(obs_dim * obs_horizon, width), nn.Mish(), nn.Linear(width, width))
        self.stem = ConvBlock(action_dim, width)
        self.block1 = ResidualBlock(width, width, width * 2)
        self.block2 = ResidualBlock(width, width, width * 2)
        self.head = nn.Conv1d(width, action_dim, 1)
        self.width = width

    def time_features(self, steps):
        half = self.width // 2
        frequencies = torch.exp(-np.log(10000) * torch.arange(half, device=steps.device) / max(half - 1, 1))
        angles = steps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        return self.time_mlp(torch.cat((angles.sin(), angles.cos()), dim=1))

    def forward(self, noisy, steps, observation):
        condition = torch.cat((self.time_features(steps), self.obs_mlp(observation.flatten(1))), dim=1)
        hidden = self.block2(self.block1(self.stem(noisy.transpose(1, 2)), condition), condition)
        return self.head(hidden).transpose(1, 2)


def _windows(state, command, grip, obs_horizon, pred_horizon):
    action = np.concatenate([command, grip[:, None]], axis=1)
    observations, actions = [], []
    for time in range(len(state) - pred_horizon + 1):
        start = time - obs_horizon + 1
        if start < 0:
            pad = np.repeat(state[:1], -start, axis=0)
            window = np.concatenate((pad, state[:time + 1]), axis=0)
        else:
            window = state[start:time + 1]
        observations.append(window)
        actions.append(action[time:time + pred_horizon])
    if not observations:
        raise ValueError("Episode is shorter than the action horizon")
    return np.stack(observations), np.stack(actions)


def load_demonstrations(directory, obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON):
    directory = Path(directory)
    paths = sorted(directory.glob("*_state.npz"))
    if not paths:
        raise ValueError(f"No state-v1 exports in {directory}")
    observations, actions, names, scene = [], [], None, None
    for path in paths:
        with np.load(path, allow_pickle=False) as episode:
            if str(episode["schema"]) != STATE_SCHEMA:
                raise ValueError(f"{path.name} is not {STATE_SCHEMA}")
            state_names = tuple(str(name) for name in episode["state_names"])
            command_names = tuple(str(name) for name in episode["command_names"])
            this_scene = str(episode["scene"])
            if names is None:
                names, scene = state_names, this_scene
            if state_names != names or command_names != COMMAND_NAMES or this_scene != scene:
                raise ValueError(f"{path.name} does not match the other demonstrations")
            window, action = _windows(
                np.asarray(episode["state"], dtype=np.float64),
                np.asarray(episode["command"], dtype=np.float64),
                np.asarray(episode["grip"], dtype=np.float64),
                obs_horizon, pred_horizon)
            observations.append(window)
            actions.append(action)
    return np.concatenate(observations), np.concatenate(actions), names, scene


def _limits(values, floor):
    low = values.min(axis=(0, 1))
    high = values.max(axis=(0, 1))
    span = np.maximum(high - low, floor)
    return low - 0.05 * span, high + 0.05 * span


def fit_checkpoint(observations, actions, checkpoint, names, scene, steps=2000, seed=0,
                   action_horizon=ACTION_HORIZON, width=WIDTH, diffusion_steps=TRAIN_STEPS,
                   inference_steps=INFERENCE_STEPS):
    """Train on window arrays shaped ``(n, obs_horizon, state_dim)`` and ``(n, pred_horizon, action_dim)``."""
    require_torch()
    obs_horizon, pred_horizon = int(observations.shape[1]), int(actions.shape[1])
    if action_horizon < 1 or action_horizon > pred_horizon:
        raise ValueError("action_horizon must be within the predicted chunk")
    if len(observations) != len(actions) or len(observations) < 2:
        raise ValueError("Need matching observation and action windows")
    generator = np.random.default_rng(seed)
    torch.manual_seed(seed)
    state_mean = observations.mean(axis=(0, 1))
    state_std = np.maximum(observations.std(axis=(0, 1)), STATE_STD_FLOOR)
    action_mean = actions.mean(axis=(0, 1))
    action_std = np.maximum(actions.std(axis=(0, 1)), ACTION_STD_FLOOR)
    action_min, action_max = _limits(actions, ACTION_STD_FLOOR)
    raw_std = actions.std(axis=(0, 1))
    # Constant joints are easy and would otherwise hide errors on the arm.
    weights = np.clip(raw_std / max(float(raw_std.mean()), 1e-6), 0.25, 4.0).astype(np.float32)
    obs_n = ((observations - state_mean) / state_std).astype(np.float32)
    act_n = ((actions - action_mean) / action_std).astype(np.float32)
    config = {
        "schema": POLICY_SCHEMA, "scene": scene, "state_names": list(names),
        "obs_horizon": obs_horizon, "pred_horizon": pred_horizon, "action_horizon": action_horizon,
        "train_steps": int(diffusion_steps), "inference_steps": int(inference_steps), "width": width,
        "prediction": "x0",
        "state_dim": int(observations.shape[-1]), "action_dim": int(actions.shape[-1]),
        "optimization_steps": int(steps), "seed": int(seed), "windows": int(len(observations)),
    }
    net = Denoiser(config["action_dim"], config["state_dim"], obs_horizon, width)
    alpha = cosine_alpha_bar(diffusion_steps)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    batch = min(64, len(obs_n))
    net.train()
    last_loss = None
    for step in range(steps):
        choice = generator.integers(0, len(obs_n), size=batch)
        obs = torch.as_tensor(obs_n[choice]) + 0.01 * torch.randn(batch, obs_horizon, config["state_dim"])
        act = torch.as_tensor(act_n[choice])
        diffusion_step = torch.randint(0, diffusion_steps, (batch,))
        noise = torch.randn_like(act)
        scale = alpha[diffusion_step].view(batch, 1, 1)
        noisy = scale.sqrt() * act + (1 - scale).sqrt() * noise
        predicted = net(noisy, diffusion_step, obs)
        loss = ((predicted - act).square() * torch.as_tensor(weights)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach())
        if step == 0 or step + 1 == steps or (step + 1) % 200 == 0:
            print(f"step {step + 1} loss {last_loss:.5f}", flush=True)
    config["final_loss"] = last_loss
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": POLICY_SCHEMA,
        "config": config,
        "model": net.state_dict(),
        "state_mean": torch.as_tensor(state_mean, dtype=torch.float32),
        "state_std": torch.as_tensor(state_std, dtype=torch.float32),
        "action_mean": torch.as_tensor(action_mean, dtype=torch.float32),
        "action_std": torch.as_tensor(action_std, dtype=torch.float32),
        "action_min": torch.as_tensor(action_min, dtype=torch.float32),
        "action_max": torch.as_tensor(action_max, dtype=torch.float32),
    }
    torch.save(payload, checkpoint)
    return config


def train(directory, checkpoint, steps=2000, seed=0, obs_horizon=OBS_HORIZON,
          pred_horizon=PRED_HORIZON, action_horizon=ACTION_HORIZON, width=WIDTH,
          diffusion_steps=TRAIN_STEPS, inference_steps=INFERENCE_STEPS):
    """Fit a checkpoint on every ``*_state.npz`` export in ``directory``."""
    observations, actions, names, scene = load_demonstrations(directory, obs_horizon, pred_horizon)
    return fit_checkpoint(
        observations, actions, checkpoint, names, scene, steps, seed, action_horizon, width,
        diffusion_steps, inference_steps)


class Policy:
    def __init__(self, checkpoint):
        require_torch()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("schema") != POLICY_SCHEMA:
            raise ValueError(f"Expected {POLICY_SCHEMA}")
        self.config = payload["config"]
        self.net = Denoiser(
            self.config["action_dim"], self.config["state_dim"], self.config["obs_horizon"], self.config["width"])
        self.net.load_state_dict(payload["model"])
        if self.config.get("prediction") != "x0":
            raise ValueError("Checkpoint does not predict the clean command")
        self.net.eval()
        self.alpha = cosine_alpha_bar(self.config["train_steps"])
        self.state_mean = payload["state_mean"].numpy()
        self.state_std = payload["state_std"].numpy()
        self.action_mean = payload["action_mean"].double().numpy()
        self.action_std = payload["action_std"].double().numpy()
        self.action_min = payload["action_min"].double().numpy()
        self.action_max = payload["action_max"].double().numpy()

    def sample(self, observation, generator):
        """One action chunk, shape ``(pred_horizon, action_dim)``, in physical units."""
        horizon, width = self.config["obs_horizon"], observation.shape[-1]
        if observation.shape != (horizon, width) or width != self.config["state_dim"]:
            raise ValueError("Observation does not match the checkpoint")
        normalized = torch.as_tensor((observation - self.state_mean) / self.state_std, dtype=torch.float32).unsqueeze(0)
        latent = torch.randn(
            1, self.config["pred_horizon"], self.config["action_dim"], generator=generator)
        times = np.linspace(self.config["train_steps"] - 1, 0, self.config["inference_steps"]).astype(int)
        with torch.no_grad():
            for index, step in enumerate(times):
                current = torch.full((1,), int(step), dtype=torch.long)
                clean = self.net(latent, current, normalized).clamp(-3, 3)
                if index + 1 == len(times):
                    latent = clean
                else:
                    bar = self.alpha[int(step)].clamp(min=1e-4)
                    epsilon = (latent - bar.sqrt() * clean) / (1 - bar).sqrt()
                    nxt = self.alpha[int(times[index + 1])].clamp(min=1e-4)
                    latent = nxt.sqrt() * clean + (1 - nxt).sqrt() * epsilon
        action = latent[0].double().numpy() * self.action_std + self.action_mean
        return np.clip(action, self.action_min, self.action_max)


class PolicySource:
    """Holds a 20 Hz command across the physics steps the tracker consumes."""

    def __init__(self, checkpoint, seed=0):
        self.policy = Policy(checkpoint)
        self.seed = int(seed)
        self.generator = torch.Generator().manual_seed(self.seed)
        self.history = deque()
        self.pending = []
        self.command = None
        self.grip = 0.0
        self.bound = False

    def bind(self, model):
        from .state import layout_names
        names = layout_names(model)
        if tuple(self.policy.config["state_names"]) != names or self.policy.config["scene"] != "cup":
            raise ValueError("Checkpoint state layout does not match this scene")
        self.bound = True

    def reset(self):
        self.generator.manual_seed(self.seed)
        self.history.clear()
        self.pending.clear()
        self.command = None
        self.grip = 0.0

    def __call__(self, model, data, controller):
        from .state import encode_state
        if not self.bound:
            self.bind(model)
        every = controller.period * (RECORD_HZ // ACTION_HZ)
        if controller.step_index % every == 0:
            state = encode_state(model, data, self.grip)
            self.history.append(state)
            horizon = self.policy.config["obs_horizon"]
            while len(self.history) > horizon:
                self.history.popleft()
            if not self.pending:
                window = list(self.history)
                while len(window) < horizon:
                    window.insert(0, window[0])
                chunk = self.policy.sample(np.stack(window), self.generator)
                self.pending = [chunk[index] for index in range(self.policy.config["action_horizon"])]
            action = self.pending.pop(0)
            if action.shape != (len(COMMAND_NAMES) + 1,) or not np.isfinite(action).all():
                raise RuntimeError("Policy returned a non-finite command")
            self.command = np.asarray(action[:-1], dtype=np.float64)
            self.grip = float(np.clip(action[-1], 0.0, 1.0))
        return self.command, self.grip


def _final(metrics):
    if "final_tip_distance_m" not in metrics:
        return "fell"
    return f"{metrics['initial_tip_distance_m']:.3f}->{metrics['final_tip_distance_m']:.3f}"


def evaluate(checkpoint, seeds, seconds):
    """Compare the policy, the script, and standing on the same cup seeds."""
    from .reach import ScriptedReach, Standing, run_episode
    rows = []
    for seed in seeds:
        sources = {
            "policy": lambda model, cup, seed=seed: PolicySource(checkpoint, seed),
            "script": lambda model, cup: ScriptedReach(model, cup),
            "standing": lambda model, cup: Standing(),
        }
        trial = {"seed": int(seed)}
        for name, factory in sources.items():
            try:
                trial[name] = run_episode(seed, seconds, source_factory=factory)
            except RuntimeError as error:
                trial[name] = {"fell": True, "error": str(error)}
        rows.append(trial)
        print(
            f"seed {seed} policy {_final(trial['policy'])} script {_final(trial['script'])} "
            f"standing {_final(trial['standing'])}",
            flush=True)
    return {"checkpoint": str(checkpoint), "trials": rows}


def main(argv=None):
    require_torch()
    parser = argparse.ArgumentParser(description="Train or roll out the low-dimensional reach policy")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("directory", type=Path)
    train_parser.add_argument("checkpoint", type=Path)
    train_parser.add_argument("--steps", type=int, default=2000)
    train_parser.add_argument("--seed", type=int, default=0)
    rollout_parser = commands.add_parser("rollout")
    rollout_parser.add_argument("checkpoint", type=Path)
    rollout_parser.add_argument("--seed", type=int, nargs="+", default=[0])
    rollout_parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args(argv)
    if args.command == "train":
        print(json.dumps(train(args.directory, args.checkpoint, args.steps, args.seed), indent=2), flush=True)
    else:
        print(json.dumps(evaluate(args.checkpoint, args.seed, args.seconds), indent=2), flush=True)


if __name__ == "__main__":
    main()
