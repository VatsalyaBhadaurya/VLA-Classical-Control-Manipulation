"""SmolVLA policy for the PiPER joint-space env (inference, post-fine-tune).

Loads a fine-tuned SmolVLA checkpoint + its processor pipeline (which does
normalization and language tokenization), and exposes `act(obs) -> [j1..j6,
gripper]` joint targets. Designed for local inference on the 6 GB 4050
(fine-tune on cloud; infer here — SmolVLA-450M inference fits 6 GB).

lerobot 0.6.x loading recipe (verified against the API):
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    policy = make_policy(cfg, pretrained_path=ckpt)
    pre, post = make_pre_post_processors(cfg, pretrained_path=ckpt)

The processor pipeline is callable: `pre(batch_dict)` normalizes images/state
and tokenizes `task`; `post(action)` unnormalizes. `select_action` returns one
action per call and re-infers a chunk internally every n_action_steps (giving
the effective 5–10 Hz VLA rate against the 30 Hz control loop).
"""
from __future__ import annotations

import numpy as np

from configs.piper import PiperConfig


class SmolVLAJointPolicy:
    def __init__(self, cfg: PiperConfig):
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import make_policy, make_pre_post_processors

        self.torch = torch
        self.instruction = cfg.instruction
        ckpt = cfg.vla_ckpt

        self.device = (cfg.vla_device
                       if (cfg.vla_device == "cpu" or torch.cuda.is_available())
                       else "cpu")

        pcfg = PreTrainedConfig.from_pretrained(ckpt)
        self.policy = make_policy(pcfg, pretrained_path=ckpt)
        self.pre, self.post = make_pre_post_processors(pcfg, pretrained_path=ckpt)
        self.policy.to(self.device)
        if cfg.vla_half and self.device != "cpu":
            self.policy.half()
        self.policy.eval()
        self._half = cfg.vla_half and self.device != "cpu"

    def reset(self):
        self.policy.reset()

    def act(self, obs) -> np.ndarray:
        torch = self.torch
        # dataset convention: images are float CHW in [0,1]; state is the 7-vec.
        batch = {
            "observation.state": torch.from_numpy(
                np.asarray(obs.raw["state"], dtype=np.float32))[None].to(self.device),
            "task": self.instruction,
        }
        for name, im in obs.images.items():               # cam_global, cam_gripper
            t = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1)
            t = t.float().div_(255.0)[None].to(self.device)
            if self._half:
                t = t.half()
            batch[f"observation.images.{name}"] = t

        with torch.no_grad():
            batch = self.pre(batch)
            action = self.policy.select_action(batch)
            action = self.post(action)

        a = np.asarray(action.detach().float().cpu().numpy()).ravel()
        return a[:7]
