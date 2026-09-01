#!/usr/bin/env python3
"""Reset and settle PourBeads-v1 without attempting to solve it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v3 as iio
import numpy as np

MANISKILL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MANISKILL_DIR))
import custom_tasks.pour_beads  # noqa: F401


def scalar(value):
    return value.item() if hasattr(value, "item") else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "PourBeads-v1",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        obs_mode="state",
    )
    _, info = env.reset(seed=args.seed)
    frames = [env.render().cpu().numpy()[0]]
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[..., -1] = 1
    for step in range(120):
        _, _, _, _, info = env.step(action)
        if step in (29, 59, 119):
            frames.append(env.render().cpu().numpy()[0])

    receipt = {
        "seed": args.seed,
        "settle_steps": 120,
        "success": bool(scalar(info["success"])),
        "num_beads_in_cup": int(scalar(info["num_beads_in_cup"])),
        "num_beads_in_bowl": int(scalar(info["num_beads_in_bowl"])),
        "cup_grasped_once": bool(scalar(info["cup_grasped_once"])),
        "cup_lifted_once": bool(scalar(info["cup_lifted_once"])),
        "cup_tilted_over_bowl_once": bool(scalar(info["cup_tilted_over_bowl_once"])),
    }
    (args.output_dir / "smoke_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    iio.imwrite(args.output_dir / "settled.png", frames[-1])
    iio.imwrite(args.output_dir / "settle.mp4", np.stack(frames), fps=2)
    env.close()
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["num_beads_in_cup"] == 8 and not receipt["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
