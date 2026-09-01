#!/usr/bin/env python3
"""Independently replay and integrity-check a FAEA ManiSkill episode."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

FORBIDDEN_MUTATORS = {
    "add_force",
    "add_joint",
    "apply_force",
    "create_drive",
    "set_angular_velocity",
    "set_linear_velocity",
    "set_pose",
    "set_qf",
    "set_qpos",
    "set_qvel",
    "set_root_pose",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value):
    return value.item() if hasattr(value, "item") else value


def static_integrity_check(episode_path: Path):
    tree = ast.parse(episode_path.read_text(), filename=str(episode_path))
    method_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    forbidden_calls = sorted(method_calls & FORBIDDEN_MUTATORS)
    return {
        "forbidden_mutator_calls": forbidden_calls,
        "uses_env_step": "step" in method_calls,
        "uses_direct_actor_mutation": bool(forbidden_calls),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--task-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    static_check = static_integrity_check(args.episode)
    if static_check["uses_direct_actor_mutation"] or not static_check["uses_env_step"]:
        raise SystemExit(
            f"Generated episode failed static integrity check: {static_check}"
        )

    spec = importlib.util.spec_from_file_location(
        "faea_generated_episode", args.episode
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    success, final_eval = module.main()
    replay = {key: scalar(value) for key, value in final_eval.items()}
    observed_success = bool(scalar(success)) and bool(replay.get("success", False))
    required_milestones = all(
        bool(replay.get(key, False))
        for key in (
            "cup_grasped_once",
            "cup_lifted_once",
            "cup_tilted_over_bowl_once",
        )
    )
    transfer_ok = int(replay.get("num_beads_in_bowl", 0)) >= 6
    verified = observed_success and required_milestones and transfer_ok

    receipt = {
        "verified": verified,
        "episode": str(args.episode.resolve()),
        "episode_sha256": sha256(args.episode),
        "task_source": str(args.task_source.resolve()),
        "task_source_sha256": sha256(args.task_source),
        "static_integrity": static_check,
        "replay": replay,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
