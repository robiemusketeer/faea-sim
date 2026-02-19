"""
LIBERO episode harness — reusable boilerplate for env setup, frame recording,
success checking, video/metadata saving.

Claude-generated policies only need to implement:

    def policy(env, obs, h) -> obs

where `h` is a helpers namespace providing execute_action, compute_action,
record_frame, GRIPPER_CLOSE, GRIPPER_OPEN, and np.
"""

import json
import os
import types

import imageio
import numpy as np

from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRIPPER_CLOSE = 1.0
GRIPPER_OPEN = -1.0


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def setup_env(benchmark_name, task_id, seed=42,
              camera_heights=256, camera_widths=256,
              render_gpu_device_id=0):
    """Create and initialise a LIBERO env, returning (env, obs, init_states).

    The environment is seeded, reset, and set to the first init state so it is
    ready for immediate use.
    """
    benchmark = get_benchmark(benchmark_name)()
    task_description = benchmark.get_task_names()[task_id]
    bddl_file_path = benchmark.get_task_bddl_file_path(task_id)
    init_states = benchmark.get_task_init_states(task_id)

    print(f"[harness] Task {task_id}: {task_description}")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_path,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
        render_gpu_device_id=render_gpu_device_id,
    )

    env.seed(seed)
    obs = env.reset()
    obs = env.set_init_state(init_states[0])

    print("[harness] Environment created and initialised.")
    return env, obs, init_states


# ---------------------------------------------------------------------------
# Episode recorder
# ---------------------------------------------------------------------------

class EpisodeRecorder:
    """Accumulates rendered frames and writes them to an mp4."""

    def __init__(self):
        self.frames = []

    def record_frame(self, obs):
        frame = obs["agentview_image"][::-1]
        self.frames.append(frame)

    def save_video(self, path, fps=20):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        imageio.mimsave(path, self.frames, fps=fps)
        print(f"[harness] Video saved to {path} ({len(self.frames)} frames)")


# ---------------------------------------------------------------------------
# Helper functions (exposed to policies via the `h` namespace)
# ---------------------------------------------------------------------------

def _make_execute_action(recorder):
    """Return an execute_action closure that auto-records frames."""

    def execute_action(env, action, steps=1):
        obs = None
        for _ in range(steps):
            obs, reward, done, info = env.step(action)
            recorder.record_frame(obs)
            if done:
                break
        return obs, done

    return execute_action


def compute_action(current_pos, target_pos, gripper, gain=15):
    """P-controller that returns a 7-d action [dx, dy, dz, 0, 0, 0, gripper]."""
    diff = target_pos - current_pos
    action_xyz = np.clip(diff * gain, -1, 1)
    return np.concatenate([action_xyz, [0, 0, 0, gripper]])


def _build_helpers(recorder):
    """Build the helpers namespace passed to policy functions."""
    h = types.SimpleNamespace(
        execute_action=_make_execute_action(recorder),
        compute_action=compute_action,
        record_frame=recorder.record_frame,
        GRIPPER_CLOSE=GRIPPER_CLOSE,
        GRIPPER_OPEN=GRIPPER_OPEN,
        np=np,
    )
    return h


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_episode(benchmark_name, task_id, policy_fn, output_dir,
                seed=42, meta_file=None, video_file=None,
                camera_heights=256, camera_widths=256,
                render_gpu_device_id=0):
    """Run a single LIBERO episode end-to-end.

    Parameters
    ----------
    benchmark_name : str
        e.g. "libero_object", "libero_spatial"
    task_id : int
    policy_fn : callable(env, obs, h) -> obs
        The task-specific action sequence.
    output_dir : str
        Directory for video and metadata.
    seed : int
    meta_file : str or None
        Filename for the metadata JSON (default: meta_{task_id}.json).
    video_file : str or None
        Filename for the video (default: episode_{task_id}.mp4).
    camera_heights, camera_widths, render_gpu_device_id :
        Passed through to OffScreenRenderEnv.

    Returns
    -------
    dict  {"success": bool, "num_frames": int}
    """
    meta_file = meta_file or f"meta_{task_id}.json"
    video_file = video_file or f"episode_{task_id}.mp4"

    env, obs, _init_states = setup_env(
        benchmark_name, task_id, seed=seed,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
        render_gpu_device_id=render_gpu_device_id,
    )

    recorder = EpisodeRecorder()
    recorder.record_frame(obs)

    helpers = _build_helpers(recorder)

    print("[harness] Running policy...")
    obs = policy_fn(env, obs, helpers)
    print("[harness] Policy finished.")

    success = env.check_success()
    print(f"[harness] Task Success: {success}")

    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.join(output_dir, video_file)
    recorder.save_video(video_path)

    meta = {"success": success, "num_frames": len(recorder.frames)}
    meta_path = os.path.join(output_dir, meta_file)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[harness] Metadata saved to {meta_path}")

    env.close()
    print("[harness] Environment closed.")

    return {"success": success, "num_frames": len(recorder.frames)}
