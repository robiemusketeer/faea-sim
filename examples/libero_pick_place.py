"""
Episode script for LIBERO pick-and-place task.

This script executes a pick-and-place task in the LIBERO simulation environment.
The robot picks up an object and places it into a target location.

Run with: micromamba run -n libero python libero_pick_place.py
"""
import numpy as np
import imageio
import json
import os

from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the LIBERO_OBJECT benchmark and task
benchmark = get_benchmark("libero_object")()
task_id = 0
bddl_file_path = benchmark.get_task_bddl_file_path(task_id)
init_states = benchmark.get_task_init_states(task_id)

# Create environment
env = OffScreenRenderEnv(
    bddl_file_name=bddl_file_path,
    camera_heights=256,
    camera_widths=256,
    render_gpu_device_id=0,
)

# Initialize environment
env.seed(42)
obs = env.reset()
obs = env.set_init_state(init_states[0])

# Video frames
frames = []

def record_frame(obs):
    frame = obs['agentview_image'][::-1]
    frames.append(frame)

record_frame(obs)

# Gripper control: +1 closes gripper, -1 opens gripper
GRIPPER_CLOSE = 1.0
GRIPPER_OPEN = -1.0

def execute_action(env, action, steps=1):
    obs = None
    for _ in range(steps):
        obs, reward, done, info = env.step(action)
        record_frame(obs)
        if done:
            break
    return obs, done

def compute_action(current_pos, target_pos, gripper, gain=15):
    diff = target_pos - current_pos
    action_xyz = np.clip(diff * gain, -1, 1)
    return np.concatenate([action_xyz, [0, 0, 0, gripper]])

# Phase 1: Open gripper
for _ in range(15):
    action = np.array([0, 0, 0, 0, 0, 0, GRIPPER_OPEN])
    obs, done = execute_action(env, action)

# Phase 2: Move above object
obj_pos = obs['alphabet_soup_1_pos'].copy()
target = np.array([obj_pos[0], obj_pos[1], 0.10])

for i in range(60):
    eef = obs['robot0_eef_pos']
    action = compute_action(eef, target, gripper=GRIPPER_OPEN, gain=20)
    obs, done = execute_action(env, action)
    if done:
        break
    dist = np.linalg.norm(eef - target)
    if dist < 0.008:
        break

# Phase 3: Lower to grasp
obj_pos = obs['alphabet_soup_1_pos'].copy()
grasp_z = obj_pos[2] + 0.035
target = np.array([obj_pos[0], obj_pos[1], grasp_z])

for i in range(80):
    eef = obs['robot0_eef_pos']
    action = compute_action(eef, target, gripper=GRIPPER_OPEN, gain=20)
    obs, done = execute_action(env, action)
    if done:
        break
    dist = np.linalg.norm(eef - target)
    if dist < 0.005:
        break

# Phase 4: Close gripper
for i in range(35):
    action = np.array([0, 0, -0.05, 0, 0, 0, GRIPPER_CLOSE])
    obs, done = execute_action(env, action)
    if done:
        break

# Phase 5: Lift
target = obs['robot0_eef_pos'].copy()
target[2] = 0.25

for i in range(60):
    eef = obs['robot0_eef_pos']
    action = compute_action(eef, target, gripper=GRIPPER_CLOSE, gain=15)
    obs, done = execute_action(env, action)
    if done:
        break
    dist = np.linalg.norm(eef - target)
    if dist < 0.01:
        break

# Phase 6: Move above target
basket_pos = obs['basket_1_pos'].copy()
target = np.array([basket_pos[0], basket_pos[1], 0.25])

for i in range(120):
    eef = obs['robot0_eef_pos']
    action = compute_action(eef, target, gripper=GRIPPER_CLOSE, gain=12)
    obs, done = execute_action(env, action)
    if done:
        break
    dist_xy = np.linalg.norm(eef[:2] - target[:2])
    if dist_xy < 0.015:
        break

# Phase 7: Lower to place
basket_pos = obs['basket_1_pos'].copy()
target = np.array([basket_pos[0], basket_pos[1], basket_pos[2] + 0.07])

for i in range(80):
    eef = obs['robot0_eef_pos']
    action = compute_action(eef, target, gripper=GRIPPER_CLOSE, gain=10)
    obs, done = execute_action(env, action)
    if done:
        break
    dist = np.linalg.norm(eef - target)
    if dist < 0.015:
        break

# Phase 8: Release
for _ in range(10):
    action = np.array([0, 0, 0, 0, 0, 0, GRIPPER_CLOSE])
    obs, done = execute_action(env, action)
    if done:
        break

for _ in range(30):
    action = np.array([0, 0, 0, 0, 0, 0, GRIPPER_OPEN])
    obs, done = execute_action(env, action)
    if done:
        break

# Phase 9: Move up
target = obs['robot0_eef_pos'].copy()
target[2] = 0.20

for i in range(40):
    eef = obs['robot0_eef_pos']
    action = compute_action(eef, target, gripper=GRIPPER_OPEN, gain=10)
    obs, done = execute_action(env, action)
    if done:
        break

# Settling
for _ in range(80):
    action = np.array([0, 0, 0, 0, 0, 0, GRIPPER_OPEN])
    obs, done = execute_action(env, action)
    if done:
        break

# Check success
success = env.check_success()
print(f"Task Success: {success}")

# Save video
video_path = os.path.join(OUTPUT_DIR, "libero_pick_place.mp4")
imageio.mimsave(video_path, frames, fps=20)
print(f"Video saved to {video_path}")

# Save metadata
meta = {
    "success": success,
    "num_tries": 1
}
meta_path = os.path.join(OUTPUT_DIR, "libero_pick_place_meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Metadata saved to {meta_path}")

env.close()
