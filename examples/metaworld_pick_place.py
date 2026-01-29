#!/usr/bin/env python3
"""
Metaworld Pick-and-Place Example

This script demonstrates a pick-and-place task using the Metaworld benchmark.
The agent picks up a puck from its initial position and places it at a goal location.

Key Implementation Notes:
    - Gripper convention: +1 = CLOSE, -1 = OPEN (different from some other sims)
    - Success is measured while HOLDING the object at the goal position
    - Goal Z can be elevated (0.05-0.30), object must be transported to that height
    - Success threshold: obj_to_target <= 0.07 meters (7cm 3D distance)

Usage:
    python metaworld_pick_place.py [seed]

    seed: Optional random seed (default: 42)

Dependencies:
    - gymnasium >= 1.1
    - mujoco >= 3.0.0
    - metaworld
    - numpy

Example:
    python metaworld_pick_place.py 42   # Run with seed 42
    python metaworld_pick_place.py 100  # Run with seed 100
"""

import numpy as np
import gymnasium as gym


def parse_observation(obs):
    """
    Parse the 39D observation vector into meaningful components.

    Observation structure:
        [0:3]   - End-effector XYZ position
        [3]     - Gripper openness (0=closed, 1=open)
        [4:7]   - Object position (XYZ)
        [7:11]  - Object quaternion
        [36:39] - Goal position (XYZ)
    """
    return {
        'eef_pos': obs[0:3].copy(),
        'gripper': obs[3],
        'obj_pos': obs[4:7].copy(),
        'obj_quat': obs[7:11].copy(),
        'goal_pos': obs[36:39].copy()
    }


def compute_action(current_pos, target_pos, gripper_state, gain=10.0):
    """
    Compute action to move end-effector toward target.

    Args:
        current_pos: Current EEF position (3D)
        target_pos: Target position (3D)
        gripper_state: 1.0 (closed) or -1.0 (open) - Metaworld convention
        gain: Control gain (higher = faster but less stable)

    Returns:
        4D action: [dx, dy, dz, gripper]
    """
    diff = target_pos - current_pos
    action_xyz = np.clip(diff * gain, -1.0, 1.0)
    return np.concatenate([action_xyz, [gripper_state]])


def move_to_position(env, obs, target_pos, gripper_state, max_steps=100,
                     threshold=0.02, gain=10.0, verbose=False):
    """
    Move end-effector to target position.

    Args:
        env: Metaworld environment
        obs: Current observation
        target_pos: Target XYZ position
        gripper_state: 1.0 (closed) or -1.0 (open) - Metaworld convention
        max_steps: Maximum steps to attempt
        threshold: Distance threshold for success
        gain: Control gain
        verbose: Print debug info

    Returns:
        obs: Final observation
        info: Final info dict
        success: Whether target was reached
    """
    info = {}
    for step in range(max_steps):
        parsed = parse_observation(obs)
        current_pos = parsed['eef_pos']
        dist = np.linalg.norm(target_pos - current_pos)

        if verbose and step % 20 == 0:
            print(f"  Step {step}: dist={dist:.4f}, eef={current_pos}")

        if dist < threshold:
            if verbose:
                print(f"  Reached target at step {step}")
            return obs, info, True

        action = compute_action(current_pos, target_pos, gripper_state, gain)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    return obs, info, False


def close_gripper(env, obs, steps=30, verbose=False):
    """Close the gripper to grasp an object."""
    if verbose:
        print("  Closing gripper...")
    for _ in range(steps):
        action = np.array([0.0, 0.0, 0.0, 1.0])  # +1 = close in Metaworld
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return obs, info


def open_gripper(env, obs, steps=20, verbose=False):
    """Open the gripper to release an object."""
    if verbose:
        print("  Opening gripper...")
    for _ in range(steps):
        action = np.array([0.0, 0.0, 0.0, -1.0])  # -1 = open in Metaworld
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return obs, info


def run_pick_and_place(env, obs, verbose=True):
    """
    Execute pick-and-place manipulation sequence.

    Key insight: Success is measured while holding the object at the goal
    position (including Z height). The goal can be at elevated positions
    (Z=0.05 to 0.30), and we must transport the object to that height.

    Phases:
        1. Open gripper
        2. Move above object (XY alignment)
        3. Lower to grasp height
        4. Close gripper (grasp)
        5. Lift object
        6. Move to goal position (holding object at goal Z)
        7. Release object
        8. Retreat upward

    Returns:
        obs: Final observation
        info: Info dict with 'success' and 'obj_to_target' keys
    """
    parsed = parse_observation(obs)
    obj_pos = parsed['obj_pos']
    goal_pos = parsed['goal_pos']

    if verbose:
        print(f"\nInitial state:")
        print(f"  EEF: {parsed['eef_pos']}")
        print(f"  Object: {obj_pos}")
        print(f"  Goal: {goal_pos}")

    # Height offsets - calibrated for Metaworld pick-place-v3
    approach_height = 0.12
    grasp_z = 0.05
    lift_height = 0.25

    # Phase 1: Open gripper
    if verbose:
        print("\nPhase 1: Opening gripper")
    obs, info = open_gripper(env, obs, steps=15, verbose=verbose)

    parsed = parse_observation(obs)
    obj_pos = parsed['obj_pos']
    if verbose:
        print(f"  Object position: {obj_pos}")

    # Phase 2: Move above object
    if verbose:
        print("\nPhase 2: Moving above object")
    above_obj = np.array([obj_pos[0], obj_pos[1], approach_height])
    obs, info, reached = move_to_position(
        env, obs, above_obj, gripper_state=-1.0,
        max_steps=100, threshold=0.015, gain=10.0, verbose=verbose
    )

    # Phase 3: Lower to object
    if verbose:
        print("\nPhase 3: Lowering to object")
    grasp_pos = np.array([obj_pos[0], obj_pos[1], grasp_z])
    obs, info, reached = move_to_position(
        env, obs, grasp_pos, gripper_state=-1.0,
        max_steps=100, threshold=0.008, gain=8.0, verbose=verbose
    )

    # Phase 4: Close gripper
    if verbose:
        print("\nPhase 4: Grasping object")
    obs, info = close_gripper(env, obs, steps=50, verbose=verbose)

    # Phase 5: Lift object
    if verbose:
        print("\nPhase 5: Lifting object")
    lift_pos = np.array([obj_pos[0], obj_pos[1], lift_height])
    obs, info, reached = move_to_position(
        env, obs, lift_pos, gripper_state=1.0,
        max_steps=100, threshold=0.02, gain=5.0, verbose=verbose
    )

    parsed = parse_observation(obs)
    obj_z_after_lift = parsed['obj_pos'][2]
    if verbose:
        print(f"  Object Z after lift: {obj_z_after_lift:.4f}")
    if obj_z_after_lift < 0.10:
        print("  WARNING: Object may not be grasped properly!")

    # Phase 6: Move to goal
    if verbose:
        print("\nPhase 6: Moving to goal (holding object)")
    obs, info, reached = move_to_position(
        env, obs, goal_pos, gripper_state=1.0,
        max_steps=200, threshold=0.015, gain=5.0, verbose=verbose
    )

    parsed = parse_observation(obs)
    obj_at_goal = parsed['obj_pos']
    dist_at_goal = np.linalg.norm(obj_at_goal - goal_pos)
    if verbose:
        print(f"  Object at goal check: dist={dist_at_goal:.4f}")
        print(f"  Success (holding): {info.get('success', dist_at_goal < 0.07)}")

    # Wait to register success
    for _ in range(30):
        action = np.array([0.0, 0.0, 0.0, 1.0])
        obs, reward, terminated, truncated, info = env.step(action)
        if info.get('success', False):
            if verbose:
                print(f"  SUCCESS registered!")
            break

    # Phase 7: Release object
    if verbose:
        print("\nPhase 7: Releasing object")
    obs, info_final = open_gripper(env, obs, steps=20, verbose=verbose)

    # Let object settle
    for _ in range(20):
        action = np.array([0.0, 0.0, 0.0, -1.0])
        obs, reward, terminated, truncated, info_final = env.step(action)

    # Phase 8: Retreat
    if verbose:
        print("\nPhase 8: Retreating")
    retreat_pos = np.array([goal_pos[0], goal_pos[1], lift_height])
    obs, info_final, reached = move_to_position(
        env, obs, retreat_pos, gripper_state=-1.0,
        max_steps=50, threshold=0.03, gain=8.0, verbose=verbose
    )

    return obs, info


def main():
    """Main entry point for pick-and-place demonstration."""
    print("=" * 60)
    print("Metaworld Pick-and-Place Example")
    print("=" * 60)

    import metaworld  # noqa: F401

    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    print(f"\nCreating environment with seed={seed}...")
    env = gym.make('Meta-World/MT1', env_name='pick-place-v3', seed=seed)

    obs, info = env.reset()
    print(f"Environment created. Observation shape: {obs.shape}")

    obs, info = run_pick_and_place(env, obs, verbose=True)

    parsed = parse_observation(obs)
    obj_pos = parsed['obj_pos']
    goal_pos = parsed['goal_pos']
    dist_to_goal = np.linalg.norm(obj_pos - goal_pos)

    print("\n" + "=" * 60)
    print("Final Result")
    print("=" * 60)
    print(f"Object position: {obj_pos}")
    print(f"Goal position: {goal_pos}")
    print(f"Distance to goal (3D): {dist_to_goal:.4f}")
    print(f"Success threshold: 0.07")
    print(f"Task success (info): {info.get('success', 'N/A')}")

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
