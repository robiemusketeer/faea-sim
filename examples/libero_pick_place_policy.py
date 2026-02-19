"""
Example policy for a LIBERO pick-and-place task (libero_object, task 0).

This file demonstrates the ONLY thing Claude needs to produce: a `policy`
function that receives (env, obs, h) and returns the final obs.

All env setup, frame recording, video saving, metadata, and teardown are
handled by the harness (libero_harness.py).
"""

import numpy as np


def policy(env, obs, h):
    """Pick up alphabet_soup_1 and place it in basket_1."""
    GRIPPER_OPEN = h.GRIPPER_OPEN
    GRIPPER_CLOSE = h.GRIPPER_CLOSE

    # Phase 1: Open gripper
    for _ in range(15):
        action = np.array([0, 0, 0, 0, 0, 0, GRIPPER_OPEN])
        obs, done = h.execute_action(env, action)

    # Phase 2: Move above object
    obj_pos = obs['alphabet_soup_1_pos'].copy()
    target = np.array([obj_pos[0], obj_pos[1], 0.10])
    for i in range(60):
        eef = obs['robot0_eef_pos']
        action = h.compute_action(eef, target, gripper=GRIPPER_OPEN, gain=20)
        obs, done = h.execute_action(env, action)
        if done or np.linalg.norm(eef - target) < 0.008:
            break

    # Phase 3: Lower to grasp
    obj_pos = obs['alphabet_soup_1_pos'].copy()
    grasp_z = obj_pos[2] + 0.035
    target = np.array([obj_pos[0], obj_pos[1], grasp_z])
    for i in range(80):
        eef = obs['robot0_eef_pos']
        action = h.compute_action(eef, target, gripper=GRIPPER_OPEN, gain=20)
        obs, done = h.execute_action(env, action)
        if done or np.linalg.norm(eef - target) < 0.005:
            break

    # Phase 4: Close gripper
    for i in range(35):
        action = np.array([0, 0, -0.05, 0, 0, 0, GRIPPER_CLOSE])
        obs, done = h.execute_action(env, action)
        if done:
            break

    # Phase 5: Lift
    target = obs['robot0_eef_pos'].copy()
    target[2] = 0.25
    for i in range(60):
        eef = obs['robot0_eef_pos']
        action = h.compute_action(eef, target, gripper=GRIPPER_CLOSE, gain=15)
        obs, done = h.execute_action(env, action)
        if done or np.linalg.norm(eef - target) < 0.01:
            break

    # Phase 6: Move above basket
    basket_pos = obs['basket_1_pos'].copy()
    target = np.array([basket_pos[0], basket_pos[1], 0.25])
    for i in range(120):
        eef = obs['robot0_eef_pos']
        action = h.compute_action(eef, target, gripper=GRIPPER_CLOSE, gain=12)
        obs, done = h.execute_action(env, action)
        if done or np.linalg.norm(eef[:2] - target[:2]) < 0.015:
            break

    # Phase 7: Lower to place
    basket_pos = obs['basket_1_pos'].copy()
    target = np.array([basket_pos[0], basket_pos[1], basket_pos[2] + 0.07])
    for i in range(80):
        eef = obs['robot0_eef_pos']
        action = h.compute_action(eef, target, gripper=GRIPPER_CLOSE, gain=10)
        obs, done = h.execute_action(env, action)
        if done or np.linalg.norm(eef - target) < 0.015:
            break

    # Phase 8: Release
    for _ in range(10):
        action = np.array([0, 0, 0, 0, 0, 0, GRIPPER_CLOSE])
        obs, done = h.execute_action(env, action)
        if done:
            break
    for _ in range(30):
        action = np.array([0, 0, 0, 0, 0, 0, GRIPPER_OPEN])
        obs, done = h.execute_action(env, action)
        if done:
            break

    # Phase 9: Retract upward and settle
    target = obs['robot0_eef_pos'].copy()
    target[2] = 0.20
    for i in range(40):
        eef = obs['robot0_eef_pos']
        action = h.compute_action(eef, target, gripper=GRIPPER_OPEN, gain=10)
        obs, done = h.execute_action(env, action)
        if done:
            break
    for _ in range(80):
        action = np.array([0, 0, 0, 0, 0, 0, GRIPPER_OPEN])
        obs, done = h.execute_action(env, action)
        if done:
            break

    return obs


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent
                          / "benchmarks" / "libero" / "src"))
    from libero_harness import run_episode

    run_episode(
        benchmark_name="libero_object",
        task_id=0,
        policy_fn=policy,
        output_dir=".",
    )
