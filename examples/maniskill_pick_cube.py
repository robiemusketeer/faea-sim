"""
Simple Pick-and-Place script for ManiSkill PickCube-v1 benchmark.

Uses IK-based delta position control (pd_ee_delta_pose) with waypoint-based trajectory.

Action Space (pd_ee_delta_pose):
- Total dimensions: 7 (6 arm + 1 gripper)
- Arm: [dx, dy, dz, drx, dry, drz] - normalized delta in [-1, 1]
  - Position maps to ~[-0.1, 0.1] meters
  - Rotation maps to ~[-0.5, 0.5] radians (axis-angle)
- Gripper: 1 value, normalized [-1, 1] where -1=closed, +1=open
"""

import gymnasium as gym
import numpy as np
import mani_skill.envs  # Register ManiSkill environments


def get_tcp_pos(env):
    """Get current TCP position."""
    return env.unwrapped.agent.tcp.pose.p.cpu().numpy()[0]


def get_cube_pos(env):
    """Get current cube position."""
    return env.unwrapped.cube.pose.p.cpu().numpy()[0]


def move_to_target(env, target_pos, gripper, frames, max_steps=100, threshold=0.015, gain=8.0):
    """Move TCP to target position using delta control.

    Note: Does not stop on env termination since termination indicates success in PickCube.
    """
    for step in range(max_steps):
        tcp_pos = get_tcp_pos(env)
        error = target_pos - tcp_pos
        dist = np.linalg.norm(error)

        if dist < threshold:
            return True, step

        # Compute delta action (clipped to [-1, 1])
        delta = np.clip(error * gain, -1, 1)

        action = np.zeros(7)
        action[0:3] = delta
        action[6] = gripper

        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render().cpu().numpy()[0])

    return False, max_steps


def actuate_gripper(env, gripper, frames, steps=50):
    """Actuate gripper for specified steps."""
    info = None
    for _ in range(steps):
        action = np.zeros(7)
        action[6] = gripper
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render().cpu().numpy()[0])
    return info


def main():
    print("Creating ManiSkill PickCube-v1 environment...")
    env = gym.make(
        "PickCube-v1",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        obs_mode="state",
    )

    obs, info = env.reset(seed=32)  # Seed 32: table-level goal, ~0.1m cube-goal distance
    frames = [env.render().cpu().numpy()[0]]

    # Get object and goal positions
    cube_pos = env.unwrapped.cube.pose.p.cpu().numpy()[0]
    goal_pos = env.unwrapped.goal_site.pose.p.cpu().numpy()[0]

    print(f"Cube position: {cube_pos}")
    print(f"Goal position: {goal_pos}")
    print(f"Initial TCP position: {get_tcp_pos(env)}")

    # Heights for waypoints
    PRE_GRASP_HEIGHT = 0.08  # Height above cube before lowering
    GRASP_HEIGHT = 0.02  # Lower to this height for grasp (cube center)
    LIFT_HEIGHT = 0.12  # Lift to this height

    # Phase 1: Open gripper and move above cube
    print("\nPhase 1: Move above cube")
    actuate_gripper(env, gripper=1, frames=frames, steps=10)
    target = np.array([cube_pos[0], cube_pos[1], PRE_GRASP_HEIGHT])
    converged, steps = move_to_target(env, target, gripper=1, frames=frames)
    print(f"  Converged: {converged} in {steps} steps, TCP: {get_tcp_pos(env)}")

    # Phase 2: Lower to grasp position
    print("\nPhase 2: Lower to grasp position")
    target = np.array([cube_pos[0], cube_pos[1], GRASP_HEIGHT])
    converged, steps = move_to_target(env, target, gripper=1, frames=frames, threshold=0.01)
    print(f"  Converged: {converged} in {steps} steps, TCP: {get_tcp_pos(env)}")

    # Phase 3: Close gripper
    print("\nPhase 3: Close gripper")
    info = actuate_gripper(env, gripper=-1, frames=frames, steps=60)
    is_grasped = info.get('is_grasped', False)
    if hasattr(is_grasped, 'item'):
        is_grasped = is_grasped.item()
    print(f"  Is grasped: {is_grasped}")

    # Phase 4: Lift with cube
    print("\nPhase 4: Lift with cube")
    tcp_pos = get_tcp_pos(env)
    target = np.array([tcp_pos[0], tcp_pos[1], LIFT_HEIGHT])
    converged, steps = move_to_target(env, target, gripper=-1, frames=frames, max_steps=50, gain=10.0)
    print(f"  Converged: {converged} in {steps} steps, TCP: {get_tcp_pos(env)}")
    print(f"  Cube position: {get_cube_pos(env)}")

    # Phase 5: Move to goal position
    print("\nPhase 5: Move to goal position")
    target = np.array([goal_pos[0], goal_pos[1], goal_pos[2]])
    converged, steps = move_to_target(env, target, gripper=-1, frames=frames, max_steps=100)
    print(f"  Converged: {converged} in {steps} steps, TCP: {get_tcp_pos(env)}")
    print(f"  Cube position: {get_cube_pos(env)}")

    # Phase 6: Open gripper to release
    print("\nPhase 6: Open gripper")
    actuate_gripper(env, gripper=1, frames=frames, steps=30)

    # Phase 7: Retreat
    print("\nPhase 7: Retreat")
    tcp_pos = get_tcp_pos(env)
    target = np.array([tcp_pos[0], tcp_pos[1], LIFT_HEIGHT])
    converged, steps = move_to_target(env, target, gripper=1, frames=frames, max_steps=30)
    print(f"  Converged: {converged} in {steps} steps")

    # Let physics settle
    for _ in range(20):
        action = np.zeros(7)
        action[6] = 1
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render().cpu().numpy()[0])

    # Check final result
    print("\n--- Final Result ---")
    cube_final = env.unwrapped.cube.pose.p.cpu().numpy()[0]
    dist_to_goal = np.linalg.norm(cube_final - goal_pos)
    print(f"Cube final position: {cube_final}")
    print(f"Goal position: {goal_pos}")
    print(f"Distance to goal: {dist_to_goal:.4f}m")
    print(f"Success threshold: 0.025m")

    success = info.get('success', False)
    if hasattr(success, 'item'):
        success = success.item()
    print(f"Environment success: {success}")
    print(f"Distance success: {dist_to_goal < 0.025}")

    # Save video
    output_path = "maniskill_pick_cube.mp4"
    print(f"\nSaving video to {output_path}...")

    import imageio
    imageio.mimsave(output_path, frames, fps=30)
    print("Done!")

    env.close()

    return success or dist_to_goal < 0.025


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
