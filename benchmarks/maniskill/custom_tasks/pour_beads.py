"""A contact-physical cup grasping and bead pouring task for FAEA.

The cup, handle, bowl, and beads are all constructed from ordinary SAPIEN
collision primitives. The task does not weld the cup to the gripper, attach
the beads to the cup, or move any task actor after reset. A successful episode
must physically grasp, lift, and tilt the cup over the bowl.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("PourBeads-v1", max_episode_steps=600)
class PourBeadsEnv(BaseEnv):
    """Grasp a handled cup and pour at least six of eight beads into a bowl."""

    SUPPORTED_ROBOTS: ClassVar[list[str]] = ["panda"]
    agent: Panda

    NUM_BEADS = 8
    REQUIRED_BEADS = 6
    BEAD_RADIUS = 0.006
    CUP_POS = np.array([-0.02, -0.18, 0.0], dtype=np.float32)
    BOWL_POS = np.array([0.18, 0.14, 0.0], dtype=np.float32)

    def __init__(
        self,
        *args,
        robot_uids: str = "panda",
        robot_init_qpos_noise: float = 0.0,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.48, -0.52, 0.42], target=[0.04, 0.0, 0.08])
        return [CameraConfig("base_camera", pose, 256, 256, 1.0, 0.01, 10)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.52, -0.62, 0.46], target=[0.04, 0.0, 0.08])
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 10)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    @staticmethod
    def _add_box(builder, pose, half_size, color, density=500):
        material = sapien.render.RenderMaterial(base_color=color)
        builder.add_box_collision(pose=pose, half_size=half_size, density=density)
        builder.add_box_visual(pose=pose, half_size=half_size, material=material)

    def _build_cup(self):
        builder = self.scene.create_actor_builder()
        silver = [0.62, 0.68, 0.72, 1.0]
        dark = [0.08, 0.10, 0.12, 1.0]

        # A flat base and four independent walls make a genuinely open vessel.
        self._add_box(
            builder, sapien.Pose(p=[0, 0, 0.004]), [0.032, 0.032, 0.004], silver
        )
        self._add_box(
            builder, sapien.Pose(p=[0.032, 0, 0.049]), [0.003, 0.032, 0.045], silver
        )
        self._add_box(
            builder, sapien.Pose(p=[-0.032, 0, 0.049]), [0.003, 0.032, 0.045], silver
        )
        self._add_box(
            builder, sapien.Pose(p=[0, 0.032, 0.049]), [0.029, 0.003, 0.045], silver
        )
        self._add_box(
            builder, sapien.Pose(p=[0, -0.032, 0.049]), [0.029, 0.003, 0.045], silver
        )

        # A three-piece loop handle faces the robot at reset (negative x).
        self._add_box(
            builder, sapien.Pose(p=[-0.052, 0, 0.076]), [0.020, 0.006, 0.006], dark
        )
        self._add_box(
            builder, sapien.Pose(p=[-0.052, 0, 0.032]), [0.020, 0.006, 0.006], dark
        )
        self._add_box(
            builder, sapien.Pose(p=[-0.072, 0, 0.054]), [0.006, 0.006, 0.028], dark
        )
        builder.set_initial_pose(sapien.Pose(p=self.CUP_POS))
        return builder.build(name="handled_cup")

    def _build_bowl(self):
        builder = self.scene.create_actor_builder()
        white = [0.92, 0.94, 0.98, 1.0]
        material = sapien.render.RenderMaterial(base_color=white)

        def add_static_box(p, half_size):
            pose = sapien.Pose(p=p)
            builder.add_box_collision(pose=pose, half_size=half_size)
            builder.add_box_visual(pose=pose, half_size=half_size, material=material)

        add_static_box([0, 0, 0.005], [0.11, 0.11, 0.005])
        add_static_box([0.105, 0, 0.035], [0.006, 0.11, 0.025])
        add_static_box([-0.105, 0, 0.035], [0.006, 0.11, 0.025])
        add_static_box([0, 0.105, 0.035], [0.099, 0.006, 0.025])
        add_static_box([0, -0.105, 0.035], [0.099, 0.006, 0.025])
        builder.set_initial_pose(sapien.Pose(p=self.BOWL_POS))
        return builder.build_static(name="bowl")

    def _build_bead(self, index: int):
        builder = self.scene.create_actor_builder()
        color = [0.12, 0.34 + 0.04 * (index % 3), 0.88, 1.0]
        material = sapien.render.RenderMaterial(base_color=color)
        builder.add_sphere_collision(radius=self.BEAD_RADIUS, density=700)
        builder.add_sphere_visual(radius=self.BEAD_RADIUS, material=material)
        builder.set_initial_pose(sapien.Pose(p=[0, 0, 0.2 + index * 0.02]))
        return builder.build(name=f"bead_{index}")

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cup = self._build_cup()
        self.bowl = self._build_bowl()
        self.beads = [self._build_bead(i) for i in range(self.NUM_BEADS)]
        self.cup_grasped_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.cup_lifted_once = torch.zeros_like(self.cup_grasped_once)
        self.cup_tilted_over_bowl_once = torch.zeros_like(self.cup_grasped_once)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            batch_size = len(env_idx)
            self.table_scene.initialize(env_idx)
            cup_pos = torch.tensor(self.CUP_POS).repeat(batch_size, 1)
            cup_q = torch.zeros((batch_size, 4))
            cup_q[:, 0] = 1
            self.cup.set_pose(Pose.create_from_pq(cup_pos, cup_q))
            self.cup.set_linear_velocity(torch.zeros((batch_size, 3)))
            self.cup.set_angular_velocity(torch.zeros((batch_size, 3)))

            offsets = [
                (-0.012, -0.012, 0.016),
                (-0.012, 0.012, 0.016),
                (0.012, -0.012, 0.016),
                (0.012, 0.012, 0.016),
                (-0.012, -0.012, 0.032),
                (-0.012, 0.012, 0.032),
                (0.012, -0.012, 0.032),
                (0.012, 0.012, 0.032),
            ]
            for bead, offset in zip(self.beads, offsets):
                bead_pos = cup_pos + torch.tensor(offset)
                bead.set_pose(Pose.create_from_pq(bead_pos, cup_q))
                bead.set_linear_velocity(torch.zeros((batch_size, 3)))
                bead.set_angular_velocity(torch.zeros((batch_size, 3)))

            self.cup_grasped_once[env_idx] = False
            self.cup_lifted_once[env_idx] = False
            self.cup_tilted_over_bowl_once[env_idx] = False

    def _bead_positions(self):
        return torch.stack([bead.pose.p for bead in self.beads], dim=1)

    def _beads_in_bowl(self, bead_positions):
        bowl_xy = torch.as_tensor(self.BOWL_POS[:2], device=self.device)
        inside_xy = torch.all(
            torch.abs(bead_positions[..., :2] - bowl_xy) < 0.098, dim=-1
        )
        inside_z = (bead_positions[..., 2] > 0.010) & (bead_positions[..., 2] < 0.070)
        return inside_xy & inside_z

    def _beads_in_cup(self, bead_positions):
        cup_matrix = self.cup.pose.to_transformation_matrix()
        relative = bead_positions - self.cup.pose.p[:, None, :]
        local = torch.einsum(
            "bij,bnj->bni", cup_matrix[:, :3, :3].transpose(1, 2), relative
        )
        inside_xy = torch.all(torch.abs(local[..., :2]) < 0.028, dim=-1)
        inside_z = (local[..., 2] > 0.008) & (local[..., 2] < 0.094)
        return inside_xy & inside_z

    def evaluate(self):
        bead_positions = self._bead_positions()
        num_beads_in_bowl = self._beads_in_bowl(bead_positions).sum(dim=1)
        num_beads_in_cup = self._beads_in_cup(bead_positions).sum(dim=1)
        is_grasped = self.agent.is_grasping(self.cup)
        cup_lifted = self.cup.pose.p[:, 2] > 0.035

        cup_up = self.cup.pose.to_transformation_matrix()[:, :3, 2]
        cup_tilted = cup_up[:, 2] < 0.60
        bowl_xy = torch.as_tensor(self.BOWL_POS[:2], device=self.device)
        cup_near_bowl = (
            torch.linalg.norm(self.cup.pose.p[:, :2] - bowl_xy, dim=1) < 0.18
        )
        cup_above_bowl = self.cup.pose.p[:, 2] > 0.08

        self.cup_grasped_once |= is_grasped
        self.cup_lifted_once |= cup_lifted
        self.cup_tilted_over_bowl_once |= cup_tilted & cup_near_bowl & cup_above_bowl

        success = (
            self.cup_grasped_once
            & self.cup_lifted_once
            & self.cup_tilted_over_bowl_once
            & (num_beads_in_bowl >= self.REQUIRED_BEADS)
        )
        return {
            "success": success,
            "is_grasped": is_grasped,
            "cup_grasped_once": self.cup_grasped_once.clone(),
            "cup_lifted_once": self.cup_lifted_once.clone(),
            "cup_tilted_over_bowl_once": self.cup_tilted_over_bowl_once.clone(),
            "num_beads_in_bowl": num_beads_in_bowl,
            "num_beads_in_cup": num_beads_in_cup,
        }

    def _get_obs_extra(self, info: dict):
        obs = {
            "tcp_pose": self.agent.tcp_pose.raw_pose,
            "cup_pose": self.cup.pose.raw_pose,
            "bowl_pose": self.bowl.pose.raw_pose,
            "bead_poses": torch.cat([bead.pose.raw_pose for bead in self.beads], dim=1),
            "num_beads_in_bowl": info["num_beads_in_bowl"],
            "num_beads_in_cup": info["num_beads_in_cup"],
        }
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        cup_to_tcp = torch.linalg.norm(self.cup.pose.p - self.agent.tcp_pose.p, dim=1)
        reach = 1 - torch.tanh(5 * cup_to_tcp)
        reward = reach
        reward += info["cup_grasped_once"].float()
        reward += info["cup_lifted_once"].float()
        reward += info["cup_tilted_over_bowl_once"].float()
        reward += info["num_beads_in_bowl"].float() / self.NUM_BEADS * 4
        reward[info["success"]] = 8
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
