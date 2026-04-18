

from __future__ import annotations

import math
import re
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.version import compare_versions

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class reset_object_poses_with_goal(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._object_goal_pose = torch.zeros(env.num_envs, 3).to(env.device)
    
    @property
    def object_goal_pose(self):
        return self._object_goal_pose
    
    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        init_pose_range: dict[str, tuple[float, float]],
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        start_table_cfg: SceneEntityCfg = SceneEntityCfg("strat_table"),
        end_table_cfg: SceneEntityCfg = SceneEntityCfg("end_table"),
    ):
        """Reset the asset root states to a random position and orientation uniformly within the given ranges.

        Args:
            env: The RL environment instance.
            env_ids: The environment IDs to reset the object poses for.
            object_cfg: The configuration for the sorting beaker asset.
            pose_range: The dictionary of pose ranges for the objects. Keys are
                        ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.
        """
        # extract the used quantities (to enable type-hinting)
        object = env.scene[object_cfg.name]

        # start pos setting 
        object_root_states = object.data.default_root_state[env_ids].clone()

        range_list = [init_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=object.device)

        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=object.device
        )
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        positions_object = (
            object_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        )
        orientations_object = math_utils.quat_mul(object_root_states[:, 3:7], orientations_delta)

        # set into the physics simulation
        object.write_root_pose_to_sim(
            torch.cat([positions_object, orientations_object], dim=-1), env_ids=env_ids
        )