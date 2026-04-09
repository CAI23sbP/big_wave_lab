

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

class push_by_setting_force(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        """
        same as push_by_setting_velocity
        """
        self._rand_push_force = torch.zeros(env.num_envs, 3).to(env.device)
        self._rand_push_torque = torch.zeros(env.num_envs, 3).to(env.device)
    
    @property
    def rand_push_force(self):
        return self._rand_push_force
    
    @property
    def rand_push_torque(self):
        return self._rand_push_torque
        
    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        vel_w = asset.data.root_vel_w[env_ids].clone()

        range_list = [
            velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=asset.device, dtype=vel_w.dtype)

        push_noise = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device
        )

        vel_w = vel_w + push_noise
        self._rand_push_force[env_ids] = push_noise[:, :3]
        self._rand_push_torque[env_ids] = push_noise[:, 3:]        # set the velocities into the physics simulation
        asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

