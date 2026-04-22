
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class wrist_box_distance(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._object: RigidObject = env.scene[cfg.params["object_cfg"].name]
        self._asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._asset_cfg = cfg.params["asset_cfg"]
        
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        ) -> torch.Tensor:
        wrist_pos = self._asset.data.body_pos_w[:, self._asset_cfg.body_ids, :3]
        box_pos = self._object.data.body_pos_w[:, 0, :3] 
        box_handle_left = box_pos.clone()
        box_handle_right = box_pos.clone()
        box_handle_pos = torch.stack([box_handle_left, box_handle_right], dim=1) # [num_envs, 2, 3]
        wrist_box_diff = wrist_pos - box_handle_pos # [num_envs, 2, 3]
        wrist_pos_diff = torch.flatten(wrist_box_diff, start_dim=1) # [num_envs, 6]
        wrist_box_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)
        return torch.exp(-4 * wrist_box_error)

class box_pos_diff(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._object: RigidObject = env.scene[cfg.params["object_cfg"].name]
        self._table: RigidObject = env.scene[cfg.params["end_table_cfg"].name]
        
    def __call__(
        self,    
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        end_table_cfg:SceneEntityCfg = SceneEntityCfg("table"),
        ) -> torch.Tensor:
        object_diff = self._object.data.root_pos_w[:, :3] 
        table_diff = self._table.data.root_pos_w[:, :3] 
        box_pos_diff = object_diff - table_diff
        box_pos_error = torch.mean(torch.abs(box_pos_diff), dim=1)
        return torch.exp(-4 * box_pos_error)


class low_levelaction_rate_l2(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous_primitive_action = torch.zeros(env.num_envs, len(env.scene[cfg.params["asset_cfg"].name].data.joint_names)).to(env.device)
        
    def reset(self, env_ids):
        self._previous_primitive_action[env_ids].zero_()
    
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        action_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
        ) -> torch.Tensor:
        current_action = env.action_manager.get_term(action_name).primitive_action
        reward = torch.sum(torch.square(current_action - self._previous_primitive_action), dim=1)
        self._previous_primitive_action[:] = env.action_manager.get_term(action_name).primitive_action
        return reward
