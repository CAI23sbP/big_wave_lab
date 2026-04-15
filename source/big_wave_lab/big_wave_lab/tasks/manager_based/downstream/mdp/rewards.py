
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

def target_joint_pos_diff(
    env: ManagerBasedRLEnv,
    action_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    
    target_pos_error = torch.sum(
        torch.square(env.action_manager.get_term(action_name).target_primitive_action - asset.data.joint_pos),
        dim=1,
    )
    return torch.exp(-4 * target_pos_error)
    
    
def box_pos_diff(    
    env: ManagerBasedRLEnv,
    action_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:

    box_pos_diff = self.box_root_states[:, :3] - self.box_goal_pos
    box_pos_error = torch.mean(torch.abs(box_pos_diff), dim=1)
    return torch.exp(-4 * box_pos_error), box_pos_error
