

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import ContactSensor
from big_wave_lab.tasks.manager_based.primitive_skill.mdp.commands.gait_command import GaitCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def target_pos_diff(
    env:ManagerBasedRLEnv,
    action_name:str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot") 
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return env.action_manager.get_term(action_name).target_primitive_action - asset.data.joint_pos