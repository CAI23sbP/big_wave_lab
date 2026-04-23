

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

def far_from_goal(
    env:ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"), 
    end_table_cfg: SceneEntityCfg = SceneEntityCfg("table") 
) -> torch.Tensor:
    target_obj = env.scene[object_cfg.name]
    end_table = env.scene[end_table_cfg.name]
    diff = target_obj.data.body_pos_w - end_table.data.body_pos_w 
    return torch.flatten(diff[:, 0, :3], start_dim=1)

def wrist_box_diff_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object") 
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    object = env.scene[object_cfg.name]
    body_pos_w = robot.data.body_pos_w
    robot_eef_pos = body_pos_w[:, asset_cfg.body_ids, :3]
    robot_eef_to_object = robot_eef_pos - object.data.body_pos_w [:, :, :3]
    return torch.flatten(robot_eef_to_object, start_dim=1)

def wrist_pos_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    body_pos_w = robot.data.body_pos_w
    robot_eef_pos = body_pos_w[:, asset_cfg.body_ids, :3]
    return torch.flatten(robot_eef_pos, start_dim=1)

def box_pos(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object") 
) -> torch.Tensor:
    object = env.scene[object_cfg.name]
    return torch.flatten(object.data.body_pos_w[:, 0] , start_dim=1)


def end_table_pos(
    env: ManagerBasedRLEnv,
    end_table_cfg: SceneEntityCfg = SceneEntityCfg("table") 
) -> torch.Tensor:
    end_table = env.scene[end_table_cfg.name]
    return torch.flatten(end_table.data.body_pos_w[:, 0] , start_dim=1)


class image_features(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)
