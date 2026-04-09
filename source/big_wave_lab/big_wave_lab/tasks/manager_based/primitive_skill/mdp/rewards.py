
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

def base_height_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    
    root_height = asset.data.root_pos_w[:, 2].unsqueeze(1)
    ref_root_height = env.command_manager.get_command(command_name)
    root_height_diff = root_height - ref_root_height # [num_envs, 1]
    root_height_error = torch.mean(torch.abs(root_height_diff), dim=1)
    return torch.exp(-4 * root_height_error)

def body_pose_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    
    body_pose_w = asset.data.body_pose_w[:, asset_cfg.body_ids, :3]
    ref_body_pose_w = env.command_manager.get_command(command_name).reshape(env.num_envs, len(asset_cfg.body_ids), -1) [:, :, :3]
    body_pose_diff = ref_body_pose_w - body_pose_w
    body_pose_diff = torch.flatten(body_pose_diff, start_dim=1) # [num_envs, 6]
    body_pose_error = torch.mean(torch.abs(body_pose_diff), dim=1)
    return torch.exp(-4 * body_pose_error)

def feet_pose_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:    
    asset: Articulation = env.scene[asset_cfg.name]
    
    feet_pose_w = asset.data.body_pose_w[:, asset_cfg.body_ids, :2]
    ref_feet_pose_w = env.command_manager.get_command(command_name).reshape(env.num_envs, len(asset_cfg.body_ids), -1) [:, :, :2]
    feet_pos_diff = feet_pose_w - ref_feet_pose_w # [num_envs, 2, 2], two feet, position only
    feet_pos_diff = torch.flatten(feet_pos_diff, start_dim=1) # [num_envs, 4]
    feet_pos_error = torch.mean(torch.abs(feet_pos_diff), dim=1)
    return torch.exp(-4 * feet_pos_error)
    



def feet_distance(
    env: ManagerBasedRLEnv,
    max_distance: float,
    min_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """
    Calculates the reward based on the distance between the feet. 
    Penilize feet get close to each other or too far away.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
    d_min = torch.clamp(foot_dist - min_distance, -0.5, 0.)
    d_max = torch.clamp(foot_dist - max_distance, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

def default_joint_pos(
    env: ManagerBasedRLEnv,
    left_cfg: SceneEntityCfg,
    right_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:

    """
    Calculates the reward for keeping joint positions close to default positions, with a focus 
    on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    joint_diff = asset.data.joint_pos - asset.data.default_joint_pos
    left_yaw_roll = joint_diff[:, left_cfg.joint_ids]
    right_yaw_roll = joint_diff[:, right_cfg.joint_ids]
    yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
    yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
    return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

def upper_body_pos(   
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:

    """
    Calculates the reward for keeping upper body joint positions close to default positions.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    upper_body_diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    upper_body_error = torch.mean(torch.abs(upper_body_diff), dim=1)
    return torch.exp(-4 * upper_body_error)


def orientation(    
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """
    Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
    from the desired base orientation using the base euler angles and the projected gravity vector.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_link_quat_w, wrap_to_2pi = True)
    
    quat_mismatch = torch.exp(-torch.sum(torch.abs(
                    torch.cat([roll[:,None], pitch[:,None]],dim=-1)
                    ), dim=1) * 10)
    orientation = torch.exp(-torch.norm(asset.data.projected_gravity_b[:, :2], dim=1) * 20)
    return (quat_mismatch + orientation) / 2.
