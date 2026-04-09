

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg, EventManager
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def base_height_diff(
    env:ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot") 
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_height = asset.data.root_pos_w[:, 2].unsqueeze(1)
    ref_root_height = env.command_manager.get_command(command_name)
    return root_height - ref_root_height

def body_pose_w_diff(
    env:ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot") 
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_pose_w = asset.data.body_pose_w[:, asset_cfg.body_ids, :7]
    ref_body_pose_w = env.command_manager.get_command(command_name).reshape(env.num_envs, len(asset_cfg.body_ids), -1)
    return (body_pose_w - ref_body_pose_w).reshape(env.num_envs, -1)

def feet_pose_w_diff(
    env:ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot") 
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_pose_w = asset.data.body_pose_w[:, asset_cfg.body_ids, :2]
    ref_body_pose_w = env.command_manager.get_command(command_name).reshape(env.num_envs, len(asset_cfg.body_ids), -1)[:, :, :2]
    return (body_pose_w - ref_body_pose_w).reshape(env.num_envs, -1)

def base_height(
    env:ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot") 
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(1)

def feet_pose_w(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    pose = asset.data.body_pose_w[:, asset_cfg.body_ids, :2]
    pose[..., :2] = pose[..., :2] - env.scene.env_origins.unsqueeze(1)[:, :, :2]
    return pose.reshape(env.num_envs, -1)

def base_euler_xyz(
    env:ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot") 
) -> torch.Tensor:
    
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(asset.data.root_link_quat_w, wrap_to_2pi = True)

    return torch.cat([roll[:,None], pitch[:,None], yaw[:,None]], dim=-1)

def base_mass(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses()[:,asset_cfg.body_ids]
    return torch.tensor(masses).to(env.device)

def feet_contact_mask(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0
    return contacts.float()

class rand_push_force(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        event_name:str
    ) -> torch.Tensor:
        
        try:
            values = self._event_manager.get_term_cfg(event_name).func.rand_push_force
        except:
            values = torch.zeros(env.num_envs, 3).to(env.device)

        return values[:, :2]

class rand_push_torque(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)
        
    def __call__(
        self,
        env: ManagerBasedEnv,
        event_name:str
    ) -> torch.Tensor:
        
        try:
            values = self._event_manager.get_term_cfg(event_name).func.rand_push_torque
        except:
            values = torch.zeros(env.num_envs, 3).to(env.device)

        return values


