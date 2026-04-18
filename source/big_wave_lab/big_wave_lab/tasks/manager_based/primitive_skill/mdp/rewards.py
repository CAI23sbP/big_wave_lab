
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

def track_lin_vel_xy(    
    env: ManagerBasedRLEnv,
    command_name: str,
    tracking_sigma:float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, 2:]
    lin_vel_diff = command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]
    lin_vel_error = torch.sum(torch.square(
        lin_vel_diff), dim=1)
    return torch.exp(-lin_vel_error * tracking_sigma)

def track_ang_vel_z(    
    env: ManagerBasedRLEnv,
    command_name: str,
    tracking_sigma:float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, 2:]
    ang_vel_diff = command[:, 2] - asset.data.root_link_ang_vel_b[:, 2]
    ang_vel_error = torch.square(
        ang_vel_diff)
    return torch.exp(-ang_vel_error * tracking_sigma)

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
    
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    
    quat_mismatch = torch.exp(-torch.sum(torch.abs(
                    torch.cat([roll[:,None], pitch[:,None]],dim=-1)
                    ), dim=1) * 10)
    orientation = torch.exp(-torch.norm(asset.data.projected_gravity_b[:, :2], dim=1) * 20)
    return (quat_mismatch + orientation) / 2.


def joint_pos_diff(
    env: ManagerBasedRLEnv,
    command_name:str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    joint_pos = asset.data.joint_pos.clone()
    pos_target = command.ref_dof_pos.clone()
    diff = joint_pos - pos_target
    return torch.exp(-2 * torch.norm(diff, dim=1)) \
            - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)


class feet_clearance(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._num_feet = len(cfg.params["asset_cfg"].body_ids)
        self._feet_height = torch.zeros(env.num_envs, self._num_feet, device=env.device)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self._cycle_time = env.command_manager.get_term(cfg.params["command_name"]).cfg.cycle_time
        self._last_feet_z = torch.ones(env.num_envs, self._num_feet, device=env.device) * cfg.params["last_feet_z"]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_feet_height: float,
        last_feet_z: float,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        contact = self.contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, 2] > 5.0

        feet_z = asset.data.body_state_w[:, asset_cfg.body_ids, 2] - 0.05
        delta_z = feet_z - self._last_feet_z
        self._feet_height += delta_z
        self._last_feet_z.copy_(feet_z)

        phase = env.episode_length_buf * env.step_dt / self._cycle_time
        sin_pos = torch.sin(2 * torch.pi * phase)

        stance_mask = torch.zeros((env.num_envs, self._num_feet), device=env.device)
        stance_mask[:, 0] = sin_pos >= 0
        stance_mask[:, 1] = sin_pos < 0
        stance_mask[torch.abs(sin_pos) < 0.1] = 1.0
        swing_mask = 1.0 - stance_mask

        rew_pos = torch.abs(self._feet_height - target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)

        self._feet_height *= (~contact).float()
        return rew_pos
    
def feet_contact_number(
    env: ManagerBasedRLEnv,
    command_name:str,
    sensor_cfg: SceneEntityCfg, 
    ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, 2] > 5.
    cycle_time = env.command_manager.get_term(command_name).cfg.cycle_time
    
    phase = env.episode_length_buf * env.step_dt / cycle_time

    # Compute swing mask
    sin_pos = torch.sin(2 * torch.pi * phase)
    # Add double support phase
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    # left foot stance
    stance_mask[:, 0] = sin_pos >= 0
    # right foot stance
    stance_mask[:, 1] = sin_pos < 0
    # Double support phase
    stance_mask[torch.abs(sin_pos) < 0.1] = 1
    reward = torch.where(contact == stance_mask, 1.0, -0.3)
    return torch.mean(reward, dim=1).float()

class feet_air_time(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        num_feet = len(cfg.params["sensor_cfg"].body_ids)
        self.last_contacts = torch.zeros(env.num_envs, num_feet, dtype=torch.bool, device=env.device)
        self.feet_air_time = torch.zeros(env.num_envs, num_feet, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        command_name: str,
        contact_threshold: float = 5.0,
        max_air_time: float = 0.5,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        feet_ids = sensor_cfg.body_ids
        cycle_time = env.command_manager.get_term(command_name).cfg.cycle_time

        phase = env.episode_length_buf * env.step_dt / cycle_time
        sin_pos = torch.sin(2 * torch.pi * phase)

        stance_mask = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)
        stance_mask[:, 0] = sin_pos >= 0
        stance_mask[:, 1] = sin_pos < 0

        double_support = torch.abs(sin_pos) < 0.1
        stance_mask[double_support] = True

        contact = torch.norm(contact_sensor.data.net_forces_w[:, feet_ids, :], dim=-1) > contact_threshold

        contact_filt = contact | stance_mask | self.last_contacts
        self.last_contacts = contact

        first_contact = (self.feet_air_time > 0.0) & contact_filt
        self.feet_air_time += env.step_dt

        air_time = torch.clamp(self.feet_air_time, 0.0, max_air_time) * first_contact.float()
        self.feet_air_time *= (~contact_filt).float()
        return air_time.sum(dim=1)
        
def foot_slip(    
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history

    contact = net_contact_forces[:, -1, sensor_cfg.body_ids, 2] > 5.
    foot_speed_norm = torch.norm(asset.data.body_com_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    rew = torch.sqrt(foot_speed_norm)
    rew *= contact
    return torch.sum(rew, dim=1)   

def knee_distance(    
    env: ManagerBasedRLEnv,
    min_dist:float, 
    max_dist:float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    
    knee_pos = asset.data.body_com_pos_w[:, asset_cfg.body_ids, :2]
    knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
    fd = min_dist
    max_df = max_dist / 2
    d_min = torch.clamp(knee_dist - fd, -0.5, 0.)
    d_max = torch.clamp(knee_dist - max_df, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


def feet_contact_forces(
    env: ManagerBasedRLEnv, max_contact_force: float, sensor_cfg: SceneEntityCfg
    ):
    """
    Calculates the reward for keeping contact forces within a specified range. Penalizes
    high contact forces on the feet.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history

    return torch.sum((torch.norm(net_contact_forces[:, -1, sensor_cfg.body_ids, :], dim=-1) \
            - max_contact_force).clip(0, 400), dim=1)


def vel_mismatch_exp(    
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Computes a reward based on the mismatch in the robot's linear and angular velocities. 
    Encourages the robot to maintain a stable velocity by penalizing large deviations.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_link_lin_vel_b
    base_ang_vel = asset.data.root_link_ang_vel_b

    lin_mismatch = torch.exp(-torch.square(base_lin_vel[:, 2]) * 10)
    ang_mismatch = torch.exp(-torch.norm(base_ang_vel[:, :2], dim=1) * 5.)

    c_update = (lin_mismatch + ang_mismatch) / 2.

    return c_update


def low_speed(
    env: ManagerBasedRLEnv,
    command_name:str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Rewards or penalizes the robot based on its speed relative to the commanded speed. 
    This function checks if the robot is moving too slow, too fast, or at the desired speed, 
    and if the movement direction matches the command.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command(command_name)[:, 2:]
    base_lin_vel = asset.data.root_link_lin_vel_b
    # Calculate the absolute value of speed and command for comparison
    absolute_speed = torch.abs(base_lin_vel[:, 0])
    absolute_command = torch.abs(commands[:, 0])

    # Define speed criteria for desired range
    speed_too_low = absolute_speed < 0.5 * absolute_command
    speed_too_high = absolute_speed > 1.2 * absolute_command
    speed_desired = ~(speed_too_low | speed_too_high)

    # Check if the speed and command directions are mismatched
    sign_mismatch = torch.sign(
        base_lin_vel[:, 0]) != torch.sign(commands[:, 0])

    # Initialize reward tensor
    reward = torch.zeros_like(base_lin_vel[:, 0])

    # Assign rewards based on conditions
    # Speed too low
    reward[speed_too_low] = -1.0
    # Speed too high
    reward[speed_too_high] = 0.
    # Speed within desired range
    reward[speed_desired] = 1.2
    # Sign mismatch has the highest priority
    reward[sign_mismatch] = -2.0
    return reward * (commands[:, 0].abs() > 0.1)


def track_vel_hard(
    env: ManagerBasedRLEnv,
    command_name:str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command(command_name)[:, 2:]
    base_lin_vel = asset.data.root_link_lin_vel_b
    base_ang_vel = asset.data.root_link_ang_vel_b

    lin_vel_error = torch.norm(
        commands[:, :2] - base_lin_vel[:, :2], dim=1)
    lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

    # Tracking of angular velocity commands (yaw)
    ang_vel_error = torch.abs(
        commands[:, 2] - base_ang_vel[:, 2])
    ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

    linear_error = 0.2 * (lin_vel_error + ang_vel_error)

    return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error


class base_acc(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.last_root_vel = torch.zeros(env.num_envs, 6, device=env.device)

    def reset(self, env_ids):
        asset_cfg = self.cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = self._env.scene[asset_cfg.name]
        self.last_root_vel[env_ids] = asset.data.root_vel_w[env_ids]

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        asset: Articulation = env.scene[asset_cfg.name]
        root_vel = asset.data.root_vel_w
        root_acc = self.last_root_vel - root_vel
        reward = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        self.last_root_vel.copy_(root_vel)
        return reward

class action_smoothness(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        action_dim = env.action_manager.action.shape[1]
        self.last_last_actions = torch.zeros(env.num_envs, action_dim, device=env.device)
        self.last_actions = torch.zeros(env.num_envs, action_dim, device=env.device)

    def reset(self, env_ids):
        self.last_last_actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
    ):
        actions = env.action_manager.action
        term_1 = torch.sum(torch.square(actions - self.last_actions), dim=1)
        term_2 = torch.sum(torch.square(actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(actions), dim=1)
        reward = term_1 + term_2 + term_3
        self.last_last_actions[:] = self.last_actions
        self.last_actions[:] = actions

        return reward
    
def base_height_exp(    
    env: ManagerBasedRLEnv,
    command_name: str,
    base_height_target: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    
    cycle_time = env.command_manager.get_term(command_name).cfg.cycle_time
    
    phase = env.episode_length_buf * env.step_dt / cycle_time

    sin_pos = torch.sin(2 * torch.pi * phase)
    # Add double support phase
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    # left foot stance
    stance_mask[:, 0] = sin_pos >= 0
    # right foot stance
    stance_mask[:, 1] = sin_pos < 0
    # Double support phase
    stance_mask[torch.abs(sin_pos) < 0.1] = 1
    
    measured_heights = torch.sum(
        asset.data.body_state_w[:, asset_cfg.body_ids, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
    base_height = asset.data.root_pos_w[:, 2] - (measured_heights - 0.05)
    return torch.exp(-torch.abs(base_height - base_height_target) * 100)
