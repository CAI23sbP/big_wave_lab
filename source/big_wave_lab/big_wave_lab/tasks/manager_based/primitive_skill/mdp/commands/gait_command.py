
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING


import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from .utils import build_leg_joint_map 

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import GaitCommandCfg

class GaitCommand(CommandTerm):
    cfg: GaitCommandCfg
    def __init__(self, cfg: GaitCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        
        self._joint_names = list(self.robot.data.joint_names)
        self._leg_joint_map = build_leg_joint_map(self._joint_names)

        missing = [k for k, v in self._leg_joint_map.items() if v is None]
        if missing:
            raise RuntimeError(
                f"Failed to find required joints for gait command: {missing}\n"
                f"joint_names={self._joint_names}"
            )
        self.env = env 
        """
        initalize target waypoints
        """
        self.step_dt = env.step_dt
        
        self._sin_pos = torch.zeros(env.num_envs).to(env.device)
        self._cos_pos = torch.zeros(env.num_envs).to(env.device)
        self._curriculum = torch.zeros(env.num_envs).to(env.device)
        self.metrics["error_vel_xy"] = torch.zeros(env.num_envs).to(env.device)
        self.metrics["error_vel_yaw"] = torch.zeros(env.num_envs).to(env.device)
        self.vel_command_b = torch.zeros(env.num_envs, 2+3).to(env.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self._ref_dof_pos = torch.zeros_like(self.robot.data.default_joint_pos)
        
    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "GaitCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """# (num_envs, 5) : gait 2 + lin_vel 2 + head 1 """
        return self.vel_command_b.clone()
    
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        self._curriculum[env_ids] = 0.
        return super().reset(env_ids)

    @property
    def curriculum(self) -> torch.Tensor:
        return self._curriculum.clone()
    
    def _update_metrics(self):
        
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_vel_xy"] += torch.norm(self.vel_command_b[:, 2:4] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        self.metrics["error_vel_yaw"] += torch.abs(self.vel_command_b[:, 4] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step

        self._curriculum += torch.exp(-torch.sum(torch.square(self.vel_command_b[:, 2:4] - self.robot.data.root_lin_vel_b[:, :2]), dim=1) * self.cfg.tracking_sigma) / max_command_step
    
    @property
    def ref_dof_pos(self):
        return self._ref_dof_pos.clone()
    
    def _update_command(self):
        phase = self.env.episode_length_buf * self.step_dt / self.cfg.cycle_time 
        self._sin_pos = torch.sin(2 * torch.pi * phase)
        self._cos_pos = torch.cos(2 * torch.pi * phase)
        self.vel_command_b[:, 0] = self._sin_pos
        self.vel_command_b[:, 1] = self._cos_pos
        heading_error = math_utils.wrap_to_pi(self.heading_target - self.robot.data.heading_w)
        self.vel_command_b[:, 4] = torch.clip(
            self.cfg.heading_control_stiffness * heading_error,
            min=self.cfg.ranges.ang_vel_z[0],
            max=self.cfg.ranges.ang_vel_z[1],
        )
        sin_pos_l = self._sin_pos.clone()
        sin_pos_r = self._sin_pos.clone()
        
        scale_1 = self.cfg.target_joint_pos_scale 
        scale_2 = 2 * self.cfg.target_joint_pos_scale
        sin_pos_l[sin_pos_l > 0] = 0
        self._ref_dof_pos = torch.zeros_like(self.robot.data.default_joint_pos)

        jm = self._leg_joint_map

        # left leg
        self._ref_dof_pos[:, jm["left_hip_pitch"]] = sin_pos_l * scale_1 # left_hip_pitch_joint   
        self._ref_dof_pos[:, jm["left_knee"]] = sin_pos_l * scale_2     # left_knee_joint
        self._ref_dof_pos[:, jm["left_ankle_pitch"]] = sin_pos_l * scale_1 # left_ankle_joint
        sin_pos_r[sin_pos_r < 0] = 0

        # right leg
        self._ref_dof_pos[:, jm["right_hip_pitch"]] = sin_pos_r * scale_1 # right_hip_pitch_joint
        self._ref_dof_pos[:, jm["right_knee"]] = sin_pos_r * scale_2 # right_knee_joint
        self._ref_dof_pos[:, jm["right_ankle_pitch"]] = sin_pos_r * scale_1 # right_ankle_joint
        self._ref_dof_pos[torch.abs(self._sin_pos) < 0.1] = 0
        
    def _resample_command(self, env_ids: Sequence[int]):
        self._curriculum[env_ids] = 0.0
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 3] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
        
    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, 2:4])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
