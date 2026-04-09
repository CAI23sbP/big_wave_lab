
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
import numpy as np 
from .utils import sample_int_from_float, sample_fp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import FeetTargetCommandCfg

class FeetTargetCommand(CommandTerm):
    cfg: FeetTargetCommandCfg
    def __init__(self, cfg: FeetTargetCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.step_dt = env.step_dt
        self.robot: Articulation = env.scene[cfg.asset_name]
        
        self.feet_indices = self.robot.find_bodies(cfg.body_names)[0]
        self.ori_feet_pos = self.robot.data.body_link_pose_w[:, self.feet_indices, :2].clone() # [num_envs, 2, 3], two hands
        
        self.target_wp, self.num_pairs, self.num_wp = sample_fp(self.device, num_points=cfg.total_num_points, num_wp=cfg.num_way_points, ranges=cfg.ranges) # relative, self.target_wp.shape=[num_pairs, num_wp, 2, 7]
        self.target_wp_i = torch.randint(0, self.num_pairs, (self.num_envs,), device=self.device) # for each env, choose one seq, [num_envs]
        self.target_wp_j = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) # for each env, the timestep in the seq is initialized to 0, [num_envs]
        self.target_wp_dt = 1 / cfg.resampling_time_range[1]
        self.target_wp_update_steps = self.target_wp_dt / self.step_dt # not necessary integer
        assert self.step_dt <= self.target_wp_dt, f"self.step_dt {self.step_dt} must be less than self.target_wp_dt {self.target_wp_dt}"
        self.target_wp_update_steps_int = sample_int_from_float(self.target_wp_update_steps)
        self.delayed_obs_target_wp_steps = 0.0
        self.delayed_obs_target_wp_steps_int = sample_int_from_float(self.delayed_obs_target_wp_steps)

        self.ref_feet_pos = self.target_wp[self.target_wp_i, self.target_wp_j] + self.ori_feet_pos # [num_envs, 2, 2], two hands
        self.delayed_obs_target_wp = self.target_wp[self.target_wp_i, torch.maximum(self.target_wp_j - self.delayed_obs_target_wp_steps_int, torch.tensor(0))]

        self.metrics["error_feet"] = torch.zeros(self.num_envs, device=self.device)
        self.env = env

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "BaseHeightCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """# (num_envs, 2, 2) -> (num_envs, 2 x 2)"""
        return self.ref_feet_pos.reshape(self.env.num_envs, -1)
    
    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # # logs data
        self.metrics["error_feet"] += (
            torch.norm(self.ref_feet_pos - self.robot.data.body_link_pose_w[:, self.feet_indices, :2], dim=(1,2)) / max_command_step
        )

    def compute(self, dt: float):
        self._update_metrics()
        self.ref_feet_pos = self.target_wp[self.target_wp_i, self.target_wp_j] + self.ori_feet_pos  # [num_envs, 2, 7], two hands
        self.delayed_obs_target_wp = self.target_wp[self.target_wp_i, torch.maximum(self.target_wp_j - self.delayed_obs_target_wp_steps_int, torch.tensor(0))]
        resample_env_ids = torch.arange(self.num_envs, dtype=torch.long, device =self.device) if self.env.common_step_counter % self.target_wp_update_steps_int== 0 else []
        resample_i = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if len(resample_env_ids) > 0:
            self.target_wp_j += 1
            wp_eps_end_bool = self.target_wp_j >= self.num_wp
            self.target_wp_j = torch.where(wp_eps_end_bool, torch.zeros_like(self.target_wp_j), self.target_wp_j)
            resample_i[wp_eps_end_bool.nonzero(as_tuple=False).flatten()] = True
            self.target_wp_update_steps_int = sample_int_from_float(self.target_wp_update_steps)
            self.delayed_obs_target_wp_steps_int = sample_int_from_float(self.delayed_obs_target_wp_steps)
        self.target_wp_i = torch.where(resample_i, torch.randint(0, self.num_pairs, (self.num_envs,), device=self.device), self.target_wp_i)

    def _resample(self, env_ids: Sequence[int]):
        if len(env_ids) != 0:
            # resample the command
            self._resample_command(env_ids)
            # increment the command counter
            self.command_counter[env_ids] += 1
    
    def _update_command(self):
        pass 
    
    def _resample_command(self, env_ids: Sequence[int]):
        self.ref_feet_pos = self.target_wp[self.target_wp_i, self.target_wp_j] + self.ori_feet_pos # [num_envs, 2, 7], two hands
        self.delayed_obs_target_wp = self.target_wp[self.target_wp_i, torch.maximum(self.target_wp_j - self.delayed_obs_target_wp_steps_int, torch.tensor(0))]
        resample_i = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.target_wp_j[env_ids] = 0
        resample_i[env_ids] = True
        self.target_wp_i = torch.where(resample_i, torch.randint(0, self.num_pairs, (self.num_envs,), device=self.device), self.target_wp_i)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "target_feet_visualizer"):
                # -- goal
                self.target_feet_visualizer = VisualizationMarkers(self.cfg.target_feet_visualizer_cfg)
                # -- current
                self.current_feet_visualizer = VisualizationMarkers(self.cfg.current_feet_visualizer_cfg)
            # set their visibility to true
            self.target_feet_visualizer.set_visibility(True)
            self.current_feet_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_feet_visualizer"):
                self.target_feet_visualizer.set_visibility(False)
                self.current_feet_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        ori_feet_pos = self.robot.data.body_link_pose_w[:, self.feet_indices, :7].clone() # [num_envs, 2, 7], two hands
        vis_ref_feet_pos = torch.ones(
            self.ref_feet_pos.shape[0],  # N
            self.ref_feet_pos.shape[1],  # 2
            3,
            device=self.ref_feet_pos.device,
            dtype=self.ref_feet_pos.dtype,
        )
        vis_ref_feet_pos[:, :, :2] = self.ref_feet_pos   # x, y
        vis_ref_feet_pos[:, :, 2] = 0.0                  # z
        self.target_feet_visualizer.visualize(vis_ref_feet_pos.reshape(-1,3))
        self.current_feet_visualizer.visualize(ori_feet_pos.clone()[:,:, :3].reshape(-1,3))

