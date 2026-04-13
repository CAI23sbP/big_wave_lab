
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
from .utils import sample_int_from_float

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import BaseHeightCommandCfg

class BaseHeightCommand(CommandTerm):
    cfg: BaseHeightCommandCfg
    def __init__(self, cfg: BaseHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        self.base_height = torch.zeros(self.num_envs, 1, device=self.device)
        """
        initalize target waypoints
        """
        self._total_num_points = cfg.total_num_points
        self._num_way_points = cfg.num_way_points
        self.step_dt = env.step_dt
        base_height = torch.randn(self._total_num_points, 1).to(env.device) * cfg.ranges.base_height_std + cfg.base_height_target
        base_height = torch.clamp(base_height, cfg.ranges.base_height_scale[0], cfg.ranges.base_height_scale[1])
        self.base_height = base_height.unsqueeze(1).repeat(1, self._num_way_points, 1) # (num_points, num_wp, 1)
        print("===> [sample_base_height] return shape:", base_height.shape)
        self.target_wp_i = torch.randint(0, self._total_num_points, (env.num_envs,), device=env.device) # for each env, choose one seq, [num_envs]
        self.target_wp_j = torch.zeros(env.num_envs, dtype=torch.long, device=env.device) # for each env, the timestep in the seq is initialized to 0, [num_envs]
        
        self.target_wp_dt = 1 / cfg.resampling_time_range[1]
        self.target_wp_update_steps = self.target_wp_dt / self.step_dt # not necessary integer
        assert self.step_dt <= self.target_wp_dt, f"self.step_dt {self.step_dt} must be less than self.target_wp_dt {self.target_wp_dt}"
        self.target_wp_update_steps_int = sample_int_from_float(self.target_wp_update_steps)
        
        self.delayed_obs_target_wp_steps = 0.0
        self.delayed_obs_target_wp_steps_int = sample_int_from_float(self.delayed_obs_target_wp_steps)
        self.ref_base_height = self.base_height[self.target_wp_i, self.target_wp_j] # [num_envs, 1]
        self.delayed_obs_target_wp = self.base_height[self.target_wp_i, torch.maximum(self.target_wp_j - self.delayed_obs_target_wp_steps_int, torch.tensor(0))]

        
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
        self.env = env
        
    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "BaseHeightCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 1)."""
        return self.ref_base_height.clone()
    
    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_height"] += torch.abs(self.ref_base_height.squeeze(-1) - self.robot.data.root_pos_w[:, 2])/ max_command_step

    def compute(self, dt: float):
        self._update_metrics()
        
        self.ref_base_height = self.base_height[self.target_wp_i, self.target_wp_j] # [num_envs, 1]
        self.delayed_obs_target_wp = self.base_height[self.target_wp_i, torch.maximum(self.target_wp_j - self.delayed_obs_target_wp_steps_int, torch.tensor(0))]
        resample_env_ids = torch.arange(self.num_envs, dtype=torch.long, device =self.device) if self.env.common_step_counter % self.target_wp_update_steps_int== 0 else []
        resample_i = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if len(resample_env_ids) > 0:
            self.ref_base_height = self.base_height[self.target_wp_i, self.target_wp_j] # [num_envs, 1]
            self.delayed_obs_target_wp = self.base_height[self.target_wp_i, torch.maximum(self.target_wp_j - self.delayed_obs_target_wp_steps_int, torch.tensor(0))]
            self.target_wp_j += 1
            wp_eps_end_bool = self.target_wp_j >= self._num_way_points
            self.target_wp_j = torch.where(wp_eps_end_bool, torch.zeros_like(self.target_wp_j), self.target_wp_j)
            resample_i[wp_eps_end_bool.nonzero(as_tuple=False).flatten()] = True
            self.target_wp_update_steps_int = sample_int_from_float(self.target_wp_update_steps)
            self.delayed_obs_target_wp_steps_int = sample_int_from_float(self.delayed_obs_target_wp_steps)
        self.target_wp_i = torch.where(resample_i, torch.randint(0, self._total_num_points, (self.num_envs,), device=self.device), self.target_wp_i)

    def _resample(self, env_ids: Sequence[int]):
        if len(env_ids) != 0:
            # resample the command
            self._resample_command(env_ids)
            # increment the command counter
            self.command_counter[env_ids] += 1
    
    def _update_command(self):
        pass 
    
    def _resample_command(self, env_ids: Sequence[int]):
        self.ref_base_height = self.base_height[self.target_wp_i, self.target_wp_j] # [num_envs, 1]
        self.delayed_obs_target_wp = self.base_height[self.target_wp_i, torch.maximum(self.target_wp_j - self.delayed_obs_target_wp_steps_int, torch.tensor(0))]
        resample_i = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.target_wp_j[env_ids] = 0
        resample_i[env_ids] = True
        self.target_wp_i = torch.where(resample_i, torch.randint(0, self._total_num_points, (self.num_envs,), device=self.device), self.target_wp_i)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "target_height_visualizer"):
                # -- goal
                self.target_height_visualizer = VisualizationMarkers(self.cfg.target_height_visualizer_cfg)
                # -- current
                self.current_height_visualizer = VisualizationMarkers(self.cfg.current_height_visualizer_cfg)
            # set their visibility to true
            self.target_height_visualizer.set_visibility(True)
            self.current_height_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_height_visualizer"):
                self.target_height_visualizer.set_visibility(False)
                self.current_height_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_quat_w = self.robot.data.root_quat_w.clone()
        flat_pos_w = base_pos_w.clone()
        flat_pos_w[:, 2] = self.command.squeeze(1)
        flat_quat_w = torch.zeros_like(base_quat_w)
        flat_quat_w[:, 0] = 1.
        # display markers
        self.target_height_visualizer.visualize(base_pos_w, base_quat_w)
        self.current_height_visualizer.visualize(flat_pos_w, flat_quat_w)

