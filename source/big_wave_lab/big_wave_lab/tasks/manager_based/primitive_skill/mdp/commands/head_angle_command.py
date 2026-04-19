from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import HeadLookTargetCommandCfg


class HeadLookTargetCommand(CommandTerm):
    cfg: HeadLookTargetCommandCfg

    def __init__(self, cfg: HeadLookTargetCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.step_dt = env.step_dt

        self.head_body_idx = self.robot.find_bodies([cfg.head_body_name])[0][0]
        self.head_joint_ids, _ = self.robot.find_joints(cfg.head_joint_names, preserve_order=True)
        assert len(self.head_joint_ids) == 2, "Expected 2 head joints: roll, pitch, yaw"

        # [N, 3] = (roll, pitch, yaw)
        self.ref_head_joint_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # sampled look-at target in world
        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self.metrics["head_angle_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.target_pos_w.clone()

    def _update_metrics(self):
        
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # # logs data
        current = self.robot.data.joint_pos[:, self.head_joint_ids]
        err = current - self.ref_head_joint_pos[:, 1:]
        self.metrics["head_angle_error"] += torch.norm(err, dim=-1)/max_command_step

    def compute(self, dt: float):
        self._update_metrics()

    def _update_command(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        head_pos_w = self.robot.data.body_state_w[env_ids, self.head_body_idx, :3]

        dist = torch.empty(env_ids.numel(), device=self.device).uniform_(
            self.cfg.ranges.distance[0], self.cfg.ranges.distance[1]
        )
        yaw = torch.empty(env_ids.numel(), device=self.device).uniform_(
            self.cfg.ranges.yaw[0], self.cfg.ranges.yaw[1]
        )
        pitch = torch.empty(env_ids.numel(), device=self.device).uniform_(
            self.cfg.ranges.pitch[0], self.cfg.ranges.pitch[1]
        )

        x = dist * torch.cos(pitch) * torch.cos(yaw)
        y = dist * torch.cos(pitch) * torch.sin(yaw)
        z = dist * torch.sin(pitch)
        offset_local = torch.stack([x, y, z], dim=-1)  # [M,3]
        self.target_pos_w[env_ids] = head_pos_w + offset_local
        roll = torch.zeros_like(yaw)
        self.ref_head_joint_pos[env_ids, 0] = roll
        self.ref_head_joint_pos[env_ids, 1] = pitch
        self.ref_head_joint_pos[env_ids, 2] = yaw
        
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "target_head_visualizer"):
                self.target_head_visualizer = VisualizationMarkers(self.cfg.target_head_visualizer_cfg)
                self.current_head_visualizer = VisualizationMarkers(self.cfg.current_head_visualizer_cfg)

            self.target_head_visualizer.set_visibility(True)
            self.current_head_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_head_visualizer"):
                self.target_head_visualizer.set_visibility(False)
                self.current_head_visualizer.set_visibility(False)
                
    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        current_head_pos = self.robot.data.body_state_w[:, self.head_body_idx, :3]   # [N, 3]
        target_head_pos = self.target_pos_w                                           # [N, 3]

        self.target_head_visualizer.visualize(target_head_pos)
        self.current_head_visualizer.visualize(current_head_pos)