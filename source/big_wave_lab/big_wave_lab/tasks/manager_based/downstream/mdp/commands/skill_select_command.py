
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
import torch.nn.functional as F

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import SkillSelectCommandCfg

class SkillSelectCommand(CommandTerm):
    cfg: SkillSelectCommandCfg
    def __init__(self, cfg: SkillSelectCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.step_dt = env.step_dt
        self._num_skills = cfg.num_skills
        
        self.select_command = F.one_hot(torch.zeros(self.num_envs, device=self.device).long(), \
                            num_classes=self._num_skills).float()
        # self.metrics["error_target_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.env = env

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "ArmTargetCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Shape is (num_envs, num_skills)."""
        return self.select_command
    
    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # self.metrics["error_target_pos"] = torch.norm((self.robot.data.joint_pos - self._env.action_manager.get_term(self.cfg.action_name).target_primitive_action), dim=-1) / max_command_step
        
    def _resample_command(self, env_ids: Sequence[int]):
        # sample select idx
        skill_ids = torch.randint(
            low=0,
            high=self._num_skills,
            size=(len(env_ids),),
            device=self.device,
        )
        self.select_command[env_ids] = F.one_hot(
            skill_ids, num_classes=self._num_skills
        ).float()
        
    def _update_command(self):
        pass 
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "selected_skill_visualizer"):
                self.selected_skill_visualizer = VisualizationMarkers(self.cfg.selected_skill_visualizer_cfg)
            self.selected_skill_visualizer.set_visibility(True)
        else:
            if hasattr(self, "selected_skill_visualizer"):
                self.selected_skill_visualizer.set_visibility(False)
                
    def _debug_vis_callback(self, event):
        robot = self.env.scene["robot"]

        root_pos = robot.data.root_pos_w.clone()   
        root_pos[:, 2] += 1.5                    

        skill_ids = torch.argmax(self.select_command, dim=-1)  
        
        self.selected_skill_visualizer.visualize(
            translations=root_pos,
            marker_indices=skill_ids,
        )
        