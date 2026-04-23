from __future__ import annotations

import torch, copy
from dataclasses import MISSING
from typing import TYPE_CHECKING
from isaaclab.managers import ObservationTermCfg as ObsTerm

from isaaclab.assets import Articulation
from isaaclab.managers import (
    ActionTerm,
    ObservationManager,
)
from isaaclab.utils.assets import check_file_path, read_file
from tensordict import TensorDict
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .pre_trained_skill_policy_action_cfg import PartWiseSkillBlenderCfg


class PartWiseSkillBlenderAction(ActionTerm):
    cfg: PartWiseSkillBlenderCfg

    def __init__(self, cfg: PartWiseSkillBlenderCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._upper_joint_ids = self.robot.find_joints(cfg.upper_joint_names)[0]
        self._lower_joint_ids = self.robot.find_joints(cfg.lower_joint_names)[0]
        
        self._scale = cfg.scale
        self.upper_body_skill_names = list(cfg.upper_body_policy_paths.keys())
        self.lower_body_skill_names = list(cfg.lower_body_policy_paths.keys())
        
        self.total_skill_names = self.upper_body_skill_names + self.lower_body_skill_names
        self.lower_idx_to_name = {i: name for i, name in enumerate(self.lower_body_skill_names)}
        
        self._command_size = {}
        for name in self.upper_body_skill_names:
            self._command_size[name] = cfg.upper_body_command_size[name]

        for name in self.lower_body_skill_names:
            self._command_size[name] = cfg.lower_body_command_size[name]

        
        self.upper_skills = {}
        for name in self.upper_body_skill_names:
            policy_path = cfg.upper_body_policy_paths[name]
            if not check_file_path(policy_path):
                raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")
            file_bytes = read_file(policy_path)
            self.upper_skills[name] = torch.jit.load(file_bytes).to(env.device).eval()

        self.lower_skills = {}
        for name in self.lower_body_skill_names:
            policy_path = cfg.lower_body_policy_paths[name]
            if not check_file_path(policy_path):
                raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")
            file_bytes = read_file(policy_path)
            self.lower_skills[name] = torch.jit.load(file_bytes).to(env.device).eval()
        
        if cfg.head_joint_names is not None:
            self._head_joint_ids = self.robot.find_joints(cfg.head_joint_names)[0]
            self.head_skill_names = list(cfg.head_policy_path.keys())
            self.total_skill_names += self.head_skill_names
            for name in self.head_skill_names:
                self._command_size[name] = cfg.head_command_size[name]
                
            self.head_skills = {}
            for name in self.head_skill_names:
                policy_path = cfg.head_policy_path[name]
                if not check_file_path(policy_path):
                    raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")
                file_bytes = read_file(policy_path)
                self.head_skills[name] = torch.jit.load(file_bytes).to(env.device).eval()

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(
            self.num_envs, self._low_level_action_term.action_dim, device=self.device
        )
        self._primitive_action = torch.zeros_like(self.low_level_actions)
        self.lower_body_actions = torch.zeros(self.num_envs, len(self.lower_skills), len(self._lower_joint_ids), device=self.device) 

        def last_action():
            # reset the low level actions if the episode was reset
            if hasattr(env, "episode_length_buf"):
                self.low_level_actions[env.episode_length_buf == 0, :] = 0
            return self.low_level_actions

        cfg.common_low_level_observations.actions.func = lambda dummy_env: last_action()
        cfg.common_low_level_observations.actions.params = dict()
        command_obs = {}
        for name, obs_cfg in cfg.low_level_command_observations.items():
            
            command_size = self._command_size[name]
            term_name = cfg.low_level_command_term_names[name]

            obs_cfg_pruned = copy.deepcopy(obs_cfg)

            # 선택된 term만 command placeholder로 교체
            obs_term = getattr(obs_cfg_pruned, term_name)
            obs_term.func = (
                lambda dummy_env, command_size=command_size: torch.zeros(
                    self.num_envs, command_size, device=self.device
                )
            )
            obs_term.params = dict()

            # ObservationTermCfg만 삭제하고 group-level 속성은 유지
            for attr_name, attr_value in list(vars(obs_cfg_pruned).items()):
                if attr_name.startswith("_"):
                    continue
                if attr_name == term_name:
                    continue
                if isinstance(attr_value, ObsTerm):
                    delattr(obs_cfg_pruned, attr_name)

            command_obs[name + "_command"] = obs_cfg_pruned

        self._low_level_obs_manager = ObservationManager(
            {
                "common_obs": cfg.common_low_level_observations,
                **command_obs,
            },
            env,
        )
        self._counter = 0 
        self._batch_idx = torch.arange(self.num_envs, device=self.device)

    @property
    def action_dim(self) -> int:
        return sum(self._command_size.values()) 

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions
    
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        
    @property
    def primitive_action(self):
        return self._primitive_action
    
    def _split_actions(self):
        cmd_dict = {}
        start = 0
        for name in self.total_skill_names:
            dim = self._command_size[name]
            cmd_dict[name] = torch.clamp(self._raw_actions[:, start : start + dim], self._scale[name][0], self._scale[name][1]) 
            start += dim
        return cmd_dict

    def apply_actions(self):
        high_level_command = self._env.command_manager.get_command(self.cfg.high_level_command_name)
        self._selected_idx = torch.argmax(high_level_command, dim=-1)
        if self._counter % self.cfg.low_level_decimation == 0:
            common_obs = self._low_level_obs_manager.compute_group("common_obs")
            cmd_dict = self._split_actions()
            if self.cfg.head_joint_names is not None:
                for name, head_skill in self.head_skills.items():
                    command_obs = cmd_dict[name]
                    policy_obs = torch.cat(
                        [
                            common_obs,
                            command_obs,
                        ],
                        dim=-1,
                    )
                    head_action = head_skill(policy_obs)[:, self._head_joint_ids]
                    self.low_level_actions[:, self._head_joint_ids] = head_action
                    
            for name, upper_skill in self.upper_skills.items():
                command_obs = cmd_dict[name]
                policy_obs = torch.cat(
                    [
                        common_obs,
                        command_obs,
                    ],
                    dim=-1,
                )
                upper_action = upper_skill(policy_obs)[:, self._upper_joint_ids]
                self.low_level_actions[:, self._upper_joint_ids] = upper_action
                
            for idx, skill_name in self.lower_idx_to_name.items():
                lower_skill = self.lower_skills[skill_name]
                lower_command = cmd_dict[skill_name]
                policy_obs = torch.cat(
                    [
                        common_obs,
                        lower_command,
                    ],
                    dim=-1,
                )
                lower_action = lower_skill(policy_obs)[:, self._lower_joint_ids]
                self.lower_body_actions[:, idx] = lower_action
            selected_lower = self.lower_body_actions[self._batch_idx, self._selected_idx]   
            self.low_level_actions[:, self._lower_joint_ids] = selected_lower
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._primitive_action[:] = self.low_level_actions[:]
        self._low_level_action_term.apply_actions()
        self._counter += 1