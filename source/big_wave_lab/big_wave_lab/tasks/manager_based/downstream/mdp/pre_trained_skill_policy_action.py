from __future__ import annotations

import torch, copy
from dataclasses import MISSING
from typing import TYPE_CHECKING
from isaaclab.managers import ObservationTermCfg as ObsTerm

from isaaclab.assets import Articulation
from isaaclab.managers import (
    ActionTerm,
    ActionTermCfg,
    ObservationGroupCfg,
    ObservationManager,
    CommandTermCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PreTrainedSkillPolicyAction(ActionTerm):
    cfg: PreTrainedSkillPolicyActionCfg

    def __init__(self, cfg: PreTrainedSkillPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        self.skill_names = list(cfg.policy_paths.keys())
        self.num_skills = len(self.skill_names)

        self._command_size = {}
        for name in self.skill_names:
            self._command_size[name] = cfg.low_level_command_size[name]

        self.policies = {}
        for name in self.skill_names:
            policy_path = cfg.policy_paths[name]
            if not check_file_path(policy_path):
                raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")
            file_bytes = read_file(policy_path)
            self.policies[name] = torch.jit.load(file_bytes).to(env.device).eval()

        # high-level output = [joint-wise alpha | all skill commands]
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(
            self.num_envs, self._low_level_action_term.action_dim, device=self.device
        )
        self._target_primitive_action = torch.zeros_like(self.low_level_actions)
        
        # transition state
        self.current_policy_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_policy_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.transition_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        def last_action():
            if hasattr(env, "episode_length_buf"):
                reset_env_ids = env.episode_length_buf == 0
                self.low_level_actions[reset_env_ids, :] = 0.0
                self.current_policy_idx[reset_env_ids] = 0
                self.target_policy_idx[reset_env_ids] = 0
                self.transition_steps[reset_env_ids] = 0
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

    @property
    def action_dim(self) -> int:
        return self.robot.data.joint_pos.shape[-1] + sum(self._command_size.values())

    @property
    def target_primitive_action(self):
        return self._target_primitive_action
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        # skill selection은 high-level command(one-hot)에서 읽음
        high_level_command = self._env.command_manager.get_command(self.cfg.high_level_command_name)
        selected_idx = torch.argmax(high_level_command, dim=-1)

        changed = selected_idx != self.target_policy_idx
        self.target_policy_idx[changed] = selected_idx[changed]
        self.transition_steps[changed] = 0

    def _split_actions(self):
        joint_dim = self.robot.data.joint_pos.shape[-1]
        raw_alpha = self._raw_actions[:, :joint_dim]
        alpha = torch.sigmoid(raw_alpha)  # [N, act_dim] in [0, 1]

        cmd_dict = {}
        start = joint_dim
        for name in self.skill_names:
            dim = self._command_size[name]
            cmd_dict[name] = torch.tanh(self._raw_actions[:, start : start + dim]) 
            start += dim

        return alpha, cmd_dict

    def _compute_target_skill_stable(self, selected_idx: torch.Tensor) -> torch.Tensor:
        """임시 안정 판정.
        지금은 최소 전이 step만 보고 True/False를 만들고,
        나중에 skill별 상태 조건으로 교체하는 것이 좋음.
        """
        stable = self.transition_steps >= self.cfg.min_transition_steps
        return stable

    def apply_actions(self):
        # if self._counter % self.cfg.low_level_decimation == 0:
            high_level_command = self._env.command_manager.get_command(self.cfg.high_level_command_name)
            selected_idx = torch.argmax(high_level_command, dim=-1)

            changed = selected_idx != self.target_policy_idx
            self.target_policy_idx[changed] = selected_idx[changed]
            self.transition_steps[changed] = 0

            alpha, cmd_dict = self._split_actions()

            common_obs = self._low_level_obs_manager.compute_group("common_obs")

            all_policy_actions = {}
            for name in self.skill_names:

                # 현재는 command obs를 high-level output command로 직접 대체
                command_obs = cmd_dict[name]

                policy_obs = torch.cat(
                    [
                        common_obs,
                        command_obs,
                    ],
                    dim=-1,
                )
                # print(f'[{name}]:{common_obs.shape,command_obs.shape}')
                all_policy_actions[name] = self.policies[name](policy_obs)

            current_skill_names = [self.skill_names[i] for i in self.current_policy_idx.tolist()]
            target_skill_names = [self.skill_names[i] for i in self.target_policy_idx.tolist()]

            current_actions = torch.stack(
                [all_policy_actions[name][env_id] for env_id, name in enumerate(current_skill_names)],
                dim=0,
            )
            target_actions = torch.stack(
                [all_policy_actions[name][env_id] for env_id, name in enumerate(target_skill_names)],
                dim=0,
            )
            self._target_primitive_action[:] = target_actions 
            # joint-wise blending
            self.low_level_actions[:] = (1.0 - alpha) * current_actions + alpha * target_actions
            self._low_level_action_term.process_actions(self.low_level_actions)

            # transition progress update
            self.transition_steps += 1

            # done 판정은 alpha가 아니라 외부 기준/상태 기준으로
            target_skill_stable = self._compute_target_skill_stable(selected_idx)
            done = target_skill_stable
            self.current_policy_idx[done] = self.target_policy_idx[done]

            # self._counter = 0

            self._low_level_action_term.apply_actions()
            # self._counter += 1


@configclass
class PreTrainedSkillPolicyActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = PreTrainedSkillPolicyAction

    asset_name: str = MISSING
    policy_paths: dict[str, str] = MISSING

    low_level_actions: ActionTermCfg = MISSING
    common_low_level_observations: ObservationGroupCfg = MISSING
    low_level_command_term_names: dict[str, str] = MISSING
    low_level_command_observations: dict[str, ObservationGroupCfg] = MISSING
    
    low_level_command_size: dict[str, int] = MISSING
    high_level_command_name: str = MISSING
    
    debug_vis: bool = False
    low_level_decimation: int = 1
    min_transition_steps: int = 2