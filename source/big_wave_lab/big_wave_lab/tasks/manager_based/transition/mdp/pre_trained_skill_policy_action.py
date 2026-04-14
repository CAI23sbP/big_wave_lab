from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PreTrainedSkillPolicyAction(ActionTerm):
    cfg: PreTrainedSkillPolicyActionCfg

    def __init__(self, cfg: PreTrainedSkillPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # 정책 로드 순서를 고정
        self.policy_names = list(cfg.policy_paths.keys())
        self.num_policies = len(self.policy_names)

        self.policies = []
        for policy_name in self.policy_names:
            policy_path = cfg.policy_paths[policy_name]
            if not check_file_path(policy_path):
                raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")
            file_bytes = read_file(policy_path)
            self.policies.append(torch.jit.load(file_bytes).to(env.device).eval())

        # high-level input action: [num_envs, num_policies]
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(
            self.num_envs, self._low_level_action_term.action_dim, device=self.device
        )

        # per-env transition state
        self.current_policy_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_policy_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.transition_alpha = torch.ones(self.num_envs, device=self.device)

        # transition speed per low-level update
        self.transition_rate = cfg.transition_rate

        def last_action():
            if hasattr(env, "episode_length_buf"):
                reset_env_ids = env.episode_length_buf == 0
                self.low_level_actions[reset_env_ids, :] = 0.0
                self.current_policy_idx[reset_env_ids] = 0
                self.target_policy_idx[reset_env_ids] = 0
                self.transition_alpha[reset_env_ids] = 1.0
            return self.low_level_actions

        # low-level obs remap
        cfg.low_level_observations.actions.func = lambda dummy_env: last_action()
        cfg.low_level_observations.actions.params = dict()

        cfg.low_level_observations.velocity_commands.func = (
            lambda dummy_env: torch.zeros(self.num_envs, self.num_policies, device=self.device)
        )
        cfg.low_level_observations.velocity_commands.params = dict()
        
        self.high_level_command = env.command_manager.get_command(cfg.high_level_command_name)
        
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)
        self._counter = 0

    @property
    def action_dim(self) -> int:
        # A/B/C 선택용
        return self.num_policies

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    def process_actions(self, actions: torch.Tensor):
        # actions shape: [num_envs, 3]
        self._raw_actions[:] = actions

        # target이 바뀌면 전이 시작
        selected_idx = self.high_level_command
        changed = selected_idx != self.target_policy_idx
        self.target_policy_idx[changed] = selected_idx[changed]
        self.transition_alpha[changed] = 0.0

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")

            all_policy_actions = []
            for policy in self.policies:
                all_policy_actions.append(policy(low_level_obs))
            all_policy_actions = torch.stack(all_policy_actions, dim=0)

            env_ids = torch.arange(self.num_envs, device=self.device)

            current_actions = all_policy_actions[self.current_policy_idx, env_ids]  # [N, act_dim]
            target_actions = all_policy_actions[self.target_policy_idx, env_ids]    # [N, act_dim]

            # A -> B 부드러운 전이
            self.low_level_actions[:] = (1.0 - self._raw_actions) * current_actions + self._raw_actions * target_actions

            self._low_level_action_term.process_actions(self.low_level_actions)

            # alpha 업데이트
            not_done = self.transition_alpha < 1.0
            self.transition_alpha[not_done] = torch.clamp(
                self.transition_alpha[not_done] + self.transition_rate, max=1.0
            )

            # 전이 완료 시 current <- target
            done = self.transition_alpha >= 1.0
            self.current_policy_idx[done] = self.target_policy_idx[done]

            self._counter = 0

        self._low_level_action_term.apply_actions()
        self._counter += 1


@configclass
class PreTrainedSkillPolicyActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = PreTrainedSkillPolicyAction

    asset_name: str = MISSING
    policy_paths: dict[str, str] = MISSING

    low_level_decimation: int = 1
    low_level_actions: ActionTermCfg = MISSING
    low_level_observations: ObservationGroupCfg = MISSING

    high_level_command_name: str = MISSING
    debug_vis: bool = True
    
    # low-level update마다 alpha가 얼마나 증가할지
    transition_rate: float = 0.05