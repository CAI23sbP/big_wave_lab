from __future__ import annotations

import torch
import copy
from typing import TYPE_CHECKING

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ObservationManager
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .pre_trained_skill_policy_action_cfg import PureSkillBlenderCfg


class PureSkillBlenderAction(ActionTerm):
    cfg: PureSkillBlenderCfg

    def __init__(self, cfg: PureSkillBlenderCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self._scale = cfg.scale

        # ------------------------------------------------------------
        # Low-level action term
        # ------------------------------------------------------------
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(
            cfg.low_level_actions,
            env,
        )

        self.low_level_action_dim = self._low_level_action_term.action_dim

        # ------------------------------------------------------------
        # Skill names
        # ------------------------------------------------------------
        self.upper_body_skill_names = list(cfg.upper_body_policy_paths.keys())
        self.lower_body_skill_names = list(cfg.lower_body_policy_paths.keys())

        self.total_skill_names = (
            self.upper_body_skill_names
            + self.lower_body_skill_names
        )

        self.has_head_skill = cfg.head_joint_names is not None

        if self.has_head_skill:
            self.head_skill_names = list(cfg.head_policy_path.keys())
            self.total_skill_names += self.head_skill_names
        else:
            self.head_skill_names = []

        self.num_skills = len(self.total_skill_names)

        # ------------------------------------------------------------
        # Command sizes
        # ------------------------------------------------------------
        self._command_size = {}

        for name in self.upper_body_skill_names:
            self._command_size[name] = cfg.upper_body_command_size[name]

        for name in self.lower_body_skill_names:
            self._command_size[name] = cfg.lower_body_command_size[name]

        if self.has_head_skill:
            for name in self.head_skill_names:
                self._command_size[name] = cfg.head_command_size[name]

        # ------------------------------------------------------------
        # Load all pretrained skills
        # ------------------------------------------------------------
        self.skills = {}

        for name in self.upper_body_skill_names:
            policy_path = cfg.upper_body_policy_paths[name]

            if not check_file_path(policy_path):
                raise FileNotFoundError(
                    f"Upper body policy file '{policy_path}' does not exist."
                )

            file_bytes = read_file(policy_path)
            self.skills[name] = torch.jit.load(file_bytes).to(env.device).eval()

        for name in self.lower_body_skill_names:
            policy_path = cfg.lower_body_policy_paths[name]

            if not check_file_path(policy_path):
                raise FileNotFoundError(
                    f"Lower body policy file '{policy_path}' does not exist."
                )

            file_bytes = read_file(policy_path)
            self.skills[name] = torch.jit.load(file_bytes).to(env.device).eval()

        if self.has_head_skill:
            for name in self.head_skill_names:
                policy_path = cfg.head_policy_path[name]

                if not check_file_path(policy_path):
                    raise FileNotFoundError(
                        f"Head policy file '{policy_path}' does not exist."
                    )

                file_bytes = read_file(policy_path)
                self.skills[name] = torch.jit.load(file_bytes).to(env.device).eval()

        # ------------------------------------------------------------
        # Buffers
        # ------------------------------------------------------------
        self._raw_actions = torch.zeros(
            self.num_envs,
            self.action_dim,
            device=self.device,
        )

        self.low_level_actions = torch.zeros(
            self.num_envs,
            self.low_level_action_dim,
            device=self.device,
        )

        self._current_ll_actions = torch.zeros_like(self.low_level_actions)
        self._prev_ll_actions = torch.zeros_like(self.low_level_actions)

        # [num_envs, num_skills, low_level_action_dim]
        self.skill_actions = torch.zeros(
            self.num_envs,
            self.num_skills,
            self.low_level_action_dim,
            device=self.device,
        )

        # [num_envs, num_skills, low_level_action_dim]
        self._mask_logits = torch.zeros(
            self.num_envs,
            self.num_skills,
            self.low_level_action_dim,
            device=self.device,
        )

        self._blend_weights = torch.zeros_like(self._mask_logits)

        # ------------------------------------------------------------
        # Common low-level observation setup
        # ------------------------------------------------------------
        def last_action():
            if hasattr(env, "episode_length_buf"):
                reset_envs = env.episode_length_buf == 0

                if reset_envs.any():
                    self.low_level_actions[reset_envs, :] = 0.0
                    self._prev_ll_actions[reset_envs, :] = 0.0
                    self._current_ll_actions[reset_envs, :] = 0.0
                    self.skill_actions[reset_envs, :, :] = 0.0
                    self._mask_logits[reset_envs, :, :] = 0.0
                    self._blend_weights[reset_envs, :, :] = 0.0

            return self.low_level_actions

        cfg.common_low_level_observations.actions.func = lambda dummy_env: last_action()
        cfg.common_low_level_observations.actions.params = dict()

        # ------------------------------------------------------------
        # Command observations for low-level policies
        # ------------------------------------------------------------
        command_obs = {}

        for name, obs_cfg in cfg.low_level_command_observations.items():
            command_size = self._command_size[name]
            term_name = cfg.low_level_command_term_names[name]

            obs_cfg_pruned = copy.deepcopy(obs_cfg)

            obs_term = getattr(obs_cfg_pruned, term_name)

            obs_term.func = (
                lambda dummy_env, command_size=command_size: torch.zeros(
                    self.num_envs,
                    command_size,
                    device=self.device,
                )
            )
            obs_term.params = dict()

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

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------
    @property
    def command_action_dim(self) -> int:
        return sum(self._command_size.values())

    @property
    def mask_action_dim(self) -> int:
        return self.num_skills * self.low_level_action_dim

    @property
    def action_dim(self) -> int:
        """
        High-level action structure:

        [
            command_for_skill_0,
            command_for_skill_1,
            ...,
            command_for_skill_N,
            mask_logits_for_all_skills
        ]

        mask logits shape after reshape:
            [num_envs, num_skills, low_level_action_dim]
        """
        return self.command_action_dim + self.mask_action_dim

    # ------------------------------------------------------------------
    # Action properties
    # ------------------------------------------------------------------
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    @property
    def prev_ll_actions(self):
        return self._prev_ll_actions

    @property
    def current_ll_actions(self):
        return self._current_ll_actions

    @property
    def blend_weights(self) -> torch.Tensor:
        """
        Shape:
            [num_envs, num_skills, low_level_action_dim]

        Meaning:
            blend_weights[:, i, j]
            = joint j에서 skill i를 얼마나 사용할지
        """
        return self._blend_weights

    # ------------------------------------------------------------------
    # Split high-level action
    # ------------------------------------------------------------------
    def _split_actions(self):
        """
        Split high-level raw action into:

        1. command for each low-level skill
        2. mask logits for pure skill blending
        """
        cmd_dict = {}
        start = 0

        # ------------------------------------------------------------
        # 1. Commands for each skill
        # ------------------------------------------------------------
        for name in self.total_skill_names:
            dim = self._command_size[name]

            cmd = self._raw_actions[:, start : start + dim]

            if name in self._scale:
                cmd = torch.clamp(
                    cmd,
                    self._scale[name][0],
                    self._scale[name][1],
                )

            cmd_dict[name] = cmd
            start += dim

        # ------------------------------------------------------------
        # 2. Mask logits
        # ------------------------------------------------------------
        mask_flat = self._raw_actions[:, start : start + self.mask_action_dim]

        self._mask_logits[:] = mask_flat.reshape(
            self.num_envs,
            self.num_skills,
            self.low_level_action_dim,
        )

        # skill dimension으로 softmax
        # 각 joint마다 모든 skill weight의 합이 1이 됨
        self._blend_weights[:] = torch.softmax(self._mask_logits, dim=1)

        return cmd_dict

    # ------------------------------------------------------------------
    # Compute all skill actions
    # ------------------------------------------------------------------
    def _compute_skill_actions(self, common_obs: torch.Tensor, cmd_dict: dict):
        """
        Compute action from every pretrained skill.

        Assumption:
            each pretrained skill outputs full low-level action dimension:
                [num_envs, low_level_action_dim]
        """
        for skill_idx, skill_name in enumerate(self.total_skill_names):
            skill_policy = self.skills[skill_name]
            command_obs = cmd_dict[skill_name]

            policy_obs = torch.cat(
                [
                    common_obs,
                    command_obs,
                ],
                dim=-1,
            )

            action = skill_policy(policy_obs)

            if action.shape[-1] != self.low_level_action_dim:
                raise RuntimeError(
                    f"Skill '{skill_name}' output dim mismatch. "
                    f"Expected {self.low_level_action_dim}, "
                    f"but got {action.shape[-1]}."
                )

            self.skill_actions[:, skill_idx, :] = action

    # ------------------------------------------------------------------
    # Pure blending
    # ------------------------------------------------------------------
    def _blend_skill_actions(self) -> torch.Tensor:
        """
        Pure joint-wise skill blending.

        skill_actions:
            [num_envs, num_skills, action_dim]

        blend_weights:
            [num_envs, num_skills, action_dim]

        output:
            [num_envs, action_dim]
        """
        blended_action = torch.sum(
            self._blend_weights * self.skill_actions,
            dim=1,
        )

        return blended_action

    # ------------------------------------------------------------------
    # Apply actions
    # ------------------------------------------------------------------
    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            common_obs = self._low_level_obs_manager.compute_group("common_obs")
            cmd_dict = self._split_actions()

            # ------------------------------------------------------------
            # 1. Compute every skill action
            # ------------------------------------------------------------
            self._compute_skill_actions(common_obs, cmd_dict)

            # ------------------------------------------------------------
            # 2. Pure blending only
            # ------------------------------------------------------------
            self.low_level_actions[:] = self._blend_skill_actions()

            # ------------------------------------------------------------
            # 3. Update action buffers
            # ------------------------------------------------------------
            self._prev_ll_actions[:] = self._current_ll_actions[:]
            self._current_ll_actions[:] = self.low_level_actions[:]

            self._low_level_action_term.process_actions(self._current_ll_actions)

            self._counter = 0

        self._low_level_action_term.apply_actions()
        self._counter += 1
        
        
    @property
    def skill_names(self) -> list[str]:
        return self.total_skill_names

    @property
    def blend_weights(self) -> torch.Tensor:
        """
        Shape:
            [num_envs, num_skills, low_level_action_dim]

        Meaning:
            blend_weights[:, i, j]
            = action dim j에서 skill i를 얼마나 사용할지
        """
        return self._blend_weights

    @property
    def mean_blend_weights(self) -> torch.Tensor:
        """
        Shape:
            [num_envs, num_skills]

        각 skill의 평균 사용 비율.
        joint-wise weight를 action dimension 방향으로 평균낸 값.
        """
        return torch.mean(self._blend_weights, dim=-1)