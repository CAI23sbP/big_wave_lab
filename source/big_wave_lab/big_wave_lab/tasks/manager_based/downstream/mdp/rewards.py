
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class wrist_box_distance(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._object: RigidObject = env.scene[cfg.params["object_cfg"].name]
        self._asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._asset_cfg = cfg.params["asset_cfg"]
        
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        ) -> torch.Tensor:
        wrist_pos = self._asset.data.body_pos_w[:, self._asset_cfg.body_ids, :3]
        box_pos = self._object.data.body_pos_w[:, 0, :3] 
        box_handle_left = box_pos.clone()
        box_handle_right = box_pos.clone()
        box_handle_pos = torch.stack([box_handle_left, box_handle_right], dim=1) # [num_envs, 2, 3]
        wrist_box_diff = wrist_pos - box_handle_pos # [num_envs, 2, 3]
        wrist_pos_diff = torch.flatten(wrist_box_diff, start_dim=1) # [num_envs, 6]
        wrist_box_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)
        return torch.exp(-4 * wrist_box_error)

class box_pos_diff(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._object: RigidObject = env.scene[cfg.params["object_cfg"].name]
        self._table: RigidObject = env.scene[cfg.params["end_table_cfg"].name]
        self._table_diff = self._table.data.root_pos_w[:, :3] 
    
    def reset(self, env_ids):
        self._table_diff[env_ids] = self._table.data.root_pos_w[env_ids, :3] 
        
    def __call__(
        self,    
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        end_table_cfg:SceneEntityCfg = SceneEntityCfg("table"),
        ) -> torch.Tensor:
        object_diff = self._object.data.root_pos_w[:, :3] 
        box_pos_diff = object_diff - self._table_diff
        box_pos_error = torch.mean(torch.abs(box_pos_diff), dim=1)
        return torch.exp(-4 * box_pos_error)


class low_level_action_rate_l2(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        action_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
        ) -> torch.Tensor:
        current_ll_actions = env.action_manager.get_term(action_name).current_ll_actions
        prev_ll_actions = env.action_manager.get_term(action_name).prev_ll_actions
        reward = torch.sum(torch.square(current_ll_actions - prev_ll_actions), dim=1)
        return reward

def command_weight_preference_for_selected_skills_and_joint_names(
    env: ManagerBasedRLEnv,
    command_name: str,
    action_term_name: str,
    skill_names: list[str],
    asset_cfg: SceneEntityCfg,
    alpha: float = 5.0,
    target_smoothing: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Same as command_weight_preference_for_selected_skills_and_joints,
    but selected action dimensions are obtained from asset_cfg.joint_ids.

    Assumption:
        asset_cfg.joint_ids correspond to low-level action indices.
    """

    action_term = env.action_manager.get_term(action_term_name)
    all_skill_names = action_term.skill_names

    weights = action_term.blend_weights
    # [num_envs, num_skills, action_dim]

    selected_skill_ids = []

    for skill_name in skill_names:
        if skill_name not in all_skill_names:
            raise RuntimeError(
                f"Skill name '{skill_name}' not found. "
                f"Available skills: {all_skill_names}"
            )

        selected_skill_ids.append(all_skill_names.index(skill_name))

    selected_skill_ids = torch.tensor(
        selected_skill_ids,
        dtype=torch.long,
        device=weights.device,
    )

    joint_ids_tensor = torch.tensor(
        asset_cfg.joint_ids,
        dtype=torch.long,
        device=weights.device,
    )

    selected_weights = weights.index_select(1, selected_skill_ids)
    selected_weights = selected_weights.index_select(2, joint_ids_tensor)

    selected_weights = selected_weights / (
        torch.sum(selected_weights, dim=1, keepdim=True) + eps
    )

    mean_selected_weights = torch.mean(selected_weights, dim=-1)
    # [num_envs, selected_num_skills]

    selected_num_skills = mean_selected_weights.shape[-1]

    command = env.command_manager.get_command(command_name)

    if command.ndim == 2 and command.shape[-1] == selected_num_skills:
        target_weights = command.float()
        target_weights = target_weights / (
            torch.sum(target_weights, dim=-1, keepdim=True) + eps
        )
    else:
        skill_idx = command.long()

        if skill_idx.ndim == 2:
            skill_idx = skill_idx.squeeze(-1)

        target_weights = torch.nn.functional.one_hot(
            skill_idx,
            num_classes=selected_num_skills,
        ).float()

    if target_smoothing > 0.0:
        uniform = torch.ones_like(target_weights) / float(selected_num_skills)

        target_weights = (
            (1.0 - target_smoothing) * target_weights
            + target_smoothing * uniform
        )

    weight_error = torch.sum(
        torch.square(mean_selected_weights - target_weights),
        dim=-1,
    )

    return torch.exp(-alpha * weight_error)