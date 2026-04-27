from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import (
    ActionTerm,
    ActionTermCfg,
    ObservationGroupCfg,
)
from isaaclab.utils import configclass

from .skill_blender_action import SkillBlenderAction
from .part_wise_skill_blender_action import PureSkillBlenderAction

@configclass
class PureSkillBlenderCfg(ActionTermCfg):
    class_type: type[ActionTerm] = PureSkillBlenderAction
    asset_name: str = MISSING

    upper_joint_names: list[str] | None = None
    lower_joint_names: list[str] | None = None
    head_joint_names: list[str] | None = None

    upper_body_policy_paths: dict[str, str] = MISSING
    lower_body_policy_paths: dict[str, str] = MISSING
    head_policy_path: dict[str, str] | None = None

    upper_body_command_size: dict[str, int] = MISSING
    lower_body_command_size: dict[str, int] = MISSING
    head_command_size: dict[str, int] | None = None
    scale: dict[str, tuple[float, float]] = MISSING

    low_level_actions: ActionTermCfg = MISSING
    common_low_level_observations: ObservationGroupCfg = MISSING
    low_level_command_observations: dict[str, ObservationGroupCfg] = MISSING
    low_level_command_term_names: dict[str, str] = MISSING
    low_level_decimation: int = 4
    debug_vis: bool = False


@configclass
class SkillBlenderActionCfg(ActionTermCfg):
    """
    TODO list
    """
    class_type: type[ActionTerm] = SkillBlenderAction
    asset_name: str = MISSING
    policy_paths: dict[str, str] = MISSING
    low_level_actions: ActionTermCfg = MISSING
    common_low_level_observations: ObservationGroupCfg = MISSING
    low_level_command_term_names: dict[str, str] = MISSING
    low_level_command_observations: dict[str, ObservationGroupCfg] = MISSING
    low_level_command_size: dict[str, int] = MISSING
    
    debug_vis: bool = False
    low_level_decimation: int = 1
