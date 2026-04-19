from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import (
    ActionTerm,
    ActionTermCfg,
    ObservationGroupCfg,
)
from isaaclab.utils import configclass

from .skill_selection_action import PreTrainedSkillPolicyAction
from .skill_blender_action import SkillBlenderAction

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
    
    
    
@configclass
class SkillBlenderActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = SkillBlenderAction
