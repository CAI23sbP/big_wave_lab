
from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from . import delay_joint_actions

@configclass
class DelayJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = delay_joint_actions.DelayJointPositionAction
    dynamic_randomization: float = MISSING 