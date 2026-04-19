from isaaclab.utils import configclass

from big_wave_lab.tasks.manager_based.downstream.downstream_env_cfg import (
DownstreamObservationsCfg, 
CommandsCfg, 
DonwStreamEnvCfg, 
RewardsCfg
)
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.scene import InteractiveSceneCfg

import big_wave_lab.tasks.manager_based.downstream.mdp as mdp
from big_wave_lab.tasks.manager_based.primitive_skill.config.h1.reach_env_cfg import H1ReachFlatEnvCfg
from big_wave_lab.tasks.manager_based.primitive_skill.config.h1.squat_env_cfg import H1SquatFlatEnvCfg
from big_wave_lab.tasks.manager_based.primitive_skill.config.h1.walk_env_cfg import H1WalkRoughEnvCfg
##
from big_wave_lab.assets.robot_cfg import H1_2_CFG
import math 

REACH_SKILL = H1ReachFlatEnvCfg()
SQUAT_SKILL = H1SquatFlatEnvCfg()
WALK_SKILL = H1WalkRoughEnvCfg()


@configclass
class TransitionCommandsCfg(CommandsCfg):
    def __post_init__(self):
        self.downstream_command = mdp.SkillSelectCommandCfg(
            asset_name="robot",
            resampling_time_range=(8., 8.),
            action_name = "downstream_joint_pos",
            num_skills = 3,
            debug_vis = True
        )
        
@configclass
class PickandPlaceSceneCfg(InteractiveSceneCfg):
    pass 