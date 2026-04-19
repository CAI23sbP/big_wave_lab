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
from big_wave_lab.tasks.manager_based.primitive_skill.primitive_skill_env_cfg import PosingFlatEnvCfg
from big_wave_lab.tasks.manager_based.downstream.downstream_env_cfg import DownstreamObservationsCfg, CommandsCfg, DonwStreamEnvCfg, ActionsCfg, RewardsCfg

LOW_LEVEL_ENV_CFG = PosingFlatEnvCfg()
REACH_SKILL = H1ReachFlatEnvCfg()
SQUAT_SKILL = H1SquatFlatEnvCfg()
WALK_SKILL = H1WalkRoughEnvCfg()


@configclass
class PickandPlaceActionsCfg(ActionsCfg):
    def __post_init__(self):
        self.downstream_joint_pos.upper_joint_names = ["torso_.*", ".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"]
        self.downstream_joint_pos.lower_joint_names = [".*_ankle_.*",".*_hip_.*"]
        self.downstream_joint_pos.upper_body_policy_paths = {
            "reach":"/home/cai/humanoid_ws/big_wave_lab/logs/rsl_rl/h1_reach/2026-04-13_15-35-08/exported/policy.pt",
        }

        self.downstream_joint_pos.lower_body_policy_paths = {
            "squat": "/home/cai/humanoid_ws/big_wave_lab/logs/rsl_rl/h1_squat/2026-04-13_14-01-50/exported/policy.pt",
            "walk": "/home/cai/humanoid_ws/big_wave_lab/logs/rsl_rl/h1_walk/2026-04-19_22-52-39/exported/policy.pt",
        }
        self.downstream_joint_pos.low_level_command_term_names = {
            "squat": "base_height_diff",
            "walk": "only_vel_generated_commands",
            "reach": "target_body_pos_w_diff"
        }
        self.downstream_joint_pos.low_level_command_observations = {
            "squat": SQUAT_SKILL.observations.policy,
            "walk": WALK_SKILL.observations.policy,
            "reach": REACH_SKILL.observations.policy,
        }
        self.downstream_joint_pos.upper_body_command_size = {
            "reach": REACH_SKILL.commands.pose_command.command_size,
        }
        self.downstream_joint_pos.lower_body_command_size = {
            "squat": SQUAT_SKILL.commands.pose_command.command_size,
            "walk": WALK_SKILL.commands.pose_command.command_size,
        }
        

@configclass
class PickandPlaceCommandsCfg(CommandsCfg):
    def __post_init__(self):
        self.downstream_command = mdp.SkillSelectCommandCfg(
            asset_name="robot",
            resampling_time_range=(8., 8.),
            action_name = "downstream_joint_pos",
            num_skills = 3,
            debug_vis = True
        )
        
@configclass
class H1PickandPlaceEnvCfg(DonwStreamEnvCfg):
    actions: PickandPlaceActionsCfg = PickandPlaceActionsCfg()
    commands: PickandPlaceCommandsCfg = PickandPlaceCommandsCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = H1_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        
@configclass
class H1PickandPlaceEnvCfg_PLAY(H1PickandPlaceEnvCfg):
    viewer = ViewerCfg(
            eye=(-0., 2.6, 1.6),
            asset_name = "robot",
            origin_type = 'asset_root',
        )
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_force_robot = None