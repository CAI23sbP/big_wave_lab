

from isaaclab.utils import configclass

from big_wave_lab.tasks.manager_based.downstream.downstream_env_cfg import DownstreamObservationsCfg, CommandsCfg, DonwStreamEnvCfg, RewardsCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup

import big_wave_lab.tasks.manager_based.downstream.mdp as mdp
# from big_wave_lab.tasks.manager_based.primitive_skill.config.h1.walk_env_cfg import WalkCommandsCfg
from big_wave_lab.tasks.manager_based.primitive_skill.config.h1.reach_env_cfg import H1ReachFlatEnvCfg
from big_wave_lab.tasks.manager_based.primitive_skill.config.h1.squat_env_cfg import H1SquatFlatEnvCfg
##
from big_wave_lab.assets.robot_cfg import H1_2_CFG
import math 

REACH_SKILL = H1ReachFlatEnvCfg()
SQUAT_SKILL = H1SquatFlatEnvCfg()

@configclass
class TransitionCommandsCfg(CommandsCfg):
    def __post_init__(self):
        self.downstream_command = mdp.SkillSelectCommandCfg(
            asset_name="robot",
            resampling_time_range=(0., 0.),
            action_name = "downstream_joint_pos",
            num_skills = 2,
            debug_vis = True
        )

@configclass
class TransitionRewardsCfg(RewardsCfg):
    target_joint_pos_diff = RewTerm(
        func=mdp.target_joint_pos_diff, 
        weight=5., 
        params={
            "action_name": "downstream_joint_pos",
            }
    )

@configclass
class TransitionObservationCfg(DownstreamObservationsCfg):
    @configclass
    class TransitionPolicyCfg(DownstreamObservationsCfg.DownstreamPolicyCfg):
        target_pos_diff = ObsTerm(
            func=mdp.target_pos_diff, 
            params={"action_name": "downstream_joint_pos"},
            scale=1.,
            clip=(-18.0, 18.0),
            )
        def __post_init__(self):
            super().__post_init__()

    @configclass
    class TransitionCriticCfg(DownstreamObservationsCfg.DownstreamCriticCfg):
        target_pos_diff = ObsTerm(
            func=mdp.target_pos_diff, 
            params={"action_name": "downstream_joint_pos"},
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        def __post_init__(self):
            super().__post_init__()
            
    policy: TransitionPolicyCfg = TransitionPolicyCfg()
    critic: TransitionCriticCfg = TransitionCriticCfg()
            
@configclass
class H1TransitionFlatEnvCfg(DonwStreamEnvCfg):
    commands: TransitionCommandsCfg = TransitionCommandsCfg()
    rewards: TransitionRewardsCfg = TransitionRewardsCfg()
    observations: TransitionObservationCfg = TransitionObservationCfg()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        robot = H1_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = robot 
        
        self.commands.downstream_command.resampling_time_range = (self.actions.downstream_joint_pos.min_transition_steps, 
                                                                  self.actions.downstream_joint_pos.min_transition_steps)
        
        ## obs set
        self.observations.critic.base_mass.params["asset_cfg"].body_names = [".*torso_link"]
        self.observations.critic.feet_contact_mask.params["sensor_cfg"].body_names = [".*_ankle_roll_.*"]
        
        ## action set
        self.actions.downstream_joint_pos.policy_paths = {
            "squat": "/home/cai/humanoid_ws/big_wave_lab/logs/rsl_rl/h1_squat/2026-04-13_14-01-50/exported/policy.pt",
            "reach": "/home/cai/humanoid_ws/big_wave_lab/logs/rsl_rl/h1_reach/2026-04-13_15-35-08/exported/policy.pt",
        }
        self.actions.downstream_joint_pos.low_level_command_term_names = {
            "squat": "base_height_diff",
            "reach": "target_body_pos_w_diff"
        }
        self.actions.downstream_joint_pos.low_level_command_observations = {
            "squat": SQUAT_SKILL.observations.policy,
            "reach": REACH_SKILL.observations.policy
        }
        self.actions.downstream_joint_pos.low_level_command_size = {
            "squat": SQUAT_SKILL.commands.pose_command.command_size,
            "reach": REACH_SKILL.commands.pose_command.command_size,
        }
        
        ## reward set 
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names=["pelvis", ".*torso_link", ".*_shoulder_.*",  ".*_elbow_.*", ".*_wrist_.*"]
        
        ## event set
        self.events.add_base_mass.params["asset_cfg"].body_names = [".*torso_link"]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [".*torso_link"]
        
        ## termination set
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "pelvis", 
            ".*torso_link", 
            ".*_shoulder_.*", 
            ".*_elbow_.*",
            ".*_wrist_.*"
        ]
        
        
@configclass
class H1TransitionFlatEnvCfg_PLAY(H1TransitionFlatEnvCfg):
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