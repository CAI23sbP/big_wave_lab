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
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

import big_wave_lab.tasks.manager_based.downstream.mdp as mdp
from big_wave_lab.tasks.manager_based.primitive_skill.config.h1.reach_env_cfg import H1ReachFlatEnvCfg
from big_wave_lab.tasks.manager_based.primitive_skill.config.h1.squat_env_cfg import H1SquatFlatEnvCfg
from big_wave_lab.tasks.manager_based.primitive_skill.config.h1.walk_env_cfg import H1WalkRoughEnvCfg
##

import math 
from big_wave_lab.tasks.manager_based.primitive_skill.primitive_skill_env_cfg import EventCfg as PrimitiveEventCfg
from big_wave_lab.tasks.manager_based.primitive_skill.primitive_skill_env_cfg import TerminationsCfg as PrimitiveTerminationsCfg
from big_wave_lab.tasks.manager_based.downstream.downstream_env_cfg import (
DownStreamSceneCfg,
DownstreamObservationsCfg, 
CommandsCfg, 
DonwStreamEnvCfg, 
ActionsCfg, 
RewardsCfg,
)
from big_wave_lab.assets.robot_cfg import H1_2_CFG
from big_wave_lab.assets.object_cfg import BREAD_BOX_CFG, TABLE_1_CFG, TABLE_2_CFG

REACH_SKILL = H1ReachFlatEnvCfg()
SQUAT_SKILL = H1SquatFlatEnvCfg()
WALK_SKILL = H1WalkRoughEnvCfg()

@configclass
class PickandPlaceSceneCfg(DownStreamSceneCfg):
    target_object = BREAD_BOX_CFG
    table_1 = TABLE_1_CFG
    table_2 = TABLE_2_CFG
    def __post_init__(self):
        super().__post_init__()
        self.table_1.spawn.scale = (1.5,1.5,1.5)
        self.table_2.spawn.scale = (1.5,1.5,1.5)
        

@configclass
class PickandPlaceObservationsCfg(DownstreamObservationsCfg):
    @configclass
    class PickandPlacePolicyCfg(DownstreamObservationsCfg.DownstreamPolicyCfg):
        
        far_from_goal = ObsTerm(
            func=mdp.far_from_goal, 
            params={
                "object_cfg": SceneEntityCfg("target_object"),
                "end_table_cfg":SceneEntityCfg("table_2") 
                },
            scale=1.,
            clip=(-18.0, 18.0),
            )
        
        wrist_command = ObsTerm(
            func=mdp.wrist_box_diff_obs, 
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names = [".*_wrist_yaw_.*"]),
                "object_cfg":SceneEntityCfg("target_object") 
                },
            scale=1.,
            clip=(-18.0, 18.0),
            )
        box_pos = ObsTerm(
            func=mdp.box_pos, 
            params={
                "object_cfg": SceneEntityCfg("target_object"),
                },
            scale=1.,
            clip=(-18.0, 18.0),
            )
        def __post_init__(self):
            super().__post_init__()
            
    @configclass
    class PickandPlaceCriticCfg(DownstreamObservationsCfg.DownstreamCriticCfg):
        end_table_pos = ObsTerm(
            func=mdp.end_table_pos, 
            params={
                "end_table_cfg":SceneEntityCfg("table_2") 
                },
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        far_from_goal = ObsTerm(
            func=mdp.far_from_goal, 
            params={
                "object_cfg": SceneEntityCfg("target_object"),
                "end_table_cfg":SceneEntityCfg("table_2") 
                },
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        box_pos = ObsTerm(
            func=mdp.box_pos, 
            params={
                "object_cfg": SceneEntityCfg("target_object"),
                },
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        wrist_pos_w = ObsTerm(
            func=mdp.wrist_pos_w, 
            params={                
                "asset_cfg": SceneEntityCfg("robot", body_names = [".*_wrist_yaw_.*"]),
                },
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        wrist_command = ObsTerm(
            func=mdp.wrist_box_diff_obs, 
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names = [".*_wrist_yaw_.*"]),
                "object_cfg":SceneEntityCfg("target_object") 
                },
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        def __post_init__(self):
            super().__post_init__()
            self.base_mass.params["asset_cfg"].body_names = [".*torso_link"]
            self.feet_contact_mask.params["sensor_cfg"].body_names = [".*_ankle_roll_.*"]

    policy: PickandPlacePolicyCfg = PickandPlacePolicyCfg()
    critic: PickandPlaceCriticCfg = PickandPlaceCriticCfg()

@configclass
class PickandPlaceRewardsCfg(RewardsCfg):
    box_pos_diff = RewTerm(
        func=mdp.box_pos_diff, 
        weight=5., 
        params={
            "object_cfg": SceneEntityCfg("target_object"),
            "end_table_cfg":SceneEntityCfg("table_2") 
            },
    )
    
    wrist_box_distance = RewTerm(
        func=mdp.wrist_box_distance, 
        weight=1., 
        params={
            "object_cfg": SceneEntityCfg("target_object"),
            "asset_cfg": SceneEntityCfg("robot", body_names = [".*_wrist_yaw_.*"]),
            },
    )
    def __post_init__(self):
        super().__post_init__()
        # self.undesired_contacts.params["sensor_cfg"].body_names=["pelvis", ".*torso_link", ".*_shoulder_.*",  ".*_elbow_.*",]

@configclass 
class PickandPlaceTerminationCfg(PrimitiveTerminationsCfg):
    
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("target_object")}
    )
    success = DoneTerm(func=mdp.task_done_pick_place, params={
                        "object_cfg": SceneEntityCfg("target_object"),
                        "end_table_cfg": SceneEntityCfg("table_2"),
                        "threshold": 0.3, # time
                        "distance": 0.2
                        })
    def __post_init__(self):
        super().__post_init__()
        self.base_contact.params["sensor_cfg"].body_names = [
            "pelvis", 
            ".*torso_link", 
            ".*_shoulder_.*", 
            ".*_elbow_.*",
        ]
        
@configclass 
class PickandPlaceEventsCfg(PrimitiveEventCfg):
    
    reset_object = EventTerm(
        func=mdp.reset_object_poses,
        mode="reset",
        params={
            "init_pose_range": {
                "x": [1.0, 1.15],
                "y": [-0.01, 0.01],
                "z": [0.8, 0.9],
            },
            "object_cfg": SceneEntityCfg("target_object"),
            "start_table_cfg": SceneEntityCfg("table_1"),
            "end_table_cfg": SceneEntityCfg("table_2"),
            "end_distance":{
                "x": [-2.0, -2.15],
                "y": [-0.01, 0.01],
                "z": [-0.4, -0.2],
            },
        },
    )
    def __post_init__(self):
        super().__post_init__()        
        ## event set
        self.add_base_mass.params["asset_cfg"].body_names = [".*torso_link"]
        self.base_external_force_torque.params["asset_cfg"].body_names = [".*torso_link"]

@configclass
class PickandPlaceActionsCfg(ActionsCfg):
    def __post_init__(self):
        super().__post_init__()        
        
        self.downstream_joint_pos.scale = {
            "squat": (-1., 1.),
            "reach": (-1., 1.),
            "walk": (-2., 2.)
        }
        self.downstream_joint_pos.upper_joint_names = [".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"]
        self.downstream_joint_pos.lower_joint_names = [".*_ankle_.*",".*_hip_.*", "torso_.*"]
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
            resampling_time_range=(4., 4.),
            action_name = "downstream_joint_pos",
            num_skills = 2, # only lower body
            debug_vis = False
        )
        
@configclass
class H1PickandPlaceEnvCfg(DonwStreamEnvCfg):
    scene: PickandPlaceSceneCfg = PickandPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: PickandPlaceActionsCfg = PickandPlaceActionsCfg()
    events: PickandPlaceEventsCfg = PickandPlaceEventsCfg()
    commands: PickandPlaceCommandsCfg = PickandPlaceCommandsCfg()
    terminations: PickandPlaceTerminationCfg = PickandPlaceTerminationCfg()
    observations: PickandPlaceObservationsCfg = PickandPlaceObservationsCfg()
    rewards: PickandPlaceRewardsCfg = PickandPlaceRewardsCfg()
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = H1_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        

@configclass
class H1PickandPlaceEnvCfg_PLAY(H1PickandPlaceEnvCfg):
    viewer = ViewerCfg(
            eye=(-0., 4.4, 3.0),
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
        self.commands.downstream_command.debug_vis = True