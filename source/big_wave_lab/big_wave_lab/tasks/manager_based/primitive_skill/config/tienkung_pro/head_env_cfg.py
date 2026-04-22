
from isaaclab.utils import configclass

from  big_wave_lab.tasks.manager_based.primitive_skill.primitive_skill_env_cfg import PosingFlatEnvCfg, RewardsCfg, ObservationsCfg, CommandsCfg
from isaaclab.managers import SceneEntityCfg

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ViewerCfg

import big_wave_lab.tasks.manager_based.primitive_skill.mdp as mdp
from big_wave_lab.assets.robot_cfg import TIENKUNG_PRO_TRAINING_CFG as PRO_CFG

"""
Tienkung pro has head joints roll,pitch,yaw
"""

@configclass
class HeadObservationsCfg(ObservationsCfg):
    
    @configclass
    class HeadPolicyCfg(ObservationsCfg.PolicyCfg):
        
        target_body_pos_w_diff = ObsTerm(
            func=mdp.head_target_dir_local,
            params={
                "command_name": "pose_command",
                "asset_cfg": SceneEntityCfg("robot", body_names=["camera_head_link"]),
            },
            scale=1.,
            clip=(-18.0, 18.0),
            )


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            
    @configclass
    class HeadCriticCfg(ObservationsCfg.CriticCfg):
        
        target_body_pos_w_diff = ObsTerm(
            func=mdp.head_target_dir_local,
            params={
                "command_name": "pose_command",
                "asset_cfg": SceneEntityCfg("robot", body_names=["camera_head_link"]),
            },
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        
        target_body_pos_w = ObsTerm(
            func=mdp.body_pos_w,
            params={"asset_cfg":SceneEntityCfg("robot", body_names=["camera_head_link"])},
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        def __post_init__(self):
            super().__post_init__()
            self.base_mass.params["asset_cfg"].body_names = ["pelvis"]
            self.feet_contact_mask.params["sensor_cfg"].body_names = ["ankle_roll_.*"]

    policy: HeadPolicyCfg = HeadPolicyCfg()
    critic: HeadCriticCfg = HeadCriticCfg()

@configclass
class HeadRewardCfg(RewardsCfg):
    target_body_position_tracking = RewTerm(
        func=mdp.head_joint_tracking,
        weight=5.0,
        params={
            "command_name": "pose_command",
            "asset_cfg": SceneEntityCfg("robot", joint_names=["head_pitch_.*", "head_yaw_.*"]),
        }
    )
    def __post_init__(self):
        self.default_joint_pos.params["left_cfg"].joint_names = ["hip_yaw_l_joint", "hip_roll_l_joint"]
        self.default_joint_pos.params["right_cfg"].joint_names = ["hip_yaw_r_joint", "hip_roll_r_joint"]
        self.upper_body_pos.params["asset_cfg"].joint_names = ["body_yaw_.*", "elbow_.*", "shoulder_.*", "wrist_.*"]
        self.feet_distance.params["asset_cfg"].body_names = ["ankle_roll_.*"]
        
@configclass
class HeadCommandsCfg(CommandsCfg):
    """Command specifications for the MDP."""
    def __post_init__(self):
        self.pose_command = mdp.HeadLookTargetCommandCfg(
            asset_name="robot",
            head_body_name = "camera_head_link",
            head_joint_names = ["head_roll_.*","head_pitch_.*", "head_yaw_.*"],
            resampling_time_range=(0, 10.),
            debug_vis=True,
            ranges=mdp.HeadLookTargetCommandCfg.Ranges(
                distance = (0.3, 0.5), yaw = (-1.0, 1.0), pitch = (-1.0, 0.0)
            ),
        )

@configclass
class ProHeadFlatEnvCfg(PosingFlatEnvCfg):
    commands: HeadCommandsCfg = HeadCommandsCfg()
    observations: HeadObservationsCfg = HeadObservationsCfg()
    rewards: HeadRewardCfg = HeadRewardCfg()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        robot = PRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = robot 
        ## event set
        self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis"]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["pelvis"]
        
        ## termination set
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "pelvis", "elbow_.*", "shoulder_.*", "wrist_.*", "head_.*", "knee_.*"
        ]

@configclass
class ProHeadFlatEnvCfg_PLAY(ProHeadFlatEnvCfg):
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
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_force_robot = None

        