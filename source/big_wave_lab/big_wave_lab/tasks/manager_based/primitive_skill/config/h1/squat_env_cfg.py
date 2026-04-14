
from isaaclab.utils import configclass

from  big_wave_lab.tasks.manager_based.primitive_skill.primitive_skill_env_cfg import PosingFlatEnvCfg, RewardsCfg, ObservationsCfg, CommandsCfg

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ViewerCfg

import big_wave_lab.tasks.manager_based.primitive_skill.mdp as mdp

##
# Pre-defined configs
##
from big_wave_lab.assets.robot_cfg import H1_2_CFG

@configclass
class SquatObservationsCfg(ObservationsCfg):
    
    @configclass
    class SquatPolicyCfg(ObservationsCfg.PolicyCfg):
        
        base_height_diff = ObsTerm(
            func=mdp.base_height_diff, 
            params={"command_name": "pose_command"},
            scale=1.,
            clip=(-18.0, 18.0),
            )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            
    @configclass
    class SquatCriticCfg(ObservationsCfg.CriticCfg):
        
        base_height_diff = ObsTerm(
            func=mdp.base_height_diff, 
            params={"command_name": "pose_command"},
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        
        base_height = ObsTerm(
            func=mdp.base_height, 
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        
        def __post_init__(self):
            super().__post_init__()
            self.base_mass.params["asset_cfg"].body_names = [".*torso_link"]
            self.feet_contact_mask.params["sensor_cfg"].body_names = [".*_ankle_roll_.*"]
        

    policy: SquatPolicyCfg = SquatPolicyCfg()
    critic: SquatCriticCfg = SquatCriticCfg()
        
@configclass
class SquatRewardCfg(RewardsCfg):
    target_height_tracking = RewTerm(
        func=mdp.base_height_tracking,
        weight=5.0,
        params={"command_name": "pose_command"},
    )

    def __post_init__(self):
        self.default_joint_pos.params["left_cfg"].joint_names = ["left_hip_yaw_.*", "left_hip_roll_.*"]
        self.default_joint_pos.params["right_cfg"].joint_names = ["right_hip_yaw_.*", "right_hip_roll_.*"]
        self.upper_body_pos.params["asset_cfg"].joint_names = ["torso_.*", ".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"]

        self.feet_distance.params["asset_cfg"].body_names = [".*_ankle_roll_.*"]
        
@configclass
class SquatCommandsCfg(CommandsCfg):
    """Command specifications for the MDP."""
    def __post_init__(self):
        self.pose_command = mdp.BaseHeightCommandCfg(
            asset_name="robot",
            resampling_time_range=(0, 10.),
            total_num_points=1000000,
            num_way_points=10,
            base_height_target=1.0,
            debug_vis=True,
            ranges=mdp.BaseHeightCommandCfg.Ranges(
                base_height_std=0.2, base_height_scale=(0.2, 1.05)
            ),
        )

@configclass
class H1SquatFlatEnvCfg(PosingFlatEnvCfg):
    commands: SquatCommandsCfg = SquatCommandsCfg()
    observations: SquatObservationsCfg = SquatObservationsCfg()
    rewards: SquatRewardCfg = SquatRewardCfg()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        robot = H1_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = robot 
        
        ## event set
        self.events.add_base_mass = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [".*torso_link"]
        
        ## termination set
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "pelvis", 
            ".*torso_link", 
            ".*_shoulder_.*", 
            ".*_elbow_.*",
        ]

@configclass
class H1SquatFlatEnvCfg_PLAY(H1SquatFlatEnvCfg):
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
        # self.observations.policy.enable_corruption = 
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
