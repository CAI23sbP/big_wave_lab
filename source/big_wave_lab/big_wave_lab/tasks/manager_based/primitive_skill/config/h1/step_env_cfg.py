
from isaaclab.utils import configclass

from  big_wave_lab.tasks.manager_based.primitive_skill.primitive_skill_env_cfg import PosingFlatEnvCfg, RewardsCfg, ObservationsCfg, CommandsCfg

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ViewerCfg
from isaaclab.managers import SceneEntityCfg

import big_wave_lab.tasks.manager_based.primitive_skill.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets import H1_MINIMAL_CFG

@configclass
class StepObservationsCfg(ObservationsCfg):
    
    @configclass
    class StepPolicyCfg(ObservationsCfg.PolicyCfg):
        
        feet_pose_w_diff = ObsTerm(
            func=mdp.feet_pose_w_diff, 
            params={
                "command_name": "pose_command",
                "asset_cfg":SceneEntityCfg("robot", body_names=[".*ankle_link"])
                },
            scale=1.,
            clip=(-18.0, 18.0),
            )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            
    @configclass
    class StepCriticCfg(ObservationsCfg.CriticCfg):
        
        feet_pose_w_diff = ObsTerm(
            func=mdp.feet_pose_w_diff, 
            params={
                "command_name": "pose_command",
                "asset_cfg":SceneEntityCfg("robot", body_names=[".*ankle_link"])
                    },
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        
        feet_pose_w = ObsTerm(
            func=mdp.feet_pose_w, 
            params={"asset_cfg":SceneEntityCfg("robot", body_names=[".*ankle_link"])},
            scale=1.,
            clip=(-18.0, 18.0),
            history_length = 3,
            )
        
        def __post_init__(self):
            super().__post_init__()
            

    policy: StepPolicyCfg = StepPolicyCfg()
    critic: StepCriticCfg = StepCriticCfg()
        
@configclass
class StepRewardCfg(RewardsCfg):
    feet_pose_tracking = RewTerm(
        func=mdp.feet_pose_tracking, 
        weight=5.0, 
        params={
            "command_name": "pose_command",
            "asset_cfg":SceneEntityCfg("robot", body_names=[".*ankle_link"])
                },
    )

@configclass
class StepCommandsCfg(CommandsCfg):
    """Command specifications for the MDP."""
    def __post_init__(self):
        self.pose_command = mdp.FeetTargetCommandCfg(
            asset_name="robot",
            resampling_time_range=(0, 10.),
            total_num_points=1000000,
            num_way_points=10,
            debug_vis=True,
            body_names= [".*ankle_link"],
            ranges=mdp.FeetTargetCommandCfg.Ranges(
                feet_max_radius=0.25
            ),
        )

@configclass
class H1StepFlatEnvCfg(PosingFlatEnvCfg):
    commands: StepCommandsCfg = StepCommandsCfg()
    observations: StepObservationsCfg = StepObservationsCfg()
    rewards: StepRewardCfg = StepRewardCfg()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        robot = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.robot = robot 
        
        ## observation set
        # self.observations.critic.base_friction.params["asset_cfg"].body_names = [".*torso_link"]
        self.observations.critic.base_mass.params["asset_cfg"].body_names = [".*torso_link"]
        self.observations.critic.feet_contact_mask.params["sensor_cfg"].body_names = [".*ankle_link"]
        
        ## reward set
        self.rewards.default_joint_pos.params["left_cfg"].joint_names = ["left_hip_yaw", "left_hip_roll"]
        self.rewards.default_joint_pos.params["right_cfg"].joint_names = ["right_hip_yaw", "right_hip_roll"]
        self.rewards.upper_body_pos.params["asset_cfg"].joint_names = ["torso"]
        self.rewards.upper_body_pos.weight = 0.5
        ## event set
        self.events.add_base_mass.params["asset_cfg"].body_names = [".*torso_link"]
        self.events.base_com.params["asset_cfg"].body_names = [".*torso_link"]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [".*torso_link"]
        
        ## termination set
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "pelvis", 
            ".*torso_link", 
            ".*_shoulder_.*", 
            ".*_elbow_.*",
            ".*_hip_yaw_.*", 
            ".*_hip_roll_.*", 
            ".*_knee_.*"
        ]

@configclass
class H1StepFlatEnvCfg_PLAY(H1StepFlatEnvCfg):
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
        self.events.push_robot = None
