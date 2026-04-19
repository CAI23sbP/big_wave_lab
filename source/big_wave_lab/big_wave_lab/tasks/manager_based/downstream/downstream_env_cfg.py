
import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg

import big_wave_lab.tasks.manager_based.downstream.mdp as mdp
import big_wave_lab.tasks.manager_based.primitive_skill.mdp as primitive_mdp

from big_wave_lab.tasks.manager_based.primitive_skill.primitive_skill_env_cfg import PosingFlatEnvCfg
LOW_LEVEL_ENV_CFG = PosingFlatEnvCfg()
##
# Pre-defined configs
##

@configclass
class CommandsCfg:
    downstream_command = mdp.DownStramCommandCfg(
        asset_name="robot",
        resampling_time_range=(0, 0.),
        debug_vis=False,
    )

@configclass
class DownstreamObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class DownstreamPolicyCfg(LOW_LEVEL_ENV_CFG.observations.PolicyCfg):
        """Observations for policy group."""
        pose_command = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "downstream_command"},
            scale=1.,
            clip=(-18.0, 18.0),
            )
        def __post_init__(self):
            super().__post_init__()
            
    @configclass
    class DownstreamCriticCfg(LOW_LEVEL_ENV_CFG.observations.CriticCfg):
        
        def __post_init__(self):
            super().__post_init__()
            self.pose_command.params["command_name"] = "downstream_command"
            
    # observation groups
    policy: DownstreamPolicyCfg = DownstreamPolicyCfg()
    critic: DownstreamCriticCfg = DownstreamCriticCfg()

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    is_alive = RewTerm(
        func=mdp.is_alive, 
        weight=5., 
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.01, 
    )
    orientation = RewTerm(
        func=primitive_mdp.orientation, 
        weight=1., 
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, 
        weight=-1e-7, 
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, 
        weight=-1e-5, 
    )
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2, 
        weight=-5e-4, 
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, 
        weight=-1., 
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[""]), "threshold": 1.0},
    )
    
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    downstream_joint_pos = mdp.PreTrainedSkillPolicyActionCfg(
        asset_name="robot", 
        policy_paths=None, 
        low_level_decimation=1, 
        high_level_command_name = "downstream_command",
        low_level_actions = LOW_LEVEL_ENV_CFG.actions.joint_pos,
        common_low_level_observations = LOW_LEVEL_ENV_CFG.observations.policy,
        low_level_command_observations = None,
        low_level_command_term_names = None,
        low_level_command_size = None
    )

##
# Environment configuration
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    

@configclass
class DonwStreamEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene = MySceneCfg()
    # Basic settings
    observations: DownstreamObservationsCfg = DownstreamObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations= LOW_LEVEL_ENV_CFG.terminations
    events= LOW_LEVEL_ENV_CFG.events
    curriculum = LOW_LEVEL_ENV_CFG.curriculum

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 24.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
