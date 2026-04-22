import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.assets import RigidObjectCfg
import os


ASSET_PATH = os.path.dirname(__file__)

BREAD_BOX_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/BreadBox",
    spawn=UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/bread_box/bread_box.usd",
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=3.0,
        ),
    ),
)

TABLE_1_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Table1",
    spawn=UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/support/support0.usd",
        activate_contact_sensors=False,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=10000.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=True,
            max_linear_velocity=0.0,
            max_angular_velocity=0.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    ),
)

TABLE_2_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Table2",
    spawn=UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/support/support1.usd",
        activate_contact_sensors=False,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=10000.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=True,
            max_linear_velocity=0.0,
            max_angular_velocity=0.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    ),
)
