import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.assets import RigidObjectCfg
import os


ASSET_PATH = os.path.dirname(__file__)

BREAD_BOX_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/BreadBox",
    init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.45, 0.6996], rot=[1, 0, 0, 0]),
    spawn=UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/bread_box/bread_box.usd",
        scale=(0.75, 0.75, 0.75),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    ),
)

TABLE_1_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Table1",
    init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.45, 0.6996], rot=[1, 0, 0, 0]),
    spawn=UsdFileCfg(
        usd_path=f"{ASSET_PATH}/support/support0.usd",
        scale=(0.75, 0.75, 0.75),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    ),
)

TABLE_2_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Table2",
    init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.45, 0.6996], rot=[1, 0, 0, 0]),
    spawn=UsdFileCfg(
        usd_path=f"{ASSET_PATH}/support/support1.usd",
        scale=(0.75, 0.75, 0.75),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    ),
)
