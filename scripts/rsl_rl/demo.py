# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates an interactive demo with the H1 rough terrain environment.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/h1_locomotion.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import cli_args  # isort: skip


from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates an interactive demo with the H1 rough terrain environment."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch

import carb
import omni
from isaacsim.core.utils.stage import get_current_stage
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf
from rsl_rl.runners import OnPolicyRunner
from modules.runners import AmpOnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path

import rsl_rl.utils.utils as utils
from rsl_rl.networks import EmpiricalNormalization
utils.Normalizer = EmpiricalNormalization
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from big_wave_lab.tasks.manager_based.amp_task.walk_env_cfg import WalkTaskFlatEnvCfg_Play
from big_wave_lab.tasks.manager_based.nominal_pos_task.nominal_env_cfg import NominalPoseTaskFlatEnvCfg_Play

WALK_TASK = "Walk-Flat-Task-v0"
STANDING_TASK = "Standing-Task-v0"
RL_LIBRARY = "rsl_rl"
HISTORY_LENGHT = 10


class TienkungDemo:
    def __init__(self):
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(WALK_TASK, args_cli)
        # load the trained jit policy
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        checkpoint = retrieve_file_path("/home/cai/humanoid_ws/big_wave_lab/logs/rsl_rl/amp_walk/2026-03-23_13-04-04/model_49999.pt")
    
        # create envionrment
        env_cfg = WalkTaskFlatEnvCfg_Play()
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1000000
        env_cfg.curriculum = None
        # wrap around environment for rsl-rl
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        # load previously trained model
        ppo_runner = AmpOnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        ppo_runner.load(checkpoint)
        # obtain the trained policy for inference
        self.walk_policy = ppo_runner.get_inference_policy(device=self.device)
        
        del ppo_runner, agent_cfg, checkpoint, log_root_path
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(STANDING_TASK, args_cli)
        # load the trained jit policy

        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        
        self.standing_policy = torch.jit.load("/home/cai/humanoid_ws/big_wave_lab/logs/rsl_rl/tienkung_nominal_pos/2026-03-31_18-47-09/exported/policy.pt", map_location="cuda:0")
        self.standing_policy.eval()
        if hasattr(self.standing_policy, "reset"):
            self.standing_policy.reset()
        self.create_camera()
        self.commands = torch.zeros(env_cfg.scene.num_envs, 4, device=self.device)
        self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")
        self.set_up_keyboard()
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._selected_action = 0
        self._previous_selected_id = None
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)

    def create_camera(self):
        """Creates a camera to be used for third-person view."""
        stage = get_current_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        # Create camera
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.perspective_path)

    def set_up_keyboard(self):
        """Sets up interface for keyboard input and registers the desired keys for control."""
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        T = 1
        R = 0.5
        self._key_to_control = {
            "UP": torch.tensor([T, 0.0, 0.0, 0.0], device=self.device),
            "DOWN": torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device),
            "LEFT": torch.tensor([T, 0.0, 0.0, -R], device=self.device),
            "RIGHT": torch.tensor([T, 0.0, 0.0, R], device=self.device),
            "ZEROS": torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device),
        }

    def _on_keyboard_event(self, event):
        """Checks for a keyboard event and assign the corresponding command control depending on key pressed."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Arrow keys map to pre-defined command vectors to control navigation of robot
            if event.input.name in self._key_to_control:
                if self._selected_id:
                    self.commands[self._selected_id] = self._key_to_control[event.input.name]
            # Escape key exits out of the current selected robot view
            elif event.input.name == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
            # C key swaps between third-person and perspective views
            elif event.input.name == "C":
                if self._selected_id is not None:
                    if self.viewport.get_active_camera() == self.camera_path:
                        self.viewport.set_active_camera(self.perspective_path)
                    else:
                        self.viewport.set_active_camera(self.camera_path)
            elif event.input.name == "M":
                if self._selected_action == 0:
                    self._selected_action = 1
                else:
                    self._selected_action = 0
        # On key release, the robot stops moving
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if self._selected_id:
                self.commands[self._selected_id] = self._key_to_control["ZEROS"]

    def update_selected_object(self):
        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            self._selected_id = None
            self.viewport.set_active_camera(self.perspective_path)
        elif len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please only select one!")
        else:
            prim_splitted_path = selected_prim_paths[0].split("/")
            # a valid robot was selected, update the camera to go into third-person view
            if len(prim_splitted_path) >= 4 and prim_splitted_path[3][0:4] == "env_":
                self._selected_id = int(prim_splitted_path[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("The selected prim was not a Tienkung robot")

        # Reset commands for previously selected robot if a new one is selected
        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            self.env.unwrapped.command_manager.reset([self._previous_selected_id])
            self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")

    def _update_camera(self):
        """Updates the per-frame transform of the third-person view camera to follow
        the selected robot's torso transform."""

        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[self._selected_id, :]  # - env.scene.env_origins
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[self._selected_id, :]

        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos

        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)

    def select_actions(self, walk_action, standing_action):
        if self._selected_action == 0 :
            walk_action[self._selected_id] = standing_action[self._selected_id]
            return walk_action
        
        if self._selected_action == 1 :
            return walk_action
        
    def remove_command_from_flattened_history(self, flat_obs, history_len, cmd_start, cmd_end):
        B = flat_obs.shape[0]
        obs = flat_obs.view(B, history_len, -1)  # [B, H, D]
        obs_wo_cmd = torch.cat(
            [obs[:, :, :cmd_start], obs[:, :, cmd_end:cmd_end+90]],
            dim=-1
        ) 
        return obs_wo_cmd.reshape(B, -1)
    
    def standing_obs(self, obs):
        origin_obs = obs.clone()
        commands_expction = self.remove_command_from_flattened_history(origin_obs, 10, 6, 9)
        # commands_expction = self.regroup_flattened_history(commands_expction, 10, 30)
        return commands_expction
    
    def regroup_flattened_history(self, flat_obs, history_len, joint_dim):
        B = flat_obs.shape[0]
        # 각 block 차원
        ang_vel_dim = 3
        gravity_dim = 3
        joint_pos_dim = joint_dim
        joint_vel_dim = joint_dim
        action_dim = joint_dim

        obs_dim = (
            ang_vel_dim
            + gravity_dim
            + joint_pos_dim
            + joint_vel_dim
            + action_dim
        )

        obs = flat_obs.view(B, history_len, obs_dim)   # [B, H, D]

        s = 0
        ang_vel = obs[:, :, s:s+ang_vel_dim]; s += ang_vel_dim
        gravity = obs[:, :, s:s+gravity_dim]; s += gravity_dim
        joint_pos = obs[:, :, s:s+joint_pos_dim]; s += joint_pos_dim
        joint_vel = obs[:, :, s:s+joint_vel_dim]; s += joint_vel_dim
        action = obs[:, :, s:s+action_dim]; s += action_dim

        # 각 block별 history를 flatten
        ang_vel_flat = ang_vel.reshape(B, -1)
        gravity_flat = gravity.reshape(B, -1)
        joint_pos_flat = joint_pos.reshape(B, -1)
        joint_vel_flat = joint_vel.reshape(B, -1)
        action_flat = action.reshape(B, -1)

        new_flat = torch.cat([
            ang_vel_flat,
            gravity_flat,
            joint_pos_flat,
            joint_vel_flat,
            action_flat,
        ], dim=-1)

        return new_flat
    
    def update_last_commands(self, flat_obs, history_len, cmd_start, cmd_end, command):
        B = flat_obs.shape[0]
        obs = flat_obs.view(B, history_len, -1)  # [B, H, D]
        obs[:, :, cmd_start:cmd_end] = torch.roll(obs[:, :, cmd_start:cmd_end], shifts=-1, dims=1)
        obs[:, -1, cmd_start:cmd_end] = command[:,:-1]
        return obs.reshape(B, -1)

        
def main():
    """Main function."""
    demo_env = TienkungDemo()
    obs, _ = demo_env.env.reset()
    while simulation_app.is_running():
        # demo_standing for selected robots
        demo_env.update_selected_object()
        with torch.inference_mode():
            walk_action = demo_env.walk_policy(obs)
            standing_action = demo_env.standing_policy(demo_env.standing_obs(obs["policy"]))
            action = demo_env.select_actions(walk_action, standing_action)
            obs, _, _, _ = demo_env.env.step(action)
            obs["policy"] = demo_env.update_last_commands(obs["policy"].clone(), 10, 6,9, demo_env.commands)

if __name__ == "__main__":
    main()
    simulation_app.close()
