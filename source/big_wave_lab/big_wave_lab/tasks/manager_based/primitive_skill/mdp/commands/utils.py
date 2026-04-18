import numpy as np 
import torch 
import re
import torch

def _find_first_match(joint_names: list[str], patterns: list[str]) -> int | None:
    for p in patterns:
        regex = re.compile(p)
        for i, name in enumerate(joint_names):
            if regex.fullmatch(name) or regex.search(name):
                return i
    return None


def build_leg_joint_map(joint_names: list[str]) -> dict[str, int | None]:
    # left/right + sagittal joints
    patterns = {
        "left_hip_pitch": [
            r".*left.*hip.*pitch.*",
            r".*left_hip_pitch.*",
            r".*L.*hip.*pitch.*",
        ],
        "right_hip_pitch": [
            r".*right.*hip.*pitch.*",
            r".*right_hip_pitch.*",
            r".*R.*hip.*pitch.*",
        ],
        "left_knee": [
            r".*left.*knee.*",
            r".*left_knee.*",
            r".*L.*knee.*",
        ],
        "right_knee": [
            r".*right.*knee.*",
            r".*right_knee.*",
            r".*R.*knee.*",
        ],
        "left_ankle_pitch": [
            r".*left.*ankle.*pitch.*",
            r".*left_ankle_pitch.*",
            r".*L.*ankle.*pitch.*",
        ],
        "right_ankle_pitch": [
            r".*right.*ankle.*pitch.*",
            r".*right_ankle_pitch.*",
            r".*R.*ankle.*pitch.*",
        ],
    }

    joint_map = {}
    for key, pats in patterns.items():
        joint_map[key] = _find_first_match(joint_names, pats)
    return joint_map

def sample_int_from_float(x):
    if int(x) == x:
        return int(x)
    return int(x) if np.random.rand() < (x - int(x)) else int(x) + 1


def sample_wp(device, num_points, num_wp, ranges):
    '''sample waypoints, relative to the starting point'''
    # position
    l_positions = torch.randn(num_points, 3) # left wrist positions
    l_positions = l_positions / l_positions.norm(dim=-1, keepdim=True) * ranges.wrist_max_radius # within a sphere, [-radius, +radius]
    l_positions = l_positions[l_positions[:,0] > ranges.l_wrist_pos_x[0]] # keep the ones that x > ranges.l_wrist_pos_x[0]
    l_positions = l_positions[l_positions[:,0] < ranges.l_wrist_pos_x[1]] # keep the ones that x < ranges.l_wrist_pos_x[1]
    l_positions = l_positions[l_positions[:,1] > ranges.l_wrist_pos_y[0]] # keep the ones that y > ranges.l_wrist_pos_y[0]
    l_positions = l_positions[l_positions[:,1] < ranges.l_wrist_pos_y[1]] # keep the ones that y < ranges.l_wrist_pos_y[1]
    l_positions = l_positions[l_positions[:,2] > ranges.l_wrist_pos_z[0]] # keep the ones that z > ranges.l_wrist_pos_z[0]
    l_positions = l_positions[l_positions[:,2] < ranges.l_wrist_pos_z[1]] # keep the ones that z < ranges.l_wrist_pos_z[1]

    r_positions = torch.randn(num_points, 3) # right wrist positions
    r_positions = r_positions / r_positions.norm(dim=-1, keepdim=True) * ranges.wrist_max_radius # within a sphere, [-radius, +radius]
    r_positions = r_positions[r_positions[:,0] > ranges.r_wrist_pos_x[0]] # keep the ones that x > ranges.r_wrist_pos_x[0]
    r_positions = r_positions[r_positions[:,0] < ranges.r_wrist_pos_x[1]] # keep the ones that x < ranges.r_wrist_pos_x[1]
    r_positions = r_positions[r_positions[:,1] > ranges.r_wrist_pos_y[0]] # keep the ones that y > ranges.r_wrist_pos_y[0]
    r_positions = r_positions[r_positions[:,1] < ranges.r_wrist_pos_y[1]] # keep the ones that y < ranges.r_wrist_pos_y[1]
    r_positions = r_positions[r_positions[:,2] > ranges.r_wrist_pos_z[0]] # keep the ones that z > ranges.r_wrist_pos_z[0]
    r_positions = r_positions[r_positions[:,2] < ranges.r_wrist_pos_z[1]] # keep the ones that z < ranges.r_wrist_pos_z[1]
    
    num_pairs = min(l_positions.size(0), r_positions.size(0))
    positions = torch.stack([l_positions[:num_pairs], r_positions[:num_pairs]], dim=1) # (num_pairs, 2, 3)
    
    # rotation (quaternion)
    quaternions = torch.randn(num_pairs, 2, 4)
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)
    
    # concat
    wp = torch.cat([positions, quaternions], dim=-1) # (num_pairs, 2, 7)
    # repeat for num_wp
    wp = wp.unsqueeze(1).repeat(1, num_wp, 1, 1) # (num_pairs, num_wp, 2, 7)
    print("===> [sample_wp] return shape:", wp.shape)
    return wp.to(device), num_pairs, num_wp


def sample_fp(device, num_points, num_wp, ranges):
    '''sample feet waypoints'''
    # left foot still, right foot move, [num_points//2, 2]
    l_positions_s = torch.zeros(num_points//2, 2) # left foot positions (xy)
    r_positions_m = torch.randn(num_points//2, 2)
    r_positions_m = r_positions_m / r_positions_m.norm(dim=-1, keepdim=True) * ranges.feet_max_radius # within a sphere, [-radius, +radius]
    # right foot still, left foot move, [num_points//2, 2]
    r_positions_s = torch.zeros(num_points//2, 2) # right foot positions (xy)
    l_positions_m = torch.randn(num_points//2, 2)
    l_positions_m = l_positions_m / l_positions_m.norm(dim=-1, keepdim=True) * ranges.feet_max_radius # within a sphere, [-radius, +radius]
    # concat
    l_positions = torch.cat([l_positions_s, l_positions_m], dim=0) # (num_points, 2)
    r_positions = torch.cat([r_positions_m, r_positions_s], dim=0) # (num_points, 2)
    wp = torch.stack([l_positions, r_positions], dim=1) # (num_points, 2, 2)
    wp = wp.unsqueeze(1).repeat(1, num_wp, 1, 1) # (num_points, num_wp, 2, 2)
    print("===> [sample_fp] return shape:", wp.shape)
    return wp.to(device), num_points, num_wp
    