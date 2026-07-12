import pickle
from typing import Dict, Optional, Sequence, List
from pathlib import Path
import json
import torch
import numpy as np
import pickle
from PIL import Image
import random
import open3d as o3d
import subprocess
import torch.nn as nn
import time
from PIL import ImageDraw, ImageFont
import os
from typing import List, Tuple
from rlbench.backend.const import (FRONT_RGB_FOLDER, FRONT_DEPTH_FOLDER,
    FRONT_MASK_FOLDER, IMAGE_FORMAT, DEPTH_SCALE, LOW_DIM_PICKLE)
from rlbench.backend import utils
from matplotlib import pyplot as plt
from rlbench.backend.utils import float_array_to_rgb_image, image_to_float_array
from rlbench.backend.const import DEPTH_SCALE

Instructions = Dict[str, Dict[int, torch.Tensor]]

def get_expert_info(task):
    task_name = task.name
    if task_name == "reach_single_moving_target_on_the_table":
        from rlbench.tasks.reach_single_moving_target_on_the_table import get_expert_info as fn
        task.disable_expert_plan()
        return fn(task, bool_return_path=False)
    elif task_name == "pick_moving_target_on_the_table":
        from rlbench.tasks.pick_moving_target_on_the_table import get_expert_info as fn
        task.disable_expert_plan()
        return fn(task, bool_return_path=False)
    elif task_name == "place_cups_on_rotating_frame":
        from rlbench.tasks.place_cups_on_rotating_frame import get_expert_info as fn
        task.disable_expert_plan()
        return fn(task, bool_return_path=False)
    elif task_name == "remove_cups_from_rotating_frame":
        from rlbench.tasks.remove_cups_from_rotating_frame import get_expert_info as fn
        task.disable_expert_plan()
        return fn(task, bool_return_path=False)
    elif task_name == "put_rubbish_in_moving_bin":
        from rlbench.tasks.put_rubbish_in_moving_bin import get_expert_info as fn
        task.disable_expert_plan()
        return fn(task, bool_return_path=False)
    elif task_name == "push_moving_button":
        from rlbench.tasks.push_moving_button import get_expert_info as fn
        task.disable_expert_plan()
        return fn(task, bool_return_path=False)
    elif task_name == 'moving_basketball_in_hoop':
        from rlbench.tasks.moving_basketball_in_hoop import get_expert_info as fn
        task.disable_expert_plan()
        return fn(task, bool_return_path=False)
    elif task_name == 'moving_basketcube_in_hoop':
        from rlbench.tasks.moving_basketball_in_hoop import get_expert_info as fn
        task.disable_expert_plan()
        return fn(task, bool_return_path=False)
    elif task_name == 'moving_colorful_basketball_in_hoop':
        from rlbench.tasks.moving_colorful_basketball_in_hoop import get_expert_info as fn
        task.disable_expert_plan()
        return fn(task, bool_return_path=False)    
    elif task_name == 'moving_basketball_in_hoop_high_speed':
        from rlbench.tasks.moving_basketball_in_hoop_high_speed import get_expert_info as fn
        task.disable_expert_plan()
        return fn(task, bool_return_path=False)
    elif task_name == "insert_onto_rotating_peg":
        from rlbench.tasks.insert_onto_rotating_peg import get_expert_info as fn
        task.disable_expert_plan()
        return fn(task, bool_return_path=False)
    elif task_name == "beat_the_rotating_buzz":
        from rlbench.tasks.beat_the_rotating_buzz import get_expert_info as fn
        task.disable_expert_plan()
        return fn(task, bool_return_path=False)
    elif task_name == "close_moving_box":
        from rlbench.tasks.close_moving_box import get_expert_info as fn
        task.disable_expert_plan()
        return fn(task, bool_return_path=False)
    elif task_name == "reach_single_moving_target_on_the_table_high_speed":
        from rlbench.tasks.reach_single_moving_target_on_the_table_high_speed import get_expert_info as fn
        task.disable_expert_plan()
        err_msg = "Error: the max_velocity is not 3 or the max_acceleration is not 12"
        assert task.robot.arm.max_velocity == 3, err_msg
        assert task.robot.arm.max_acceleration == 12, err_msg
        return fn(task, bool_return_path=False)
    elif task_name == "pick_moving_target_on_the_table_high_speed":
        from rlbench.tasks.pick_moving_target_on_the_table_high_speed import get_expert_info as fn
        task.disable_expert_plan()
        err_msg = "Error: the max_velocity is not 3 or the max_acceleration is not 12"
        assert task.robot.arm.max_velocity == 3, err_msg
        assert task.robot.arm.max_acceleration == 12, err_msg
        return fn(task, bool_return_path=False)
    elif task_name == "place_cups_on_rotating_frame_high_speed":
        from rlbench.tasks.place_cups_on_rotating_frame_high_speed import get_expert_info as fn
        task.disable_expert_plan()
        err_msg = "Error: the max_velocity is not 3 or the max_acceleration is not 12"
        assert task.robot.arm.max_velocity == 3, err_msg
        assert task.robot.arm.max_acceleration == 12, err_msg
        return fn(task, bool_return_path=False)
    elif task_name == "remove_cups_from_rotating_frame_high_speed":
        from rlbench.tasks.remove_cups_from_rotating_frame_high_speed import get_expert_info as fn
        task.disable_expert_plan()
        err_msg = "Error: the max_velocity is not 3 or the max_acceleration is not 12"
        assert task.robot.arm.max_velocity == 3, err_msg
        assert task.robot.arm.max_acceleration == 12, err_msg
        return fn(task, bool_return_path=False)
    elif task_name == "beat_the_rotating_buzz_high_speed":
        from rlbench.tasks.beat_the_rotating_buzz_high_speed import get_expert_info as fn
        task.disable_expert_plan()
        err_msg = "Error: the max_velocity is not 3 or the max_acceleration is not 12"
        assert task.robot.arm.max_velocity == 3, err_msg
        assert task.robot.arm.max_acceleration == 12, err_msg
        return fn(task, bool_return_path=False)
    elif task_name == "insert_onto_rotating_peg_high_speed":
        from rlbench.tasks.insert_onto_rotating_peg_high_speed import get_expert_info as fn
        task.disable_expert_plan()
        err_msg = "Error: the max_velocity is not 3 or the max_acceleration is not 12"
        assert task.robot.arm.max_velocity == 3, err_msg
        assert task.robot.arm.max_acceleration == 12, err_msg
        return fn(task, bool_return_path=False)
    elif task_name == "push_moving_button_high_speed":
        from rlbench.tasks.push_moving_button_high_speed import get_expert_info as fn
        task.disable_expert_plan()
        err_msg = "Error: the max_velocity is not 3 or the max_acceleration is not 12"
        assert task.robot.arm.max_velocity == 3, err_msg
        assert task.robot.arm.max_acceleration == 12, err_msg
        return fn(task, bool_return_path=False)
    elif task_name == "put_rubbish_in_moving_bin_high_speed":
        from rlbench.tasks.put_rubbish_in_moving_bin_high_speed import get_expert_info as fn
        task.disable_expert_plan()
        err_msg = "Error: the max_velocity is not 3 or the max_acceleration is not 12"
        assert task.robot.arm.max_velocity == 3, err_msg
        assert task.robot.arm.max_acceleration == 12, err_msg
        return fn(task, bool_return_path=False)
    else:
        raise ValueError(f"Invalid task name: {task_name}")

def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def save_observations(
    obs,
    save_dir,
    frame_id,
    camera_names,
    camera_list,
    camera_mask_list,
    ):
    def _get_camera(target_cam_name: str) -> Tuple:
        idx = [itm.get_name() for itm in camera_list].index(target_cam_name)
        cam_sensor = camera_list[idx]
        cam_mask_sensor = camera_mask_list[idx]
        return cam_sensor, cam_mask_sensor

    # save fixed cameras
    # save nerf cameras (maybe you need to update)
    example_path = Path(save_dir)
    front_rgb_path = os.path.join(example_path, FRONT_RGB_FOLDER)
    front_depth_path = os.path.join(example_path, FRONT_DEPTH_FOLDER)
    front_mask_path = os.path.join(example_path, FRONT_MASK_FOLDER)
    check_and_make(front_rgb_path)
    check_and_make(front_depth_path)
    check_and_make(front_mask_path)

    front_rgb = Image.fromarray(obs.front_rgb)
    front_depth = utils.float_array_to_rgb_image(obs.front_depth, scale_factor=DEPTH_SCALE)
    front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8))
    front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % frame_id))
    front_depth.save(os.path.join(front_depth_path, IMAGE_FORMAT % frame_id))
    front_mask.save(os.path.join(front_mask_path, IMAGE_FORMAT % frame_id))

    # 采集 nerf camera rgb, pcd, depth, camera pose, mask
    for cam_name in camera_names:
        if not cam_name.isdigit():
            continue
        nerf_rgb_path = os.path.join(example_path, f"nerf_data/{frame_id}/images/")
        nerf_depth_path = os.path.join(example_path, f"nerf_data/{frame_id}/depths/")
        nerf_mask_path = os.path.join(example_path, f"nerf_data/{frame_id}/masks/")
        nerf_pose_path = os.path.join(example_path, f"nerf_data/{frame_id}/poses/")
        check_and_make(nerf_rgb_path)
        check_and_make(nerf_depth_path)
        check_and_make(nerf_mask_path)
        check_and_make(nerf_pose_path)
        nerf_rgb = Image.fromarray(getattr(obs, f"cam{cam_name}_rgb").astype(np.uint8))
        nerf_depth = utils.float_array_to_rgb_image(getattr(obs, f"cam{cam_name}_depth"), scale_factor=DEPTH_SCALE)
        nerf_mask = Image.fromarray((getattr(obs, f"cam{cam_name}_mask") * 255).astype(np.uint8))
        cam, _ = _get_camera(cam_name)
        extrinsic = cam.get_matrix()
        intrinsic = cam.get_intrinsic_matrix()
        near, far = cam.get_near_clipping_plane(), cam.get_far_clipping_plane()
        nerf_rgb.save(os.path.join(nerf_rgb_path, f"{cam_name}.png"))
        nerf_depth.save(os.path.join(nerf_depth_path, f"{cam_name}.png"))
        nerf_mask.save(os.path.join(nerf_mask_path, f"{cam_name}.png"))
        write_pkl({
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
            "near": near,
            "far": far,
        }, os.path.join(nerf_pose_path, f"{cam_name}.pkl"))
    return

def save_expert_info(expert_info_dict, save_dir, frame_id):
    example_path = Path(save_dir)
    expert_info_path = example_path/ "expert_info"
    check_and_make(expert_info_path)
    write_pkl(expert_info_dict, expert_info_path / f"{frame_id}.pkl")
    return


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def quaternion_xyzw_to_matrix(r: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix
    Args:
        r: (K, 4) [x, y, z, w]
    Return:
        R: (K, 3, 3)
    """
    return quaternion_wxyz_to_matrix(r[..., [3, 0, 1, 2]])

def quaternion_wxyz_to_matrix(r: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix
    Args:
        r: (K, 4) [w, x, y, z]
    Return:
        R: (K, 3, 3)
    """
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]
    device = r.device

    R = torch.zeros((q.size(0), 3, 3), device=device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def construct_pose_gs_dict(
    eepose: torch.Tensor,
    ) -> Dict:
    '''
    Args:
        action: torch.Tensor (7+) [x,y,z,qx,qy,qz,qw,...]
    Return:
        xyz_rgb: torch.Tensor (300, 6) [x,y,z,r,g,b]
    '''
    # Extract translation and rotation from action
    t = eepose[:3]
    q = eepose[3:7]

    # Convert quaternion to rotation matrix
    R = quaternion_xyzw_to_matrix(q.unsqueeze(0))[0]

    # Create the base coordinate axes
    num_points = 100
    x_axis = torch.zeros((num_points, 3))
    y_axis = torch.zeros((num_points, 3))
    z_axis = torch.zeros((num_points, 3))

    x_axis[:, 0] = torch.linspace(0, 0.1, num_points)
    y_axis[:, 1] = torch.linspace(0, 0.1, num_points)
    z_axis[:, 2] = torch.linspace(0, 0.1, num_points)

    # Concatenate the axes
    base_xyz = torch.cat([x_axis, y_axis, z_axis], dim=0).to(R.device)

    # Apply rotation and translation
    rotated_xyz = (R @ base_xyz.T).T + t

    # Create the colors for the axes
    x_rgb = torch.zeros((num_points, 3))
    y_rgb = torch.zeros((num_points, 3))
    z_rgb = torch.zeros((num_points, 3))

    # Red for x, Green for y, Blue for z
    x_rgb[:, 0] = 1.0
    y_rgb[:, 1] = 1.0
    z_rgb[:, 2] = 1.0

    base_rgb = torch.cat([x_rgb, y_rgb, z_rgb], dim=0).to(R.device)

    # Combine the transformed points and colors
    xyz_rgb = torch.cat([rotated_xyz, base_rgb], dim=1)

    means3D = rotated_xyz
    means2D = torch.zeros_like(means3D, requires_grad=True)
    try:
        means2D.retain_grad()
    except:
        pass
    shs = RGB2SH(base_rgb)
    shs_16 = torch.zeros(
        (shs.shape[0], 16, 3),
        dtype=shs.dtype, device=shs.device)
    shs_16[:, 0, :] = shs
    opacity = torch.ones_like(base_rgb[:, :1])
    scales = torch.ones_like(base_rgb) * 0.004
    rotations = torch.zeros((shs_16.shape[0], 4), dtype=shs.dtype, device=shs.device)
    rotations[:, 0] = 1
    valid = torch.ones_like(base_rgb[:, 0]).bool()
    return {
        "means3D": means3D,
        "means2D": means2D,
        "sh": shs_16,
        "rgb": base_rgb,
        "opacity": opacity,
        "scales": scales,
        "rotations": rotations,
        "valid": valid,
}
class LowDimObsDemo:
    def __init__(self):
        self._observations = []
        self._expert_trajectories = []

    def __getitem__(self, i):
        return self._observations[i]
    
    def __len__(self):
        return len(self._observations)

    def restore_state(self):
        return


class LowDimObservation:
    def __init__(self, gripper_pose, gripper_open, joint_positions, misc):
        self.gripper_pose = gripper_pose
        self.gripper_open = gripper_open
        self.joint_positions = joint_positions
        self.misc = misc

def save_low_dim_obs(obs, save_dir, low_dim_obs_demo):
    # 采集 low_dim_observation
    obs.misc['gripper_joint_positions'] = obs.gripper_joint_positions
    low_dim_obs = LowDimObservation(
        gripper_pose=obs.gripper_pose,
        gripper_open=obs.gripper_open,
        joint_positions=obs.joint_positions,
        misc=obs.misc,
    )
    low_dim_obs_demo._observations.append(low_dim_obs)
    with open(os.path.join(save_dir, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(low_dim_obs_demo, f)
    return low_dim_obs_demo

def save_variation_number(variation_number, save_dir):
    with open(os.path.join(save_dir, "variation_number.pkl"), 'wb') as f:
        pickle.dump(int(variation_number), f)
    return

def save_variation_description(description, save_dir):
    with open(os.path.join(save_dir, "variation_descriptions.pkl"), 'wb') as f:
        pickle.dump(description, f)
    return

def save_target_state(target_state_list, save_dir):
    with open(os.path.join(save_dir, "target_state.pkl"), 'wb') as f:
        pickle.dump(target_state_list, f)
    return


def _world_points_to_uv(
    points_xyz: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    z_eps: float = 1e-5,
) -> tuple:
    """Project world points to image uv coordinates."""
    R = np.asarray(extrinsics, dtype=np.float64)[:3, :3]
    t = np.asarray(extrinsics, dtype=np.float64)[:3, 3]
    K = np.asarray(intrinsics, dtype=np.float64)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    pw = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    pc = (pw - t) @ R
    z = pc[:, 2]
    valid = z > z_eps
    u = np.zeros(len(pc), dtype=np.float64)
    v = np.zeros(len(pc), dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        u[valid] = fx * pc[valid, 0] / z[valid] + cx
        v[valid] = fy * pc[valid, 1] / z[valid] + cy
    return u, v, valid


def _draw_gs_axes_on_rgb_hwc(
    rgb_hwc: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    axis_rgb: np.ndarray,
    num_per_axis: int = 100,
    line_width: int = 2,
) -> np.ndarray:
    """Draw projected axis polylines from construct_pose_gs_dict on RGB image."""
    h, w = rgb_hwc.shape[:2]
    im = Image.fromarray(np.ascontiguousarray(rgb_hwc))
    draw = ImageDraw.Draw(im)
    n_axis_rgb = axis_rgb.shape[0]
    assert n_axis_rgb == num_per_axis * 3
    for a in range(3):
        lo = a * num_per_axis
        hi = lo + num_per_axis
        col = axis_rgb[lo]
        fill = tuple(int(np.clip(float(c) * 255.0, 0, 255)) for c in col[:3])
        for i in range(lo, hi - 1):
            if not (valid[i] and valid[i + 1]):
                continue
            ui, vi = int(round(u[i])), int(round(v[i]))
            uj, vj = int(round(u[i + 1])), int(round(v[i + 1]))
            if not (0 <= ui < w and 0 <= vi < h and 0 <= uj < w and 0 <= vj < h):
                continue
            draw.line([(ui, vi), (uj, vj)], fill=fill, width=line_width)
    return np.asarray(im)


def visualize_demo_data(episode_dir, frame_id, vis_dir, camera_names):
    from datasets_module.dataset_engine import RLBenchReachMovingTargetDataset
    cam_param_dict = RLBenchReachMovingTargetDataset.get_cam_param_dict(
        camera_names=camera_names,
        episode_dir=episode_dir,
        episode_hf=None,
    )
    observations = {}
    for cam_name in camera_names:
        # (256, 256, 3) in range [-1,1]
        rgb = RLBenchReachMovingTargetDataset.read_rgb_data(
            episode_dir, cam_name, frame_id)
        # (256, 256)
        depth = RLBenchReachMovingTargetDataset.read_depth_data(
            episode_dir, cam_name, frame_id, cam_param_dict)
        # (256, 256, 3)
        pcd = RLBenchReachMovingTargetDataset.read_pcd_data(
            episode_dir, cam_name, frame_id, cam_param_dict)
        mask = RLBenchReachMovingTargetDataset.read_mask_data(
            episode_dir, cam_name, frame_id)
        observations[cam_name] = {
            'rgb': rgb, # (h, w, 3) in range [-1, 1]
            'depth': depth, # (h, w)
            'pcd': pcd, # (h, w, 3)
            'mask': mask,
        }
    # visualize expert trajectory
    expert_info = read_pkl(episode_dir / "expert_info" / f"{frame_id}.pkl")
    expert_traj = expert_info["trajectory"]
    if isinstance(expert_traj, torch.Tensor):
        expert_pose_8 = expert_traj.detach().cpu().to(torch.float32).reshape(-1)[:8]
    else:
        expert_pose_8 = torch.as_tensor(
            np.asarray(expert_traj, dtype=np.float32).reshape(-1)[:8],
            dtype=torch.float32,
        )
    gs_dict = construct_pose_gs_dict(expert_pose_8)
    means_gt = gs_dict["means3D"].detach().cpu().numpy()
    axis_rgb_gt = gs_dict["rgb"].detach().cpu().numpy()
    # visualize current gripper
    low_dim_obs = read_pkl(episode_dir / "low_dim_obs.pkl")._observations[frame_id]
    # gripper_pose = low_dim_obs.gripper_pose
    # gripper_open = low_dim_obs.gripper_open
    # joint_positions = low_dim_obs.joint_positions

    # visualize image with the following
    vis_rgb = []
    for cam_name in camera_names:
        rgb_hwc = (
            (np.asarray(observations[cam_name]['rgb'], dtype=np.float32) / 2.0 + 0.5) * 255.0
        ).clip(0, 255).astype(np.uint8)
        cam_j = cam_param_dict[cam_name]
        u, v, valid = _world_points_to_uv(
            means_gt, cam_j["extrinsics"], cam_j["intrinsics"])
        rgb_hwc = _draw_gs_axes_on_rgb_hwc(
            rgb_hwc, u, v, valid, axis_rgb_gt)
        vis_rgb.append(rgb_hwc)
    vis_depth = [observations[cam_name]['depth'][...,None].repeat(1,1,3) for cam_name in camera_names]
    vis_mask = [observations[cam_name]['mask'] for cam_name in camera_names]
    # concate images into three rows and save
    vis_rgb = np.concatenate(vis_rgb, axis=1)
    vis_depth = np.concatenate(vis_depth, axis=1)
    vis_depth = (vis_depth - vis_depth.min()) / (vis_depth.max() - vis_depth.min())
    vis_depth = vis_depth * 255
    vis_mask = np.concatenate(vis_mask, axis=1)
    vis_image = np.concatenate([vis_rgb, vis_depth, vis_mask], axis=0).astype(np.uint8)

    front_rgb = observations['front']['rgb'] / 2 + 0.5
    front_pcd = observations['front']['pcd']
    front_rgb = front_rgb.reshape(-1, 3)
    front_pcd = front_pcd.reshape(-1, 3)
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(front_pcd[:, 0], front_pcd[:, 1], front_pcd[:, 2], 
            c=front_rgb, s=20, alpha=0.3, label='Point Cloud')
    gs_dict = construct_pose_gs_dict(expert_pose_8)
    means3D = gs_dict['means3D'].numpy()
    rgb = gs_dict['rgb'].numpy()
    ax.scatter(
        means3D[:, 0],
        means3D[:, 1],
        means3D[:, 2], 
        c=rgb,
        s=20,
        alpha=0.3,
        label='Expert Trajectory')
    center_x, center_y, center_z = expert_traj.reshape(-1)[:3]
    zoom_range = 0.1
    stage = expert_info['stage']
    tip_cur_position = expert_info['debug_info']['tip_cur_position']
    target_position = expert_info['debug_info']['tar_position']
    t = expert_info['debug_info']['t']
    dist_tar_to_wp = np.linalg.norm(target_position - expert_traj.reshape(-1)[:3].numpy())
    dist_tip_to_tar = np.linalg.norm(tip_cur_position - target_position)
    ax.set_xlim(center_x - zoom_range, center_x + zoom_range)
    ax.set_ylim(center_y - zoom_range, center_y + zoom_range)
    ax.set_zlim(center_z - zoom_range, center_z + zoom_range)
    ax.set_title(f"Stage: {stage}, dist_tar_to_wp: {dist_tar_to_wp:.2f}, t: {t:.2f}")
    ax = fig.add_subplot(122)
    ax.imshow(front_rgb.reshape(256, 256, 3))
    ax.set_title(f"dist_tip_to_tar: {dist_tip_to_tar:.2f}")

    # convert fig into np.ndarray
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    # resize and concate data and vis_image vertically
    data = Image.fromarray(data)
    data = data.resize((vis_image.shape[1], int(vis_image.shape[0]/2)))
    vis_image = np.concatenate([vis_image, np.array(data)], axis=0)
    vis_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(vis_image).save(vis_dir / f"{frame_id}.png")
    return

def write_json(obj: dict, path: str) -> None:
    """
    Writes a dictionary to a JSON file.

    Args:
        obj (dict): The dictionary to write to the file.
        path (str): The file path where the JSON will be saved.

    Returns:
        None
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
    return

def encode_image_from_float_to_uint8(image: np.ndarray) -> np.ndarray:
    '''
    Args:
        image: Any Shape in range [0, 1]
    Returns:
        image: Same Shape in range [0, 255]
    '''
    assert image.min() >= 0.0 and image.max() <= 1.0
    return (image * 255.0).astype(np.uint8)

def decode_image_from_uin8_to_float(image: np.ndarray) -> np.ndarray:
    '''
    Args:
        image: Any Shape in range [0, 255]
    Returns:
        image: Same Shape in range [0, 1]
    '''
    assert image.min() >= 0 and image.max() <= 255
    image = image.astype(np.float32)
    return image / 255.0

def encode_depth_from_float_to_uint8(
    depth: np.ndarray,
    bool_metric: bool = False,
    near: float = None,
    far: float = None,
) -> np.ndarray:
    '''
    Args:
        depth (H, W): 
    Returns:
        depth (H, W, 3): Depth in RGB format uint8
    '''
    if not bool_metric:
        # depth 已经是 [0, 1] 的归一化深度
        assert depth.min() >= 0.0 and depth.max() <= 1.0
        depth_norm = depth
    else:
        # 将 metric 深度 [near, far] 线性归一化到 [0, 1]
        assert near is not None and far is not None
        depth_norm = (depth - near) / (far - near)
        # 理论上应当已经在 [0, 1]，数值边缘做一下 clamp 更安全
        depth_norm = np.clip(depth_norm, 0.0, 1.0)
    depth_rgb = float_array_to_rgb_image(depth_norm, scale_factor=DEPTH_SCALE)
    return np.array(depth_rgb).astype(np.uint8)


def decode_depth_from_uin8_to_float(
    depth_rgb: np.ndarray,
    bool_metric: bool = False,
    near: float = None,
    far: float = None,
    ) -> np.ndarray:
    '''
    Args:
        depth_rgb (H, W, 3): Depth in RGB format uint8
    Returns:
        depth (H, W): Depth in meters
    '''
    assert depth_rgb.min() >= 0 and depth_rgb.max() <= 255
    assert len(depth_rgb.shape) == 3 and depth_rgb.shape[2] == 3
    # 先把 RGB 编码恢复成归一化深度（对应 encode 里的 depth_norm）
    depth_norm = image_to_float_array(depth_rgb, scale_factor=DEPTH_SCALE)
    if not bool_metric:
        # 返回 [0, 1] 归一化深度
        return depth_norm
    # 将 [0, 1] 归一化深度还原回 metric 深度 [near, far]
    assert near is not None and far is not None
    depth_m = depth_norm * (far - near) + near
    return depth_m

def read_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_npz(obj, path: str) -> None:
    np.savez_compressed(path, **obj)
    return

def read_npz(path: str) -> dict:
    # 1. 使用 'with' 确保 np.load 的文件句柄会被自动关闭
    with np.load(path) as f:
        # 2. 【关键】显式读取数据到内存字典中 (Deep Copy)
        # 这样 rtn 就变成了纯粹的内存数据，不再依赖打开的文件
        rtn = {key: f[key] for key in f.files}
    
    # 到这里，文件已经被 with 语句自动关闭了，数据都在 rtn 里
    
    # 3. 现在可以安全地清理缓存了
    # 以只读模式获取文件描述符
    fd = os.open(path, os.O_RDONLY)
    # 告诉内核：这个文件我读完了，数据都在内存里了，磁盘缓存可以扔了
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
        
    return rtn

def write_text_on_image(image: np.ndarray, text: str, font_size: int = 20, color: tuple = (255, 0, 0)) -> np.ndarray:
    """
    Write text on an image using Pillow.

    Args:
        image (np.ndarray): Input image (H, W, 3), dtype: uint8.
        text (str): Text to write on the image.
        font_size (int, optional): Font size of the text. Default is 20.
        color (tuple, optional): Text color in RGB format. Default is (255, 0, 0) (red).

    Returns:
        np.ndarray: Image with text written on it.
    """
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # Load a default font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Define text position
    text_x, text_y = 10, 10

    # Add text to the image
    draw.text((text_x, text_y), text, fill=color, font=font)

    # Convert PIL Image back to numpy array
    return np.array(pil_image)

def sample_points(N: int, target_N: int) -> List[int]:
    """
    Uniformly samples target_N indices from range(N).
    If target_N >= N, sample with replacement.
    Otherwise, sample without replacement.
    """
    if target_N >= N:
        # Sample with replacement
        return np.random.choice(N, target_N, replace=True).tolist()
    else:
        # Sample without replacement
        return np.random.choice(N, target_N, replace=False).tolist()

class Logger:
    def __init__(self, log_dir: Path, bool_enable: bool = False):
        self._data = {}
        self.log_dir = Path(log_dir)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._t0_dict = {}
        self.bool_enable = bool_enable
        return
    
    def log_demo(self, obs, scene, task, action):
        from utils.utils_with_rlbench import update_obs
        if not self.bool_enable:
            return
        save_dir = Path(self.log_dir) / "demo"
        save_dir.mkdir(parents=True, exist_ok=True)
        variation = task.variation
        episode_index = task.real_episode_dir
        step_id = task.step_id - 1
        camera_names = task.cam_names
        camera_list = task.camera_list
        camera_mask_list = task.camera_mask_list
        descriptions = task.descriptions
        
        # get expert information
        expert_info_dict = get_expert_info(task)
        # update nerf cam
        update_obs(
            obs,
            camera_names=camera_names,
            cam_list=camera_list,
            cam_mask_list=camera_mask_list,
        )
        save_observations(
            obs,
            save_dir=save_dir/episode_index,
            frame_id=step_id,
            camera_names=camera_names,
            camera_list=camera_list,
            camera_mask_list=camera_mask_list,
        )
        if step_id == 0:
            self.low_dim_obs_demo = LowDimObsDemo()
        save_low_dim_obs(
            obs, 
            save_dir=save_dir/episode_index,
            low_dim_obs_demo=self.low_dim_obs_demo,
        )
        save_variation_number(
            variation_number=variation,
            save_dir=save_dir/episode_index,
        )
        save_variation_description(
            description=descriptions,
            save_dir=save_dir/episode_index,
        )
        save_target_state(
            target_state_list=task.var2target_state_list,
            save_dir=save_dir/episode_index,
        )
        save_expert_info(
            expert_info_dict,
            save_dir=save_dir/episode_index,
            frame_id=step_id,
        )
        if step_id % 5 == 0:
            visualize_demo_data(
                episode_dir=save_dir/episode_index,
                frame_id=step_id,
                vis_dir=save_dir/episode_index/"vis",
                camera_names=camera_names,
            )
        return

    def log_training_dataset_model_forward(
        self,
        input_dict,
        output_dict,
        step_id,
        model,
        rank=0,
    ):
        if not self.bool_enable:
            return
        save_dir = Path(self.log_dir) / "training_dataset_model_forward"
        save_dir.mkdir(parents=True, exist_ok=True)
        write_pkl(input_dict, save_dir / f"input_dict_step{step_id}_rank{rank}.pkl")
        write_pkl(output_dict, save_dir / f"output_dict_step{step_id}_rank{rank}.pkl")
        torch.save({
            "weight": model.state_dict(),
            "iter": step_id + 1,
        }, save_dir / f"weight_state_dict_batchedstep{step_id}_rank{rank}.pth")
        return

    def log_evaluation_dataset_model_forward(
        self,
        input_dict,
        output_dict,
        eval_step,
        rank=0,
    ):
        if not self.bool_enable:
            return
        save_dir = Path(self.log_dir) / "evaluation_dataset_model_forward"
        save_dir.mkdir(parents=True, exist_ok=True)
        write_pkl(input_dict, save_dir / f"input_dict_batchedstep{eval_step}_rank{rank}.pkl")
        write_pkl(output_dict, save_dir / f"output_dict_batchedstep{eval_step}_rank{rank}.pkl")
        return
    
    def log_evaluation_simulation_model_forward(
        self,
        input_dict,
        output_dict,
        step_id,
    ):
        if not self.bool_enable:
            return
        save_dir = Path(self.log_dir) / "evaluation_simulation_model_forward"
        save_dir.mkdir(parents=True, exist_ok=True)
        write_pkl(input_dict, save_dir / f"input_dict_step{step_id}.pkl")
        write_pkl(output_dict, save_dir / f"output_dict_step{step_id}.pkl")
        return
    
    def log_world_model_init(
        self,
        data_dict_cano: Dict,
        model: nn.Module,
    ) -> None:
        '''
        Args:
            data_dict_cano: Dict,
            model: nn.Module,
        '''
        if not self.bool_enable:
            return
        output_dict = model.evaluate(
            data_dict_cano,
            deform_tracker=None,
            bool_canonical=True,
            cam_names_eval=data_dict_cano['cam_names'],
            metric_logger=self,
        )
        save_dir = Path(self.log_dir) / "world_model_init"
        save_dir.mkdir(parents=True, exist_ok=True)
        write_pkl(data_dict_cano, save_dir / "data_dict_cano.pkl")
        model.save_ckpt(str(save_dir / "model.pth"))
        write_pkl(output_dict, save_dir / "evaluate_output_dict.pkl")
        self.log("psnr_cano", output_dict["psnr_list"])
        self.log("derr_cano", output_dict["depth_err_list"])
        self.save()
        return
    
    def log_world_model_update(
        self,
        data_dict: Dict,
        world_model,
        cur_t: float,
        num_future_frames: int,
        dt: float,
    ) -> None:
        if not self.bool_enable:
            return
        save_dir = Path(self.log_dir) / "world_model_update"
        save_dir.mkdir(parents=True, exist_ok=True)
        write_pkl(data_dict, save_dir / f"data_dict_cur_t{cur_t:.3f}.pkl")
        world_model._model.save_ckpt(str(save_dir / f"model_cur_t{cur_t:.3f}.pth"))

        num_cam = world_model.num_cam
        device = world_model.device
        deform_tracker = {
            "network_t": world_model._model.network_t.clone(),
            "dxyz": world_model._model.dxyz.clone(),
            "drot": world_model._model.drot.clone(),
            "dxyz_node": world_model._model.dxyz_node.clone(),
            "drot_node": world_model._model.drot_node.clone(),
            "eepose": world_model._model.eepose.clone(),
            "openness": world_model._model.openness.clone(),
        }
        image_list, depth_list, eepose_list = [], [], []
        for i in range(num_future_frames):
            data_dict_ = {
                "frame_ids": torch.ones(num_cam, device=device) * (cur_t + dt * (i+1)),
                "views": world_model._views_dict["views"],
                "viewmats": world_model._views_dict["viewmats"],
                "Ks": world_model._views_dict["Ks"],
                "image_height": world_model._views_dict["image_height"],
                "image_width": world_model._views_dict["image_width"],
                "dt": torch.ones(num_cam, device=device) * (dt),
                "cam_names": world_model._cam_names,
            }
            output_dict = world_model._model.evaluate(
                data_dict_,
                deform_tracker=deform_tracker,
                bool_canonical=False,
                cam_names_eval=world_model._cam_names,
                metric_logger=self,
            )
            deform_tracker.update({
                "network_t": output_dict["network_t"],
                "dxyz": output_dict["dxyz"],
                "drot": output_dict["drot"],
                "dxyz_node": output_dict["dxyz_node"],
                "drot_node": output_dict["drot_node"],
                "eepose": output_dict["eepose"],
                "openness": output_dict["openness"],
            })
            image_list.append(output_dict["image_list"])
            depth_list.append(output_dict["depth_list"])
            eepose_list.append(output_dict["eepose"])
        eval_dict = {
            "image_list": image_list,
            "depth_list": depth_list,
            "eepose_list": eepose_list,
        }
        write_pkl(eval_dict, save_dir / f"eval_dict_cur_t{cur_t:.3f}.pkl")
        return
    
    def log_world_model_predict(
        self,
        world_model,
        eval_dict: Dict,
        cur_t: float,
        num_future_frames: int,
        dt: float,
    ) -> None:
        if not self.bool_enable:
            return
        save_dir = Path(self.log_dir) / "world_model_predict"
        save_dir.mkdir(parents=True, exist_ok=True)
        world_model._model.save_ckpt(str(save_dir / "model.pth"))
        write_pkl(eval_dict, save_dir / f"eval_dict_cur_t{cur_t:.3f}.pkl")
        kwargs = {
            "cur_t": cur_t,
            "num_future_frames": num_future_frames,
            "dt": dt,
        }
        write_pkl(kwargs, save_dir / "kwargs.pkl")
        return
    
    def tik(self, name: str):
        if not self.bool_enable:
            return
        assert name not in self._t0_dict
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0_dict[name] = time.time()
        return
    
    def tok(self, name: str):
        if not self.bool_enable:
            return
        assert name in self._t0_dict
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        t0 = self._t0_dict[name]
        t = t1 - t0
        if f"time_{name}" not in self._data:
            self._data[f"time_{name}"] = []
        self._data[f"time_{name}"].append(t)
        del self._t0_dict[name]
        return

    def log(self, name: str, val: float):
        if name not in self._data:
            self._data[name] = []
        self._data[name].append(val)
        return

    def save(self, path: str=None):
        if path is None:
            path = Path(self.log_dir) / "metric_dict.json"
        write_json(self._data, path)
        return

def write_txt(path: Path, text: List[str]):
    with open(path, "w") as fid:
        for line in text:
            fid.write(line + "\n")
    return

def read_ply(path: Path):
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points), np.asarray(pcd.colors)

def fix_random_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def disable_deterministic_algorithms() -> None:
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)
    return

def log_git_status(log_file: Path) -> None:
    """
    Logs the current Git status, branch, latest commit, and full diff against HEAD to a log file.

    Args:
        log_file (Path): Path to the log file where the Git status will be written.

    Returns:
        None
    """
    try:
        # Ensure the log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, "w") as f:
            # Write the current branch name
            branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
            f.write(f"Branch: {branch_name}\n")

            # Write the latest commit hash and message
            latest_commit = subprocess.check_output(["git", "log", "-1", "--pretty=format:%H %s"], text=True).strip()
            f.write(f"Latest Commit: {latest_commit}\n")

            # Write the full diff against HEAD
            diff_process = subprocess.Popen(
                ["git", "diff", "HEAD"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = diff_process.communicate()

            f.write("Diff against HEAD:\n")
            if stdout:
                f.write(stdout)
            else:
                f.write("No changes\n")

            if stderr:
                print(f"Error while generating diff: {stderr}")

        print(f"Git status and diff logged to {log_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing Git command: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return

def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(str(path), pcd)
    return

def write_pkl(data: Dict, path: Path,):
    with open(path, "wb") as fid:
        pickle.dump(data, fid)
    return

def read_pkl(path: Path):
    with open(path, "rb") as fid:
        data: Instructions = pickle.load(fid)
    return data

def read_image(path: Path):
    return np.array(Image.open(path))

def write_image(image: np.ndarray, path: Path):
    Image.fromarray(image).save(path)
    return

def write_gif(
    image_list: List[np.ndarray],
    path: Path,
    duration_ms: int = 100,
):
    image_list = [Image.fromarray(image) for image in image_list]
    image_list[0].save(
        path,
        save_all=True,
        append_images=image_list[1:],
        duration=duration_ms,
        loop=0,
    )
    return


def write_webp(
    image_list: List[np.ndarray],
    path: Path,
    duration_ms: int = 100,
    loop: int = 0,
    lossless: bool = True,
    method: int = 6,
):
    """Animated WebP（真彩色/无损可选），避免 GIF 256 色调色板偏色。"""
    if not image_list:
        return
    frames = [Image.fromarray(image) for image in image_list]
    save_kw = dict(
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
        method=method,
    )
    if lossless:
        save_kw["lossless"] = True
    else:
        save_kw["lossless"] = False
        save_kw["quality"] = 90
    frames[0].save(path, **save_kw)
    return

def round_floats(o):
    if isinstance(o, float): return round(o, 2)
    if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [round_floats(x) for x in o]
    return o


def normalise_quat(x: torch.Tensor):
    return x / x.square().sum(dim=-1).sqrt().unsqueeze(-1)


def get_gripper_loc_bounds(path: str, buffer: float = 0.0, task: Optional[str] = None):
    gripper_loc_bounds = json.load(open(path, "r"))
    if task is not None and task in gripper_loc_bounds:
        gripper_loc_bounds = gripper_loc_bounds[task]
        gripper_loc_bounds_min = np.array(gripper_loc_bounds[0]) - buffer
        gripper_loc_bounds_max = np.array(gripper_loc_bounds[1]) + buffer
        gripper_loc_bounds = np.stack([gripper_loc_bounds_min, gripper_loc_bounds_max])
    else:
        # Gripper workspace is the union of workspaces for all tasks
        gripper_loc_bounds = json.load(open(path, "r"))
        gripper_loc_bounds_min = np.min(np.stack([bounds[0] for bounds in gripper_loc_bounds.values()]), axis=0) - buffer
        gripper_loc_bounds_max = np.max(np.stack([bounds[1] for bounds in gripper_loc_bounds.values()]), axis=0) + buffer
        gripper_loc_bounds = np.stack([gripper_loc_bounds_min, gripper_loc_bounds_max])
    print("Gripper workspace size:", gripper_loc_bounds_max - gripper_loc_bounds_min)
    return gripper_loc_bounds


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def norm_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / torch.linalg.norm(tensor, ord=2, dim=-1, keepdim=True)


def load_instructions(
    instructions: Optional[Path],
    tasks: Optional[Sequence[str]] = None,
    variations: Optional[Sequence[int]] = None,
) -> Optional[Instructions]:
    if instructions is not None:
        with open(instructions, "rb") as fid:
            data: Instructions = pickle.load(fid)
        if tasks is not None:
            data = {task: var_instr for task, var_instr in data.items() if task in tasks}
        if variations is not None:
            data = {
                task: {
                    var: instr for var, instr in var_instr.items() if var in variations
                }
                for task, var_instr in data.items()
            }
        return data
    return None
