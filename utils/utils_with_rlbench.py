import os
import glob
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json

import open3d
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import einops

from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.task_environment import TaskEnvironment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from rlbench.demo import Demo
from rlbench.video_utils import CircleCameraMotion
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode
from pyrep.objects import VisionSensor, Dummy

import math
from pyrep.objects import VisionSensor
from typing import Optional, Dict
from scipy.spatial.transform import Rotation
from utils import common_utils
from utils.common_utils import Logger
import torch
from pathlib import Path
from diffuser_actor.world_model.freegave import FreeGave
from utils import common_utils, gs_utils
from tqdm import tqdm
from utils.common_utils import get_expert_info
from rlbench.action_modes.arm_action_modes import assert_action_shape, assert_unit_quaternion, calculate_delta_pose, ObjectType 
from pyrep.const import ConfigurationPathAlgorithms as Algos
from diffuser_actor.keypose_optimization.act3d import Act3D
from pyrep.backend import sim, utils

ALL_RLBENCH_TASKS = [
    'basketball_in_hoop', 'beat_the_buzz', 'change_channel', 'change_clock', 'close_box',
    'close_door', 'close_drawer', 'close_fridge', 'close_grill', 'close_jar', 'close_laptop_lid',
    'close_microwave', 'hang_frame_on_hanger', 'insert_onto_square_peg', 'insert_usb_in_computer',
    'lamp_off', 'lamp_on', 'lift_numbered_block', 'light_bulb_in', 'meat_off_grill', 'meat_on_grill',
    'move_hanger', 'open_box', 'open_door', 'open_drawer', 'open_fridge', 'open_grill',
    'open_microwave', 'open_oven', 'open_window', 'open_wine_bottle', 'phone_on_base',
    'pick_and_lift', 'pick_and_lift_small', 'pick_up_cup', 'place_cups', 'place_hanger_on_rack',
    'place_shape_in_shape_sorter', 'place_wine_at_rack_location', 'play_jenga',
    'plug_charger_in_power_supply', 'press_switch', 'push_button', 'push_buttons', 'put_books_on_bookshelf',
    'put_groceries_in_cupboard', 'put_item_in_drawer', 'put_knife_on_chopping_board', 'put_money_in_safe',
    'put_rubbish_in_bin', 'put_umbrella_in_umbrella_stand', 'reach_and_drag', 'reach_target',
    'scoop_with_spatula', 'screw_nail', 'setup_checkers', 'slide_block_to_color_target',
    'slide_block_to_target', 'slide_cabinet_open_and_place_cups', 'stack_blocks', 'stack_cups',
    'stack_wine', 'straighten_rope', 'sweep_to_dustpan', 'sweep_to_dustpan_of_size', 'take_frame_off_hanger',
    'take_lid_off_saucepan', 'take_money_out_safe', 'take_plate_off_colored_dish_rack', 'take_shoes_out_of_box',
    'take_toilet_roll_off_stand', 'take_umbrella_out_of_umbrella_stand', 'take_usb_out_of_computer',
    'toilet_seat_down', 'toilet_seat_up', 'tower3', 'turn_oven_on', 'turn_tap', 'tv_on', 'unplug_charger',
    'water_plants', 'wipe_desk'
]
TASK_TO_ID = {task: i for i, task in enumerate(ALL_RLBENCH_TASKS)}
ARM2Joint = {
    (46, 0, 0): -1, # base
    (45, 0, 0): 0,
    (44, 0, 0): 1,
    (43, 0, 0): 2,
    (42, 0, 0): 3,
    (41, 0, 0): 4, 
    (40, 0, 0): 5,
    (39, 0, 0): 6,
    (35, 0, 0): 6, 
    (31, 0, 0): 7, # left tip
    (34, 0, 0): 8, # right tip
}

Joint2ARM = {v: [] for k, v in ARM2Joint.items()}
for k, v in ARM2Joint.items():
    Joint2ARM[v].append(k)

def get_mask_with_obj_indices_ts(
        obj: torch.Tensor,
        indices: List[List[int]],
    ) -> torch.BoolTensor:
    '''
    Args:
        obj: torch.Tensor (B, N, 3) or (N, 3)
        indices: List[List[int]] (n_indices, 3)
    Return:
        mask: torch.BoolTensor (B, N) or (N)
    '''
    indices_tensor = torch.tensor(indices, dtype=obj.dtype, device=obj.device)  # (n_indices, 3)
    if obj.dim() == 3:  # (B, N, 3)
        obj_exp = obj.unsqueeze(2)  # (B, N, 1, 3)
        indices_exp = indices_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, n_indices, 3)
        match = (obj_exp == indices_exp).all(dim=-1)  # (B, N, n_indices)
        mask = match.any(dim=-1)  # (B, N)
    elif obj.dim() == 2:  # (N, 3)
        obj_exp = obj.unsqueeze(1)  # (N, 1, 3)
        indices_exp = indices_tensor.unsqueeze(0)  # (1, n_indices, 3)
        match = (obj_exp == indices_exp).all(dim=-1)  # (N, n_indices)
        mask = match.any(dim=-1)  # (N,)
    else:
        raise ValueError(f"obj must be of shape (B, N, 3) or (N, 3), but got {obj.shape}")
    return mask

def update_obs(obs, camera_names, cam_list, cam_mask_list):
    from datasets_module.dataset_engine import pointcloud_from_depth_and_camera_params_inplace
    for cam_name in camera_names:
        if not cam_name.isdigit():
            continue
        idx = [itm.get_name() for itm in cam_list].index(cam_name)
        cam_sensor = cam_list[idx]
        cam_sensor.handle_explicitly()
        cam_mask_sensor = cam_mask_list[idx]
        cam_mask_sensor.handle_explicitly()
        rgb = cam_sensor.capture_rgb() * 255
        extrinsics = cam_sensor.get_matrix()
        intrinsics = cam_sensor.get_intrinsic_matrix()
        near = cam_sensor.get_near_clipping_plane()
        far = cam_sensor.get_far_clipping_plane()
        depth = cam_sensor.capture_depth()
        depth_m = near + depth * (far - near)
        point_cloud = pointcloud_from_depth_and_camera_params_inplace(
            depth_m,
            extrinsics,
            intrinsics)
        mask = cam_mask_sensor.capture_rgb()
        setattr(obs, f'cam{cam_name}_point_cloud', point_cloud)
        setattr(obs, f'cam{cam_name}_rgb', rgb)
        setattr(obs, f'cam{cam_name}_depth', depth)
        setattr(obs, f'cam{cam_name}_mask', mask)
    
    # check
    for cam_name in camera_names:
        if not cam_name.isdigit():
            print(cam_name, 'rgb', getattr(obs, f'{cam_name}_rgb').min(), getattr(obs, f'{cam_name}_rgb').max())
            print(cam_name, 'depth', getattr(obs, f'{cam_name}_depth').min(), getattr(obs, f'{cam_name}_depth').max())
            print(cam_name, 'mask', getattr(obs, f'{cam_name}_mask').min(), getattr(obs, f'{cam_name}_mask').max())
        else:
            print(cam_name, 'rgb', getattr(obs, f'cam{cam_name}_rgb').min(), getattr(obs, f'cam{cam_name}_rgb').max())
            print(cam_name, 'depth', getattr(obs, f'cam{cam_name}_depth').min(), getattr(obs, f'cam{cam_name}_depth').max())
            print(cam_name, 'mask', getattr(obs, f'cam{cam_name}_mask').min(), getattr(obs, f'cam{cam_name}_mask').max())
    return

def init_multiple_cameras(
    cam_name_list: List,
    camera_resolution: Tuple[int, int],
):
    def _gen_pose_list(num_cam):
        assert num_cam < 63
        _cam_placeholder = Dummy('cam_cinematic_placeholder')
        _cam = VisionSensor.create(
            resolution=camera_resolution,
            explicit_handling=True,
            render_mode=RenderMode.OPENGL,
        )
        _cam.set_parent(_cam_placeholder)
        pose = VisionSensor('cam_front').get_pose()
        pose[2] += 0.1
        _cam.set_pose(pose)
        rotate_speed = 0.1
        _cam_motion = CircleCameraMotion(
            _cam,
            Dummy('cam_cinematic_base'),
            rotate_speed,
        )
        pose_list = []
        num_all = 63
        for _ in range(num_all):
            _cam_motion.step()
            _cam_motion.save_pose()
            _cam_motion.restore_pose()
            _pose = _cam.get_pose()
            pose_list.append(_pose)
        inter = num_all // num_cam
        pose_list = pose_list[0:len(pose_list):inter]
        return pose_list[:num_cam]

    cam_list, cam_mask_list = [], []
    max_num_cam = 40
    err_msg = f"Error: the default camera views is 0-{max_num_cam}"
    assert all([itm < max_num_cam for itm in cam_name_list]), err_msg
    pose_list = _gen_pose_list(num_cam=max_num_cam)
    pose_list = [pose_list[idx] for idx in cam_name_list]
    for i in range(len(pose_list)):
        cam_placeholder = Dummy('cam_cinematic_placeholder')
        cam = VisionSensor.create(
            resolution=camera_resolution,
            explicit_handling=True,
            render_mode=RenderMode.OPENGL,
        )
        cam.set_name(f'{cam_name_list[i]}')
        cam.set_pose(pose_list[i])
        cam.set_parent(cam_placeholder)
        cam_list.append(cam)

        cam_placeholder = Dummy('cam_cinematic_placeholder')
        cam_mask = VisionSensor.create(
            resolution=camera_resolution,
            explicit_handling=True,
            render_mode=RenderMode.OPENGL_COLOR_CODED,
        )
        cam_mask.set_pose(pose_list[i])
        cam_mask.set_parent(cam_placeholder)
        cam_mask_list.append(cam_mask)
    return cam_list, cam_mask_list

def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


def load_episodes() -> Dict[str, Any]:
    with open(Path(__file__).parent.parent / "data_preprocessing/episodes.json") as fid:
        return json.load(fid)


class Mover:

    def __init__(self, task, env):
        self._task = task
        self._env = env

    def __call__(self, action):
        action_collision = np.ones(action.shape[0]+1)
        action_collision[:-1] = action
        obs, reward, terminate = self._task.step(action_collision)
        return obs, reward, terminate, []


def get_cam_param(low_dim_obs_path: str, cam_name: str, frame_id: int = 0) -> Dict:
    data = common_utils.read_pkl(low_dim_obs_path)
    misc_dict = data._observations[frame_id].misc
    return {
        "intrinsics": misc_dict[f'{cam_name}_camera_intrinsics'],
        "extrinsics": misc_dict[f'{cam_name}_camera_extrinsics'], 
        "cam_name": cam_name,
        "near": misc_dict[f'{cam_name}_camera_near'],
        "far": misc_dict[f'{cam_name}_camera_far'],
    }

def get_cam_param_from_pkl(pose_path: str) -> Dict:
    view_dict = common_utils.read_pkl(pose_path)
    return {
        'intrinsics': view_dict['intrinsic'],
        'extrinsics': view_dict['extrinsic'],
        'near': view_dict['near'],
        'far': view_dict['far'],
    }

def extrinsic_rlbench_to_colmap(extrinsics):
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]

    R_inv = R.T
    R_flip = Rotation.from_euler('z', 180, degrees=True).as_matrix()
    R_inv = R_flip @ R_inv
    t_inv = -R_inv @ t

    return R_inv.T, t_inv


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def normalize_rgb(rgb):
    '''
    Args:
        rgb in range [-1,1]
    Return:
        rgb in range [0,1]
    '''
    return (rgb + 1) / 2

def convert_cam_param_to_gs_format(
        cam_param_dict: Dict,
        device: torch.device,
    ) -> Dict:
    cam_extrinsics = cam_param_dict['extrinsics']
    cam_intrinsics = cam_param_dict['intrinsics']
    znear = cam_param_dict['near']
    zfar = cam_param_dict['far']
    R, T = extrinsic_rlbench_to_colmap(cam_extrinsics)
    focal_length_x, focal_length_y = abs(cam_intrinsics[0,0]), abs(cam_intrinsics[1,1])
    height, width = 256, 256
    FoVy = focal2fov(focal_length_y, height)
    FoVx = focal2fov(focal_length_x, width)
    trans=np.array([0.0, 0.0, 0.0])
    scale=1.0
    world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
    projection_matrix = getProjectionMatrix(
        znear=znear,
        zfar=zfar, fovX=FoVx,
        fovY=FoVy).transpose(0, 1)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    
    # 计算内参 K（从 FoV 和分辨率恢复 fx, fy, cx, cy）
    fx = 0.5 * width / math.tan(FoVx * 0.5)
    fy = 0.5 * height / math.tan(FoVy * 0.5)
    cx = width * 0.5
    cy = height * 0.5
    K = torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    
    view_dict = {
        "FoVx": FoVx,
        "FoVy": FoVy,
        "image_height": height,
        "image_width": width,
        "world_view_transform": world_view_transform.to(device),
        "full_proj_transform": full_proj_transform.to(device),
        "camera_center": camera_center.to(device),
        "K": K,  # 内参矩阵 [3, 3]
        "R": R,
        "T": T,
        "znear": znear,
        "zfar": zfar,
        "rlbench_cam_param_dict": cam_param_dict,
    }
    return view_dict

def convert_depth_to_pcd(depth_m: torch.Tensor, view_dict: Dict) -> torch.Tensor:
    cam_param_dict = view_dict['rlbench_cam_param_dict']
    return VisionSensor.pointcloud_from_depth_and_camera_params(
        depth_m,
        cam_param_dict['extrinsics'],
        cam_param_dict['intrinsics'],
    )

class WorldModel:

    def __init__(
        self,
        logger: Logger,
        device: torch.device,
        episode_dir: Path,
        cam_names: List[str],
        # initialization
        num_init_gaussian: int = 25_000,
        max_iter_cano_gaussian: int = 3500,
        # update
        max_iter_gaussian: int = 10,
        max_iter_network: int = 10,
        num_future_frames: int = 30,
        bool_mask: bool = False,
        bool_wm_predict_velmappc: bool = False,
    ):
        '''
        Args:
            logger: (Logger)
            device: (torch.device)
            episode_dir: (Path)
            cam_names: (List[str])
            num_init_gaussian: (int)
            max_iter_cano_gaussian: (int)
            max_iter_gaussian: (int)
            max_iter_network: (int)
            num_future_frames: (int)
            bool_mask: (bool)
        '''
        self.num_future_frames = num_future_frames
        self.bool_mask = bool_mask
        self.bool_wm_predict_velmappc = bool_wm_predict_velmappc
        self.logger = logger
        self._model = None
        # initialization
        self.num_init_gaussian = num_init_gaussian
        self.max_iter_cano_gaussian = max_iter_cano_gaussian
        # update
        self.max_iter_gaussian = max_iter_gaussian
        self.max_iter_network = max_iter_network
        self.device = device
        self._cam_names = cam_names
        self._views_dict = self.get_cam_views(episode_dir, cam_names)

        assert not bool_mask, "Error: mask is not yet supported."

    def get_cam_views(self, episode_dir: Path, cam_names: List[str]) -> List[Dict]:
        '''
        Args:
            episode_dir: (Path) use frame_id 0 camera parameters
            cam_names: (List[str])
        Return:
            views: (List[Dict])
        '''
        frame_id = 0
        device = self.device
        views_dict = {}
        views = []
        for cam_name in cam_names:
            if cam_name in [
                "front", "left_shoulder", "right_shoulder", "overhead", "wrist"]:
                # fixed view camera
                low_dim_obs_path = Path(episode_dir) / "low_dim_obs.pkl"
                cam_param_dict = get_cam_param(str(low_dim_obs_path), cam_name, frame_id)
                view_dict = convert_cam_param_to_gs_format(cam_param_dict, device=device)
            elif cam_name.isdigit():
                # new view camera
                pose_path = Path(episode_dir) / f"nerf_data/{frame_id}/poses/{cam_name}.pkl"
                cam_param_dict = get_cam_param_from_pkl(str(pose_path))
                view_dict = convert_cam_param_to_gs_format(cam_param_dict, device=device)
            else:
                err_msg = f"Error: Cannot recognize cam_name={cam_name}"
                raise RuntimeError(err_msg)
            view_dict.update({"cam_name": cam_name})
            views.append(view_dict)
        views_dict['views'] = views
        # viewmats
        views_dict['viewmats'] = torch.stack([v["world_view_transform"].T for v in views], dim=0)  # (M, 4, 4)
        views_dict['Ks'] = torch.stack([v["K"] for v in views], dim=0)  # (M, 3, 3)
        views_dict['image_height'] = views[0]["image_height"]
        views_dict['image_width'] = views[0]["image_width"]
        return views_dict
    
    @property
    def num_cam(self) -> int:
        return len(self._views_dict["views"])

    def init(
        self,
        rgb: torch.Tensor,
        pcd: torch.Tensor,
        depth: torch.Tensor,
        gripper: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ) -> None:
        '''
        Args:
            rgb [1, num_cam, 3, H, W] (range [-1,1])
            pcd [1, num_cam, 3, H, W]
            depth [1,num_cam, 1, H, W] (metric in meters)
            gripper [1, 8]
            mask [1, num_cam, 3, H, W] (range [0-1]), default = None
        Note:
            - init gaussian from pcd
            - optimize gaussian with rgb, depth
            - rendering mask is not yet supported
        '''
        device = self.device
        num_init_gaussian = self.num_init_gaussian
        max_iter_cano_gaussian = self.max_iter_cano_gaussian
        gripper = gripper[0]
        gripper[3:7] = gs_utils.normalize_quaternion(gripper[3:7])
        data_dict_cano = {
            "images": normalize_rgb(rgb[0]),
            "pcds": pcd[0],
            "depths": depth[0],
            "masks": mask[0] * 255 if mask is not None else None,
            "views": self._views_dict["views"],
            "cam_names": self._cam_names,
            "frame_ids": torch.zeros(self.num_cam, device=device),
            "viewmats": self._views_dict["viewmats"],
            "Ks": self._views_dict["Ks"],
            "image_height": self._views_dict["image_height"],
            "image_width": self._views_dict["image_width"],
            "dt": torch.ones(self.num_cam, device=device) * 0.05,
            "eepose": gripper[:7],
            "openness": gripper[7:],
        }
        for k, v in data_dict_cano.items():
            if isinstance(v, torch.Tensor):
                data_dict_cano[k] = v.to(device)
        gs_dict = gs_utils.construct_init_gaussian(
            data_dict_cano, num_gaussian=num_init_gaussian
        )
        gaussians = gs_utils.train_canonical_gaussian(
            gs_dict,
            data_dict_cano,
            max_iter=max_iter_cano_gaussian,
            device=device,
            densify_grad_threshold=0.002,
            percent_dense=0.4,
            sh_degree=0,
            lambda_rigid=0.0,
            rigid_k=0,
        )
        cur_t = data_dict_cano["frame_ids"][0]
        model = FreeGave(
            cano_gaussians=gaussians,
            cano_t=cur_t,
            num_control_nodes=10_000,
            position_lr_init=0.00016,
            position_lr_final=0.0000016,
            position_lr_delay_mult=0.01,
            position_lr_max_steps=30_000,
            lambda_rigid=5.0,
            rigid_k=20,
            feature_lr=0.0025,
            opacity_lr=0.05,
            scaling_lr=0.001,
            rotation_lr=0.001,
            spatial_lr_scale=5,
            control_node_lr=0.001,
            percent_dense=0.01,
            lambda_dssim=0.2,
            lambda_depth=1.0,
        ).to(device)
        model.segment_gaussian(data_dict_cano)
        model.init_gaussian_weights()
        model.update_control_node()
        self._model = model
        self.warmup_model(model, data_dict_cano)
        # self.logger.log_world_model_init(
        #     data_dict_cano=data_dict_cano,
        #     model=model,
        # )
        return
    
    def warmup_model(self, model, data_dict_cano):
        # Warmup: 预热 CUDA kernels 和 cuDNN 算法选择
        # 避免第一帧因 JIT 编译和算法选择导致的延迟
        device = self.device
        num_samples = 3
        print("[Warmup] Running warmup iterations...")
        warmup_data = data_dict_cano
        cur_t = warmup_data["frame_ids"][0]
        warmup_network_steps = 3
        warmup_gaussian_steps = 2
        # 预热 network step (前几次会触发 JIT 编译和 cuDNN benchmark)
        for _ in range(warmup_network_steps):
            output_dict = model.train_network_one_step(warmup_data, num_samples=num_samples, idx_iter=_)
        # 预热 gaussian step
        for _ in range(warmup_gaussian_steps):
            _ = model.train_gaussian_one_step(
                warmup_data,
                dxyz_nograd=output_dict['dxyz'].detach(),
                drot_nograd=output_dict['drot'].detach(),
                num_samples=num_samples,
            )
        
        # 预热 evaluate
        output_dict = model.transform_gaussian_to(
            t=cur_t,
            dxyz=output_dict['dxyz'],
            drot=output_dict['drot'],
            dxyz_node=output_dict['dxyz_node'],
            drot_node=output_dict['drot_node'],
            eepose=warmup_data["eepose"],
            openness=warmup_data["openness"],
        )
        # eval current & future data
        deform_tracker = {
            "network_t": model.network_t.clone(),
            "dxyz": model.dxyz.clone(),
            "drot": model.drot.clone(),
            "dxyz_node": model.dxyz_node.clone(),
            "drot_node": model.drot_node.clone(),
            "eepose": model.eepose.clone(),
            "openness": model.openness.clone(),
        }
        # select next obs
        warmup_evaluate_steps = 10
        cam_names = self._cam_names
        cam_names_eval = self._cam_names
        dt = 0.05
        num_cam = len(cam_names)
        for i in range(warmup_evaluate_steps):
            data_dict_ = {
                "frame_ids": torch.ones(num_cam, device=device) * (cur_t + dt * (i+1)),
                "views": warmup_data["views"],
                "viewmats": warmup_data["viewmats"],
                "Ks": warmup_data["Ks"],
                "image_height": warmup_data["image_height"],
                "image_width": warmup_data["image_width"],
                "cam_names": cam_names_eval,
                "dt": torch.ones(num_cam, device=device) * (dt),
            }
            tar_t = data_dict_["frame_ids"][0]
            output_dict = model.evaluate(
                data_dict_,
                deform_tracker=deform_tracker,
                bool_canonical=False,
                cam_names_eval=cam_names_eval,
                metric_logger=None,
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
        # 再预热一次 train_network_one_step
        for _ in range(warmup_network_steps):
            output_dict = model.train_network_one_step(warmup_data, num_samples=num_samples, idx_iter=_)

        torch.cuda.synchronize()
        print("[Warmup] Done.")
        return

    def update(
        self,
        rgb: torch.Tensor,
        pcd: torch.Tensor,
        last_t: float,
        cur_t: float,
        depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        gripper: torch.Tensor = None,
        num_samples: int = 3,
    ) -> None:
        '''
        Args:
            rgb [num_cam, 3, H, W] (range [0-1])
            pcd [num_cam, 3, H, W]
            last_t: (float)
            cur_t: (float)
            depth [num_cam, 1, H, W] (metric in meters), default = None
            mask [num_cam, 3, H, W] (range [0-1]), default = None
            gripper [8,]
        '''
        assert self._model is not None, "Error: World model is not initialized."
        max_iter_network = self.max_iter_network
        max_iter_gaussian = self.max_iter_gaussian
        device = self.device
        assert len(gripper.shape) == 1
        assert gripper.shape[0] == 8
        gripper[3:7] = gs_utils.normalize_quaternion(gripper[3:7])
        data_dict = {
            "images": normalize_rgb(rgb[0]),
            "pcds": pcd[0],
            "depths": depth[0],
            "masks": mask[0] * 255 if mask is not None else None,
            "views": self._views_dict["views"],
            "frame_ids": torch.ones(self.num_cam, device=device) * cur_t,
            "cam_names": self._cam_names,
            "dt": torch.ones(self.num_cam, device=device) * (cur_t - last_t),
            "eepose": gripper[:7],
            "openness": gripper[7:],
        }
        # Stack viewmats and Ks from views for batched rendering
        data_dict['viewmats'] = self._views_dict["viewmats"]
        data_dict['Ks'] = self._views_dict["Ks"]
        data_dict['image_height'] = self._views_dict["image_height"]
        data_dict['image_width'] = self._views_dict["image_width"]
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.to(device)
        
        for idx_iter in range(max_iter_network):
            output_dict = self._model.train_network_one_step(
                data_dict,
                num_samples=num_samples,
                idx_iter=idx_iter,
            )
        for _ in range(max_iter_gaussian):
            gs_output_dict = self._model.train_gaussian_one_step(
                data_dict,
                dxyz_nograd=output_dict['dxyz'].detach(),
                drot_nograd=output_dict['drot'].detach(),
                num_samples=num_samples,
            )
        self._model.transform_gaussian_to(
                t=cur_t,
                dxyz=output_dict['dxyz'],
                drot=output_dict['drot'],
                dxyz_node=output_dict['dxyz_node'],
                drot_node=output_dict['drot_node'],
                eepose=data_dict['eepose'],
                openness=data_dict['openness'],
            )
        # self.logger.log_world_model_update(
        #     data_dict=data_dict,
        #     world_model=self,
        #     cur_t=cur_t,
        #     num_future_frames=self.num_future_frames,
        #     dt=cur_t - last_t,
        # )
        return
    
    def predict(
        self,
        num_future_frames: int,
        dt: float,
        ) -> Dict:
        if self.bool_wm_predict_velmappc:
            return self.predict_velmappc(dt)
        else:
            return self.predict_images(num_future_frames, dt)

    def predict_images(
        self,
        num_future_frames: int,
        dt: float,
        ) -> Dict:
        '''
        Args:
            num_future_frames: (int)
            dt: (float)
        Return:
            Dict[str, torch.Tensor]
                'image_list': [num_future_frames, num_cam, 3, H, W] (range [0-1])
                'depth_list': [num_future_frames, num_cam, 3, H, W]
                'mask_list': [num_future_frames, num_cam, 1, H, W]
        '''
        assert self._model is not None, "Error: World model is not initialized."
        num_cam = self.num_cam
        device = self.device
        cur_t = self._model.network_t.item()
        deform_tracker = {
            "network_t": self._model.network_t.clone(),
            "dxyz": self._model.dxyz.clone(),
            "drot": self._model.drot.clone(),
            "dxyz_node": self._model.dxyz_node.clone(),
            "drot_node": self._model.drot_node.clone(),
            "eepose": self._model.eepose.clone(),
            "openness": self._model.openness.clone(),
        }
        target_time_list, image_list, depth_list = [], [], []
        eepose_list, openness_list = [], []
        for i in range(num_future_frames):
            data_dict_ = {
                "frame_ids": torch.ones(num_cam, device=device) * (cur_t + dt * (i+1)),
                "views": self._views_dict["views"],
                "viewmats": self._views_dict["viewmats"],
                "Ks": self._views_dict["Ks"],
                "image_height": self._views_dict["image_height"],
                "image_width": self._views_dict["image_width"],
                "dt": torch.ones(num_cam, device=device) * (dt),
                "cam_names": self._cam_names,
            }
            output_dict = self._model.evaluate(
                data_dict_,
                deform_tracker=deform_tracker,
                bool_canonical=False,
                cam_names_eval=self._cam_names,
                metric_logger=self.logger,
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
            target_time_list.append(output_dict["network_t"])
            eepose_list.append(output_dict["eepose"])
            openness_list.append(output_dict["openness"])

        eval_dict = {
            "image_list": image_list,
            "depth_list": depth_list,
            "semantic_mask_list": None,
            "target_time_list": target_time_list,
            "eepose_list": eepose_list,
            "openness_list": openness_list,
        }
        # self.logger.log_world_model_predict(
        #     world_model=self,
        #     eval_dict=eval_dict,
        #     cur_t=cur_t,
        #     num_future_frames=num_future_frames,
        #     dt=dt,
        # )
        return eval_dict
    
    def predict_velmappc(self, dt:float) -> Dict:
        '''
        Args:
            num_future_frames: (int)
            dt: (float)
        Return:
            Dict[str, torch.Tensor]
                'image_list': [num_future_frames, num_cam, 3, H, W] (range [0-1])
                'depth_list': [num_future_frames, num_cam, 3, H, W]
                'mask_list': [num_future_frames, num_cam, 1, H, W]
        '''
        assert self._model is not None, "Error: World model is not initialized."
        num_cam = self.num_cam
        device = self.device
        cur_t = self._model.network_t.item()
        # eval current & future data
        deform_tracker = {
            "network_t": self._model.network_t.clone(),
            "dxyz": self._model.dxyz.clone(),
            "drot": self._model.drot.clone(),
            "dxyz_node": self._model.dxyz_node.clone(),
            "drot_node": self._model.drot_node.clone(),
            "eepose": self._model.eepose.clone(),
            "openness": self._model.openness.clone(),
        }
        velmappc_list = []
        data_dict_ = {
            "frame_ids": torch.ones(num_cam, device=device) * (cur_t),
            "views": self._views_dict["views"],
            "viewmats": self._views_dict["viewmats"],
            "Ks": self._views_dict["Ks"],
            "image_height": self._views_dict["image_height"],
            "image_width": self._views_dict["image_width"],
            "dt": torch.ones(num_cam, device=device) * (dt),
            "cam_names": self._cam_names,
        }
        output_dict = self._model.evaluate(
            data_dict_,
            deform_tracker=deform_tracker,
            bool_canonical=False,
            cam_names_eval=self._cam_names,
            metric_logger=self.logger,
        )
        velmappc_list.append(output_dict["velmap_pc"])

        eval_dict = {
            "velmappc_list": velmappc_list,
        }
        # self.logger.log_world_model_predict(
        #     world_model=self,
        #     eval_dict=eval_dict,
        #     cur_t=cur_t,
        #     num_future_frames=num_future_frames,
        #     dt=dt,
        # )
        return eval_dict


class Actioner:

    def __init__(
        self,
        policy=None,
        instructions=None,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
        action_dim=7,
        predict_trajectory=True
    ):
        self._policy = policy
        self._instructions = instructions
        self._apply_cameras = apply_cameras
        self._action_dim = action_dim
        self._predict_trajectory = predict_trajectory

        self._actions = {}
        self._instr = None
        self._task_str = None

        self._policy.eval()

    def load_episode(self, task_str, variation):
        self._task_str = task_str
        instructions = list(self._instructions[task_str][variation])
        self._instr = random.choice(instructions).unsqueeze(0)
        # self._task_id = torch.tensor(TASK_TO_ID[task_str]).unsqueeze(0)
        self._actions = {}

    def get_action_from_demo(self, demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :return: a list of obs and action
        """
        key_frame = keypoint_discovery(demo)

        action_ls = []
        trajectory_ls = []
        for i in range(len(key_frame)):
            obs = demo[key_frame[i]]
            action_np = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
            action = torch.from_numpy(action_np)
            action_ls.append(action.unsqueeze(0))

            trajectory_np = []
            for j in range(key_frame[i - 1] if i > 0 else 0, key_frame[i]):
                obs = demo[j]
                trajectory_np.append(np.concatenate([
                    obs.gripper_pose, [obs.gripper_open]
                ]))
            trajectory_ls.append(np.stack(trajectory_np))

        trajectory_mask_ls = [
            torch.zeros(1, key_frame[i] - (key_frame[i - 1] if i > 0 else 0)).bool()
            for i in range(len(key_frame))
        ]

        return action_ls, trajectory_ls, trajectory_mask_ls

    @torch.no_grad()
    def predict(self, rgbs, pcds, gripper,
                interpolation_length=None, step_id=None, save_dir=None, real_episode_dir=None, logger=None, **kwargs):
        """
        Args:
            rgbs: (bs, num_hist, num_cameras, 3, H, W)
            pcds: (bs, num_hist, num_cameras, 3, H, W)
            gripper: (B, nhist, output_dim)
            interpolation_length: an integer

        Returns:
            {"action": torch.Tensor, "trajectory": torch.Tensor}
        """
        output = {"action": None, "trajectory": None}

        rgbs = rgbs / 2 + 0.5  # in [0, 1]
        if 'next_rgb_obs' in kwargs:
            kwargs['next_rgb_obs'] = kwargs['next_rgb_obs'] / 2 + 0.5  # in [0, 1]

        if self._instr is None:
            raise ValueError()

        self._instr = self._instr.to(rgbs.device)
        # self._task_id = self._task_id.to(rgbs.device)

        # Predict trajectory
        if self._predict_trajectory:
            print('Predict Trajectory')
            fake_traj = torch.full(
                [1, interpolation_length - 1, gripper.shape[-1]], 0
            ).to(rgbs.device)
            traj_mask = torch.full(
                [1, interpolation_length - 1], False
            ).to(rgbs.device)
            if isinstance(self._policy, Act3D):
                output_dict = self._policy(
                    rgbs[:, -1],
                    pcds[:, -1],
                    self._instr,
                    gripper[..., 0, :7],
                    **kwargs,
                )
                output["trajectory"] = self._policy.prepare_action(output_dict)
                output["trajectory"] = output["trajectory"].unsqueeze(1)
            else:
                output_dict = self._policy(
                    fake_traj,
                    traj_mask,
                    rgbs[:, -1],
                    pcds[:, -1],
                    self._instr,
                    gripper[..., :7],
                    run_inference=True,
                    **kwargs,
                )
                output["trajectory"] = output_dict['action']
            if logger is not None:
                pass
                # logger.log_evaluation_simulation_model_forward(
                #     input_dict=dict(
                #         gt_trajectory=fake_traj,
                #         trajectory_mask=traj_mask,
                #         rgb_obs=rgbs[:, -1],
                #         pcd_obs=pcds[:, -1],
                #         instruction=self._instr,
                #         curr_gripper=gripper[..., :7],
                #         **kwargs,
                #     ),
                #     output_dict=dict(
                #         out=output_dict,
                #     ),
                #     step_id=step_id
                # )
        else:
            print('Predict Keypose')
            pred = self._policy(
                rgbs[:, -1],
                pcds[:, -1],
                self._instr,
                gripper[:, -1, :self._action_dim],
            )
            # Hackish, assume self._policy is an instance of Act3D
            output["action"] = self._policy.prepare_action(pred)

        return output

    @property
    def device(self):
        if self._policy.__class__.__name__ != 'PolicyClient':
            return next(self._policy.parameters()).device
        else:
            return torch.device('cuda:0')

def obs_to_attn_from_sensor(obs, camera_sensor):
    extrinsics_44 = torch.from_numpy(
        camera_sensor.get_matrix()
    ).float()
    extrinsics_44 = torch.linalg.inv(extrinsics_44)
    intrinsics_33 = torch.from_numpy(
        camera_sensor.get_intrinsic_matrix()
    ).float()
    intrinsics_34 = F.pad(intrinsics_33, (0, 1, 0, 0))
    gripper_pos_3 = torch.from_numpy(obs.gripper_pose[:3]).float()
    gripper_pos_41 = F.pad(gripper_pos_3, (0, 1), value=1).unsqueeze(1)
    points_cam_41 = extrinsics_44 @ gripper_pos_41

    proj_31 = intrinsics_34 @ points_cam_41
    proj_3 = proj_31.float().squeeze(1)
    u = int((proj_3[0] / proj_3[2]).round())
    v = int((proj_3[1] / proj_3[2]).round())

    return u, v

def get_action_traj(path) -> np.ndarray:
    '''
    Args:
        path: ArmConfigurationPath
    Returns:
        np.ndarray, shape: (N, 7)
    '''
    def _set_joints(path, positions):
        [sim.simSetJointPosition(jh, p)  # type: ignore
         for jh, p in zip(path._arm._joint_handles, positions)]
        [j.set_joint_target_position(p)  # type: ignore
         for j, p in zip(path._arm.joints, positions)]
        return

    if len(path._path_points) <= 0:
        raise RuntimeError("Can't visualise a path with no points.")
    tip = path._arm.get_tip()
    init_angles = path._arm.get_joint_positions()
    action_traj = []
    joint_traj = []
    is_model = path._arm.is_model()
    if not is_model:
        path._arm.set_model(True)
    prior = sim.simGetModelProperty(path._arm.get_handle())
    p = prior | sim.sim_modelproperty_not_dynamic
    # Disable the dynamics
    sim.simSetModelProperty(path._arm._handle, p)
    with utils.step_lock:
        sim.simExtStep(True)  # Have to step for changes to take effect
    _set_joints(path, path._path_points[0: len(path._arm.joints)])
    for i in range(len(path._arm.joints), len(path._path_points),
                    len(path._arm.joints)):
        points = path._path_points[i:i + len(path._arm.joints)]
        _set_joints(path, points)
        p = list(tip.get_pose()) # x,y,z,qx,qy,qz,qw
        action_traj.append(p)
        joint_traj.append(points)
    _set_joints(path, init_angles)
    with utils.step_lock:
        sim.simExtStep(True)  # Have to step for changes to take effect
    # Re-enable the dynamics
    sim.simSetModelProperty(path._arm._handle, prior)
    path._arm.set_model(is_model)
    return np.array(action_traj), np.array(joint_traj)

class DiscreteZeroStep(Discrete):

    def _actuate(self, scene, action):
        done = False
        while not done:
            done = scene.robot.gripper.actuate(action, velocity=0.2)
            scene.pyrep.step()
            # scene.task.step()

    def action(self, scene, action):
        assert_action_shape(action, self.action_shape(scene.robot))
        if 0.0 > action[0] > 1.0:
            raise InvalidActionError(
                'Gripper action expected to be within 0 and 1.')
        open_condition = all(
            x > 0.9 for x in scene.robot.gripper.get_open_amount())
        current_ee = 1.0 if open_condition else 0.0
        action = float(action[0] > 0.5)

        if current_ee != action:
            done = False
            if not self._detach_before_open:
                self._actuate(scene, action)
            if action == 0.0 and self._attach_grasped_objects:
                # If gripper close action, the check for grasp.
                for g_obj in scene.task.get_graspable_objects():
                    scene.robot.gripper.grasp(g_obj)
            else:
                # If gripper open action, the check for un-grasp.
                scene.robot.gripper.release()
            if self._detach_before_open:
                self._actuate(scene, action)
            if action == 1.0:
                # Step a few more times to allow objects to drop
                for _ in range(10):
                    scene.pyrep.step()
                    # scene.task.step()
class MoveArmThenGripperFixedSteps(MoveArmThenGripper):
    """The arm action is first applied, followed by the gripper action. """

    def action(self, scene, action: np.ndarray, N: int = 10):
        np.set_printoptions(precision=3, suppress=True)
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:arm_act_size+1])
        ignore_collisions = bool(action[arm_act_size+1:arm_act_size+2])
        # Move arm
        try:
            # Here N = N-1 is to rest 1 step for moving gripper
            after_arm_N = self.arm_action_mode.action(scene, arm_action, ignore_collisions, N=N-1)
        except:
            print(f"Warning: failed to move arm. N = {N}")
            after_arm_N = N-1
        print(f"after_arm_N = {after_arm_N}")
        # Check if task already succeeded after arm movement
        # If so, we don't need to execute resting steps or gripper action
        success, terminate = scene.task.success()
        if success:
            print(f"Task succeeded after arm movement, skipping resting steps and gripper action")
            return
        # Here range(N-1) is to reset 1 step for moving gripper
        for i in range(after_arm_N, 0, -1):
            # Move the rest steps
            scene.step()
            logger = scene.task.logger
            logger.log_demo(
                obs=scene.get_observation(),
                scene=scene,
                task=scene.task,
                action=action,
            )
            success, terminate = scene.task.success()
            # If the task succeeds while traversing path, then break early
            if success:
                break
            print(f"resting {i} steps")
        # Move gripper
        self.gripper_action_mode.action(scene, ee_action)
        scene.step()
        logger = scene.task.logger
        logger.log_demo(
            obs=scene.get_observation(),
            scene=scene,
            task=scene.task,
            action=action,
        )
        N = N - 1
        print(f"one step gripper")
        return

class EndEffectorPoseViaPlanningFixedSteps(EndEffectorPoseViaPlanning):
    def __init__(self, add_noise=False, bool_log_demo=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_noise = add_noise
        self.bool_log_demo = bool_log_demo

    def action(
        self,
        scene,
        action,
        ignore_collisions: bool = True,
        N: int = 10,
        min_dist: float = 0.01,
    ):
        def modify_path(path):
            # take 10 to 10+N action points
            cartesian_wp_array, joint_wp_array = get_action_traj(path)
            cartesian_wp_array = cartesian_wp_array[10:]
            joint_wp_array = joint_wp_array[10:]
            modified_cartesian_wp_array = []
            modified_joint_wp_array = []
            for i in range(len(cartesian_wp_array)):
                if i == 0:
                    modified_cartesian_wp_array.append(cartesian_wp_array[i])
                    modified_joint_wp_array.append(joint_wp_array[i])
                    continue
                dist = np.linalg.norm(
                    np.array(cartesian_wp_array[i][:3]) - np.array(modified_cartesian_wp_array[-1][:3]))
                if dist > min_dist:
                    if self.add_noise:
                        try:
                            ee_noise_std = 0.02 # 2 cm for x/y/z dim
                            noise = np.random.normal(0, ee_noise_std, 7)
                            cartesian_wp = cartesian_wp_array[i] + noise
                            joint_wp = scene.robot.arm.solve_ik_via_jacobian(cartesian_wp[:3], quaternion=cartesian_wp[3:])
                        except IKError:
                            print(f"Warning: Failed to add noise in execution. Using original cartesian_wp.")
                            cartesian_wp, joint_wp = cartesian_wp_array[i], joint_wp_array[i]
                    else:
                        cartesian_wp, joint_wp = cartesian_wp_array[i], joint_wp_array[i]
                    modified_cartesian_wp_array.append(cartesian_wp)
                    modified_joint_wp_array.append(joint_wp)
            cartesian_wp_array = np.array(modified_cartesian_wp_array)
            joint_wp_array = np.array(modified_joint_wp_array)
            if joint_wp_array.shape[0] < N:
                print(f"Warning: joint_wp_array.shape[0] < N, {joint_wp_array.shape[0]} < {N}")
            path._path_points = np.asarray(joint_wp_array.reshape(-1))
            return path

        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])
        if not self._absolute_mode and self._frame != 'end effector':
            action = calculate_delta_pose(scene.robot, action)
        relative_to = None if self._frame == 'world' else scene.robot.arm.get_tip()
        self._quick_boundary_check(scene, action)

        colliding_shapes = []
        if not ignore_collisions:
            if self._robot_shapes is None:
                self._robot_shapes = scene.robot.arm.get_objects_in_tree(
                    object_type=ObjectType.SHAPE)
            # First check if we are colliding with anything
            colliding = scene.robot.arm.check_arm_collision()
            if colliding:
                # Disable collisions with the objects that we are colliding with
                grasped_objects = scene.robot.gripper.get_grasped_objects()
                colliding_shapes = [
                    s for s in scene.pyrep.get_objects_in_tree(
                        object_type = ObjectType.SHAPE) if (
                            s.is_collidable() and
                            s not in self._robot_shapes and
                            s not in grasped_objects and
                            scene.robot.arm.check_arm_collision(
                                s))]
                [s.set_collidable(False) for s in colliding_shapes]

        try:
            # try once with collision checking (if ignore_collisions is true)
            try:
                path = scene.robot.arm.get_path(
                    action[:3],
                    quaternion=action[3:],
                    ignore_collisions=ignore_collisions,
                    relative_to=relative_to,
                    trials=100,
                    max_configs=10,
                    max_time_ms=10,
                    trials_per_goal=5,
                    algorithm=Algos.RRTConnect
                )
            except ConfigurationPathError as e:
                if ignore_collisions:
                    raise InvalidActionError(
                        'A path could not be found. Most likely due to the target '
                        'being inaccessible or a collison was detected.') from e
                else:
                    # try once more with collision checking disabled
                    path = scene.robot.arm.get_path(
                        action[:3],
                        quaternion=action[3:],
                        ignore_collisions=True,
                        relative_to=relative_to,
                        trials=100,
                        max_configs=10,
                        max_time_ms=10,
                        trials_per_goal=5,
                        algorithm=Algos.RRTConnect
                    )
        except ConfigurationPathError as e:
            # raise InvalidActionError(
            #     'A path could not be found. Most likely due to the target '
            #     'being inaccessible or a collison was detected.') from e
            print('A path could not be found. Most likely due to the target being inaccessible or a collison was detected.')
            path = None
        if path is not None:
            path = modify_path(path)
        done = False
        while N > 0 and not done:
            if path is not None:
                done = path.step()
            else:
                done = False
            scene.step()
            if self._callable_each_step is not None:
                # Record observations
                self._callable_each_step(scene.get_observation())
            logger = scene.task.logger
            logger.log_demo(
                obs=scene.get_observation(),
                scene=scene,
                task=scene.task,
                action=action,
            )
            success, terminate = scene.task.success()
            # If the task succeeds while traversing path, then break early
            # IMPORTANT: We need to decrement N before break, because we've already executed a step
            # The return value should reflect the remaining steps after executing the current step
            N -= 1
            if success and self._callable_each_step is None:
                print(f"moving arm, success=True, remaining N = {N}")
                break
            print(f"moving arm, N = {N}")
        return N

class ExpertActioner(Actioner):

    @torch.no_grad()
    def predict(self, task,
                interpolation_length=None, step_id=None, save_dir=None, real_episode_dir=None, logger=None, **kwargs):
        """
        Args:
            rgbs: (bs, num_hist, num_cameras, 3, H, W)
            pcds: (bs, num_hist, num_cameras, 3, H, W)
            gripper: (B, nhist, output_dim)
            interpolation_length: an integer

        Returns:
            {"action": torch.Tensor, "trajectory": torch.Tensor}
        """
        print("Predicting expert action")
        task = task._task
        expert_info = get_expert_info(task)
        task.stage = expert_info['stage']
        return {
            "trajectory": expert_info['trajectory'],
            "stage": expert_info['stage'],
        }

def obs_to_attn(obs, camera):
    extrinsics_44 = torch.from_numpy(
        obs.misc[f"{camera}_camera_extrinsics"]
    ).float()
    extrinsics_44 = torch.linalg.inv(extrinsics_44)
    intrinsics_33 = torch.from_numpy(
        obs.misc[f"{camera}_camera_intrinsics"]
    ).float()
    intrinsics_34 = F.pad(intrinsics_33, (0, 1, 0, 0))
    gripper_pos_3 = torch.from_numpy(obs.gripper_pose[:3]).float()
    gripper_pos_41 = F.pad(gripper_pos_3, (0, 1), value=1).unsqueeze(1)
    points_cam_41 = extrinsics_44 @ gripper_pos_41

    proj_31 = intrinsics_34 @ points_cam_41
    proj_3 = proj_31.float().squeeze(1)
    u = int((proj_3[0] / proj_3[2]).round())
    v = int((proj_3[1] / proj_3[2]).round())

    return u, v

def update_actioner_kwargs_init_frames(
    rgb,
    pcd,
    gripper,
    num_future_obs,
    num_future_downsample_factor,
    bool_wm_predict_velmappc,
) -> Dict:
    if not bool_wm_predict_velmappc:
        raise NotImplementedError
        # B,ncam,3+1,h,w
        # (1,T,ncam,c,h,w)
        actioner_kwargs = {
            # range -1, 1
            "next_rgb_obs": rgb[:,:,:3].unsqueeze(1)\
                .repeat(1, num_future_obs, 1, 1, 1, 1),
            "next_pcd_obs": pcd.unsqueeze(1)\
                .repeat(1, num_future_obs, 1, 1, 1, 1),
            "next_mask_obs": None,
            "next_gripper": gripper.unsqueeze(1)\
                .repeat(1, num_future_obs, 1),
        }
        actioner_kwargs['next_mask_obs'] = torch.ones_like(
            actioner_kwargs['next_rgb_obs'][:, :, :, :1]
        )
        actioner_kwargs["next_frame_relative_id"] = torch.tensor(
            [
                (i+1)*num_future_downsample_factor
                for i in range(actioner_kwargs['next_rgb_obs'].shape[1])
            ]
        )[None]
    else:
        num_samples = 30000
        pcd = einops.rearrange(pcd, "b n c h w -> (b n h w) c")
        num_points = pcd.shape[0]
        if num_points > num_samples:
            # Random permutation and take first num_samples
            indices = torch.randperm(num_points)[:num_samples]
            sampled_pcd = pcd[indices]
        else:
            # Sample with replacement
            indices = torch.randint(0, num_points, (num_samples,))
            sampled_pcd = pcd[indices]
        # Pad to 9 dimensions (xyz + 6 zeros for velocity)
        velmap_pc = torch.zeros((num_samples, 9), dtype=pcd.dtype)
        velmap_pc[:, :3] = sampled_pcd
        actioner_kwargs = {
            'velmap_pc': velmap_pc.unsqueeze(0)
        }
    return actioner_kwargs
    
def update_actioner_kwargs_subsequent_frames(
        wm_output,
        world_model,
        num_future_downsample_factor,
        dt,
        num_future_obs,
        bool_wm_predict_velmappc,
    ) -> Dict:
    if not bool_wm_predict_velmappc:
        raise NotImplementedError
        next_rgb, next_pcd, next_mask, next_gripper = select_next_obs(
            wm_output=wm_output,
            views=world_model._views_dict["views"],
            cur_t=world_model.cur_t,
            num_future_downsample_factor=num_future_downsample_factor,
            dt=dt,
            num_future_obs=num_future_obs,
        )
        actioner_kwargs = {
            "next_rgb_obs": next_rgb[None], # [1, num_future_frames, ncam, 3, H, W]
            "next_pcd_obs": next_pcd[None], # [1, num_future_frames, ncam, 3, H, W]
            "next_mask_obs": next_mask, # Nones
            "next_gripper": next_gripper[None], # [1, num_future_frames, 8]
        }
        actioner_kwargs['next_mask_obs'] = torch.ones_like(
            actioner_kwargs['next_rgb_obs'][:, :, :, :1]
        )
        actioner_kwargs["next_frame_relative_id"] = torch.tensor(
            [
                (i+1)*num_future_downsample_factor
                for i in range(actioner_kwargs['next_rgb_obs'].shape[1])
            ]
        )[None]
    else:
        actioner_kwargs = {
            'velmap_pc': torch.stack(wm_output['velmappc_list']),
        }
        return actioner_kwargs

def select_next_obs(
    wm_output: Dict,
    views: List[Dict],
    cur_t: float,
    num_future_downsample_factor: int,
    dt: float,
    num_future_obs,
):
    '''
    Args:
        wm_output: dict
        views: List[Dict]
        cur_t: float, e.g. 0.50
        num_future_downsample_factor: int e.g. 10
        dt: float, e,g. 0.05
        num_future_obs: int, e.g. 3
    Return:
        next_rgb: torch.Tensor [num_future_obs, ncam, 3, H, W] (range [-1, 1])
        next_pcd: torch.Tensor [num_future_obs, ncam, 3, H, W]
        next_mask: None
        next_gripper: torch.Tensor [num_future_obs, 8]
    '''
    # select next obs
    rel_timesteps = torch.tensor(wm_output['target_time_list']) - cur_t
    select_rel_timesteps = torch.tensor([(itm+1)* num_future_downsample_factor * dt for itm in range(num_future_obs)])
    diff = torch.abs(rel_timesteps.unsqueeze(0) - select_rel_timesteps.unsqueeze(1)) 
    select_indices = torch.argmin(diff, dim=1) 
    selected_diffs = diff[torch.arange(diff.shape[0]), select_indices]
    err_msg = "Error: cannot find suitable next obs according to rel_timesteps"
    assert not torch.any(selected_diffs > 1e-3), err_msg
    next_rgb = [wm_output['image_list'][itm] for itm in select_indices]
    next_depth = [wm_output['depth_list'][itm] for itm in select_indices]
    next_pcd = []
    next_gripper = [
        torch.cat([wm_output['eepose_list'][itm], wm_output['openness_list'][itm]], dim=0)
        for itm in select_indices]
    for itm in next_depth:
        next_pcd_multi_cam = []
        for itm_, view_dict in zip(itm, views):
            # itm_ [1, 1, H, W] -> depth_m [H,W]
            next_pcd_single_cam = convert_depth_to_pcd(
                depth_m=itm_[0,0].cpu().numpy(),
                view_dict=view_dict,
            )
            next_pcd_single_cam = einops.rearrange(
                next_pcd_single_cam,
                "h w c -> 1 1 c h w",
            )
            # next_pcd_single_cam [1, 1, 3, H, W]
            next_pcd_multi_cam.append(torch.from_numpy(next_pcd_single_cam))
        next_pcd.append(next_pcd_multi_cam)
    # next_rgb: List[List[torch.Tensor]] : (1,1,3,h,w) -> torch.Tensor (b, ncam, 3 h w)
    next_rgb_tensor = torch.stack(
        [torch.stack(itm, dim=0) for itm in next_rgb],
        dim=0)
    next_rgb_tensor = next_rgb_tensor.squeeze(2).squeeze(2)
    next_rgb_tensor = next_rgb_tensor * 2 - 1 # in [-1, 1]
    next_pcd_tensor = torch.stack(
        [torch.stack(itm, dim=0) for itm in next_pcd],
        dim=0)
    next_pcd_tensor = next_pcd_tensor.squeeze(2).squeeze(2)
    next_mask_tensor = None
    next_gripper_tensor = torch.stack(next_gripper, dim=0)
    return next_rgb_tensor, next_pcd_tensor, next_mask_tensor, next_gripper_tensor
class RLBenchEnv:

    def __init__(
        self,
        data_path,
        image_size=(128, 128),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        headless=False,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
        fine_sampling_ball_diameter=None,
        collision_checking=False,
        bool_expert_actioner=False,
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_cameras = apply_cameras
        self.apply_mask = True
        self.fine_sampling_ball_diameter = fine_sampling_ball_diameter
        self.image_size = image_size
        # setup RLBench environments
        self.obs_config = self.create_obs_config(
            image_size, apply_rgb, apply_depth, apply_pc, apply_cameras
        )

        self.action_mode = MoveArmThenGripperFixedSteps(
            arm_action_mode=EndEffectorPoseViaPlanningFixedSteps(
                collision_checking=collision_checking,
                add_noise=False,
                bool_log_demo=True),
            gripper_action_mode=DiscreteZeroStep()
        )
        self.env = Environment(
            self.action_mode, str(data_path), self.obs_config,
            headless=headless
        )
        self.image_size = image_size
        self.bool_expert_actioner = bool_expert_actioner

    def update_obs(self, obs):
        update_obs(obs, self.apply_cameras, self.cam_list, self.cam_mask_list)
        return

    def get_obs_action(self, obs):
        """
        Fetch the desired state and action based on the provided demo.
            :param obs: incoming obs
            :return: required observation and action list
        """

        # fetch state
        state_dict = {"rgb": [], "depth": [], "pc": [], "mask": []}
        for cam in self.apply_cameras:
            if cam in ['front', 'left_shoulder', 'right_shoulder', 'wrist', 'overhead']:
                if self.apply_rgb:
                    rgb = getattr(obs, "{}_rgb".format(cam))
                    state_dict["rgb"] += [rgb]

                if self.apply_depth:
                    depth = getattr(obs, "{}_depth".format(cam))
                    extrinsics = obs.misc[f"{cam}_camera_extrinsics"]
                    intrinsics = obs.misc[f"{cam}_camera_intrinsics"]
                    near = obs.misc[f"{cam}_camera_near"]
                    far = obs.misc[f"{cam}_camera_far"]
                    depth_m = near + depth * (far - near)
                    state_dict["depth"] += [depth_m[..., None].repeat(3, axis=2)]

                if self.apply_pc:
                    pc = getattr(obs, "{}_point_cloud".format(cam))
                    state_dict["pc"] += [pc]
                
                if self.apply_mask:
                    mask = getattr(obs, "{}_mask".format(cam))
                    state_dict["mask"] += [mask]
            if cam.isdigit():
                idx = [itm.get_name() for itm in self.cam_list].index(cam)
                cam_sensor = self.cam_list[idx]
                if self.apply_rgb:
                    # rgb = cam_sensor.capture_rgb() * 255
                    rgb = getattr(obs, "cam{}_rgb".format(cam))
                    state_dict["rgb"] += [rgb]

                if self.apply_depth:
                    depth = getattr(obs, "cam{}_depth".format(cam))
                    # depth = cam_sensor.capture_depth()
                    extrinsics = cam_sensor.get_matrix()
                    intrinsics = cam_sensor.get_intrinsic_matrix()
                    near = cam_sensor.get_near_clipping_plane()
                    far = cam_sensor.get_far_clipping_plane()
                    depth_m = near + depth * (far - near)
                    state_dict["depth"] += [depth_m[..., None].repeat(3, axis=2)]

                if self.apply_pc:
                    depth = getattr(obs, "cam{}_depth".format(cam))
                    # depth = cam_sensor.capture_depth()
                    extrinsics = cam_sensor.get_matrix()
                    intrinsics = cam_sensor.get_intrinsic_matrix()
                    near = cam_sensor.get_near_clipping_plane()
                    far = cam_sensor.get_far_clipping_plane()
                    depth_m = near + depth * (far - near)
                    pc = VisionSensor.pointcloud_from_depth_and_camera_params(depth_m, extrinsics, intrinsics)
                    state_dict["pc"] += [pc]
                if self.apply_mask:
                    mask = getattr(obs, "cam{}_mask".format(cam))
                    state_dict["mask"] += [mask]

        # fetch action
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        return state_dict, torch.from_numpy(action).float()

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        state_dict, gripper = self.get_obs_action(obs)
        state = transform(state_dict, augmentation=False)
        state = einops.rearrange(
            state,
            "(m n ch) h w -> n m ch h w",
            ch=3,
            n=len(self.apply_cameras),
            m=4
        )
        rgb = state[:, 0].unsqueeze(0)  # 1, N, C, H, W
        depth = state[:, 1].unsqueeze(0)  # 1, N, C, H, W
        depth = depth[:,:,:1]
        pcd = state[:, 2].unsqueeze(0)  # 1, N, C, H, W
        mask = state[:, 3].unsqueeze(0)  # 1, N, C, H, W
        gripper = gripper.unsqueeze(0)  # 1, D

        attns = torch.Tensor([])
        for cam in self.apply_cameras:
            if cam in ['front', 'left_shoulder', 'right_shoulder', 'wrist', 'overhead']:
                u, v = obs_to_attn(obs, cam)
            elif cam.isdigit():
                idx = [itm.get_name() for itm in self.cam_list].index(cam)
                cam_sensor = self.cam_list[idx]
                u, v = obs_to_attn_from_sensor(obs, cam_sensor)
            else:
                raise ValueError(f"Invalid camera name: {cam}")
            attn = torch.zeros(1, 1, 1, self.image_size[0], self.image_size[1])
            if not (u < 0 or u > self.image_size[1] - 1 or v < 0 or v > self.image_size[0] - 1):
                attn[0, 0, 0, v, u] = 1
            attns = torch.cat([attns, attn], 1)
        rgb = torch.cat([rgb, attns], 2)
        return rgb, pcd, depth, gripper, mask

    def get_obs_action_from_demo(self, demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :param normalise_rgb: normalise rgb to (-1, 1)
            :return: a list of obs and action
        """
        key_frame = keypoint_discovery(demo)
        key_frame.insert(0, 0)
        state_ls = []
        action_ls = []
        for f in key_frame:
            state, action = self.get_obs_action(demo._observations[f])
            state = transform(state, augmentation=False)
            state_ls.append(state.unsqueeze(0))
            action_ls.append(action.unsqueeze(0))
        return state_ls, action_ls

    def get_gripper_matrix_from_action(self, action):
        action = action.cpu().numpy()
        position = action[:3]
        quaternion = action[3:7]
        rotation = open3d.geometry.get_rotation_matrix_from_quaternion(
            np.array((quaternion[3], quaternion[0], quaternion[1], quaternion[2]))
        )
        gripper_matrix = np.eye(4)
        gripper_matrix[:3, :3] = rotation
        gripper_matrix[:3, 3] = position
        return gripper_matrix

    def get_demo(self, task_name, variation, episode_index):
        """
        Fetch a demo from the saved environment.
            :param task_name: fetch task name
            :param variation: fetch variation id
            :param episode_index: fetch episode index: 0 ~ 99
            :return: desired demo
        """
        demos = self.env.get_demos(
            task_name=task_name,
            variation_number=variation,
            amount=1,
            from_episode_number=episode_index,
            random_selection=False
        )
        return demos

    def evaluate_task_on_multiple_variations(
        self,
        task_str: str,
        max_steps: int,
        num_variations: int,  # -1 means all variations
        num_demos: int,
        actioner: Actioner,
        max_tries: int = 1,
        verbose: bool = False,
        dense_interpolation=False,
        interpolation_length=100,
        num_history=1,
        num_future_frames=10,
        vis_save_dir: str = None,
        bool_future_info: bool = False,
        bool_use_ground_truth_action: bool = False,
        bool_world_server: bool = False,
        world_server_port: int = 8766,
    ):
        self.env.launch()

        camera_resolution = self.image_size
        cam_name_list = [0, 8, 16, 24, 32, 36]
        self.cam_list, self.cam_mask_list = init_multiple_cameras(cam_name_list, camera_resolution)

        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task_variations = task.variation_count()

        if num_variations > 0:
            task_variations = np.minimum(num_variations, task_variations)
            task_variations = range(task_variations)
        else:
            task_variations = glob.glob(os.path.join(self.data_path, task_str, "variation*"))
            task_variations = [int(n.split('/')[-1].replace('variation', '')) for n in task_variations]

        var_success_rates = {}
        var_num_valid_demos = {}

        for variation in task_variations:
            task.set_variation(variation)
            success_rate, valid, num_valid_demos = (
                self._evaluate_task_on_one_variation(
                    task_str=task_str,
                    task=task,
                    max_steps=max_steps,
                    variation=variation,
                    num_demos=num_demos // len(task_variations) + 1,
                    actioner=actioner,
                    max_tries=max_tries,
                    verbose=verbose,
                    dense_interpolation=dense_interpolation,
                    interpolation_length=interpolation_length,
                    num_history=num_history,
                    num_future_frames=num_future_frames,
                    vis_save_dir=vis_save_dir,
                    bool_future_info=bool_future_info,
                    bool_use_ground_truth_action=bool_use_ground_truth_action,
                    bool_world_server=bool_world_server,
                    world_server_port=world_server_port,
                )
            )
            if valid:
                var_success_rates[variation] = success_rate
                var_num_valid_demos[variation] = num_valid_demos

        self.env.shutdown()

        var_success_rates["mean"] = (
            sum(var_success_rates.values()) /
            sum(var_num_valid_demos.values())
        )

        return var_success_rates

    def _evaluate_task_on_one_variation(
        self,
        task_str: str,
        task: TaskEnvironment,
        max_steps: int,
        variation: int,
        num_demos: int,
        actioner: Actioner,
        max_tries: int = 1,
        verbose: bool = False,
        dense_interpolation=False,
        interpolation_length=50,
        num_history=0,
        num_future_frames=10,
        vis_save_dir: Optional[str] = None,
        bool_future_info: bool = False,
        bool_use_ground_truth_action: bool = False,
        bool_world_server: bool = False,
        world_server_port: int = 8766,
    ):
        device = actioner.device

        success_rate = 0
        num_valid_demos = 0
        total_reward = 0
        bool_use_expert_action = isinstance(actioner, ExpertActioner)
        for demo_id in range(num_demos):
            print(f"Starting demo {demo_id}")
            if bool_use_expert_action:
                try:
                    task._task.init_task()
                    descriptions, obs = task.reset()
                except:
                    print(f"Encounter error in reset the environment, will skip this episode.")
                    continue
                self.update_obs(obs)
                num_valid_demos += 1
                real_episode_dir = Path(f'episode{demo_id}')
            else:
                episode_dir = Path(self.data_path) / task_str / f"variation{variation}" / "episodes" / f"episode{demo_id}"
                if not episode_dir.exists():
                    print(f"Episode directory {episode_dir} does not exist, skipping demo {demo_id}")
                    continue
                real_episode_dir = Path(os.readlink(episode_dir))
                demo = self.get_demo(task_str, variation, episode_index=demo_id)[0]
                if task_str == "push_moving_button_high_speed":
                    # handle the case that task._task.var2target_state_list is with len==1
                    var2target_state_list = common_utils.read_pkl(episode_dir / "target_state.pkl")
                    task._task.var2target_state_list = var2target_state_list
                descriptions, _ = task.reset_to_demo(demo)
                num_valid_demos += 1
                if task_str in [
                    "reach_single_moving_target_on_the_table",
                    "reach_single_moving_target_on_the_table_high_speed",
                    "push_moving_button",
                    "push_moving_button_high_speed",
                    "moving_basketball_in_hoop",
                    "moving_basketball_in_hoop_high_speed",
                    'pick_moving_target_on_the_table',
                    'pick_moving_target_on_the_table_high_speed',
                    'put_rubbish_in_moving_bin',
                    'put_rubbish_in_moving_bin_high_speed',
                ]:
                    var2target_state_list = common_utils.read_pkl(episode_dir / "target_state.pkl")
                    task._task.var2target_state_list = var2target_state_list
                    task._task.init_episode(variation)
                elif task_str in [
                    "place_cups_on_rotating_frame",
                    "place_cups_on_rotating_frame_high_speed",
                    "remove_cups_from_rotating_frame",
                    "remove_cups_from_rotating_frame_high_speed",
                    "beat_the_rotating_buzz",
                    "beat_the_rotating_buzz_high_speed",
                    "close_moving_box",
                    ]:
                    var2target_state_list = common_utils.read_pkl(episode_dir / "target_state.pkl")
                    target_state = var2target_state_list[variation]
                    task._task.target_state_list = [{
                        "yaw_speed": target_state["yaw_speed"],
                        "t0": 0,
                    }]
                    task._task.var2target_state_list = var2target_state_list
                    task._task.init_episode(variation)
                elif task_str in [
                    "insert_onto_rotating_peg",
                    "insert_onto_rotating_peg_high_speed",
                ]:
                    var2target_state_list = common_utils.read_pkl(episode_dir / "target_state.pkl")
                    target_state = var2target_state_list[variation]
                    task._task.target_state_list = [{
                        "yaw_speed": target_state["yaw_speed"],
                        "t0": 0,
                    }]
                    task._task.var2target_state_list = var2target_state_list
                    task._task.init_episode(variation, bool_random_place=False)
                else:
                    raise NotImplementedError
                task._scene.pyrep.step()
                obs = task.get_observation()
                self.update_obs(obs)
                task._task.disable_expert_plan()
                actioner.load_episode(task_str, variation) 
            reward = 0.0
            max_reward = 0.0
            rgbs = torch.Tensor([]).to(device)
            pcds = torch.Tensor([]).to(device)
            grippers = torch.Tensor([]).to(device)
            move = Mover(task, self)
            logger = Logger(
                log_dir=Path(vis_save_dir)/task_str/f"variation{variation}"/real_episode_dir.name,
                bool_enable=bool_use_expert_action,
            )
            if (Path(vis_save_dir)/task_str/f"variation{variation}"/real_episode_dir.name/'metric_dict.json').exists():
                print(f"Metric dict file already exists, will skip this episode.")
                continue
            if bool_future_info:
                dt = 0.05
                num_future_downsample_factor = 10
                num_future_obs = int(np.round(num_future_frames / num_future_downsample_factor))
                max_iter_network = 50
                max_iter_gaussian = 7
                num_init_gaussian = 25000
                max_iter_cano_gaussian = 3500
                # common_utils.fix_random_seed(seed=123)
                # device = torch.device("cuda:0")
                episode_dir = Path(self.data_path) / task_str / f"variation{variation}" / "episodes" / f"episode{demo_id}"
                bool_wm_predict_velmappc = False
                if actioner._policy.__class__.__name__ in [
                    'ForesightDiffuserActorV4',
                    'ForesightDiffuserActorV5',
                    'ForesightDiffuserActorV6']:
                    bool_wm_predict_velmappc = True
                elif actioner._policy.__class__.__name__ == 'PolicyClient' and \
                    actioner._policy._model_name in [
                        'foresight_diffuser_actor_v4',
                        'foresight_diffuser_actor_v5',
                        'foresight_diffuser_actor_v6',
                    ]:
                    bool_wm_predict_velmappc = True
                if bool_world_server:
                    from online_evaluation_rlbench.world_client import WorldModelClient
                    world_model = WorldModelClient(
                        host="127.0.0.1",
                        port=world_server_port,
                        logger=logger,
                        device=device,
                        episode_dir=episode_dir,
                        cam_names=self.apply_cameras,
                        num_init_gaussian=num_init_gaussian,
                        max_iter_cano_gaussian=max_iter_cano_gaussian,
                        max_iter_gaussian=max_iter_gaussian,
                        max_iter_network=max_iter_network,
                        num_future_frames=num_future_frames,
                        bool_mask=False,
                        bool_wm_predict_velmappc=bool_wm_predict_velmappc,
                    )
                else:
                    world_model = WorldModel(
                        logger=logger,
                        device=device,
                        episode_dir=episode_dir,
                        cam_names=self.apply_cameras,
                        num_init_gaussian=num_init_gaussian,
                        max_iter_cano_gaussian=max_iter_cano_gaussian,
                        max_iter_gaussian=max_iter_gaussian,
                        max_iter_network=max_iter_network,
                        num_future_frames=num_future_frames,
                        bool_mask=False,
                        bool_wm_predict_velmappc=bool_wm_predict_velmappc,
                    )
                world_model.last_t = 0
                world_model.cur_t = 0
                rgb, pcd, depth, gripper, mask = self.get_rgb_pcd_gripper_from_obs(obs)
                world_model.init(
                    rgb[:, :, :3, ...],
                    pcd,
                    depth,
                    gripper=gripper,
                    mask=mask,
                )
                def scene_call_back_func_wrapper(rlbench_env, world_model):
                    def scene_call_back_func():
                        if world_model is not None:
                            obs_ = task._scene.get_observation()
                            rlbench_env.update_obs(obs_)
                            rgb_, pcd_, depth_, gripper_, _ = rlbench_env.get_rgb_pcd_gripper_from_obs(obs_)
                            world_model.cur_t = world_model.cur_t + dt
                            # try:
                            world_model.update(
                                rgb=rgb_[:, :, :3],
                                pcd=pcd_,
                                last_t=world_model.last_t,
                                cur_t=world_model.cur_t,
                                depth=depth_,
                                mask=None,
                                gripper=gripper_[0].clone(),
                            )
                            # except:
                            #     reward = 0
                            #     terminate = 1
                            #     break
                            world_model.last_t = world_model.cur_t
                            return
                    return scene_call_back_func
                task._scene.register_step_callback(
                    scene_call_back_func_wrapper(self, world_model)
                )
            else:
                world_model = None
            for step_id in range(max_steps):
                rgb, pcd, depth, gripper, _ = self.get_rgb_pcd_gripper_from_obs(obs)
                if world_model is not None:
                    if step_id == 0:
                        actioner_kwargs = update_actioner_kwargs_init_frames(
                            rgb,
                            pcd,
                            gripper,
                            num_future_obs,
                            num_future_downsample_factor,
                            bool_wm_predict_velmappc = bool_wm_predict_velmappc,
                        )
                    else:
                        assert (world_model.cur_t - world_model._model.network_t.item()) < 1e-5
                        try:
                            wm_output = world_model.predict(
                                num_future_frames=num_future_frames,
                                dt=dt,
                            )
                        except:
                            print(f"Encounter error in world_model.predict at step {step_id}, will terminate this episode.")
                            terminate = True
                            break
                        actioner_kwargs = update_actioner_kwargs_subsequent_frames(
                            wm_output,
                            world_model,
                            num_future_downsample_factor,
                            dt=dt,
                            num_future_obs=num_future_obs,
                            bool_wm_predict_velmappc = bool_wm_predict_velmappc,
                        )
                    if 'next_gripper' in actioner_kwargs:
                        if actioner._policy.__class__.__name__ == 'PolicyClient':
                            if actioner._policy._model_name != 'foresight_diffuser_actor_v3':
                                actioner_kwargs.pop('next_gripper')
                        elif actioner._policy.__class__.__name__ != 'ForesightDiffuserActorV3':
                            actioner_kwargs.pop('next_gripper')
                else:
                    actioner_kwargs = {}
                for k, v in actioner_kwargs.items():
                    actioner_kwargs[k] = v.to(device)
                rgb = rgb.to(device)
                pcd = pcd.to(device)
                gripper = gripper.to(device)

                rgbs = torch.cat([rgbs, rgb.unsqueeze(1)], dim=1)
                pcds = torch.cat([pcds, pcd.unsqueeze(1)], dim=1)
                grippers = torch.cat([grippers, gripper.unsqueeze(1)], dim=1)

                # Prepare proprioception history
                rgbs_input = rgbs[:, -1:][:, :, :, :3]
                pcds_input = pcds[:, -1:]
                pcds_input[:,:,:,0] = torch.clamp(pcds_input[:,:,:,0], -2.5, 2.5)
                pcds_input[:,:,:,1] = torch.clamp(pcds_input[:,:,:,1], -2.5, 2.5)
                pcds_input[:,:,:,2] = torch.clamp(pcds_input[:,:,:,2], 0, 2)
                if num_history < 1:
                    gripper_input = grippers[:, -1]
                else:
                    gripper_input = grippers[:, -num_history:]
                    npad = num_history - gripper_input.shape[1]
                    gripper_input = F.pad(
                        gripper_input, (0, 0, npad, 0), mode='replicate'
                    )
                try:
                    if bool_use_expert_action:
                        output = actioner.predict(
                            task,
                            interpolation_length=interpolation_length,
                            step_id=step_id,
                            save_dir=Path(f'{vis_save_dir}/eval_sim/{task_str}_var{variation}_epi{demo_id}'),
                            real_episode_dir=real_episode_dir,
                            logger=logger,
                        )
                    else:
                        output = actioner.predict(
                            rgbs_input,
                            pcds_input,
                            gripper_input,
                            interpolation_length=interpolation_length,
                            step_id=step_id,
                            save_dir=Path(f'{vis_save_dir}/eval_sim/{task_str}_var{variation}_epi{demo_id}'),
                            real_episode_dir=real_episode_dir,
                            logger=logger,
                            **actioner_kwargs,
                        )
                except:
                    print(f"Encounter error in actioner.predict at step {step_id}, will terminate this episode.")
                    terminate = True
                    break
                print(f"Step {step_id}")
                # Update the observation based on the predicted action
                # Execute entire predicted trajectory step by step
                trajectory = output["trajectory"][-1].cpu().numpy()
                trajectory[:, -1] = trajectory[:, -1].round()
                # execute
                for action in tqdm(trajectory):
                    if bool_use_ground_truth_action:
                        idx_demo = int(step_id * 10)
                        expert_dir = Path(real_episode_dir) / "expert_info"
                        expert_pkl = expert_dir / f"{idx_demo}.pkl"
                        if expert_pkl.exists():
                            expert_info = common_utils.read_pkl(expert_pkl)
                        else:
                            # fallback: use the last pkl under expert_info/
                            pkls = sorted(expert_dir.glob("*.pkl"))
                            if len(pkls) == 0:
                                raise FileNotFoundError(f"No *.pkl found under {expert_dir}")
                            # prefer numeric filename ordering when possible
                            def _pkl_key(p: Path):
                                s = p.stem
                                return int(s) if s.isdigit() else s
                            last_pkl = sorted(pkls, key=_pkl_key)[-1]
                            expert_info = common_utils.read_pkl(last_pkl)
                        action = expert_info['trajectory'].reshape(-1)
                    task._task.logger = logger
                    task._task.variation = variation
                    task._task.real_episode_dir = real_episode_dir.name
                    task._task.cam_names = self.apply_cameras
                    task._task.camera_list = self.cam_list
                    task._task.camera_mask_list = self.cam_mask_list
                    task._task.descriptions = descriptions
                    try:
                        obs, reward, terminate, _ = move(action)
                    except InvalidActionError:
                        print(f"Encounter invalid action error at step {step_id}, will terminate this episode.")
                        terminate = True
                        break
                    self.update_obs(obs)
                    if reward == 1:
                        break
                # Update the observation based on the predicted action
                max_reward = max(max_reward, reward)

                if reward == 1:
                    success_rate += 1
                    if logger.bool_enable:
                        logger.log_demo(
                            obs=task._scene.get_observation(),
                            scene=task._scene,
                            task=task._task,
                            action=action,
                        )
                    break

                if terminate:
                    print("The episode has terminated!")
                    break


            total_reward += max_reward
            if reward == 0:
                step_id += 1

            print(
                task_str,
                "Variation",
                variation,
                "Demo",
                demo_id,
                "Reward",
                f"{reward:.2f}",
                "max_reward",
                f"{max_reward:.2f}",
                f"SR: {success_rate}/{demo_id+1}",
                f"SR: {total_reward:.2f}/{demo_id+1}",
                "# valid demos", num_valid_demos,
            )
            logger.log(f"{task_str}_variation{variation}_epi{demo_id}_reward", reward)
            logger.save()

        # Compensate for failed demos
        if num_valid_demos == 0:
            assert success_rate == 0
            valid = False
        else:
            valid = True

        return success_rate, valid, num_valid_demos

    def create_obs_config(
        self, image_size, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs
    ):
        """
        Set up observation config for RLBench environment.
            :param image_size: Image size.
            :param apply_rgb: Applying RGB as inputs.
            :param apply_depth: Applying Depth as inputs.
            :param apply_pc: Applying Point Cloud as inputs.
            :param apply_cameras: Desired cameras.
            :return: observation config
        """
        unused_cams = CameraConfig()
        unused_cams.set_all(False)
        used_cams = CameraConfig(
            rgb=apply_rgb,
            point_cloud=apply_pc,
            depth=apply_depth,
            mask=True,
            image_size=image_size,
            render_mode=RenderMode.OPENGL,
            depth_in_meters=False,
            **kwargs,
        )

        camera_names = apply_cameras
        kwargs = {}
        for n in camera_names:
            kwargs[n] = used_cams

        obs_config = ObservationConfig(
            front_camera=kwargs.get("front", unused_cams),
            left_shoulder_camera=kwargs.get("left_shoulder", unused_cams),
            right_shoulder_camera=kwargs.get("right_shoulder", unused_cams),
            wrist_camera=kwargs.get("wrist", unused_cams),
            overhead_camera=kwargs.get("overhead", unused_cams),
            joint_forces=False,
            joint_positions=True,
            joint_velocities=True,
            task_low_dim_state=False,
            gripper_touch_forces=False,
            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=True,
            gripper_joint_positions=True,
        )

        return obs_config


# Identify way-point in each RLBench Demo
def _is_stopped(demo, i, obs, stopped_buffer, delta):
    next_is_not_final = i == (len(demo) - 2)
    # gripper_state_no_change = i < (len(demo) - 2) and (
    #     obs.gripper_open == demo[i + 1].gripper_open
    #     and obs.gripper_open == demo[i - 1].gripper_open
    #     and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    # )
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[max(0, i - 1)].gripper_open
        and demo[max(0, i - 2)].gripper_open == demo[max(0, i - 1)].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )
    return stopped


def keypoint_discovery(demo: Demo, stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0

    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open

    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)

    return episode_keypoints


def transform(obs_dict, scale_size=(0.75, 1.25), augmentation=False):
    apply_depth = len(obs_dict.get("depth", [])) > 0
    apply_pc = len(obs_dict["pc"]) > 0
    apply_mask = len(obs_dict["mask"]) > 0
    num_cams = len(obs_dict["rgb"])

    obs_rgb = []
    obs_depth = []
    obs_pc = []
    obs_mask = []

    for i in range(num_cams):
        rgb = torch.tensor(obs_dict["rgb"][i]).float().permute(2, 0, 1)
        mask = torch.tensor(obs_dict["mask"][i]).float().permute(2, 0, 1) if apply_mask else None
        depth = (
            torch.tensor(obs_dict["depth"][i]).float().permute(2, 0, 1)
            if apply_depth
            else None
        )
        pc = (
            torch.tensor(obs_dict["pc"][i]).float().permute(2, 0, 1) if apply_pc else None
        )

        if augmentation:
            raise NotImplementedError()  # Deprecated

        # normalise to [-1, 1]
        rgb = rgb / 255.0
        rgb = 2 * (rgb - 0.5)

        obs_rgb += [rgb.float()]
        if depth is not None:
            obs_depth += [depth.float()]
        if pc is not None:
            obs_pc += [pc.float()]
        if mask is not None:
            obs_mask += [mask.float()]
    obs = obs_rgb + obs_depth + obs_pc + obs_mask
    return torch.cat(obs, dim=0)
