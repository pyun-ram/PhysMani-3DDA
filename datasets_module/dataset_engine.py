from collections import defaultdict, Counter
import itertools
import math
import random
from pathlib import Path
from time import time
from typing import List, Dict, Tuple
import h5py
import time
import torch
import einops
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets_module.utils import loader, Resize, TrajectoryInterpolator
from utils import common_utils
import re
from rlbench.backend.utils import DEFAULT_RGB_SCALE_FACTOR, DEFAULT_GRAY_SCALE_FACTOR
from rlbench.backend.const import DEPTH_SCALE

_UV_CACHE = {}

STAGE2IDX = {
    "reach": 0,
    "grasp": 1,
}
IDX2STAGE = {v: k for k, v in STAGE2IDX.items()}

def image_to_float_array_inplace(image, scale_factor=None):
    """
    Args:
        image: np.ndarray, shape (H, W, 3) or (H, W) [0-255] np.uint8
    Consistent with (have checked with 1k data frames)
        from rlbench.backend.utils import image_to_float_array
        depth = image_to_float_array(depth_uint8)
        depth = np.ascontiguousarray(depth, dtype=np.float32)
    """
    arr = np.asarray(image)  # 避免不必要拷贝
    if arr.ndim == 3 and arr.shape[2] == 3:
        if scale_factor is None:
            scale_factor = DEFAULT_RGB_SCALE_FACTOR
        # 避免大广播: 通道分开+位移组合，控制 dtype
        r = arr[..., 0].astype(np.uint32, copy=False)
        g = arr[..., 1].astype(np.uint32, copy=False)
        b = arr[..., 2].astype(np.uint32, copy=False)
        val = (r << 16) | (g << 8) | b
        out = val.astype(np.float32)  # 转 float 一次
    else:
        if scale_factor is None:
            scale_factor = DEFAULT_GRAY_SCALE_FACTOR[arr.dtype.type]
        out = arr.astype(np.float32, copy=False)
    out /= scale_factor  # 就地除法，避免再分配
    return out

def pointcloud_from_depth_and_camera_params_inplace(
        depth: np.ndarray, extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        cache_key: tuple = None) -> np.ndarray:
    """
    Memory-friendly conversion from depth (meters) to world point cloud.
    Avoids large concatenations/reshapes and computes in-place in float32.
    Consistent with (have checked with 1k data frames)
        from pyrep.objects import VisionSensor
        pcd = VisionSensor.pointcloud_from_depth_and_camera_params()
        pcd = np.ascontiguousarray(pcd, dtype=np.float32)
    """
    assert depth.ndim == 2, "depth must be HxW"
    H, W = depth.shape
    # Extract intrinsics
    if intrinsics.shape[0] >= 3 and intrinsics.shape[1] >= 3:
        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])
    else:
        raise ValueError(f'Invalid intrinsics shape: {intrinsics.shape}')
    # Normalized pixel grid (cached)
    key = cache_key if cache_key is not None else (H, W, fx, fy, cx, cy)
    if key in _UV_CACHE:
        u_norm, v_norm = _UV_CACHE[key]
    else:
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)  # (H, W)
        u_norm = (uu - cx) / fx
        v_norm = (vv - cy) / fy
        _UV_CACHE[key] = (u_norm, v_norm)
    # Camera coordinates
    depth = depth.astype(np.float32, copy=False)
    x = u_norm * depth
    y = v_norm * depth
    z = depth
    # Camera-to-world transform from extrinsics
    R = extrinsics[:3, :3].astype(np.float32, copy=False)
    t = extrinsics[:3, 3].astype(np.float32, copy=False)
    # Allocate output and fill in-place
    pcd = np.empty((H, W, 3), dtype=np.float32)
    pcd[..., 0] = R[0, 0] * x + R[0, 1] * y + R[0, 2] * z + t[0]
    pcd[..., 1] = R[1, 0] * x + R[1, 1] * y + R[1, 2] * z + t[1]
    pcd[..., 2] = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z + t[2]
    return pcd

class RLBenchDataset(Dataset):
    """RLBench dataset."""

    def __init__(
        self,
        # required
        root,
        instructions=None,
        # dataset specification
        taskvar=[('close_door', 0)],
        max_episode_length=5,
        cache_size=0,
        max_episodes_per_task=100,
        num_iters=None,
        cameras=("wrist", "left_shoulder", "right_shoulder"),
        # for augmentations
        training=True,
        image_rescale=(1.0, 1.0),
        # for trajectories
        return_low_lvl_trajectory=False,
        dense_interpolation=False,
        interpolation_length=100,
        relative_action=False
    ):
        self._cache = {}
        self._cache_size = cache_size
        self._cameras = cameras
        self._max_episode_length = max_episode_length
        self._num_iters = num_iters
        self._training = training
        self._taskvar = taskvar
        self._return_low_lvl_trajectory = return_low_lvl_trajectory
        if isinstance(root, (Path, str)):
            root = [Path(root)]
        self._root = [Path(r).expanduser() for r in root]
        self._relative_action = relative_action

        # For trajectory optimization, initialize interpolation tools
        if return_low_lvl_trajectory:
            assert dense_interpolation
            self._interpolate_traj = TrajectoryInterpolator(
                use=dense_interpolation,
                interpolation_length=interpolation_length
            )

        # Keep variations and useful instructions
        self._instructions = defaultdict(dict)
        self._num_vars = Counter()  # variations of the same task
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if data_dir.is_dir():
                if instructions is not None:
                    self._instructions[task][var] = instructions[task][var]
                self._num_vars[task] += 1

        # If training, initialize augmentation classes
        if self._training:
            self._resize = Resize(scales=image_rescale)

        # File-names of episodes per task and variation
        episodes_by_task = defaultdict(list)  # {task: [(task, var, filepath)]}
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if not data_dir.is_dir():
                print(f"Can't find dataset folder {data_dir}")
                continue
            npy_episodes = [(task, var, ep) for ep in data_dir.glob("*.npy")]
            dat_episodes = [(task, var, ep) for ep in data_dir.glob("*.dat")]
            pkl_episodes = [(task, var, ep) for ep in data_dir.glob("*.pkl")]
            episodes = npy_episodes + dat_episodes + pkl_episodes
            # Split episodes equally into task variations
            if max_episodes_per_task > -1:
                episodes = episodes[
                    :max_episodes_per_task // self._num_vars[task] + 1
                ]
            if len(episodes) == 0:
                print(f"Can't find episodes at folder {data_dir}")
                continue
            episodes_by_task[task] += episodes

        # Collect and trim all episodes in the dataset
        self._episodes = []
        self._num_episodes = 0
        for task, eps in episodes_by_task.items():
            if len(eps) > max_episodes_per_task and max_episodes_per_task > -1:
                eps = random.sample(eps, max_episodes_per_task)
            episodes_by_task[task] = sorted(
                eps, key=lambda t: int(str(t[2]).split('/')[-1][2:-4])
            )
            self._episodes += eps
            self._num_episodes += len(eps)
        print(f"Created dataset from {root} with {self._num_episodes}")
        self._episodes_by_task = episodes_by_task

    def read_from_cache(self, args):
        if self._cache_size == 0:
            return loader(args)

        if args in self._cache:
            return self._cache[args]

        value = loader(args)

        if len(self._cache) == self._cache_size:
            key = list(self._cache.keys())[int(time()) % self._cache_size]
            del self._cache[key]

        if len(self._cache) < self._cache_size:
            self._cache[args] = value

        return value

    @staticmethod
    def _unnormalize_rgb(rgb):
        # (from [-1, 1] to [0, 1]) to feed RGB to pre-trained backbone
        assert rgb.min() >= -1 and rgb.max() <= 1
        return rgb / 2 + 0.5

    def __getitem__(self, episode_id):
        """
        the episode item: [
            [frame_ids],  # we use chunk and max_episode_length to index it
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors],  # wrt frame_ids, (1, 8)
            [trajectories]  # wrt frame_ids, (N_i, 8)
        ]
        """
        episode_id %= self._num_episodes
        task, variation, file = self._episodes[episode_id]

        # Load episode
        episode = self.read_from_cache(file)
        if episode is None:
            return None

        # Dynamic chunking so as not to overload GPU memory
        chunk = random.randint(
            0, math.ceil(len(episode[0]) / self._max_episode_length) - 1
        )

        # Get frame ids for this chunk
        frame_ids = episode[0][
            chunk * self._max_episode_length:
            (chunk + 1) * self._max_episode_length
        ]

        # Get the image tensors for the frame ids we got
        states = torch.stack([
            episode[1][i] if isinstance(episode[1][i], torch.Tensor)
            else torch.from_numpy(episode[1][i])
            for i in frame_ids
        ])

        # Camera ids
        if episode[3]:
            cameras = list(episode[3][0].keys())
            assert all(c in cameras for c in self._cameras)
            index = torch.tensor([cameras.index(c) for c in self._cameras])
            # Re-map states based on camera ids
            states = states[:, index]

        # Split RGB and XYZ
        rgbs = states[:, :, 0]
        pcds = states[:, :, 1]
        rgbs = self._unnormalize_rgb(rgbs)

        # Get action tensors for respective frame ids
        action = torch.cat([episode[2][i] for i in frame_ids])

        # Sample one instruction feature
        if self._instructions:
            instr = random.choice(self._instructions[task][variation])
            instr = instr[None].repeat(len(rgbs), 1, 1)
        else:
            instr = torch.zeros((rgbs.shape[0], 53, 512))

        # Get gripper tensors for respective frame ids
        gripper = torch.cat([episode[4][i] for i in frame_ids])

        # gripper history
        gripper_history = torch.stack([
            torch.cat([episode[4][max(0, i-2)] for i in frame_ids]),
            torch.cat([episode[4][max(0, i-1)] for i in frame_ids]),
            gripper
        ], dim=1)

        # Low-level trajectory
        traj, traj_lens = None, 0
        if self._return_low_lvl_trajectory:
            if len(episode) > 5:
                traj_items = [
                    self._interpolate_traj(episode[5][i]) for i in frame_ids
                ]
            else:
                traj_items = [
                    self._interpolate_traj(
                        torch.cat([episode[4][i], episode[2][i]], dim=0)
                    ) for i in frame_ids
                ]
            max_l = max(len(item) for item in traj_items)
            traj = torch.zeros(len(traj_items), max_l, 8)
            traj_lens = torch.as_tensor(
                [len(item) for item in traj_items]
            )
            for i, item in enumerate(traj_items):
                traj[i, :len(item)] = item
            traj_mask = torch.zeros(traj.shape[:-1])
            for i, len_ in enumerate(traj_lens.long()):
                traj_mask[i, len_:] = 1

        # Augmentations
        if self._training:
            if traj is not None:
                for t, tlen in enumerate(traj_lens):
                    traj[t, tlen:] = 0
            modals = self._resize(rgbs=rgbs, pcds=pcds)
            rgbs = modals["rgbs"]
            pcds = modals["pcds"]

        ret_dict = {
            "task": [task for _ in frame_ids],
            "frame_ids": frame_ids,
            "rgbs": rgbs,  # e.g. tensor (n_frames, n_cam, 3+1, H, W)
            "pcds": pcds,  # e.g. tensor (n_frames, n_cam, 3, H, W)
            "action": action,  # e.g. tensor (n_frames, 8), target pose
            "instr": instr,  # a (n_frames, 53, 512) tensor
            "curr_gripper": gripper,
            "curr_gripper_history": gripper_history
        }
        if self._return_low_lvl_trajectory:
            ret_dict.update({
                "trajectory": traj,  # e.g. tensor (n_frames, T, 8)
                "trajectory_mask": traj_mask.bool()  # tensor (n_frames, T)
            })
        return ret_dict

    def __len__(self):
        if self._num_iters is not None:
            return self._num_iters
        return self._num_episodes

class RLBenchReachMovingTargetDataset(RLBenchDataset):
    def __init__(
        self,
        # required
        root,
        instructions=None,
        instructions_str=None,
        # dataset specification
        taskvar=[('close_door', 0)],
        max_episodes_per_task=100,
        num_iters=None,
        cameras=("front", "left_shoulder", "right_shoulder", "overhead"),
        # for augmentations
        training=True,
        image_rescale=(1.0, 1.0),
        # for trajectories
        num_waypoints_traj=10,
        num_frames_interval_traj=1,
        num_waypoints_history_traj=2,
        num_frames_interval_history_traj=10,
        # for observation
        num_future_frames_obs=3,
        num_history_frames_obs=0,
        # for evaluation
        bool_eval_only=False,
        bool_use_wm_predictions=False,
        ):
        self._cameras = cameras
        self._num_iters = num_iters
        self._training = training
        self._taskvar = taskvar
        self._num_waypoints_traj = num_waypoints_traj
        self._num_frames_interval_traj = num_frames_interval_traj
        self._num_waypoints_history_traj = num_waypoints_history_traj
        self._num_frames_interval_history_traj = num_frames_interval_history_traj
        self._num_future_frames_obs = num_future_frames_obs
        self._num_history_frames_obs = num_history_frames_obs
        self._root = root
        self._instructions = defaultdict(dict)
        self._instructions_str = defaultdict(dict)
        self._num_vars = Counter()  # variations of the same task
        self._bool_use_wm_predictions = bool_use_wm_predictions
        if not Path(root).exists():
            raise ValueError(f"Dataset root {root} does not exist")
        for root, (task, var) in itertools.product([self._root], taskvar):
            if instructions is not None:
                self._instructions[task][var] = instructions[task][var]
                self._instructions_str[task][var] = instructions_str[task][var]
                self._num_vars[task] += 1
        # If training, initialize augmentation classes
        if self._training:
            self._resize = Resize(scales=image_rescale)
        task2var = {
            task: list(vars.keys())
            for task, vars in self._instructions.items()}
        self._episodes, self._num_episodes = self.get_all_episodes(
            root=self._root,
            task2var=task2var,
            max_episodes_per_task=max_episodes_per_task,
        )
        self._cam_param_dict = self.get_cam_param_dict(
            camera_names=self._cameras,
            episode_dir=self._episodes[0][2],
            episode_hf=None,
        )
        self._interpolate_traj = TrajectoryInterpolator(
                use=True,
                interpolation_length=num_waypoints_traj+1,
            )
        self._bool_eval_only = bool_eval_only
        if self._bool_eval_only:
            # (task, variation, episode_dir, all_frame_ids)
            episodes = []
            for task, variation, episode_dir, all_frame_ids in self._episodes:
                for frame_id in all_frame_ids:
                    episodes.append((task, variation, episode_dir, all_frame_ids, frame_id))
            self._episodes = episodes
        self._h5_file_handles = None
        return

    @staticmethod
    def get_cam_param_dict(camera_names, episode_dir: str, episode_hf: h5py.File = None) -> Dict:
        '''
        Args:
            episode: str
        Returns:
            cam_param_dict: Dict,
        '''
        assert 'wrist' not in camera_names, 'wrist camera is not supported'
        if episode_hf is not None:
            cam_param_h5 = episode_hf['cam_params']
            cam_param_dict = {}
            for cam_name in camera_names:
                cam_param_dict[cam_name] = {
                    'extrinsics': cam_param_h5[cam_name]['extrinsics'][:],
                    'intrinsics': cam_param_h5[cam_name]['intrinsics'][:],
                    'near': cam_param_h5[cam_name]['near'],
                    'far': cam_param_h5[cam_name]['far'],
                }
            return cam_param_dict
        elif (Path(episode_dir)/'data.h5').exists():
            with h5py.File(Path(episode_dir) / 'data.h5', 'r') as episode_hf:
                cam_param_h5 = episode_hf['cam_params']
                cam_param_dict = {}
                for cam_name in camera_names:
                    cam_param_dict[cam_name] = {
                        'extrinsics': cam_param_h5[cam_name]['extrinsics'][:],
                        'intrinsics': cam_param_h5[cam_name]['intrinsics'][:],
                        'near': cam_param_h5[cam_name]['near'],
                        'far': cam_param_h5[cam_name]['far'],
                    }
                return cam_param_dict
        low_dim_obs = common_utils.read_pkl(Path(episode_dir) / 'low_dim_obs.pkl')
        cam_param_dict = {}
        for cam_name in camera_names:
            if cam_name in ['front', 'left_shoulder', 'right_shoulder', 'overhead']:
                cam_param_dict[cam_name] = {
                    'extrinsics': low_dim_obs[0].misc[f'{cam_name}_camera_extrinsics'],
                    'intrinsics': low_dim_obs[0].misc[f'{cam_name}_camera_intrinsics'],
                    'near': low_dim_obs[0].misc[f'{cam_name}_camera_near'],
                    'far': low_dim_obs[0].misc[f'{cam_name}_camera_far'],
                }
            elif cam_name.isdigit():
                pose_dict = common_utils.read_pkl(Path(episode_dir) / f'nerf_data/0/poses/{cam_name}.pkl')
                extrinsics = pose_dict['extrinsic']
                intrinsics = pose_dict['intrinsic']
                near = pose_dict['near']
                far = pose_dict['far']
                cam_param_dict[cam_name] = {
                    'extrinsics': extrinsics,
                    'intrinsics': intrinsics,
                    'near': near,
                    'far': far,
                }
            else:
                err_msg = f"Error: cannot recognize cam_name={cam_name}"
                raise RuntimeError(err_msg)
        return cam_param_dict
    
    def get_all_episodes(
        self,
        root: Path,
        task2var: Dict[str, List[int]],
        max_episodes_per_task: int,
    ) -> Tuple[List, int]:
        '''
        Args:
            root: Path,
            task2var: Dict[str, List[int]],
            max_episodes_per_task: int,
        Returns:
            episodes: List[Tuple[str, int, str, List[int]]
                0 task: str
                1 variation: int
                2 episode_dir: str
                3 all_frame_ids: List[int]
            num_episodes: int,
        '''
        episodes = []
        episode_dict = {}
        episode_dirs = Path(root).glob("*/all_variations/episodes/episode*")
        for episode_dir in sorted(episode_dirs):
            task = episode_dir.parents[2].name
            if (episode_dir / f"meta_info.pkl").exists():
                pkl_data = common_utils.read_pkl(episode_dir / f"meta_info.pkl")
                variation = pkl_data['variation']
                all_frame_ids = pkl_data['frame_ids']
            else:
                variation = common_utils.read_pkl(episode_dir / f"variation_number.pkl")
                all_frame_ids = [int(itm.stem) for itm in episode_dir.glob("front_rgb/*.png")]
            if task not in task2var:
                continue
            if variation not in task2var[task]:
                continue
            if not self._is_valid_task(episode_dir):
                continue
            episode_dir = str(episode_dir.absolute())
            episode = (task, variation, episode_dir, all_frame_ids)
            if task in episode_dict:
                episode_dict[task].append(episode)
            else:
                episode_dict[task] = [episode]
        for task, episodes_ in episode_dict.items():
            if max_episodes_per_task > -1 and len(episodes_) > max_episodes_per_task:
                # TODO: 现在这种方式 每个 variation 的 episode 数量不一致
                episodes_ = random.sample(episodes_, max_episodes_per_task)
            episodes.extend(episodes_)
        num_episodes = len(episodes)
        return episodes, num_episodes

    def read_from_episode_dir(
        self,
        episode_dir: Path,
        episode_hf: h5py.File,
        all_frame_ids: List[int],
        cur_frame_id: int,
        num_waypoints_traj: int,
        num_frames_interval_traj: int,
        num_waypoints_history_traj: int,
        num_frames_interval_history_traj: int,
        num_future_frames_obs: int,
        num_history_frames_obs: int,
        bool_use_wm_predictions: bool = False
    ) -> Dict:
        '''
        Args:
            episode_dir: Path,
            all_frame_ids: List[int],
            cur_frame_id: int,
            num_waypoints_traj: int,
            num_frames_interval_traj: int,
            num_waypoints_history_traj: int,
            num_frames_interval_history_traj: int,
            num_future_frames_obs: int,
            num_history_frames_obs: int,
        Returns:
            episode:
                frame_id: int, # current frame id
                status: List[torch.Tensor], # rgb, pcd [view, mode, 3, 256, 256]
                curr_gripper: torch.Tensor, # current gripper pose [1, 8]
                history_grippers: List[torch.Tensor], # history gripper poses 
                    (num_history_frames + 1, 8)
                trajectory: List[torch.Tensor], # future gripper poses 
                    (num_waypoints_traj, 8)
                trajectory_mask: List[torch.Tensor], # future gripper poses mask, 1 for masking it out
                    (num_waypoints_traj)
        '''
        frame_id = cur_frame_id
        history_frame_ids = [
            max(frame_id - (1 + i) * num_frames_interval_history_traj, 0)
            for i in range(num_waypoints_history_traj)
        ]
        # future_frame_ids = [
        #     min(frame_id + (1 + i) * num_frames_interval_traj, len(all_frame_ids) - 1)
        #     for i in range(num_waypoints_traj)
        # ]
        hf = episode_hf
        status, next_status = [], []
        next_data, next_data_velmappc = None, None
        for cam_name in self._cameras:
            # 读取当前帧的 observations
            rgb = self.get_rgb_from_hdf5(hf, cam_name, frame_id)
            rgb = einops.rearrange(torch.from_numpy(rgb), 'h w c -> 1 1 c h w')
            pcd = self.get_pcd_from_hdf5(hf, cam_name, frame_id)
            pcd = einops.rearrange(torch.from_numpy(pcd), 'h w c -> 1 1 c h w')
            status.append(torch.cat([rgb, pcd], dim=1))
            # 读取未来帧的 wm_predictions
            err_msg = f"Error: {cam_name} has less than {num_future_frames_obs} future frames"
            assert hf[f'wm_predictions/{cam_name}/rgb'][frame_id].shape[0] >= num_future_frames_obs, err_msg
        if bool_use_wm_predictions:
            velmappc, next_data_velmappc = self.get_wm_predictions_velmappc(episode_dir, hf, self._cameras, frame_id, num_future_frames_obs)

        status = torch.cat(status, dim=0)
        # if bool_use_wm_predictions:
        #     next_status = torch.cat(next_status, dim=1)
        # (1, 8)
        curr_gripper = self.get_eepose_from_hdf5(hf, [cur_frame_id])
        # (num_history_frames + 1, 8)
        history_grippers = self.get_eepose_from_hdf5(hf, sorted(history_frame_ids + [cur_frame_id]))
        # (num_waypoints_traj, 8)
        # assert all([itm in all_frame_ids for itm in future_frame_ids])

        expert_info = self.get_expert_traj_from_hdf5(hf, cur_frame_id)
        trajectory = expert_info['trajectory']
        stage = expert_info['stage']
        target_position = expert_info['target_position']
        # TODO: change the 10 to the ngv_robot hyperparameter
        # if bool_use_wm_predictions:
        #     next_frame_relative_id = torch.tensor([(i+1)*10 for i in range(num_future_frames_obs)])
        assert num_history_frames_obs == 0, "num_history_frames_obs is not supported"
        # (num_waypoints_traj)
        trajectory_mask = torch.zeros(num_waypoints_traj)
        episode = {
            "frame_id": frame_id,
            "status": status,
            # "next_status": next_status if bool_use_wm_predictions else None,
            "curr_gripper": curr_gripper,
            "curr_gripper_history": history_grippers,
            # "next_gripper": next_gripper if bool_use_wm_predictions else None,
            "velmappc": velmappc if bool_use_wm_predictions else None,
            "trajectory": trajectory,
            "trajectory_mask": trajectory_mask,
            # "next_frame_relative_id": next_frame_relative_id if bool_use_wm_predictions else None,
            "stage": stage,
            "target_position": target_position,
        }
        return episode

    def get_rgb_from_hdf5(self, hf, cam_name, frame_id) -> np.ndarray:
        '''
        Return:
            rgb: np.ndarray, (h, w, 3) in range [-1, 1]
        '''
        rgb = hf[f'observations/{cam_name}/rgb'][frame_id]
        rgb = (rgb / 255.0).astype(np.float32)
        rgb = rgb * 2 - 1
        return rgb

    def get_pcd_from_hdf5(self, hf, cam_name, frame_id) -> np.ndarray:
        '''
        Return:
            pcd: np.ndarray, (h, w, 3) in meters
        '''
        pcd = self.read_pcd_data(
            None,
            cam_name, frame_id,
            cam_param_dict=hf[f'cam_params'],
            depth_uint8=hf[f'observations/{cam_name}/depth'][frame_id],
        )
        return pcd.numpy()

    def get_wm_predictions_rgb(self, episode_dir, hf, cam_name, frame_id, num_future_frames_obs, next_data=None):
        wm_pred_dir_ = episode_dir.replace('_package_compressed', '_wm_predictions')
        assert Path(wm_pred_dir_).exists()
        if frame_id == 0:
            rgb = self.get_rgb_from_hdf5(hf, cam_name, frame_id)
            rgb = torch.from_numpy(rgb).unsqueeze(0).repeat(num_future_frames_obs, 1, 1, 1)
            return rgb, None
        if next_data is None:
            next_data_paths = list(Path(wm_pred_dir_).glob(f"eval_dict_cur_step{frame_id}_*.npz"))
            next_data_paths = sorted(
                next_data_paths,
                key=lambda x: int(re.search(r'tar_step(\d+)', x.stem).group(1)),
            )
            next_data = [
                common_utils.read_npz(next_data_path)
                for next_data_path in next_data_paths[:num_future_frames_obs]
            ]
        err_msg = f"Error: {cam_name} has less than {num_future_frames_obs} future frames"
        assert len(next_data) > 0, err_msg
        rgb_list = []
        for next_data_ in next_data:
            cam_name_list = next_data_['camera_name_list'].tolist()
            idx = cam_name_list.index(cam_name)
            rgb = next_data_['image'][idx] # [1, c, h, w] in range (0,255)
            rgb = einops.rearrange(rgb, '1 c h w -> h w c')
            rgb = (rgb / 255.0).astype(np.float32)
            rgb = rgb * 2 - 1
            rgb_list.append(torch.from_numpy(rgb))
        return torch.stack(rgb_list, dim=0), next_data

    def get_wm_predictions_pcd(self, episode_dir, hf, cam_name, frame_id, num_future_frames_obs,next_data=None):
        wm_pred_dir_ = episode_dir.replace('_package_compressed', '_wm_predictions')
        assert Path(wm_pred_dir_).exists()
        if frame_id == 0:
            pcd = self.get_pcd_from_hdf5(hf, cam_name, frame_id)
            pcd = torch.from_numpy(pcd).unsqueeze(0).repeat(num_future_frames_obs, 1, 1, 1)
            return  pcd, None
        if next_data is None:
            next_data_paths = list(Path(wm_pred_dir_).glob(f"eval_dict_cur_step{frame_id}_*.npz"))
            next_data_paths = sorted(
                next_data_paths,
                key=lambda x: int(re.search(r'tar_step(\d+)', x.stem).group(1)),
            )
            next_data = [
                common_utils.read_npz(next_data_path)
                for next_data_path in next_data_paths[:num_future_frames_obs]
            ]
        err_msg = f"Error: {cam_name} has less than {num_future_frames_obs} future frames"
        assert len(next_data) > 0, err_msg
        pcd_list = []
        for next_data_ in next_data:
            cam_name_list = next_data_['camera_name_list'].tolist()
            idx = cam_name_list.index(cam_name)
            depth_rgb = next_data_['depth'][idx,0] # (3, h, w) [0-255]
            depth_rgb = einops.rearrange(depth_rgb, 'c h w -> h w c')
            depth_m = common_utils.decode_depth_from_uin8_to_float(
                depth_rgb,
                bool_metric=True,
                near=hf['cam_params'][cam_name]['near'][...],
                far=hf['cam_params'][cam_name]['far'][...],
            )
            extrinsics = hf['cam_params'][cam_name]['extrinsics']
            intrinsics = hf['cam_params'][cam_name]['intrinsics']
            pcd = pointcloud_from_depth_and_camera_params_inplace(
                depth_m,
                extrinsics,
                intrinsics,
            ) # (h, w, 3)
            # clamp x, y, z channels
            # should be align with self.read_pcd_data()
            pcd[..., 0] = np.clip(pcd[..., 0], -2.5, 2.5)
            pcd[..., 1] = np.clip(pcd[..., 1], -2.5, 2.5)
            pcd[..., 2] = np.clip(pcd[..., 2], 0, 2)
            pcd_list.append(torch.from_numpy(pcd))
        return torch.stack(pcd_list, dim=0).float(), next_data

    def clean_velmap(self, velmap_pc, workspace_bounds=None, max_vel_mag=2.0):
        """
        Clean velmap_pc by removing position outliers and clipping velocity outliers.
        
        Args:
            velmap_pc: np.ndarray, (N, 9) [x, y, z, vx, vy, vz, wx, wy, wz]
            workspace_bounds: list [x_min, y_min, z_min, x_max, y_max, z_max]
            max_vel_mag: float, maximum velocity magnitude (default: 2.0 m/s)
        
        Returns:
            cleaned_velmap_pc: np.ndarray, (M, 9) where M <= N
        """
        pos = velmap_pc[:, :3]  # (N, 3)
        vel = velmap_pc[:, 3:9]  # (N, 6) - 6 velocity components
        
        # 1. 处理位置 Outlier (XYZ) -> 使用 Box Filter 丢弃，不要 Clip!
        if workspace_bounds is not None:
            x_min, y_min, z_min, x_max, y_max, z_max = workspace_bounds
            buffer = 0.0
            mask_x = (pos[:, 0] > x_min - buffer) & (pos[:, 0] < x_max + buffer)
            mask_y = (pos[:, 1] > y_min - buffer) & (pos[:, 1] < y_max + buffer)
            mask_z = (pos[:, 2] > z_min - buffer) & (pos[:, 2] < z_max + buffer)
            keep_mask = mask_x & mask_y & mask_z
            
            velmap_pc = velmap_pc[keep_mask]
            # 更新 pos, vel 用于后续处理
            pos = velmap_pc[:, :3]
            vel = velmap_pc[:, 3:9]
        
        # 2. 处理速度 Outlier (Vel) -> Hard Clamp 到合理范围
        # 策略 B (推荐): 保留点，但把速度 Clip 到合理范围，防止 tanh 饱和
        velmap_pc[:, 3:9] = np.clip(velmap_pc[:, 3:9], -max_vel_mag, max_vel_mag)
        
        return velmap_pc

    def get_wm_predictions_velmappc(self, episode_dir, hf, cam_names, frame_id, num_future_frames_obs):
        num_samples = 30000
        wm_pred_dir_ = episode_dir.replace('_package_compressed', '_wm_predictions')
        assert Path(wm_pred_dir_).exists()
        if frame_id < 3 :
            pcd_list = []
            for cam_name in cam_names:
                pcd = self.get_pcd_from_hdf5(hf, cam_name, frame_id)
                pcd_list.append(torch.from_numpy(pcd))
            pcd = torch.stack(pcd_list, dim=0)
            # Reshape to (num_future_frames_obs * h * w, 3)
            pcd = pcd.reshape(-1, 3)
            # Sample num_samples points from pcd
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
            # 【Checkpoint: 检查 frame_id 2 和 3 输出 velmap pc 区别】
            return velmap_pc.unsqueeze(0), None

        next_data_paths = list(Path(wm_pred_dir_).glob(f"vmap_dict_cur_step{frame_id}_*.npz"))
        next_data_paths = sorted(
            next_data_paths,
            key=lambda x: int(re.search(r'tar_step(\d+)', x.stem).group(1)),
        )
        next_data = [
            common_utils.read_npz(next_data_path)
            for next_data_path in next_data_paths
        ]
        err_msg = f"Error: find more than one velmappc file"
        assert len(next_data) == 1, err_msg
        # Read velmap_pc from NpzFile (cannot assign back to NpzFile object)
        velmap_pc_raw = next_data[0]['velmap_pc']
        # Clean velmap_pc: remove position outliers and clip velocity outliers
        # x_min, y_min, z_min, x_max, y_max, z_max
        workspace_bounds = [-2.5, -2.5, 0.0, 2.5, 2.5, 2.0]
        velmap_pc_cleaned = self.clean_velmap(
            velmap_pc_raw,
            workspace_bounds=workspace_bounds,
            max_vel_mag=1.0
        )
        # 9 dim:
        # x y z,
        # line_velocity_weight_x, line_velocity_weight_y, line_velocity_weight_z,
        # anguler_velocity_weight_x, anguler_velocity_weight_y, anguler_velocity_weight_z
        if velmap_pc_cleaned.shape[0] > num_samples:
            valmappc = np.random.permutation(velmap_pc_cleaned)[:num_samples]
        else:
            # 有放回采样：先随机选择索引，然后根据索引选择点
            num_points = velmap_pc_cleaned.shape[0]
            indices = np.random.choice(num_points, size=num_samples, replace=True)
            valmappc = velmap_pc_cleaned[indices]
        return torch.from_numpy(valmappc).float().unsqueeze(0), next_data

    def get_wm_predictions_gripper(self, episode_dir, hf, cam_name, frame_id, num_future_frames_obs, next_data=None):
        wm_pred_dir_ = episode_dir.replace('_package_compressed', '_wm_predictions')
        assert Path(wm_pred_dir_).exists()
        if frame_id == 0:
            cur_gripper = np.concatenate([
                hf['low_dim_obs/gripper_pose'][frame_id],
                [hf['low_dim_obs/gripper_open'][frame_id]],
            ], axis=-1)
            cur_gripper = torch.from_numpy(cur_gripper)
            return  cur_gripper.unsqueeze(0).repeat(num_future_frames_obs, 1)[None], None
        if next_data is None:
            next_data_paths = list(Path(wm_pred_dir_).glob(f"eval_dict_cur_step{frame_id}_*.npz"))
            next_data_paths = sorted(
                next_data_paths,
                key=lambda x: int(re.search(r'tar_step(\d+)', x.stem).group(1)),
            )
            next_data = [
                common_utils.read_npz(next_data_path)
                for next_data_path in next_data_paths[:num_future_frames_obs]
            ]
        err_msg = f"Error: {cam_name} has less than {num_future_frames_obs} future frames"
        assert len(next_data) > 0, err_msg
        gripper_list = []
        for next_data_ in next_data:
            gripper = np.concatenate([
                [next_data_['eepose']],
                [next_data_['openness']],
            ], axis=-1)
            gripper_list.append(torch.from_numpy(gripper))
        return torch.stack(gripper_list, dim=0), next_data

    @staticmethod
    def read_rgb_data(
        episode_dir: Path,
        cam_name: str,
        frame_id: int,
    ) -> torch.Tensor:
        '''
        Args:
            episode_dir: Path,
            cam_name: str,
            frame_id: int,
        Returns:
            rgb: torch.Tensor, (256, 256, 3) [-1,1]
        '''
        if cam_name in ['front', 'left_shoulder', 'right_shoulder', 'wrist', 'overhead']:
            rgb_path = Path(episode_dir) / f'{cam_name}_rgb/{frame_id}.png'
        elif cam_name.isdigit():
            rgb_path = Path(episode_dir) / f'nerf_data/{frame_id}/images/{cam_name}.png'
        else:
            raise ValueError(f'Invalid cam_name: {cam_name}')
        rgb = common_utils.read_image(rgb_path)
        rgb = rgb / 255.0 * 2 - 1
        return torch.from_numpy(rgb)
    
    @staticmethod
    def read_expert_info(
        episode_dir: Path,
        frame_id: int,
    ) -> Dict:
        '''
        '''
        expert_info = common_utils.read_pkl(Path(episode_dir) / f'expert_info/{frame_id}.pkl')
        return expert_info
    
    @staticmethod
    def read_mask_data(
        episode_dir: Path,
        cam_name: str,
        frame_id: int,
    ) -> torch.Tensor:
        '''
        Args:
            episode_dir: Path,
            cam_name: str,
            frame_id: int,
        Returns:
            mask: torch.Tensor, (256, 256, 3) [0, 255]
        '''
        if cam_name in ['front', 'left_shoulder', 'right_shoulder', 'wrist', 'overhead']:
            mask_path = Path(episode_dir) / f'{cam_name}_mask/{frame_id}.png'
        elif cam_name.isdigit():
            mask_path = Path(episode_dir) / f'nerf_data/{frame_id}/masks/{cam_name}.png'
        else:
            raise ValueError(f'Invalid cam_name: {cam_name}')
        mask = common_utils.read_image(mask_path)
        return torch.from_numpy(mask)

    @staticmethod
    def read_pcd_data(
            episode_dir: Path,
            cam_name: str,
            frame_id: int,
            cam_param_dict: Dict,
            depth_uint8: np.ndarray = None,
        ) -> torch.Tensor:
        '''
        Args:
            episode_dir: Path,
            cam_name: str,
            frame_id: int,
            cam_param_dict: Dict,
        Returns:
            pcd: torch.Tensor, (256, 256, 3)
        '''
        if depth_uint8 is None:
            if cam_name in ['front', 'left_shoulder', 'right_shoulder', 'wrist', 'overhead']:
                depth_path = Path(episode_dir) / f'{cam_name}_depth/{frame_id}.png'
            elif cam_name.isdigit():
                depth_path = Path(episode_dir) / f'nerf_data/{frame_id}/depths/{cam_name}.png'
            else:
                raise ValueError(f'Invalid cam_name: {cam_name}')
            depth = image_to_float_array_inplace(common_utils.read_image(depth_path), DEPTH_SCALE)
        else:
            depth = image_to_float_array_inplace(depth_uint8, DEPTH_SCALE)

        if isinstance(cam_param_dict, dict):
            extrinsics = cam_param_dict[cam_name][f'extrinsics']
            intrinsics = cam_param_dict[cam_name][f'intrinsics']
            near = cam_param_dict[cam_name][f'near']
            far = cam_param_dict[cam_name][f'far']
        elif isinstance(cam_param_dict, h5py._hl.group.Group):
            extrinsics = cam_param_dict[cam_name][f'extrinsics'][...]
            intrinsics = cam_param_dict[cam_name][f'intrinsics'][...]
            near = cam_param_dict[cam_name][f'near'][...]
            far = cam_param_dict[cam_name][f'far'][...]
        else:
            raise ValueError(f'Invalid cam_param_dict: {cam_param_dict}')
        depth_m = near + depth * (far - near)
        pcd = pointcloud_from_depth_and_camera_params_inplace(
            depth_m,
            extrinsics,
            intrinsics,
        )
        # clamp x and y channels
        pcd[..., 0] = np.clip(pcd[..., 0], -2.5, 2.5)
        pcd[..., 1] = np.clip(pcd[..., 1], -2.5, 2.5)
        pcd[..., 2] = np.clip(pcd[..., 2], 0, 2)
        return torch.from_numpy(pcd)

    @staticmethod
    def read_depth_data(
            episode_dir: Path,
            cam_name: str,
            frame_id: int,
            cam_param_dict: Dict,
            bool_read_uint8: bool = False,
        ) -> torch.Tensor:
        '''
        Args:
            episode_dir: Path,
            cam_name: str,
            frame_id: int,
            cam_param_dict: Dict,
        Returns:
            depth: torch.Tensor, (256, 256) in unit meters / (256, 256, 3) in range [0, 255]
        '''
        if cam_name in ['front', 'left_shoulder', 'right_shoulder', 'wrist', 'overhead']:
            depth_path = Path(episode_dir) / f'{cam_name}_depth/{frame_id}.png'
        elif cam_name.isdigit():
            depth_path = Path(episode_dir) / f'nerf_data/{frame_id}/depths/{cam_name}.png'
        else:
            raise ValueError(f'Invalid cam_name: {cam_name}')
        if bool_read_uint8:
            depth_uint8 = common_utils.read_image(depth_path)
            return torch.from_numpy(depth_uint8)
        else:
            depth = image_to_float_array_inplace(common_utils.read_image(depth_path), DEPTH_SCALE)
            near = cam_param_dict[cam_name][f'near']
            far = cam_param_dict[cam_name][f'far']
            depth_m = near + depth * (far - near)
            return torch.from_numpy(depth_m).float()

    def get_eepose_from_hdf5(self, hf_file, frame_ids):
        gripper_poses, gripper_opens = [], []
        for f_id in frame_ids:
            pose = hf_file[f'low_dim_obs/gripper_pose'][f_id]
            open_state = hf_file[f'low_dim_obs/gripper_open'][f_id]
            gripper_poses.append(pose)
            gripper_opens.append(open_state)
        # 这里需要根据你的 get_eepose_from_low_dim_obs 逻辑，构建一个等价的输入
        # 为了保持一致性，我假设你将 gripper_pose 和 open_state 合并为一个 np.array
        # 并返回一个类似的结构
        gripper_data = np.concatenate([np.stack(gripper_poses), np.stack(gripper_opens)[:, np.newaxis]], axis=-1)
        return torch.from_numpy(gripper_data).float()

    def get_expert_traj_from_hdf5(self, hf_file, cur_frame_id):
        expert_traj = hf_file['low_dim_obs/expert_trajectory'][cur_frame_id]
        stage = hf_file['low_dim_obs/expert_trajectory_stage'][cur_frame_id]
        target_position = hf_file['low_dim_obs/expert_trajectory_target_position'][cur_frame_id]
        return {
            "trajectory": torch.from_numpy(expert_traj).float(),
            "stage": stage,
            "target_position": target_position,
        }

    def get_eepose_from_low_dim_obs(
        self,
        low_dim_obs,
        frame_ids: List[int],
    ) -> torch.Tensor:
        obs_list = low_dim_obs._observations
        eepose_list = []
        all_frame_ids = list(range(len(obs_list)))
        for frame_id_ in frame_ids:
            if frame_id_ not in all_frame_ids:
                action = torch.zeros(1, 8)
            else:
                obs = obs_list[frame_id_]
                action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
                action = torch.from_numpy(action)[None]
            eepose_list.append(action)
        eepose_arr = torch.cat(eepose_list, dim=0)
        return eepose_arr
    
    def _is_valid_task(self, episode_dir: Path) -> bool:
        '''
        Return False, if the wm_predictions is not valid
        '''
        if not self._bool_use_wm_predictions:
            return True
        wm_pred_dir_ = str(episode_dir).replace('_package_compressed', '_wm_predictions')
        if (Path(wm_pred_dir_)/'.success').exists():
            return True
        print(f"Error: {wm_pred_dir_} is not valid, since WM did not success.")
        return False

    def __getitem__(self, episode_id: int) -> Dict[str, torch.Tensor]:
        num_waypoints_traj = self._num_waypoints_traj
        num_frames_interval_traj = self._num_frames_interval_traj
        num_waypoints_history_traj = self._num_waypoints_history_traj
        num_frames_interval_history_traj = self._num_frames_interval_history_traj
        num_future_frames_obs = self._num_future_frames_obs
        num_history_frames_obs = self._num_history_frames_obs
        bool_use_wm_predictions = self._bool_use_wm_predictions
        if not self._bool_eval_only:
            # get episode dir
            episode_id %= self._num_episodes
            task, variation, episode_dir, all_frame_ids = self._episodes[episode_id]
            # Load episode
            cur_frame_id = random.choice(all_frame_ids)
        else:
            task, variation, episode_dir, all_frame_ids, cur_frame_id = self._episodes[episode_id]
            # print(task, variation, episode_dir, cur_frame_id)
        if self._training:
            # training not use cache
            # it will not explode memory, due to the calling of self.resize
            with h5py.File(Path(episode_dir)/'data.h5', 'r') as episode_hf:
                episode = self.read_from_episode_dir(
                    episode_dir=episode_dir,
                    episode_hf=episode_hf,
                    all_frame_ids=all_frame_ids,
                    cur_frame_id=cur_frame_id,
                    num_waypoints_traj=num_waypoints_traj,
                    num_frames_interval_traj=num_frames_interval_traj,
                    num_waypoints_history_traj=num_waypoints_history_traj,
                    num_frames_interval_history_traj=num_frames_interval_history_traj,
                    num_future_frames_obs=num_future_frames_obs,
                    num_history_frames_obs=num_history_frames_obs,
                    bool_use_wm_predictions=bool_use_wm_predictions,
                )
        else:
            # test use cache
            # it will not explode memory, if it uses h5_cache
            if getattr(self, '_h5_cache', None) is None:
                self._h5_cache = {}
            if episode_dir not in self._h5_cache:
                self._h5_cache[episode_dir] = h5py.File(Path(episode_dir)/'data.h5', 'r')
            episode_hf = self._h5_cache[episode_dir]
            episode = self.read_from_episode_dir(
                episode_dir=episode_dir,
                episode_hf=episode_hf,
                all_frame_ids=all_frame_ids,
                cur_frame_id=cur_frame_id,
                num_waypoints_traj=num_waypoints_traj,
                num_frames_interval_traj=num_frames_interval_traj,
                num_waypoints_history_traj=num_waypoints_history_traj,
                num_frames_interval_history_traj=num_frames_interval_history_traj,
                num_future_frames_obs=num_future_frames_obs,
                num_history_frames_obs=num_history_frames_obs,
                bool_use_wm_predictions=bool_use_wm_predictions,
            )

        frame_id = episode['frame_id']
        states = episode['status']
        # next_states = episode['next_status']
        # next_frame_relative_id = episode['next_frame_relative_id']
        velmappc = episode['velmappc']
        # Split RGB and XYZ
        # states (ncam, mode, c, h, w)
        # rgbs (ncam, c, h, w)
        # pcds (ncam, c, h, w)
        rgbs = states[:, 0][None]
        pcds = states[:, 1][None]
        rgbs = self._unnormalize_rgb(rgbs)
        # next_states (num_future_frames_obs, ncam, mode, c, h, w)
        # next_rgbs (num_future_frames_obs, ncam, c, h, w)
        # next_pcds (num_future_frames_obs, ncam, c, h, w)
        if self._bool_use_wm_predictions:
            # next_rgbs = next_states[:, :, 0]
            # next_pcds = next_states[:, :, 1]
            # next_rgbs = self._unnormalize_rgb(next_rgbs)
            # next_rgbs = einops.rearrange(next_rgbs, 't ncam c h w -> 1 ncam (t c) h w')
            # next_pcds = einops.rearrange(next_pcds, 't ncam c h w -> 1 ncam (t c) h w')
            velmappc = einops.rearrange(velmappc, 't n c -> t c n')
        # Sample one instruction feature
        if self._instructions:
            instr_idx = random.choice(list(range(len(self._instructions[task][variation]))))
            instr = self._instructions[task][variation][instr_idx]
            instr_str = self._instructions_str[task][variation][instr_idx]
            instr = instr[None].repeat(len(rgbs), 1, 1)
            instr_str = [instr_str] * len(rgbs)
        else:
            instr = torch.zeros((rgbs.shape[0], 53, 512))
            instr_str = [""] * len(rgbs)
        
        # Get gripper tensors for respective frame ids
        gripper = episode['curr_gripper']
        gripper_history = episode['curr_gripper_history']
        # next_gripper = episode['next_gripper']
        traj = episode['trajectory']
        traj_mask = episode['trajectory_mask']
        stage = episode['stage'].astype(str)[0]
        stage_idx = 0
        target_position = episode['target_position']
        # Augmentations
        if self._training:
            if self._bool_use_wm_predictions:
                modals = self._resize(
                    rgbs=rgbs, pcds=pcds,
                    # next_rgbs=next_rgbs, next_pcds=next_pcds,
                )
            else:
                modals = self._resize(
                    rgbs=rgbs, pcds=pcds,
                )
            rgbs = modals["rgbs"]
            pcds = modals["pcds"]
            # if self._bool_use_wm_predictions:
            #     next_rgbs = modals["next_rgbs"]
            #     next_pcds = modals["next_pcds"]
        # if self._bool_use_wm_predictions:
        #     next_rgbs = einops.rearrange(
        #         next_rgbs, 'b ncam (t c) h w -> b t ncam c h w',
        #         t=num_future_frames_obs,
        #     )
        #     next_pcds = einops.rearrange(
        #         next_pcds,
        #         'b ncam (t c) h w -> b t ncam c h w',
        #         t=num_future_frames_obs,
        #     )

        # add current gripper to trajectory at the first position
        # it is to compatible with the trajectory in the dataset
        # traj = torch.cat([gripper, traj], dim=0)
        # traj_mask = torch.cat([torch.zeros(1), traj_mask], dim=0)
        # if self._bool_use_wm_predictions:
        #     next_masks = torch.ones_like(next_rgbs[:, :, :, :1])

        ret_dict = {
            "task": [task],
            "variation": [variation],
            "episode": [int(Path(episode_dir).stem[7:])],
            "frame_id": [frame_id],
            "rgbs": rgbs,  # e.g. tensor (1, n_cam, 3, H, W)
            "pcds": pcds,  # e.g. tensor (1, n_cam, 3, H, W)
            "instr": instr,  # (1, 53, 512) tensor
            "instr_str": instr_str, # (1, ) string
            "curr_gripper": gripper[None], # (1, 8)
            "curr_gripper_history": gripper_history[None], # (1, num_history_frames+1, 8)
            "trajectory": traj[None],  # e.g. tensor (1, 1+num_waypoints_traj, 8)
            "trajectory_mask": traj_mask.bool()[None],  # tensor (1, 1+num_waypoints_traj)
            "action": traj[None][:,-1,...], # (1, 8)
            "stage": stage_idx,
        }
        if self._bool_use_wm_predictions:
            ret_dict.update({
                # "next_rgbs": next_rgbs,  # e.g. tensor (1, n_fut, n_cam, 3, H, W)
                # "next_pcds": next_pcds,  # e.g. tensor (1, n_fut, n_cam, 3, H, W)
                # "next_masks": next_masks, # e.g. tensor (1, n_fut, n_cam, 3, H, W)
                # "next_gripper": next_gripper, # e.g. tensor (1, n_fut, 8)
                # "next_frame_relative_id": next_frame_relative_id[None], # e.g. tensor (1, n_fut)
                "velmappc": velmappc.permute(0, 2, 1).cpu().numpy(), # e.g. numpy array (1, n_fut, 9)
            })
        return ret_dict

    def __len__(self):
        if self._bool_eval_only:
            return len(self._episodes)
        else:
            return super().__len__()
    
    @staticmethod
    def package_dataset(
        root: Path,
        wm_pred_dir: Path,
        save_dir: Path,
        camera_names: List[str],
        num_future_frames_obs: int,
    ):
        '''
        Args:
            root: Path,
            save_dir: Path,
        '''
        bool_use_wm_predictions = wm_pred_dir is not None
        if Path(save_dir).exists():
            raise ValueError(f"Save directory {save_dir} already exists, you have to remove it first")
        episode_dirs = sorted(list(Path(root).glob("*/all_variations/episodes/episode*")))
        for episode_dir in tqdm(episode_dirs, desc="Packaging dataset", total=len(episode_dirs)):
            task = episode_dir.parents[2].name
            variation = common_utils.read_pkl(episode_dir / f"variation_number.pkl")
            low_dim_obs = common_utils.read_pkl(episode_dir / f"low_dim_obs.pkl")
            episode_id = episode_dir.name[7:]
            all_frame_ids = [int(itm.stem) for itm in episode_dir.glob("front_rgb/*.png")]
            all_frame_ids = sorted(all_frame_ids)
            meta_info = {
                'task': task,
                'variation': int(variation),
                'episode': int(episode_id),
                'frame_ids': all_frame_ids,
            }
            assert all_frame_ids == list(range(len(all_frame_ids))), \
                f"Frame IDs are not continuous: {all_frame_ids}"
            if bool_use_wm_predictions:
                wm_pred_dir_ = Path(wm_pred_dir) / f'{task}/all_variations/episodes/episode{episode_id}'
            save_dir_ = Path(save_dir) / f'{task}/all_variations/episodes/episode{episode_id}'
            save_dir_.mkdir(parents=True, exist_ok=True)
            # (save_dir_/"frames").mkdir(parents=True, exist_ok=True)
            common_utils.write_pkl(meta_info, Path(save_dir_)/"meta_info.pkl")
            hdf5_file_path = Path(save_dir_)/"data.h5"
            with h5py.File(hdf5_file_path, 'a') as hf:
                # 存储元数据
                hf.attrs['task'] = task
                hf.attrs['variation'] = int(variation)
                hf.attrs['episode'] = int(episode_id)
                hf.attrs['num_frames'] = len(all_frame_ids)
                num_frames = len(all_frame_ids)
                
                # 存储摄像头参数
                cam_param_dict = RLBenchReachMovingTargetDataset.get_cam_param_dict(
                    camera_names=camera_names,
                    episode_dir=episode_dir,
                    episode_hf=None,
                )
                cam_params_group = hf.create_group('cam_params')
                for cam_name, params in cam_param_dict.items():
                    cam_group = cam_params_group.create_group(cam_name)
                    cam_group.create_dataset('extrinsics', data=params['extrinsics'])
                    cam_group.create_dataset('intrinsics', data=params['intrinsics'])
                    hf['cam_params'][cam_name].create_dataset('near', data=params['near'])
                    hf['cam_params'][cam_name].create_dataset('far', data=params['far'])

                # 从front_rgb获取图像尺寸，如果没有则使用默认值256
                front_rgb_path = Path(episode_dir) / "front_rgb" / "0.png"
                if front_rgb_path.exists():
                    front_rgb = common_utils.read_image(front_rgb_path)
                    image_h, image_w = front_rgb.shape[:2]
                else:
                    image_h, image_w = 256, 256
                # 创建 observations 组
                observations_group = hf.create_group('observations')
                for cam_name in camera_names:
                    cam_group = observations_group.create_group(cam_name)
                    cam_group.create_dataset(
                        'rgb', 
                        shape=(num_frames, image_h, image_w, 3), 
                        dtype=np.uint8, 
                        chunks=(1, image_h, image_w, 3), 
                        compression="gzip",
                        compression_opts=4,
                    )
                    cam_group.create_dataset(
                        'depth',
                        shape=(num_frames, image_h, image_w, 3),
                        dtype=np.uint8,
                        chunks=(1, image_h, image_w, 3),
                        compression="gzip",
                        compression_opts=4,
                    )

                # 创建 wm_predictions 组
                wm_predictions_group = hf.create_group('wm_predictions')
                for cam_name in camera_names:
                    cam_group = wm_predictions_group.create_group(cam_name)
                    cam_group.create_dataset(
                        'rgb', 
                        shape=(num_frames, num_future_frames_obs, image_h, image_w, 3),
                        dtype=np.uint8, 
                        chunks=(1, num_future_frames_obs, image_h, image_w, 3),
                        compression="gzip",
                        compression_opts=4,
                    )
                    cam_group.create_dataset(
                        'depth',
                        shape=(num_frames, num_future_frames_obs, image_h, image_w, 3),
                        dtype=np.uint8,
                        chunks=(1, num_future_frames_obs, image_h, image_w, 3),
                        compression="gzip",
                        compression_opts=4,
                    )
                    cam_group.create_dataset(
                        'gripper',
                        shape=(num_frames, num_future_frames_obs, 8),
                        dtype=np.float32,
                        chunks=(1, num_future_frames_obs, 8),
                    )

                # 创建 low_dim_obs 组
                low_dim_group = hf.create_group('low_dim_obs')
                low_dim_group.create_dataset(
                    'gripper_pose', 
                    shape=(num_frames, 7), # 假设 gripper pose 有 7 个值
                    dtype=np.float32, 
                    chunks=(1, 7),
                )
                low_dim_group.create_dataset(
                    'gripper_open',
                    shape=(num_frames,),
                    dtype=np.float32,
                    chunks=True,
                )
                low_dim_group.create_dataset(
                    'joint_positions',
                    shape=(num_frames, 7), # 假设 joint positions 有 7 个值
                    dtype=np.float32,
                    chunks=(1, 7),
                )
                low_dim_group.create_dataset(
                    'expert_trajectory',
                    shape=(num_frames, 1, 8),
                    dtype=np.float32,
                    chunks=(1, 1, 7),
                )
                low_dim_group.create_dataset(
                    'expert_trajectory_stage',
                    shape=(num_frames, 1),
                    dtype=h5py.string_dtype(encoding='utf-8'),
                    chunks=(1, 1),
                )
                low_dim_group.create_dataset(
                    'expert_trajectory_target_position',
                    shape=(num_frames, 3),
                    dtype=np.float32,
                    chunks=(1, 3),
                )

                for frame_id in tqdm(all_frame_ids):
                    observations = {}
                    for cam_name in camera_names:
                        # (256, 256, 3) in range [-1,1]
                        rgb = RLBenchReachMovingTargetDataset.read_rgb_data(
                            episode_dir, cam_name, frame_id)
                        # (256, 256, 3) in range [0, 255]
                        depth = RLBenchReachMovingTargetDataset.read_depth_data(
                            episode_dir, cam_name, frame_id, cam_param_dict, bool_read_uint8=True)
                        observations[cam_name] = {
                            'rgb': rgb, # (h, w, 3) in range [-1, 1]
                            'depth': depth, # (h, w)
                        }
                    expert_info = RLBenchReachMovingTargetDataset.read_expert_info(
                        episode_dir, frame_id)

                    if frame_id == 0 and bool_use_wm_predictions:
                        wm_predictions = {}
                        cur_gripper = np.concatenate([
                            low_dim_obs._observations[frame_id].gripper_pose,
                            [low_dim_obs._observations[frame_id].gripper_open],
                        ], axis=-1)
                        cur_gripper = torch.from_numpy(cur_gripper)
                        for cam_name, v in observations.items():
                            wm_predictions[cam_name] = {
                                # (n_fut, h, w, 3) in range [-1, 1]
                                'rgb': v['rgb'].unsqueeze(0).repeat(num_future_frames_obs, 1, 1, 1),
                                # (n_fut, h, w, 3) in range [0, 255]
                                'depth': v['depth'].unsqueeze(0).repeat(num_future_frames_obs, 1, 1, 1),
                                'gripper': cur_gripper.unsqueeze(0).repeat(num_future_frames_obs, 1),
                            }
                    elif bool_use_wm_predictions:
                        wm_predictions = {}
                        next_data_paths = list(Path(wm_pred_dir_).glob(f"eval_dict_cur_step{frame_id}_*.npz"))
                        next_data_paths = sorted(
                            next_data_paths,
                            key=lambda x: int(re.search(r'tar_step(\d+)', x.stem).group(1)),
                        )
                        next_data = [
                            common_utils.read_npz(next_data_path)
                            for next_data_path in next_data_paths[:num_future_frames_obs]
                        ]
                        assert len(next_data) == num_future_frames_obs
                        for cam_name in camera_names:
                            rgb_list, depth_list, pcd_list, gripper_list = [], [], [], []
                            for next_data_ in next_data:
                                cam_name_list = next_data_['camera_name_list'].tolist()
                                idx = cam_name_list.index(cam_name)
                                rgb = next_data_['image'][idx] # [1, c, h, w] in range (0,255)
                                rgb = (rgb / 255.0).astype(np.float32)
                                rgb = einops.rearrange(rgb, '1 c h w -> h w c')
                                rgb = rgb * 2 - 1
                                assert False, "Not implemented due to depth_m"
                                depth_m = next_data_['depth_m'][idx][0,0] # (1, 1 h, w)
                                extrinsics = cam_param_dict[cam_name][f'extrinsics']
                                intrinsics = cam_param_dict[cam_name][f'intrinsics']
                                pcd = pointcloud_from_depth_and_camera_params_inplace(
                                    depth_m,
                                    extrinsics,
                                    intrinsics,
                                ) # (h, w, 3)
                                # clamp x, y, z channels
                                # should be align with self.read_pcd_data()
                                pcd[..., 0] = np.clip(pcd[..., 0], -2.5, 2.5)
                                pcd[..., 1] = np.clip(pcd[..., 1], -2.5, 2.5)
                                pcd[..., 2] = np.clip(pcd[..., 2], 0, 2)
                                gripper = np.concatenate([
                                    [next_data_['eepose']],
                                    [next_data_['openness']],
                                ], axis=-1)
                                rgb_list.append(torch.from_numpy(rgb))
                                depth_list.append(torch.from_numpy(depth_m))
                                pcd_list.append(torch.from_numpy(pcd))
                                gripper_list.append(torch.from_numpy(gripper))
                            wm_predictions[cam_name] = {
                                'rgb': torch.stack(rgb_list, dim=0), # (n_fut, h, w, 3) in range [-1, 1]
                                'depth': torch.stack(depth_list, dim=0), # (n_fut, h, w)
                                # 'pcd': torch.stack(pcd_list, dim=0), # (n_fut, h, w, 3)
                                'gripper': torch.stack(gripper_list, dim=0), # (n_fut, 8)
                            }
                    # 3. 将数据写入 HDF5 文件
                    for cam_name in camera_names:
                        # range [0-255] uint8
                        hf['observations'][cam_name]['rgb'][frame_id] = common_utils.encode_image_from_float_to_uint8(
                            observations[cam_name]['rgb'].numpy()/2.0+0.5
                        )
                        # range [0-255] uint8
                        hf['observations'][cam_name]['depth'][frame_id] = observations[cam_name]['depth'].numpy()
                        if bool_use_wm_predictions:
                            hf['wm_predictions'][cam_name]['rgb'][frame_id] = common_utils.encode_image_from_float_to_uint8(
                                wm_predictions[cam_name]['rgb'].numpy()/2.0+0.5
                            )
                            hf['wm_predictions'][cam_name]['depth'][frame_id] = wm_predictions[cam_name]['depth'].numpy()
                            hf['wm_predictions'][cam_name]['gripper'][frame_id] = wm_predictions[cam_name]['gripper']
                    
                    obs_item = low_dim_obs._observations[frame_id]
                    hf['low_dim_obs']['gripper_pose'][frame_id] = obs_item.gripper_pose
                    hf['low_dim_obs']['gripper_open'][frame_id] = obs_item.gripper_open
                    hf['low_dim_obs']['joint_positions'][frame_id] = obs_item.joint_positions
                    expert_traj = expert_info['trajectory']
                    expert_stage = expert_info['stage']
                    expert_target_position = expert_info['debug_info']['tar_position']
                    hf['low_dim_obs']['expert_trajectory'][frame_id] = expert_traj[0]
                    hf['low_dim_obs']['expert_trajectory_stage'][frame_id] = np.array(expert_stage)
                    if np.array(expert_target_position).shape == (2,3):
                        expert_target_position = np.array(expert_target_position)[0]
                    else:
                        expert_target_position = np.array(expert_target_position)
                    hf['low_dim_obs']['expert_trajectory_target_position'][frame_id] = expert_target_position

        print(f"Dataset packaged successfully")
