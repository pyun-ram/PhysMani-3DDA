"""WorldModel 客户端 - 运行在 conda_env_a (Python 3.8)"""
import logging
import torch
from typing import Dict, Any, List, Optional
from pathlib import Path

from .ipc_core import WebsocketClientPolicy
from utils.utils_with_rlbench import (
    get_cam_param,
    get_cam_param_from_pkl,
    convert_cam_param_to_gs_format,
)

logging.basicConfig(level=logging.INFO, format='[WORLD_CLIENT] %(message)s')


class _DummyModel:
    """伪模型对象，用于支持 _model.network_t 访问"""
    def __init__(self):
        self.network_t = None  # 从服务器同步


class WebsocketClientWorld(WebsocketClientPolicy):
    """WorldModel WebSocket 客户端，复用 WebsocketClientPolicy 的基础功能"""
    pass


class WorldModelClient:
    """WorldModel 客户端，兼容现有 WorldModel 接口"""
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8766,
        logger=None,
        device: torch.device = None,
        episode_dir: Path = None,
        cam_names: List[str] = None,
        num_init_gaussian: int = 25_000,
        max_iter_cano_gaussian: int = 3500,
        max_iter_gaussian: int = 10,
        max_iter_network: int = 10,
        num_future_frames: int = 30,
        bool_mask: bool = False,
        bool_wm_predict_velmappc: bool = False,
    ):
        """初始化 WorldModel 客户端
        
        Args:
            host: 服务器地址
            port: 服务器端口
            logger: Logger 对象
            device: torch.device
            episode_dir: episode 目录路径
            cam_names: 相机名称列表
            num_init_gaussian: 初始高斯数量
            max_iter_cano_gaussian: 规范高斯最大迭代次数
            max_iter_gaussian: 高斯最大迭代次数
            max_iter_network: 网络最大迭代次数
            num_future_frames: 未来帧数量
            bool_mask: 是否使用 mask
        """
        # 客户端本地属性
        self.logger = logger
        self.device = device
        self._cam_names = cam_names
        self.num_init_gaussian = num_init_gaussian
        self.max_iter_cano_gaussian = max_iter_cano_gaussian
        self.max_iter_gaussian = max_iter_gaussian
        self.max_iter_network = max_iter_network
        self.num_future_frames = num_future_frames
        self.bool_mask = bool_mask
        self.bool_wm_predict_velmappc = bool_wm_predict_velmappc
        
        # 保存 episode_dir 用于后续使用
        self._episode_dir = episode_dir
        
        # 保存 log_dir（从 logger 中获取，如果 logger 存在）
        self._log_dir = logger.log_dir if logger is not None else None
        
        # 本地计算 _views_dict（与 WorldModel 相同逻辑）
        self._views_dict = self.get_cam_views(episode_dir, cam_names)
        
        # 伪模型对象，用于支持 _model.network_t 访问
        self._model = _DummyModel()
        
        # 时间状态（客户端维护）
        self.last_t = 0.0
        self.cur_t = 0.0
        
        # WebSocket 客户端
        self._client = WebsocketClientWorld(host=host, port=port)
        
        # 发送初始化请求（传递配置参数）
        self._initialize_world_model()
    
    def get_cam_views(self, episode_dir: Path, cam_names: List[str]) -> Dict:
        """获取相机视图（完全复制 WorldModel 的逻辑）
        
        Args:
            episode_dir: episode 目录路径
            cam_names: 相机名称列表
            
        Returns:
            views_dict: 包含 views, viewmats, Ks 等的字典
        """
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
        """相机数量（property，与 WorldModel 兼容）"""
        return len(self._views_dict["views"])
    
    def _initialize_world_model(self):
        """初始化服务器端的 WorldModel"""
        logging.info("发送 WorldModel 初始化请求...")
        
        # 注意：episode_dir 在 __init__ 时传入，需要转换为字符串
        episode_dir_str = str(self._episode_dir) if self._episode_dir else None
        
        init_request = {
            "type": "initialize",
            "model_name": "world_model",  # 为兼容性保留
            "model_args": {
                "num_init_gaussian": self.num_init_gaussian,
                "max_iter_cano_gaussian": self.max_iter_cano_gaussian,
                "max_iter_gaussian": self.max_iter_gaussian,
                "max_iter_network": self.max_iter_network,
                "num_future_frames": self.num_future_frames,
                "bool_mask": self.bool_mask,
                "episode_dir": episode_dir_str,
                "cam_names": self._cam_names,
                "bool_wm_predict_velmappc": self.bool_wm_predict_velmappc,
                # logger 对象无法序列化，服务器端使用 None
            },
            "checkpoint": "",  # 为兼容性保留
            "device": str(self.device) if self.device else "cuda",
        }
        
        response = self._client.infer(init_request)
        
        if response.get("status") == "success":
            logging.info(f"WorldModel 初始化成功: {response.get('message')}")
        else:
            error_msg = response.get("message", "未知错误")
            raise RuntimeError(f"WorldModel 初始化失败: {error_msg}")
    
    def init(
        self,
        rgb: torch.Tensor,
        pcd: torch.Tensor,
        depth: torch.Tensor,
        gripper: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """初始化 WorldModel（远程调用）
        
        Args:
            rgb [1, num_cam, 3, H, W] (range [-1,1])
            pcd [1, num_cam, 3, H, W]
            depth [1, num_cam, 1, H, W] (metric in meters)
            gripper [8,]
            mask [1, num_cam, 3, H, W] (range [0-1]), default = None
        """
        # 发送 init 请求，传递 episode_dir 和 cam_names 以确保服务器端可以创建实例
        # 将 episode_dir 转换为绝对路径，以确保服务器端可以访问
        episode_dir_str = str(self._episode_dir.resolve()) if self._episode_dir else None
        
        # 将 log_dir 转换为字符串（如果存在）
        log_dir_str = str(self._log_dir) if self._log_dir is not None else None
        
        request = {
            "type": "init",
            "rgb": rgb,
            "pcd": pcd,
            "depth": depth,
            "gripper": gripper,
            "mask": mask,
            "episode_dir": episode_dir_str,  # 传递绝对路径
            "cam_names": self._cam_names,  # 传递 cam_names
            "log_dir": log_dir_str,  # 传递 log_dir
        }
        response = self._client.infer(request)
        
        # 检查状态
        if response.get("status") != "success":
            error_msg = response.get("message", "未知错误")
            raise RuntimeError(f"WorldModel init 失败: {error_msg}")
        
        # 同步 network_t（如果存在且不为 None）
        # 注意：在 init() 后，network_t 可能为 None，这是正常的，因为 transform_gaussian_to 还未被调用
        if "network_t" in response:
            network_t_value = response["network_t"]
            if network_t_value is not None:
                if isinstance(network_t_value, torch.Tensor):
                    self._model.network_t = network_t_value.to(self.device)
                else:
                    self._model.network_t = torch.tensor(network_t_value, device=self.device)
    
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
        """更新 WorldModel（远程调用）
        
        Args:
            rgb [num_cam, 3, H, W] (range [0-1])
            pcd [num_cam, 3, H, W]
            last_t: (float)
            cur_t: (float)
            depth [num_cam, 1, H, W] (metric in meters)
            mask [num_cam, 3, H, W] (range [0-1]), default = None
            gripper [8,]
            num_samples: (int)
        """
        request = {
            "type": "update",
            "rgb": rgb,
            "pcd": pcd,
            "last_t": last_t,
            "cur_t": cur_t,
            "depth": depth,
            "mask": mask,
            "gripper": gripper,
            "num_samples": num_samples,
        }
        response = self._client.infer(request)
        
        # 检查状态
        if response.get("status") != "success":
            error_msg = response.get("message", "未知错误")
            raise RuntimeError(f"WorldModel update 失败: {error_msg}")
        
        # 同步 network_t（如果存在且不为 None）
        # 注意：在 update() 后，network_t 应该有值（因为 transform_gaussian_to 被调用）
        if "network_t" in response:
            network_t_value = response["network_t"]
            if network_t_value is not None:
                if isinstance(network_t_value, torch.Tensor):
                    self._model.network_t = network_t_value.to(self.device)
                else:
                    self._model.network_t = torch.tensor(network_t_value, device=self.device)
    
    def predict(
        self,
        num_future_frames: int,
        dt: float,
    ) -> Dict:
        """预测未来帧（远程调用）
        
        Args:
            num_future_frames: (int)
            dt: (float)
        
        Returns:
            Dict[str, torch.Tensor]
                'image_list': [num_future_frames, num_cam, 3, H, W] (range [0-1])
                'depth_list': [num_future_frames, num_cam, 3, H, W]
                'semantic_mask_list': None
                'target_time_list': [num_future_frames]
                'eepose_list': [num_future_frames, 7]
                'openness_list': [num_future_frames, 1]
        """
        if not self.bool_wm_predict_velmappc:
            request = {
                "type": "predict",
                "num_future_frames": num_future_frames,
                "dt": dt,
            }
            response = self._client.infer(request)
            
            # 检查状态
            if response.get("status") != "success":
                error_msg = response.get("message", "未知错误")
                raise RuntimeError(f"WorldModel predict 失败: {error_msg}")
            
            # 返回结果字典（与 WorldModel.predict 相同格式）
            return {
                "image_list": response["image_list"],
                "depth_list": response["depth_list"],
                "semantic_mask_list": response.get("semantic_mask_list", None),
                "target_time_list": response["target_time_list"],
                "eepose_list": response["eepose_list"],
                "openness_list": response["openness_list"],
            }
        else:
            request = {
                "type": "predict",
                "num_future_frames": num_future_frames,
                "dt": dt,
            }
            response = self._client.infer(request)
            
            # 检查状态
            if response.get("status") != "success":
                error_msg = response.get("message", "未知错误")
                raise RuntimeError(f"WorldModel predict 失败: {error_msg}")
            
            # 返回结果字典（与 WorldModel.predict 相同格式）
            return {
                "velmappc_list": response["velmappc_list"],
            }

