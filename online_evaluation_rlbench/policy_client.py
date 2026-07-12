"""策略客户端 - 运行在 conda_env_a (Python 3.8)"""
import logging
import torch
from typing import Dict, Any

from .ipc_core import WebsocketClientPolicy

logging.basicConfig(level=logging.INFO, format='[CLIENT] %(message)s')

class PolicyClient:
    """策略客户端，兼容现有模型接口"""
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        model_name: str = "3d_diffuser_actor",
        model_args: Dict[str, Any] = None,
        model_checkpoint: str = "",
        device: str = "cuda"
    ):
        """初始化策略客户端
        
        Args:
            host: 服务器地址
            port: 服务器端口
            model_name: 模型名称
            model_args: 模型参数字典
            model_checkpoint: checkpoint 路径
            device: 设备（服务器端使用）
        """
        self._model_name = model_name
        self._model_args = model_args or {}
        self._model_checkpoint = model_checkpoint
        self._device_str = device
        self._device = torch.device(device)
        
        # 创建 WebSocket 客户端
        self._client = WebsocketClientPolicy(host=host, port=port)
        
        # 发送初始化请求
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化服务器端的模型"""
        logging.info("发送模型初始化请求...")
        
        init_request = {
            "type": "initialize",
            "model_name": self._model_name,
            "model_args": self._model_args,
            "checkpoint": self._model_checkpoint,
            "device": self._device_str  # 使用字符串而不是 torch.device 对象
        }
        
        response = self._client.infer(init_request)
        
        if response.get("status") == "success":
            logging.info(f"模型初始化成功: {response.get('message')}")
            logging.info(f"模型: {response.get('model_name')}, 设备: {response.get('device')}")
        else:
            error_msg = response.get("message", "未知错误")
            raise RuntimeError(f"模型初始化失败: {error_msg}")
    
    def __call__(self, *args, **kwargs):
        """调用模型进行推理
        
        仅支持 DiffuserActor 模式的调用:
        policy(fake_traj, traj_mask, rgbs, pcds, instr, gripper, run_inference=True, **kwargs)
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            包含 "action" 字段的字典
        """
        # 检查是否是 Act3D 模式的调用（不支持）
        # Act3D 模式: policy(rgbs, pcds, instr, gripper, **kwargs)
        # 如果 args 数量为 4 且没有 fake_traj 和 traj_mask，可能是 Act3D 模式
        if len(args) == 4 and "fake_traj" not in kwargs and "traj_mask" not in kwargs:
            raise NotImplementedError("PolicyClient 不支持 Act3D 模式，仅支持 DiffuserActor 模式")
        
        # DiffuserActor 模式: policy(fake_traj, traj_mask, rgbs, pcds, instr, gripper, run_inference=True, **kwargs)
        if len(args) < 6:
            raise ValueError(f"DiffuserActor 模式需要至少 6 个参数，收到 {len(args)} 个")
        
        fake_traj, traj_mask, rgbs, pcds, instr, gripper = args[:6]
        run_inference = kwargs.pop("run_inference", True)
        
        # 构建推理请求
        obs = {
            "fake_traj": fake_traj,
            "traj_mask": traj_mask,
            "rgbs": rgbs,
            "pcds": pcds,
            "instr": instr,
            "gripper": gripper,
            "run_inference": run_inference,
            **kwargs
        }
        
        # 发送推理请求
        result = self._client.infer(obs)
        
        # 确保返回的 action 是 torch.Tensor
        if "action" in result:
            if isinstance(result["action"], torch.Tensor):
                return result
            else:
                # 如果是 numpy array，转换为 tensor
                result["action"] = torch.from_numpy(result["action"])
                return result
        else:
            raise RuntimeError("服务器返回结果中缺少 'action' 字段")
    
    def eval(self):
        """设置为评估模式（兼容性方法，无需实际实现）"""
        pass
    
    def prepare_action(self, output_dict: Dict) -> torch.Tensor:
        raise NotImplementedError
    
    @property
    def device(self):
        return self._device

