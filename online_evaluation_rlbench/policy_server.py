"""策略服务器 - 运行在 conda_env_b (Python 3.10)"""
import logging
import os
import torch
import time
from pathlib import Path
from typing import Dict

from diffuser_actor.trajectory_optimization.diffuser_actor import DiffuserActor
from diffuser_actor.trajectory_optimization.foresight_diffuser_actor_v3 import ForesightDiffuserActorV3
from diffuser_actor.trajectory_optimization.foresight_diffuser_actor_v4 import ForesightDiffuserActorV4
from diffuser_actor.trajectory_optimization.foresight_diffuser_actor_v5 import ForesightDiffuserActorV5
from diffuser_actor.trajectory_optimization.foresight_diffuser_actor_v6 import ForesightDiffuserActorV6
from utils.common_utils import get_gripper_loc_bounds
from .ipc_core import BasePolicy, WebsocketPolicyServer

logging.basicConfig(level=logging.INFO, format='[SERVER] %(message)s')

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

class PolicyServer(BasePolicy):
    """策略服务器实现"""
    def __init__(self):
        self._model = None
        self._device = None
        self._initialized = False
        self._init_error = None
        self._model_name = None

    def initialize(self, model_name: str, model_args: Dict, checkpoint: str, device: str = "cuda") -> Dict:
        """初始化模型
        
        Args:
            model_name: 模型名称 ("3d_diffuser_actor" 或 "foresight_diffuser_actor_v3")
            model_args: 模型参数字典
            checkpoint: checkpoint 路径
            device: 设备 ("cuda" 或 "cpu")
        
        Returns:
            包含初始化状态的字典
        """
        try:
            self._device = torch.device(device)
            self._model_name = model_name
            logging.info(f"初始化模型: {model_name}, checkpoint: {checkpoint}")
            
            # 创建模型
            if model_name == "3d_diffuser_actor":
                self._model = DiffuserActor(
                    backbone=model_args['backbone'],
                    image_size=tuple(model_args['image_size']),
                    embedding_dim=model_args['embedding_dim'],
                    num_vis_ins_attn_layers=model_args['num_vis_ins_attn_layers'],
                    use_instruction=model_args['use_instruction'],
                    fps_subsampling_factor=model_args['fps_subsampling_factor'],
                    gripper_loc_bounds=model_args['gripper_loc_bounds'],
                    rotation_parametrization=model_args['rotation_parametrization'],
                    quaternion_format=model_args['quaternion_format'],
                    diffusion_timesteps=model_args['diffusion_timesteps'],
                    denoise_model=model_args['denoise_model'],
                    num_inference_steps=model_args['num_inference_steps'],
                    nhist=model_args['nhist'],
                    relative=model_args['relative'],
                    lang_enhanced=model_args['lang_enhanced'],
                ).to(self._device)
            elif model_name == "foresight_diffuser_actor_v3":
                self._model = ForesightDiffuserActorV3(
                    backbone=model_args['backbone'],
                    image_size=tuple(model_args['image_size']),
                    embedding_dim=model_args['embedding_dim'],
                    num_vis_ins_attn_layers=model_args['num_vis_ins_attn_layers'],
                    use_instruction=model_args['use_instruction'],
                    fps_subsampling_factor=model_args['fps_subsampling_factor'],
                    gripper_loc_bounds=model_args['gripper_loc_bounds'],
                    rotation_parametrization=model_args['rotation_parametrization'],
                    quaternion_format=model_args['quaternion_format'],
                    diffusion_timesteps=model_args['diffusion_timesteps'],
                    denoise_model=model_args['denoise_model'],
                    num_inference_steps=model_args['num_inference_steps'],
                    nhist=model_args['nhist'],
                    relative=model_args['relative'],
                    lang_enhanced=model_args['lang_enhanced'],
                    bool_classifier_free_guidance=model_args['bool_classifier_free_guidance'],
                    classifier_free_guidance_w=model_args['classifier_free_guidance_w'],
                    classifier_free_guidance_dropout_prob=model_args['classifier_free_guidance_dropout_prob'],
                ).to(self._device)
            elif model_name == 'foresight_diffuser_actor_v4':
                self._model = ForesightDiffuserActorV4(
                    backbone=model_args['backbone'],
                    image_size=tuple(model_args['image_size']),
                    embedding_dim=model_args['embedding_dim'],
                    num_vis_ins_attn_layers=model_args['num_vis_ins_attn_layers'],
                    use_instruction=model_args['use_instruction'],
                    fps_subsampling_factor=model_args['fps_subsampling_factor'],
                    gripper_loc_bounds=model_args['gripper_loc_bounds'],
                    rotation_parametrization=model_args['rotation_parametrization'],
                    quaternion_format=model_args['quaternion_format'],
                    diffusion_timesteps=model_args['diffusion_timesteps'],
                    denoise_model=model_args['denoise_model'],
                    num_inference_steps=model_args['num_inference_steps'],
                    nhist=model_args['nhist'],
                    relative=model_args['relative'],
                    lang_enhanced=model_args['lang_enhanced'],
                    prob_dropout_vel_features=0.0,
                ).to(self._device)
            elif model_name == 'foresight_diffuser_actor_v5':
                self._model = ForesightDiffuserActorV5(
                    backbone=model_args['backbone'],
                    image_size=tuple(model_args['image_size']),
                    embedding_dim=model_args['embedding_dim'],
                    num_vis_ins_attn_layers=model_args['num_vis_ins_attn_layers'],
                    use_instruction=model_args['use_instruction'],
                    fps_subsampling_factor=model_args['fps_subsampling_factor'],
                    gripper_loc_bounds=model_args['gripper_loc_bounds'],
                    rotation_parametrization=model_args['rotation_parametrization'],
                    quaternion_format=model_args['quaternion_format'],
                    diffusion_timesteps=model_args['diffusion_timesteps'],
                    denoise_model=model_args['denoise_model'],
                    num_inference_steps=model_args['num_inference_steps'],
                    nhist=model_args['nhist'],
                    relative=model_args['relative'],
                    lang_enhanced=model_args['lang_enhanced'],
                    prob_dropout_vel_features=0.0,
                ).to(self._device)
            elif model_name == 'foresight_diffuser_actor_v6':
                self._model = ForesightDiffuserActorV6(
                    backbone=model_args['backbone'],
                    image_size=tuple(model_args['image_size']),
                    embedding_dim=model_args['embedding_dim'],
                    num_vis_ins_attn_layers=model_args['num_vis_ins_attn_layers'],
                    use_instruction=model_args['use_instruction'],
                    fps_subsampling_factor=model_args['fps_subsampling_factor'],
                    gripper_loc_bounds=model_args['gripper_loc_bounds'],
                    rotation_parametrization=model_args['rotation_parametrization'],
                    quaternion_format=model_args['quaternion_format'],
                    diffusion_timesteps=model_args['diffusion_timesteps'],
                    denoise_model=model_args['denoise_model'],
                    num_inference_steps=model_args['num_inference_steps'],
                    nhist=model_args['nhist'],
                    relative=model_args['relative'],
                    lang_enhanced=model_args['lang_enhanced'],
                    prob_dropout_vel_features=0.0,
                    bool_use_gating_and_adapter=model_args['bool_use_gating_and_adapter'],
                ).to(self._device)
            else:
                raise ValueError(f"不支持的模型类型: {model_name}，仅支持 '3d_diffuser_actor' 和 'foresight_diffuser_actor_v3'")
            
            # 加载 checkpoint
            # 将相对路径转换为绝对路径（相对于项目根目录）
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = PROJECT_ROOT / checkpoint_path
            
            if checkpoint_path.is_file():
                model_dict = torch.load(str(checkpoint_path), map_location="cpu")
                model_dict_weight = {}
                for key in model_dict["weight"]:
                    _key = key[7:]  # 移除 "module." 前缀
                    model_dict_weight[_key] = model_dict["weight"][key]
                self._model.load_state_dict(model_dict_weight)
                logging.info(f"模型权重加载成功: {checkpoint_path}")
            else:
                assert False, f"Checkpoint 文件不存在: {checkpoint_path}"
            
            # 设置为 eval 模式
            self._model.eval()
            self._initialized = True
            self._init_error = None
            
            logging.info("模型初始化完成")
            return {
                "status": "success",
                "message": "模型初始化成功",
                "model_name": model_name,
                "device": str(self._device)
            }
        except Exception as e:
            self._initialized = False
            self._init_error = str(e)
            logging.error(f"模型初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"模型初始化失败: {str(e)}",
                "error": str(e)
            }

    def infer(self, obs: Dict) -> Dict:
        """推理接口
        
        Args:
            obs: 包含以下字段的字典:
                - "fake_traj": torch.Tensor
                - "traj_mask": torch.Tensor
                - "rgbs": torch.Tensor
                - "pcds": torch.Tensor
                - "instr": torch.Tensor
                - "gripper": torch.Tensor
                - "run_inference": bool
                - 其他 kwargs
        
        Returns:
            包含 "action" 字段的字典
        """
        if not self._initialized:
            raise RuntimeError("模型未初始化")
        
        if self._model is None:
            raise RuntimeError("模型未创建")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        print("Run inference ... ")
        # 提取参数
        fake_traj = obs["fake_traj"]
        traj_mask = obs["traj_mask"]
        rgbs = obs["rgbs"]
        pcds = obs["pcds"]
        instr = obs["instr"]
        gripper = obs["gripper"]
        run_inference = obs.get("run_inference", True)
        
        # 提取其他 kwargs（包括 next_rgb_obs, next_pcd_obs, next_mask_obs, next_gripper, next_frame_relative_id 等）
        kwargs = {k: v for k, v in obs.items() 
                 if k not in ["fake_traj", "traj_mask", "rgbs", "pcds", "instr", "gripper", "run_inference"]}
        
        # 确保张量在正确的设备上
        fake_traj = fake_traj.to(self._device)
        traj_mask = traj_mask.to(self._device)
        rgbs = rgbs.to(self._device)
        pcds = pcds.to(self._device)
        instr = instr.to(self._device)
        gripper = gripper.to(self._device)
        
        # 转换 kwargs 中的张量
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.to(self._device)
            elif v is None:
                # 保持 None 值不变（如 next_mask_obs 可能为 None）
                pass
        
        # 推理
        with torch.no_grad():
            output_dict = self._model(
                fake_traj,
                traj_mask,
                rgbs,
                pcds,
                instr,
                gripper,
                run_inference=run_inference,
                **kwargs
            )
        
        # 返回结果（确保 action 在 CPU 上以便序列化）
        result = {
            "action": output_dict["action"].cpu() if isinstance(output_dict["action"], torch.Tensor) else output_dict["action"]
        }
        
        # 如果 output_dict 中有其他字段，也返回
        for k, v in output_dict.items():
            if k != "action":
                if isinstance(v, torch.Tensor):
                    result[k] = v.cpu()
                else:
                    result[k] = v
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[SERVER] infer: {(time.time() - t0)*1000:.2f}ms")
        print("Inference done ... ")
        return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8765, help="服务器端口")
    args = parser.parse_args()
    
    policy = PolicyServer()
    server = WebsocketPolicyServer(policy, host=args.host, port=args.port)
    print("🚀 策略服务器运行中...")
    server.serve_forever()
