"""WorldModel 服务器 - 运行在 conda_env_b (Python 3.10)"""
import logging
import os
import torch
import time
from pathlib import Path
from typing import Dict, Optional

from utils.utils_with_rlbench import WorldModel
from utils.common_utils import Logger
from .ipc_core import BasePolicy, WebsocketPolicyServer

logging.basicConfig(level=logging.INFO, format='[WORLD_SERVER] %(message)s')

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


class WorldModelServer(BasePolicy):
    """WorldModel 服务器实现"""
    def __init__(self):
        self._world_model = None
        self._device = None
        self._initialized = False
        self._init_error = None
        self._episode_dir = None
        self._cam_names = None

    def initialize(
        self,
        model_name: str = None,
        model_args: Dict = None,
        checkpoint: str = None,
        device: str = "cuda",
        **kwargs
    ) -> Dict:
        """初始化 WorldModel 配置并创建实例
        
        Args:
            model_name: 模型名称（未使用，为兼容性保留）
            model_args: 模型参数字典，包含：
                - num_init_gaussian: 初始高斯数量
                - max_iter_cano_gaussian: 规范高斯最大迭代次数
                - max_iter_gaussian: 高斯最大迭代次数
                - max_iter_network: 网络最大迭代次数
                - num_future_frames: 未来帧数量
                - bool_mask: 是否使用 mask
                - episode_dir: episode 目录路径（字符串）
                - cam_names: 相机名称列表
                - logger: Logger 对象（可选）
            checkpoint: checkpoint 路径（未使用，为兼容性保留）
            device: 设备 ("cuda" 或 "cpu")
            **kwargs: 其他参数（从请求中直接传递的参数）
        
        Returns:
            包含初始化状态的字典
        """
        try:
            self._device = torch.device(device)
            
            # 从 model_args 或 kwargs 中提取参数
            if model_args is None:
                model_args = {}
            
            # 合并 kwargs 到 model_args（kwargs 优先级更高）
            model_args = {**model_args, **kwargs}
            
            self._num_init_gaussian = model_args.get("num_init_gaussian", 25_000)
            self._max_iter_cano_gaussian = model_args.get("max_iter_cano_gaussian", 3500)
            self._max_iter_gaussian = model_args.get("max_iter_gaussian", 10)
            self._max_iter_network = model_args.get("max_iter_network", 10)
            self._num_future_frames = model_args.get("num_future_frames", 30)
            self._bool_mask = model_args.get("bool_mask", False)
            self._bool_wm_predict_velmappc = model_args.get("bool_wm_predict_velmappc", False)
            
            episode_dir = model_args.get("episode_dir")
            cam_names = model_args.get("cam_names")
            
            # 保存配置参数，但不立即创建 WorldModel 实例
            # 因为 episode_dir 路径可能是相对路径，且服务器和客户端的工作目录可能不同
            # 延迟到第一次 init() 调用时才创建实例，此时可以传递绝对路径或正确的相对路径
            if episode_dir is not None:
                # 保存为字符串，稍后使用
                self._episode_dir_str = str(episode_dir)
            else:
                self._episode_dir_str = None
            
            self._cam_names = cam_names if cam_names is not None else None
            
            logging.info("WorldModel 配置已保存，等待首次 init 调用时创建实例")
            
            self._initialized = True
            self._init_error = None
            
            logging.info("WorldModel 配置初始化完成")
            return {
                "status": "success",
                "message": "WorldModel 配置初始化成功",
                "device": str(self._device)
            }
        except Exception as e:
            self._initialized = False
            self._init_error = str(e)
            logging.error(f"WorldModel 配置初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"WorldModel 配置初始化失败: {str(e)}",
                "error": str(e)
            }

    def infer(self, request: Dict) -> Dict:
        """处理推理请求
        
        Args:
            request: 包含以下字段的字典:
                - "type": "init" | "update" | "predict"
                - 其他字段根据类型不同而不同
        
        Returns:
            包含结果的字典
        """
        if not self._initialized:
            raise RuntimeError("WorldModel 配置未初始化")
        
        request_type = request.get("type")
        
        if request_type == "init":
            return self._handle_init(request)
        elif request_type == "update":
            return self._handle_update(request)
        elif request_type == "predict":
            return self._handle_predict(request)
        else:
            raise ValueError(f"未知的请求类型: {request_type}")

    def _handle_init(self, request: Dict) -> Dict:
        """处理 init 请求"""
        logging.info("[_handle_init] 进入方法")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        try:
            # 每次 init 都重新创建 WorldModel 实例，避免遗留参数
            logging.info("[_handle_init] 重新创建 WorldModel 实例")
            
            # 清理旧的实例（如果有）
            if self._world_model is not None:
                logging.info("[_handle_init] 清理旧的 WorldModel 实例")
                del self._world_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 从请求中获取 episode_dir（如果提供），否则使用 initialize 时保存的
            episode_dir_str = request.get("episode_dir", self._episode_dir_str)
            cam_names = request.get("cam_names", self._cam_names)
            
            if episode_dir_str is None or cam_names is None:
                logging.info(f"[_handle_init] 退出方法 (错误: 缺少必要参数)")
                return {
                    "status": "error",
                    "message": f"无法创建 WorldModel 实例：缺少必要参数。episode_dir={episode_dir_str}, cam_names={cam_names}"
                }
            
            # 转换为 Path 对象，如果是相对路径，尝试转换为绝对路径
            episode_dir = Path(episode_dir_str)
            if not episode_dir.is_absolute():
                # 如果是相对路径，尝试相对于项目根目录
                episode_dir = PROJECT_ROOT / episode_dir
                # 如果还是不存在，尝试相对于当前工作目录
                if not episode_dir.exists():
                    episode_dir = Path(episode_dir_str).resolve()
            
            logging.info(f"创建 WorldModel 实例: episode_dir={episode_dir}, cam_names={cam_names}")
            
            # 创建 Logger 实例（logger 对象无法通过 WebSocket 传输，在服务器端创建）
            # log_dir 基于 episode_dir，如果请求中提供了 log_dir 则使用它
            log_dir_str = request.get("log_dir")
            if log_dir_str:
                log_dir = Path(log_dir_str)
                if not log_dir.is_absolute():
                    log_dir = PROJECT_ROOT / log_dir
            else:
                # 默认使用 episode_dir 的父目录下的 logs 目录
                log_dir = episode_dir.parent.parent / "logs" / episode_dir.name
            log_dir.mkdir(parents=True, exist_ok=True)
            logger = Logger(log_dir=log_dir, bool_enable=True)
            logging.info(f"Logger 已创建: log_dir={log_dir}")
            
            self._world_model = WorldModel(
                logger=logger,
                device=self._device,
                episode_dir=episode_dir,
                cam_names=cam_names,
                num_init_gaussian=self._num_init_gaussian,
                max_iter_cano_gaussian=self._max_iter_cano_gaussian,
                max_iter_gaussian=self._max_iter_gaussian,
                max_iter_network=self._max_iter_network,
                num_future_frames=self._num_future_frames,
                bool_mask=self._bool_mask,
                bool_wm_predict_velmappc=self._bool_wm_predict_velmappc
            )
            logging.info("WorldModel 实例创建成功")
            
            rgb = request["rgb"]
            pcd = request["pcd"]
            depth = request["depth"]
            gripper = request["gripper"]
            mask = request.get("mask", None)
            
            # 调用 init
            logging.info("[_handle_init] 调用 world_model.init()")
            self._world_model.init(rgb, pcd, depth, gripper, mask)
            
            # 返回 network_t（需要检查 _model 和 network_t 是否存在）
            network_t = None
            if self._world_model._model is not None:
                if hasattr(self._world_model._model, 'network_t') and self._world_model._model.network_t is not None:
                    network_t = self._world_model._model.network_t.cpu()
            
            logging.info(f"[_handle_init] 退出方法 (成功, network_t={network_t is not None})")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"[WORLD_SERVER] init: {(time.time() - t0)*1000:.2f}ms")
            return {
                "status": "success",
                "message": "WorldModel init 成功",
                "network_t": network_t,
            }
        except Exception as e:
            logging.error(f"WorldModel init 失败: {e}")
            logging.info(f"[_handle_init] 退出方法 (异常: {str(e)})")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"[WORLD_SERVER] init: {(time.time() - t0)*1000:.2f}ms")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"WorldModel init 失败: {str(e)}",
                "error": str(e)
            }

    def _handle_update(self, request: Dict) -> Dict:
        """处理 update 请求"""
        logging.info("[_handle_update] 进入方法")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        try:
            if self._world_model is None:
                logging.info("[_handle_update] 退出方法 (错误: WorldModel 未初始化)")
                return {
                    "status": "error",
                    "message": "WorldModel 未初始化，请先调用 init"
                }
            
            rgb = request["rgb"]
            pcd = request["pcd"]
            last_t = request["last_t"]
            cur_t = request["cur_t"]
            depth = request["depth"]
            mask = request.get("mask", None)
            gripper = request["gripper"]
            num_samples = request.get("num_samples", 3)
            
            logging.info(f"[_handle_update] 参数: last_t={last_t}, cur_t={cur_t}, num_samples={num_samples}")
            
            # 调用 update
            logging.info("[_handle_update] 调用 world_model.update()")
            self._world_model.update(
                rgb=rgb,
                pcd=pcd,
                last_t=last_t,
                cur_t=cur_t,
                depth=depth,
                mask=mask,
                gripper=gripper,
                num_samples=num_samples,
            )
            
            # 返回 network_t（需要检查 _model 和 network_t 是否存在）
            network_t = None
            if self._world_model._model is not None:
                if hasattr(self._world_model._model, 'network_t') and self._world_model._model.network_t is not None:
                    network_t = self._world_model._model.network_t.cpu()
            
            logging.info(f"[_handle_update] 退出方法 (成功, network_t={network_t is not None})")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"[WORLD_SERVER] update: {(time.time() - t0)*1000:.2f}ms")
            return {
                "status": "success",
                "message": "WorldModel update 成功",
                "network_t": network_t,
            }
        except Exception as e:
            logging.error(f"WorldModel update 失败: {e}")
            logging.info(f"[_handle_update] 退出方法 (异常: {str(e)})")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"[WORLD_SERVER] update: {(time.time() - t0)*1000:.2f}ms")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"WorldModel update 失败: {str(e)}",
                "error": str(e)
            }

    def _handle_predict(self, request: Dict) -> Dict:
        """处理 predict 请求"""
        logging.info("[_handle_predict] 进入方法")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        try:
            if self._world_model is None:
                logging.info("[_handle_predict] 退出方法 (错误: WorldModel 未初始化)")
                return {
                    "status": "error",
                    "message": "WorldModel 未初始化，请先调用 init"
                }
            
            num_future_frames = request["num_future_frames"]
            dt = request["dt"]
            
            logging.info(f"[_handle_predict] 参数: num_future_frames={num_future_frames}, dt={dt}")
            
            # 调用 predict
            logging.info("[_handle_predict] 调用 world_model.predict()")
            eval_dict = self._world_model.predict(
                num_future_frames=num_future_frames,
                dt=dt,
            )
            
            # 将结果中的 tensor 移到 CPU 以便序列化
            logging.info("[_handle_predict] 处理返回结果，将 tensor 移到 CPU")
            if not self._bool_wm_predict_velmappc:
                result = {
                    "status": "success",
                    "image_list": [img.cpu() if isinstance(img, torch.Tensor) else img for img in eval_dict["image_list"]],
                    "depth_list": [depth.cpu() if isinstance(depth, torch.Tensor) else depth for depth in eval_dict["depth_list"]],
                    "semantic_mask_list": eval_dict.get("semantic_mask_list", None),
                    "target_time_list": [t.cpu() if isinstance(t, torch.Tensor) else t for t in eval_dict["target_time_list"]],
                    "eepose_list": [eepose.cpu() if isinstance(eepose, torch.Tensor) else eepose for eepose in eval_dict["eepose_list"]],
                    "openness_list": [openness.cpu() if isinstance(openness, torch.Tensor) else openness for openness in eval_dict["openness_list"]],
                }
                logging.info(f"[_handle_predict] 退出方法 (成功, image_list长度={len(result['image_list'])}, depth_list长度={len(result['depth_list'])})")
            else:
                result = {
                    "status": "success",
                    "velmappc_list": [img.cpu() if isinstance(img, torch.Tensor) else img for img in eval_dict["velmappc_list"]],
                }
                logging.info(f"[_handle_predict] 退出方法 (成功, velmappc_list={len(result['velmappc_list'])})")
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"[WORLD_SERVER] predict: {(time.time() - t0)*1000:.2f}ms")
            return result
        except Exception as e:
            logging.error(f"WorldModel predict 失败: {e}")
            logging.info(f"[_handle_predict] 退出方法 (异常: {str(e)})")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"[WORLD_SERVER] predict: {(time.time() - t0)*1000:.2f}ms")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"WorldModel predict 失败: {str(e)}",
                "error": str(e)
            }


if __name__ == "__main__":
    import argparse
    #disable_deterministic_algorithms()    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8766, help="服务器端口")
    args = parser.parse_args()
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision('medium')
    world_model_server = WorldModelServer()
    server = WebsocketPolicyServer(world_model_server, host=args.host, port=args.port)
    print("🚀 WorldModel 服务器运行中...")
    server.serve_forever()
