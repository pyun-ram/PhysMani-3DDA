from torch import nn
import torch
from typing import Dict, Tuple, Optional
from pathlib import Path
import random
from functorch import vmap, jacrev
from utils import gs_utils, metric_utils, common_utils, utils_with_rlbench
import einops
import torch.nn.functional as F
import pytorch3d.ops
from typing import List
import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

def compute_icp_svd(
    source: torch.Tensor,
    target: torch.Tensor,
)-> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Args:
        source: torch.Tensor, shape (N, 3)
        target: torch.Tensor, shape (N, 3)
    Returns:
        R: torch.Tensor, shape (3, 3)
        t: torch.Tensor, shape (3,)
    '''
    # 计算质心
    source_center = source.mean(dim=0)
    target_center = target.mean(dim=0)
    # 去中心化
    source_centered = source - source_center
    target_centered = target - target_center
    # SVD 求解旋转
    H = source_centered.t() @ target_centered
    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    R = Vh.t() @ U.t()
    # 保证右手系
    if torch.det(R) < 0:
        Vh[-1, :] *= -1
        R = Vh.t() @ U.t()
    # 平移
    t = target_center - R @ source_center
    return R.float(), t.float()

def sample_for_rigidity_loss(
    canon_means_full: torch.Tensor,
    deform_means_full: torch.Tensor,
    weights_full: Optional[torch.Tensor] = None,
    sample_size: int = 2000,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Sample canonical and deformed means for rigidity loss computation.
    Sampling is done outside torch.compile to avoid CUDAGraphs issues.
    
    Args:
        canon_means_full: (N, 3) canonical gaussian centers
        deform_means_full: (N, 3) deformed gaussian centers (canonical + dxyz)
        sample_size: target number of samples (default 2000)
    
    Returns:
        (canon_means_sampled, deform_means_sampled, weights_sampled)
    """
    N = canon_means_full.shape[0]
    M = int(min(max(sample_size, 2), N))
    
    if M < N:
        idx = torch.randperm(N, device=canon_means_full.device)[:M]
        canon_means_sampled = canon_means_full.detach()[idx]
        deform_means_sampled = deform_means_full[idx]
        weights_sampled = None if weights_full is None else weights_full.detach()[idx]
    else:
        canon_means_sampled = canon_means_full.detach()
        deform_means_sampled = deform_means_full
        weights_sampled = None if weights_full is None else weights_full.detach()
    
    return canon_means_sampled, deform_means_sampled, weights_sampled

class FreeGave(nn.Module):
    def __init__(
            self,
            cano_gaussians: gs_utils.GaussianModel,
            cano_t: float,
            num_control_nodes: int,
            lambda_dssim: float,
            lambda_depth: float,
            lambda_rigid: float,
            rigid_k: int,
            position_lr_init: float,
            position_lr_final: float,
            position_lr_delay_mult: float,
            position_lr_max_steps: float,
            feature_lr: float,
            opacity_lr: float,
            scaling_lr: float,
            rotation_lr: float,
            spatial_lr_scale: float,
            control_node_lr: float,
            percent_dense: float,
            roi_min: Optional[List[float]] = None,
            roi_max: Optional[List[float]] = None,
            ckpt_path: str=None,
            network_ckpt_path: str=None,
        ):
        '''
        Args:
            cano_gaussians: gs_utils.GaussianModel
            cano_t: float,
            dt: float,
            num_control_nodes: int,
            lambda_dssim: float,
            position_lr_init: float,
            position_lr_final: float,
            position_lr_delay_mult: float,
            position_lr_max_steps: float,
            feature_lr: float,
            opacity_lr: float,
            scaling_lr: float,
            rotation_lr: float,
            spatial_lr_scale: float,
            control_node_lr: float,
            percent_dense: float,
            ckpt_path: str="",
        '''
        super().__init__()
        self._cano_gaussians = None
        self._cano_t = None
        self._network = None
        self._control_nodes = None
        self._optim_network = None
        self._optim_gaussian = None
        self._optim_cnode = None
        self.dxyz = None
        self.drot = None
        self.network_t = None
        self.dxyz_node = None
        self.drot_node = None
        self.eepose = None
        self.openness = None
        self.background = torch.tensor(
            [1, 1, 1],
            dtype=torch.float32,
            device="cuda",
        )
        # 预分配 xyzt tensor，根据 num_control_nodes 的大小
        # 预分配连续的 CUDA tensor，确保内存布局最优
        self._cudagraph_dict = {
            '_compute_increment_dxyz_drot_xyzt': torch.empty(
                num_control_nodes, 4, dtype=torch.float32, device=self.background.device).contiguous(),
        }
        self.lambda_dssim = lambda_dssim
        self.lambda_depth = lambda_depth
        self.lambda_rigid = lambda_rigid
        self.rigid_k = rigid_k
        # ROI config (default to hardcoded bbox from comment)
        self.roi_min = roi_min if roi_min is not None else [-0.6, -0.6, 0.5]
        self.roi_max = roi_max if roi_max is not None else [1.2, 0.6, 0.9]
        self.gaussian_weights = None  # (N,) on device
        self.init_gaussians(cano_gaussians, cano_t)
        self.init_network()
        self.init_control_nodes(num_control_nodes)
        self.init_optimizer(
            position_lr_init=position_lr_init,
            position_lr_final=position_lr_final,
            position_lr_delay_mult=position_lr_delay_mult,
            position_lr_max_steps=position_lr_max_steps,
            feature_lr=feature_lr,
            opacity_lr=opacity_lr,
            scaling_lr=scaling_lr,
            rotation_lr=rotation_lr,
            spatial_lr_scale=spatial_lr_scale,
            control_node_lr=control_node_lr,
            percent_dense=percent_dense,
        )
        if ckpt_path is not None and Path(ckpt_path).exists():
            self.load_ckpt(ckpt_path)
            print(f"Load from {ckpt_path}")
        if network_ckpt_path is not None and Path(network_ckpt_path).exists():
            self.load_network_ckpt(network_ckpt_path)
            print(f"Load network weights from {network_ckpt_path}")
        
        return

    @property
    def num_gaussian(self):
        return self._cano_gaussians.get_xyz.shape[0]

    def init_optimizer(
        self,
        position_lr_init,
        position_lr_final,
        position_lr_delay_mult,
        position_lr_max_steps,
        feature_lr,
        opacity_lr,
        scaling_lr,
        rotation_lr,
        spatial_lr_scale,
        control_node_lr,
        percent_dense,
    ):
        position_lr_init = position_lr_init * 2
        l_adam = [
            {'params': list(self._network.vel_weight.parameters()),
                'lr': position_lr_init/4, 'name': 'vel'},
            {'params': 
                list(self._network.code_linear.parameters()) + \
                list(self._network.code_output.parameters()) + \
                list(self._network.code_seg.parameters()),
                'lr': position_lr_init/4 * spatial_lr_scale, 'name': 'deform'}
        ]
        if hasattr(torch, 'compile'):
            self._optim_network = torch.optim.Adam(l_adam, lr=0.0, eps=1e-15, fused=True)
        else:
            self._optim_network = torch.optim.Adam(l_adam, lr=0.0, eps=1e-15, fused=False)
        self._params_network = [
            p
            for group in self._optim_network.param_groups
            for p in group["params"]
        ]
        self._cano_gaussians.training_setup(
            percent_dense=percent_dense,
            position_lr_init=position_lr_init,
            position_lr_final=position_lr_final,
            position_lr_delay_mult=position_lr_delay_mult,
            position_lr_max_steps=position_lr_max_steps,
            feature_lr=feature_lr,
            opacity_lr=opacity_lr,
            scaling_lr=scaling_lr,
            rotation_lr=rotation_lr
        )
        self._optim_gaussian = self._cano_gaussians.optimizer
        # don't change scaling parameter
        for group in self._optim_gaussian.param_groups:
            if group["name"] == "scaling":
                group["lr"] = 0.0
        self._params_gaussian = [
            p
            for group in self._optim_gaussian.param_groups
            for p in group["params"]
        ]
        l = [{'params': self._control_nodes.parameters(), 'lr': control_node_lr , "name": "nodes"}]
        if hasattr(torch, 'compile'):
            self._optim_cnode = torch.optim.Adam(l, lr=0.0, eps=1e-15, fused=True)
        else:
            self._optim_cnode = torch.optim.Adam(l, lr=0.0, eps=1e-15, fused=False)
        return

    def init_network(self):
        m = FreeGavePhysicsNetwork(D=4, W=128, input_ch=3, output_ch=16, multires=8)
        if hasattr(torch, 'compile'):
            self._network = torch.compile(m, mode='reduce-overhead')
            self.compute_dxyz_drot = torch.compile(self.compute_dxyz_drot, mode='reduce-overhead')
            self.render_image_gsplat = torch.compile(self.render_image_gsplat, mode='reduce-overhead')
            self.compute_render_loss = torch.compile(self.compute_render_loss, mode='reduce-overhead')
        else:
            self._network = m
        return

    def init_gaussians(
            self,
            gaussians: gs_utils.GaussianModel,
            t: int,
        ) -> None:
        '''
        Args:
            gaussians: gs_utils.GaussianModel,
            t: int
        '''
        self._cano_gaussians = gaussians
        self._cano_t = t
        return

    def init_control_nodes(self, num_control_nodes) -> None:
        m = ControlNodes(K=3)
        self._control_nodes = m
        indices = common_utils.sample_points(
            N=self._cano_gaussians.get_xyz.shape[0],
            target_N=num_control_nodes)
        self._control_nodes.init(self._cano_gaussians.get_xyz[indices].cpu())
        return

    def save_ckpt(self, path: str) -> None:
        """
        Save the model, optimizer states, and Gaussian parameters to a checkpoint file.
        """
        checkpoint = {
            "network_state_dict": self._network.state_dict(),  # 保存网络参数
            "control_nodes_state_dict": self._control_nodes.state_dict(),  # 保存控制节点参数
            "optim_network_state_dict": self._optim_network.state_dict(),  # 保存网络优化器状态
            "optim_gaussian_state_dict": self._optim_gaussian.state_dict(),  # 保存高斯优化器状态
            "optim_cnode_state_dict": self._optim_cnode.state_dict(),  # 保存控制节点优化器状态
        }
        checkpoint['control_nodes_tensor_dict'] = {
            "node_weights": self._control_nodes._node_weights.clone(),
            "node_indices": self._control_nodes._node_indices.clone(),
        }
        # 只有当变形跟踪器属性不为None时才保存
        if (self.network_t is not None and self.dxyz is not None and 
            self.drot is not None and self.dxyz_node is not None and 
            self.drot_node is not None):
            checkpoint['deform_tracker_tensor_dict'] = {
                "network_t": self.network_t.clone(),
                "dxyz": self.dxyz.clone(),
                "drot": self.drot.clone(),
                "dxyz_node": self.dxyz_node.clone(),
                "drot_node": self.drot_node.clone(),
            }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
        # save gaussian ckpt
        name = f"{Path(path).stem}_gaussian.pkl" 
        path = Path(path).parent / name
        self._cano_gaussians.save_ckpt(str(path))
        print(f"Gaussians saved to {path}")
        return

    def load_ckpt(self, path: str) -> None:
        """
        Load the model, optimizer states, and Gaussian parameters from a checkpoint file.
        """
        checkpoint = torch.load(path, map_location=torch.device('cpu'))  # 加载到 CPU
        device = next(self._network.parameters()).device
        self._network.load_state_dict(checkpoint["network_state_dict"])  # 加载网络参数
        # 加载变形跟踪器状态（如果存在）
        if "deform_tracker_tensor_dict" in checkpoint:
            deform_tracker = checkpoint["deform_tracker_tensor_dict"]
            self.network_t = deform_tracker['network_t'].to(device)
            self.dxyz = deform_tracker['dxyz'].to(device)
            self.drot = deform_tracker['drot'].to(device)
            self.dxyz_node = deform_tracker['dxyz_node'].to(device)
            self.drot_node = deform_tracker['drot_node'].to(device)
        else:
            # 如果checkpoint中没有变形跟踪器状态，初始化为None
            print("No deform tracker in checkpoint")
            self.network_t = None
            self.dxyz = None
            self.drot = None
            self.dxyz_node = None
            self.drot_node = None
        
        # 初始化控制节点的权重和索引（如果需要）
        self._control_nodes._node_weights =  checkpoint['control_nodes_tensor_dict']['node_weights'].to(device)
        self._control_nodes._node_indices = checkpoint['control_nodes_tensor_dict']['node_indices'].to(device)
        self._control_nodes.load_state_dict(checkpoint["control_nodes_state_dict"])  # 加载控制节点参数
        self._optim_network.load_state_dict(checkpoint["optim_network_state_dict"])  # 加载网络优化器状态
        self._optim_gaussian.load_state_dict(checkpoint["optim_gaussian_state_dict"])  # 加载高斯优化器状态
        self._optim_cnode.load_state_dict(checkpoint["optim_cnode_state_dict"])  # 加载控制节点优化器状态
        print(f"Checkpoint loaded from {path}")
        # load gaussian ckpt
        name = f"{Path(path).stem}_gaussian.pkl" 
        path = Path(path).parent / name
        self._cano_gaussians.load_ckpt(str(path))
        print(f"Gaussians loaded from {path}")
        return

    def load_network_ckpt(self, path: str) -> None:
        """
        Load the model, optimizer states, and Gaussian parameters from a checkpoint file.
        """
        checkpoint = torch.load(path, map_location=torch.device('cpu'))  # 加载到 CPU
        if 'model_state_dict' in checkpoint:
            self._network.load_state_dict(checkpoint['model_state_dict'])
        if 'network_state_dict' in checkpoint:
            self._network.load_state_dict(checkpoint['network_state_dict'])
        print(f"Checkpoint loaded from {path}")
        return


    def train_network_one_step(self, data_dict: Dict, num_samples: int=1, idx_iter: int=0) -> Dict:
        '''
        Args:
            data_dict:
                "images": torch.Tensor (M, 3, H, W) [0,1],
                "depths": torch.Tensor (M, 1, H, W) ,
                "pcds": torch.Tensor (M, 3, H, W) ,
                "views": List[Dict] (M, ),
                "frame_ids": List[int] (M,),
                "cam_names": List[str] (M,),
        Return:
            out_dict (Dict)
                "loss"
                "dxyz_node": torch.Tensor, (num_node, 3)
                "drot_node": torch.Tensor, (num_node, 4)
                "dxyz": torch.Tensor, (num_gaussian, 3)
                "drot: torch.Tensor, (num_gaussian, 4)
        '''
        self.train()
        self._optim_network.zero_grad()
        num_views = data_dict['viewmats'].shape[0]
        sample_indices = torch.randperm(num_views)[:num_samples]
        loss, aux = self._train_network_one_step_core(data_dict, sample_indices=sample_indices)
        grad_norm = torch.nn.utils.clip_grad_norm_(self._params_network, 1.0)
        if grad_norm > 10 or grad_norm.isnan():
            print("Warning: NaN or too large grad norm detected — zeroing grad and skipping step")
            self._optim_network.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            return {
                "loss": loss,
                "dxyz_node": aux["dxyz_node"],
                "drot_node": aux["drot_node"],
                "dxyz": aux["dxyz"],
                "drot": aux["drot"],
            }
        self._optim_network.step()
        return {
            "loss": loss,
            "dxyz_node": aux["dxyz_node"],
            "drot_node": aux["drot_node"],
            "dxyz": aux["dxyz"],
            "drot": aux["drot"],
        }

    def _train_network_one_step_core(self, data_dict: Dict, sample_indices: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        '''
        Core training step: forward + backward (without optimizer operations)
        This function is compiled to optimize the forward and backward pass.
        '''
        loss, aux = self.forward_multi_sample(data_dict, sample_indices=sample_indices)
        loss.backward()
        return loss, aux

    def forward_multi_sample(self, data_dict: Dict, sample_indices: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        idx = sample_indices[0]
        dt = data_dict['dt'][idx]
        t = data_dict['frame_ids'][idx] - self._cano_t - dt
        xyz_node = self._control_nodes.nodes
        dxyz_node = self.dxyz_node \
            if self.dxyz_node is not None \
            else torch.zeros_like(xyz_node)
        if self.drot_node is None:
            drot_node = torch.zeros(
                (xyz_node.shape[0], 4),
                device=xyz_node.device, dtype=torch.float32)
            drot_node[:, 0] = 1
        else:
            drot_node = self.drot_node
        dxyz_, drot_, dxyz_node_, drot_node_ = self.compute_dxyz_drot(
            t,
            dt,
            dxyz_node=dxyz_node,
            drot_node=drot_node,
        )
        dxyz = dxyz_.clone()
        drot = drot_.clone()
        dxyz_node = dxyz_node_.clone()
        drot_node = drot_node_.clone()
        # Batch render all views at once        
        # Sample for rigidity loss OUTSIDE of torch.compile to avoid CUDAGraphs issues
        if self.lambda_rigid > 0.0:
            canon_means_full = self._cano_gaussians.get_xyz
            deform_means_full = canon_means_full + dxyz
            canon_means_sampled, deform_means_sampled, rigid_weights_sampled = sample_for_rigidity_loss(
                canon_means_full=canon_means_full,
                deform_means_full=deform_means_full,
                weights_full=self.gaussian_weights,
            )
        else:
            canon_means_sampled, deform_means_sampled, rigid_weights_sampled = None, None, None

        # Render RGB + Weight + Depth in one pass if weights are available
        weight_image = None
        if self.gaussian_weights is not None:
            # 1. 获取 SH DC 分量并转换为 RGB
            sh_dc = self._cano_gaussians.get_features  # (N, 1, 3) or (N, 3)
            if sh_dc.dim() == 3:
                sh_dc = sh_dc[:, 0, :]  # (N, 3)
            rgb = gs_utils.SH2RGB(sh_dc)  # (N, 3)
            
            # 2. 拼接 weight
            colors_with_weight = torch.cat([
                rgb, 
                self.gaussian_weights.unsqueeze(-1)
            ], dim=-1)  # (N, 4)
            
            # 3. 单次渲染（RGB + Weight + Depth）
            render_dict = self.render_image_gsplat(
                viewmats=data_dict['viewmats'][sample_indices],
                Ks=data_dict['Ks'][sample_indices],
                image_height=data_dict['image_height'],
                image_width=data_dict['image_width'],
                bg_color=self.background,
                dxyz=dxyz,
                drot=drot,
                colors_precomputed=colors_with_weight,  # 传入预计算的颜色
            )
            
            # 4. 提取结果
            image = render_dict['image']  # (B, 3, H, W)
            depth = render_dict['depth']  # (B, 1, H, W)
            weight_image = render_dict['weight'].detach()  # (B, 1, H, W)
        else:
            # 原有逻辑：只渲染 RGB + Depth
            render_dict = self.render_image_gsplat(
                viewmats=data_dict['viewmats'][sample_indices],
                Ks=data_dict['Ks'][sample_indices],
                image_height=data_dict['image_height'],
                image_width=data_dict['image_width'],
                bg_color=self.background,
                dxyz=dxyz,
                drot=drot,
            )
            image = render_dict['image']
            depth = render_dict['depth']

        # Compute loss with batched inputs
        loss = self.compute_render_loss(
            image=image,
            depth=depth,
            gt_image=data_dict['images'][sample_indices],
            gt_depth=data_dict['depths'][sample_indices],
            lambda_dssim=self.lambda_dssim,
            lambda_depth=self.lambda_depth,
            weight_image=weight_image.detach(),
            canon_means_sampled=canon_means_sampled,
            deform_means_sampled=deform_means_sampled,
            rigid_weights_sampled=rigid_weights_sampled,
            lambda_rigid=self.lambda_rigid,
            rigid_k=self.rigid_k,
        )
        aux = {
            "dxyz_node": dxyz_node,
            "drot_node": drot_node,
            "dxyz": dxyz,
            "drot": drot,
        }
        return loss, aux

    def train_gaussian_one_step(
            self,
            data_dict: Dict,
            num_samples: int=1,
            dxyz_nograd: Optional[torch.Tensor]=None,
            drot_nograd: Optional[torch.Tensor]=None,
        ) -> Dict:
        '''
        Args:
            data_dict:
                "images": torch.Tensor (M, 3, H, W) [0,1],
                "depths": torch.Tensor (M, 1, H, W) ,
                "pcds": torch.Tensor (M, 3, H, W) ,
                "views": List[Dict] (M, ),
                "frame_ids": List[int] (M,),
                "cam_names": List[str] (M,),
            dxyz_nograd: Optional[torch.Tensor], if not None (N, 3)
            drot_nograd: Optional[torch.Tensor], if not None (N, 4)
        Return:
            out_dict (Dict)
                "loss"
        '''
        self.train()
        self._optim_gaussian.zero_grad()
        sample_indices = random.choices(range(len(data_dict['cam_names'])), k=num_samples)
        render_dict = self.render_image_gsplat(
            viewmats=data_dict['viewmats'][sample_indices],
            Ks=data_dict['Ks'][sample_indices],
            image_height=data_dict['image_height'],
            image_width=data_dict['image_width'],
            bg_color=self.background,
            dxyz=dxyz_nograd,
            drot=drot_nograd,
            colors_precomputed=None,
        )
        loss = self.compute_render_loss(
            image=render_dict['image'],
            depth=render_dict['depth'],
            gt_image=data_dict['images'][sample_indices],
            gt_depth=data_dict['depths'][sample_indices],
            lambda_dssim=self.lambda_dssim,
            lambda_depth=self.lambda_depth,
            weight_image=None,
            canon_means_sampled=None,
            deform_means_sampled=None,
            rigid_weights_sampled=None,
            lambda_rigid=0.0,
            rigid_k=20,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self._params_gaussian, 1.0)
        if grad_norm > 10 or grad_norm.isnan():
            print("Warning: NaN grad norm detected — zeroing grad and skipping step")
            self._optim_gaussian.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            return {
                "loss": loss,
            }
        # print(f'gaussian grad_norm: {grad_norm :.3f}, loss: {loss.item():.3f}')
        self._optim_gaussian.step()
        return {
            "loss": loss,
        }
    
    def transform_gaussian_to(
        self,
        t: torch.Tensor,
        dxyz: torch.Tensor,
        drot: torch.Tensor,
        dxyz_node: torch.Tensor,
        drot_node: torch.Tensor,
        eepose: torch.Tensor,
        openness: torch.Tensor,
    ) -> None:
        '''
        Update deformation tracker to time t
        Args:
            t: torch.Tensor, (1,)
            dxyz: torch.Tensor, (N, 3)
            drot: torch.Tensor, (N, 4)
            dxyz_node: torch.Tensor, (M, 3)
            drot_node: torch.Tensor, (M, 3)
        '''
        self.network_t = t - self._cano_t
        self.dxyz = dxyz
        self.drot = drot
        self.dxyz_node = dxyz_node
        self.drot_node = drot_node
        self.eepose = eepose
        self.openness = openness
        return

    def evaluate(
        self,
        data_dict: Dict,
        deform_tracker: Dict,
        bool_canonical: bool,
        cam_names_eval: List,
        metric_logger=None,
    ) -> Dict:
        '''
        Args:
            data_dict: Dict,
            deform_tracker: Dict,
                "network_t": torch.Tensor (1)
                "dxyz_node": torch.Tensor (num_node, 3)
                "drot_node": torch.Tensor (num_node, 4)
            bool_canonical: bool, True, if only evaluate canonical gaussian.
        Return:
            "psnr_list": List[float]
            "image_list": List[torch.Tensor]
            "depth_list": List[torch.Tensor]
            "velmap_list": List[torch.Tensor]
            "gt_image_list": List[torch.Tensor]
            "gt_depth_list": List[torch.Tensor]
            "network_t": torch.Tensor(1,)
            "dxyz": torch.Tensor(num_gaussian, 3)
            "drot": torch.Tensor(num_gaussian, 4)
            "dxyz_node": torch.Tensor(num_node, 3)
            "drot_node": torch.Tensor(num_node, 4)
        '''
        self.eval()
        if bool_canonical:
            dxyz = drot = None
            eepose = None
            openness = None
            gs_dict_add = None
            velmap_pc = None
        else:
            t = deform_tracker['network_t']
            dt = data_dict['dt'][0]
            with torch.no_grad():
                if metric_logger is not None:
                    metric_logger.tik('extropolate')
                dxyz_, drot_, dxyz_node_, drot_node_ = self.compute_dxyz_drot(
                    t,
                    dt,
                    dxyz_node=deform_tracker['dxyz_node'],
                    drot_node=deform_tracker['drot_node'],
                )
                dxyz = dxyz_.clone()
                drot = drot_.clone()
                dxyz_node = dxyz_node_.clone()
                drot_node = drot_node_.clone()
                eepose, openness = self.get_eepose_openness(
                    current_xyzs=self._cano_gaussians.get_xyz + deform_tracker['dxyz'],
                    current_eepose=deform_tracker['eepose'],
                    target_xyzs=self._cano_gaussians.get_xyz + dxyz,
                    mask=self._cano_semantic_mask,
                )
                if eepose.isnan().any() or eepose.isinf().any():
                    raise ValueError('eepose is nan or inf')
                if metric_logger is not None:
                    metric_logger.tok('extropolate')
                velmap = self._network.get_vel_map(
                    deform_seg=self._network(self._cano_gaussians.get_xyz),
                    xyzt=torch.cat([self._cano_gaussians.get_xyz, t.repeat(self._cano_gaussians.get_xyz.shape[0], 1)], dim=1))
                # vel = self._network.get_vel(
                #     deform_seg=self._network(self._cano_gaussians.get_xyz),
                #     xyzt=torch.cat([self._cano_gaussians.get_xyz, t.repeat(self._cano_gaussians.get_xyz.shape[0], 1)], dim=1))
                sh_dc = self._cano_gaussians.get_features  # (N, 1, 3)
                rgb = gs_utils.SH2RGB(sh_dc[:, 0, :])  # (N, 3)
                velmap_pc = torch.cat([self._cano_gaussians.get_xyz + deform_tracker['dxyz'], velmap], dim=-1)

                gs_dict_add = None
                # gs_dict_add = self.construct_pose_gs_dict(eepose)
        # Collect valid view indices
        eval_indices = [i for i, cam_name in enumerate(data_dict['cam_names']) if cam_name in cam_names_eval]

        if metric_logger is not None:
            metric_logger.tik('render')
        # Batch render all views
        with torch.no_grad():
            render_dict = self.render_image_gsplat(
                viewmats=data_dict['viewmats'][eval_indices],
                Ks=data_dict['Ks'][eval_indices],
                image_height=data_dict['image_height'],
                image_width=data_dict['image_width'],
                bg_color=self.background,
                dxyz=dxyz,
                drot=drot,
                gs_dict_add=gs_dict_add,
                colors_precomputed=None,  # 传入预计算的颜色
            )
            est_images = torch.clamp(render_dict['image'], 0, 1)  # (B, 3, H, W)
            est_depths = render_dict['depth']  # (B, 1, H, W)

        if metric_logger is not None:
            metric_logger.tok('render')

        # Process results
        eval_dict = {
            "eepose": eepose,
            "openness": openness,
        }
        if 'images' in data_dict:
            gt_images = data_dict['images'][eval_indices]  # (B, 3, H, W)
            gt_depths = data_dict['depths'][eval_indices]  # (B, 1, H, W)
            psnr = metric_utils.compute_psnr_ts(gt_images, est_images)  # (B,)
            depth_err = metric_utils.compute_l2error_ts(gt_depths, est_depths)  # (B,)
            eval_dict.update({
                "psnr_list": psnr.tolist(),
                "depth_err_list": depth_err.tolist(),
                "gt_image_list": [gt_images[j:j+1] for j in range(len(eval_indices))],
                "gt_depth_list": [gt_depths[j:j+1] for j in range(len(eval_indices))],
            })
        # Clone slices and release batched tensors to avoid memory fragmentation
        eval_dict["image_list"] = [est_images[j:j+1].clone() for j in range(len(eval_indices))]
        eval_dict["depth_list"] = [est_depths[j:j+1].clone() for j in range(len(eval_indices))]
        eval_dict["velmap_pc"] = velmap_pc
        if not bool_canonical:
            eval_dict.update({
                "network_t": data_dict['frame_ids'][0] - self._cano_t,
                "dxyz": dxyz,
                "drot": drot,
                "dxyz_node": dxyz_node,
                "drot_node": drot_node,
            })
        return eval_dict

    def segment_gaussian(
        self,
        data_dict_cano: Dict,
        downsample_factor: int=20,
    ):
        '''
        Args:
            data_dict_cano:
                "images": torch.Tensor (M, 3, H, W) [0,1],
                "masks": torch.Tensor (M, 3, H, W) ,
                "pcds": torch.Tensor (M, 3, H, W),
            downsample_factor: int, default=20
        '''
        gs_pcds = self._cano_gaussians.get_xyz  # (num_gaussian, 3)
        gs_pcds_expand = einops.rearrange(gs_pcds, 'g c -> g 1 c')
        pc = einops.rearrange(
            data_dict_cano['pcds'], 'm c h w -> 1 (m h w) c')[:, ::downsample_factor, :]
        mask = einops.rearrange(
            data_dict_cano['masks'], 'm c h w -> (m h w) c')[::downsample_factor, :]  # (N, 3)
        dist = torch.norm(gs_pcds_expand - pc, dim=2)
        nearest_idx = torch.argmin(dist, dim=1)  # (num_gaussian,)
        gs_mask = mask[nearest_idx]
        self._cano_semantic_mask = gs_mask # (num_gaussian, 3)
        return

    
    def _merge_excess_clusters(self, labels, max_clusters, start_id=2):
        """
        纯 CPU 逻辑运算：迭代将最小的无效簇合并到最小的有效簇。
        不操作 N 个点，只操作 Cluster 列表，速度极快。
        """
        # 1. 统计 valid clusters
        # unique_labels 会自动排序 (例如: 0, 1, 5, 8...)
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        num_valid = len(unique_labels)
        
        if num_valid <= max_clusters:
            # 如果没超标，直接偏移 ID 返回
            new_labels = labels.copy()
            mask = (labels >= 0)
            new_labels[mask] += start_id
            return new_labels
        
        # 2. 构建可变列表用于模拟合并
        # 格式: {'id': original_label_id, 'count': num_points}
        cluster_list = []
        for uid, c in zip(unique_labels, counts):
            cluster_list.append({'id': uid, 'count': c})
            
        # 3. 初始化映射表: original_id -> new_id (初始指向自己)
        # 我们最终只需要修改 labels 数组一次
        mapping = {uid: uid for uid in unique_labels}
        
        # 4. 迭代合并逻辑
        # 因为我们只需要减少 cluster 数量，每次减少 1 个，循环次数 = num_valid - max_clusters
        # 相比 while 循环，这里因为 cluster 数量少 (几十个)，完全可以用 sort
        
        while len(cluster_list) > max_clusters:
            # A. 排序: Count 大 -> 小
            # 这样 list[-1] 是最小的 (Victim)， list[max_clusters-1] 是最小的保留簇 (Target)
            cluster_list.sort(key=lambda x: x['count'], reverse=True)
            
            # B. 选出 Victim (最小的溢出簇)
            victim = cluster_list.pop() # 移除最后一个
            
            # C. 选出 Target (保留组里最小的簇)
            # 保留组是前 max_clusters 个。最小的就是 index = max_clusters - 1
            target_idx = max_clusters - 1
            target = cluster_list[target_idx]
            
            # D. 模拟合并
            # 将 Victim 的点数加到 Target 上
            target['count'] += victim['count']
            
            # E. 更新映射表
            # 所有之前指向 victim['id'] 的，现在都要指向 target['id']
            # 注意：可能存在 A->B, B->C 的链条，所以要遍历 mapping 更新
            victim_original_id = victim['id']
            target_original_id = target['id']
            
            # 这里的 mapping key 是原始 DBSCAN label，value 是当前的归属
            for k, v in mapping.items():
                if v == victim_original_id:
                    mapping[k] = target_original_id
        
        # 5. 应用映射
        # 此时 mapping 中只有 max_clusters 个 unique values
        new_labels = labels.copy()
        
        # 向量化应用映射 (使用 np.vectorize 或 简单的循环，因为 unique labels 很少)
        # 为了极速，直接用 look-up table 数组
        # 但 labels 不连续，所以用 dict 遍历更安全
        
        # 我们可以只对 unique labels 做映射，不需要遍历 N 个点
        for old_id, final_target_id in mapping.items():
            # 找到原始标签是 old_id 的点，设为 final_target_id
            # 这里需要偏移 start_id
            mask = (labels == old_id)
            new_labels[mask] = final_target_id + start_id
            
        return new_labels

    @torch.no_grad()
    def init_gaussian_weights(self, min_weight=1.0, max_weight=100.0):
        """
        计算 per-Gaussian 权重。
        
        Args:
            min_weight (float): 背景/大簇的基础权重 (默认 1.0)
            max_weight (float): 小物体/稀有簇的最大权重 (默认 15.0)
        """
        # --- 0. 参数配置 ---
        roi_min = np.array(self.roi_min, dtype=np.float32)
        roi_max = np.array(self.roi_max, dtype=np.float32)
        
        conf_spatial_scale = 0.5    # 颜色/空间平衡系数
        dbscan_eps = 0.1           # 5cm
        dbscan_min_samples = 10     # 最小点数
        max_object_clusters = 20    # 最大保留的物体簇数量
        
        # --- 1. 数据准备 ---
        xyz = self._cano_gaussians.get_xyz.detach()
        device = xyz.device
        N = xyz.shape[0]
        
        xyz_np = xyz.cpu().numpy()
        sh_dc = self._cano_gaussians.get_features.detach().squeeze(1)
        rgb_feat = torch.clamp(gs_utils.SH2RGB(sh_dc), 0, 1).cpu().numpy()
        
        # 初始化标签: -1 (待定/噪声), 0 (ROI外)
        final_labels = np.full(N, -1, dtype=np.int32)
        
        # --- 2. 筛选 ROI (Cluster 0) ---
        in_roi_mask = np.all((xyz_np >= roi_min) & (xyz_np <= roi_max), axis=1)
        final_labels[~in_roi_mask] = 0
        
        if in_roi_mask.sum() == 0:
            # 如果没有 ROI 内的点，返回基础权重
            self.gaussian_weights = torch.full((N,), min_weight, device=device)
            return

        roi_indices = np.where(in_roi_mask)[0]
        roi_xyz = xyz_np[roi_indices]
        roi_rgb = rgb_feat[roi_indices]
        
        # --- 3. 平面提取 (Cluster 1) ---
        # Filter table points: z < 0.8
        table_mask = roi_xyz[:, 2] < 0.8
        table_xyz = roi_xyz[table_mask]
        table_indices_in_roi = np.where(table_mask)[0]  # Indices in roi_xyz
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(table_xyz)
        _, inliers_idx = pcd.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=1000)
        
        # Map plane inliers back to roi_xyz indices
        is_plane = np.zeros(len(roi_xyz), dtype=bool)
        plane_indices_in_table = inliers_idx  # Indices in table_xyz
        plane_indices_in_roi = table_indices_in_roi[plane_indices_in_table]  # Map to roi_xyz indices
        is_plane[plane_indices_in_roi] = True
        final_labels[roi_indices[is_plane]] = 1
        
        # --- 4. 物体 DBSCAN 聚类 (Cluster 2+) ---
        is_object = ~is_plane
        obj_indices = roi_indices[is_object]
        if len(obj_indices) > 0:
            features = np.hstack([roi_xyz[is_object], roi_rgb[is_object] * conf_spatial_scale])
            
            clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='euclidean', n_jobs=-1)
            raw_labels = clusterer.fit_predict(features)
            
            # --- 平衡合并逻辑 ---
            # 将溢出的 Cluster 均匀合并到保留组中
            refined_labels = self._merge_excess_clusters(raw_labels, max_clusters=max_object_clusters, start_id=2)
            
            mask_valid = np.ones(len(refined_labels), dtype=bool)  # All True
            final_labels[obj_indices[mask_valid]] = refined_labels[mask_valid]

        # --- 5. 权重计算 ---
        # 映射为连续 ID: 0, 1, 2... M
        unique_ids, inverse_indices = np.unique(final_labels, return_inverse=True)
        num_clusters = len(unique_ids)
        
        counts = np.bincount(inverse_indices, minlength=num_clusters).astype(np.float32)
        counts = np.maximum(counts, 1.0)

        # 逆类簇大小权重
        avg_count = N / num_clusters
        raw_weights = avg_count / counts
        point_weights = raw_weights[inverse_indices]
        
        # 对数平滑
        point_weights = np.log1p(point_weights)
        w_min_curr, w_max_curr = point_weights.min(), point_weights.max()
        
        # 动态归一化到 [min_weight, max_weight]
        if w_max_curr - w_min_curr > 1e-6:
            norm_weights = (point_weights - w_min_curr) / (w_max_curr - w_min_curr)
            final_weights = min_weight + norm_weights * (max_weight - min_weight)
        else:
            final_weights = np.full(N, min_weight, dtype=np.float32)
            
        self.gaussian_weights = torch.from_numpy(final_weights).float().to(device).detach()
        return

    def get_eepose_openness(
        self,
        current_xyzs: torch.Tensor,
        current_eepose: torch.Tensor,
        target_xyzs: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
            current_xyzs: torch.Tensor, (num_gaussian, 3)
            current_eepose: torch.Tensor, (7,)
            target_xyzs: torch.Tensor, (num_gaussian, 3)
            mask: torch.Tensor, (num_gaussian, 3)
        Return:
            eepose: torch.Tensor, (7,)
            openness: torch.Tensor, (1,)
        '''
        dtype = current_eepose.dtype
        device = current_eepose.device
        current_eepose_xyz = current_eepose[:3]
        current_eepose_qxyzw = current_eepose[3:7]
        current_eepose_R = gs_utils.quaternion_xyzw_to_matrix(
            current_eepose_qxyzw.unsqueeze(0))[0]
        head_mask = utils_with_rlbench.get_mask_with_obj_indices_ts(
            mask,
            utils_with_rlbench.Joint2ARM[6],
        )
        left_tip_mask = utils_with_rlbench.get_mask_with_obj_indices_ts(
            mask,
            utils_with_rlbench.Joint2ARM[7],
        )
        right_tip_mask = utils_with_rlbench.get_mask_with_obj_indices_ts(
            mask,
            utils_with_rlbench.Joint2ARM[8],
        )
        R_src2tar, t_src2tar = compute_icp_svd(
            source=current_xyzs[head_mask],
            target=target_xyzs[head_mask],
        )
        lefttip_xyz_tar = target_xyzs[left_tip_mask] # N, 3
        righttip_xyz_tar = target_xyzs[right_tip_mask] # M, 3
        dist_xyz_tar = lefttip_xyz_tar.unsqueeze(1) - righttip_xyz_tar.unsqueeze(0) # N, M, 3
        dist_xyz_tar = torch.norm(dist_xyz_tar, dim=2) # N, M
        target_action_open = 1 if dist_xyz_tar.min() > 0.04 else 0
        target_action_open = torch.tensor(
            [target_action_open],
            dtype=dtype,
            device=device,
        )
        target_eepose_xyz = R_src2tar @ current_eepose_xyz + t_src2tar
        target_eepose_R =  R_src2tar @ current_eepose_R
        target_eepose_qxyzw = gs_utils.rotation_to_quaternion_xyzw(target_eepose_R.unsqueeze(0))[0]
        eepose = torch.cat([
            target_eepose_xyz,
            target_eepose_qxyzw,
        ], dim=0)
        return eepose, target_action_open

    def compute_dxyz_drot(
        self,
        t: torch.Tensor,
        dt: torch.Tensor,
        dxyz_node: torch.Tensor,
        drot_node: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Args:
            t (torch.Tensor): (1,)
            dt (torch.Tensor): (1,)
            dxyz_node (torch.Tensor): (num_node, 3) accumulated deformation (control node)
            drot_node (torch.Tensor): (num_node, 4) accumulated deformation (control node)
        Return:
            dxyz (torch.Tensor): (num_gaussian, 3) updated deformation accumulation
            drot (torch.Tensor): (num_gaussian, 4) updated deformation accumulation
            dxyz_node (torch.Tensor): (num_node, 3) updated deformation accumulation (contorl node)
            drot_node (torch.Tensor): (num_node, 4) updated deformation accumulation (contorl node)
        '''
        cnode = self._control_nodes
        xyz_node = cnode.nodes
        xyz_node.requires_grad = False
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            deform_seg_node = self._network(xyz_node)
            dxyz_incre_node, drot_incre_node = \
            self._compute_increment_dxyz_drot(
                deform_seg_node,
                xyz_node.detach() + dxyz_node.detach(),
                t,
                dt=dt,
            )
        dxyz_total_node = dxyz_node.detach() + dxyz_incre_node
        drot_total_node = gs_utils.quaternion_multiply(
            drot_incre_node, drot_node.detach())
        dxyz_total, drot_total = cnode.upsample(
            dxyz_total_node,
            drot_total_node,
        )
        return dxyz_total, drot_total, dxyz_total_node, drot_total_node

    def _compute_increment_dxyz_drot(
        self,
        deform_seg: torch.Tensor,
        xyz: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
    ):
        ## get vel, omega of cnode
        ## rgkt-2 on cnode
        # CUDA 优化：使用静态预分配 tensor + slice assignment，避免每次分配内存和 kernel 调用
        # 直接使用 __init__ 中预分配的 xyzt tensor（根据 num_control_nodes 大小）
        xyzt = self._cudagraph_dict['_compute_increment_dxyz_drot_xyzt']
        xyzt[:, :3] = xyz; xyzt[:, 3:4] = t
        # Runge-Kutta 2
        # 获得当前时刻每个点的 vel
        v_cur = self._network.get_vel(deform_seg, xyzt)
        p_mid = (xyz + 0.5 * dt * v_cur).detach()
        xyzt_mid = xyzt # 复用 xyzt 作为 xyzt_mid，避免再次分配
        xyzt_mid[:, :3] = p_mid; xyzt_mid[:, 3:4] = t + 0.5 * dt
        # jac_v , v_mid = vmap(jacrev(self.u_func,argnums = -1 , has_aux = True))(deform_seg, xyzt_mid)
        jac_v, v_mid = self.get_vel_jac(deform_seg, xyzt_mid)
        dxyz_incre = dt * v_mid
        drot_incre = dt * jac_v[..., :3, :3]
        # in-place 操作，等价于 drot_incre = identity + dt * jac_v[..., :3, :3]
        drot_incre.diagonal(dim1=1, dim2=2).add_(1)
        drot_incre = gs_utils.rotation_to_quaternion_wxyz(drot_incre)
        return dxyz_incre, drot_incre

    def get_vel_jac(self, deform_code, xt):
        v_basis, jac_basis = self.get_basis_jac(xt)
        # t_embed = self.embedder(xt[..., -1:])
        t_embed = self._network.embedder(xt[..., -1:])
        weights = self._network.vel_weight(t_embed)
        weights = einops.rearrange(weights, '... (K dim) -> ... K dim', K=self._network.K)
        v = torch.einsum('...ij,...ki->...kj', v_basis, weights)
        v = torch.einsum('...k,...kj->...j', deform_code, v)
        jac = torch.einsum('...imn,...ki->...kmn', jac_basis, weights)
        jac = torch.einsum('...k,...kmn->...mn', deform_code, jac)
        return jac, v

    def get_basis_jac(self, xt):
        x, y, z = xt[..., 0], xt[..., 1], xt[..., 2]
        zeros = xt[..., -1] * 0.
        ones = zeros + 1.

        b1 = torch.stack([ones, zeros, zeros], dim=-1)
        b2 = torch.stack([zeros, ones, zeros], dim=-1)
        b3 = torch.stack([zeros, zeros, ones], dim=-1)
        b4 = torch.stack([zeros, z, -y], dim=-1)
        b5 = torch.stack([-z, zeros, x], dim=-1)
        b6 = torch.stack([y, -x, zeros], dim=-1)

        zeros_vec = torch.stack([zeros, zeros, zeros], dim=-1)

        jac_1 = torch.stack([zeros_vec, zeros_vec, zeros_vec], dim=-2)
        jac_4 = torch.stack([zeros_vec, b3, -b2], dim=-2)
        jac_5 = torch.stack([-b3, zeros_vec, b1], dim=-2)
        jac_6 = torch.stack([b2, -b1, zeros_vec], dim=-2)

        # jac_1 = torch.stack([
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        # ], dim=-2)
        # jac_2 = torch.stack([
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        # ], dim=-2)
        # jac_3 = torch.stack([
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        # ], dim=-2)
        # jac_4 = torch.stack([
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        #     torch.stack([zeros, zeros, ones], dim=-1),
        #     torch.stack([zeros, -ones, zeros], dim=-1),
        # ], dim=-2)
        # jac_5 = torch.stack([
        #     torch.stack([zeros, zeros, -ones], dim=-1),
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        #     torch.stack([ones, zeros, zeros], dim=-1),
        # ], dim=-2)
        # jac_6 = torch.stack([
        #     torch.stack([zeros, ones, zeros], dim=-1),
        #     torch.stack([-ones, zeros, zeros], dim=-1),
        #     torch.stack([zeros, zeros, zeros], dim=-1),
        # ], dim=-2)

        return torch.stack([b1, b2, b3, b4, b5, b6], dim=-2), torch.stack([jac_1, jac_1, jac_1, jac_4, jac_5, jac_6], dim=-3)

    def u_func(self, deform_code,  xyzt):
        u = self._network.get_vel(deform_code, xyzt)
        return u , u
    
    def render_image(
        self,
        view_dict_list,
        bg_color: torch.Tensor,
        dxyz=None,
        drot=None,
        gs_dict_add=None,
    ) -> Dict:
        '''
        Args:
            view_dict (List[Dict])
            dxyz (None, torch.Tensor): (N, 3)
            drot (None, torch.Tensor): (N, 4)
        Return:
            render_dict (Dict)
                "image": (N, 3)
                "depth": (N, 4)
        '''
        image_list, depth_list = [], []
        for view_dict in view_dict_list:
            render_dict = gs_utils.render_image(
                self._cano_gaussians,
                view_dict,
                device=bg_color.device,
                bg_color=bg_color,
                dxyz=dxyz,
                drot=drot,
                gs_dict_add=gs_dict_add,
            )
            image = render_dict['render']
            depth = render_dict['depth']
            image_list.append(image)
            depth_list.append(depth)
        return {
            "image": torch.stack(image_list),
            "depth": torch.stack(depth_list),
        }
    
    def render_image_gsplat(
        self,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        image_height: int,
        image_width: int,
        bg_color: torch.Tensor,
        dxyz=None,
        drot=None,
        gs_dict_add=None,
        colors_precomputed=None,  # 新增参数
    ) -> Dict:
        """Render using gs_utils.render_image_gsplat.

        Args:
            viewmats: [B, 4, 4] batched view matrices
            Ks: [B, 3, 3] batched intrinsics
            image_height: image height
            image_width: image width
            bg_color: background color tensor
            dxyz, drot, gs_dict_add: optional rendering parameters
            colors_precomputed: optional [N, D] precomputed colors (RGB+Weight)

        Returns:
            {
                "image": (B, 3, H, W),
                "depth": (B, 1, H, W),
                "weight": (B, 1, H, W) (if colors_precomputed is provided),
            }
        """
        device = bg_color.device

        render_dict = gs_utils.render_image_gsplat(
            gaussians=self._cano_gaussians,
            viewmats=viewmats,
            Ks=Ks,
            device=device,
            bg_color=bg_color,
            dxyz=dxyz,
            drot=drot,
            gs_dict_add=gs_dict_add,
            height=image_height,
            width=image_width,
            colors_precomputed=colors_precomputed,  # 传递参数
        )

        result = {
            "image": render_dict["render"],  # (B, 3, H, W)
            "depth": render_dict["depth"],   # (B, 1, H, W)
        }
        if render_dict.get("weight") is not None:
            result["weight"] = render_dict["weight"]  # (B, 1, H, W)
        return result

    def densify_and_prune_gaussians(
        self,
        densify_grad_threshold: float,
        cameras_extent: float,
        size_threshold: float,
    ):
        return self._cano_gaussians.densify_and_prune(
            densify_grad_threshold,
            0.005,
            cameras_extent,
            size_threshold,
        )

    def embed_time(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def update_control_node(self) -> None:
        xyzs = self._cano_gaussians.get_xyz.detach()
        self._control_nodes.update(xyzs)
        return
    
    def compute_render_loss(
        self,
        image,
        depth,
        gt_image,
        gt_depth,
        lambda_dssim,
        lambda_depth,
        weight_image=None,
        canon_means_sampled=None,
        deform_means_sampled=None,
        rigid_weights_sampled=None,
        lambda_rigid: float = 0.0,
        rigid_k: int = 20,
        ) -> torch.Tensor:
        loss = gs_utils.compute_render_loss(
            image=image,
            depth=depth,
            gt_image=gt_image,
            gt_depth=gt_depth,
            lambda_dssim=lambda_dssim,
            lambda_depth=lambda_depth,
            weight_image=weight_image,
            canon_means_sampled=canon_means_sampled,
            deform_means_sampled=deform_means_sampled,
            rigid_weights_sampled=rigid_weights_sampled,
            lambda_rigid=lambda_rigid,
            rigid_k=rigid_k,
        )
        return loss


    def construct_pose_gs_dict(
        self,
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
        R = gs_utils.quaternion_xyzw_to_matrix(q.unsqueeze(0))[0]

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
        shs = gs_utils.RGB2SH(base_rgb)
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
            "opacity": opacity,
            "scales": scales,
            "rotations": rotations,
            "valid": valid,
        }

class FreeGavePhysicsNetwork(nn.Module):
    def __init__(self, D: int, W: int, input_ch: int, output_ch: int, multires: int = 8):
        super().__init__()
        self.skips = [D // 2]

        # 直接创建 embedder，已知 multires=8, input_ch=3
        # out_dim = input_ch + num_freqs * 2 * input_ch = 3 + 8 * 2 * 3 = 51
        self.embed_fn = Embedder(
            include_input=True,
            input_dims=input_ch,
            max_freq_log2=multires - 1,  # 7
            num_freqs=multires,  # 8
            log_sampling=True,
        )
        xyz_input_ch = self.embed_fn.out_dim  # 51
        self.code_linear = nn.ModuleList(
            [nn.Linear(xyz_input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch, W)
                for i in range(D - 1)]
        )
        self.code_output = nn.Sequential(
            nn.Linear(W, output_ch),
        )
        self.code_seg = nn.Sequential(
            nn.Linear(output_ch, output_ch * 4),
            nn.ReLU(inplace=False),
            nn.Linear(output_ch * 4, output_ch * 4),
            nn.ReLU(inplace=False),
            nn.Linear(output_ch * 4, output_ch),
        )

        self.K = output_ch
        encode_dim = 3
        hidden_dim = 128
        layers = 5
        in_dim = 1 + 1 * 2 * encode_dim
        self.embedder = PositionEncoder(encode_dim)
        self.vel_weight = WeightNet_zeropadding(in_dim,hidden_dim,layers,self.K)
        return
    
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        x_emb = self.embed_fn(xyz)
        h = torch.cat([x_emb], dim=-1)
        for i, l in enumerate(self.code_linear):
            h = self.code_linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, h], -1)
        motion_code = self.code_output(h)
        return self.code_seg(motion_code)

    def get_vel(self, deform_seg: torch.Tensor, xyzt: torch.Tensor) -> torch.Tensor:
        v_basis, _ = self.get_basis(xyzt)
        t_embed = self.embedder(xyzt[..., -1:])
        weights = self.vel_weight(t_embed)
        weights = einops.rearrange(weights, '... (K dim) -> ... K dim', K=self.K)
        v = torch.einsum('...ij,...ki->...kj', v_basis, weights)
        v = torch.einsum('...k,...kj->...j', deform_seg, v)
        return v

    def get_vel_map(self, deform_seg: torch.Tensor, xyzt: torch.Tensor) -> torch.Tensor:
        v_basis, _ = self.get_basis(xyzt)
        t_embed = self.embedder(xyzt[..., -1:])
        weights = self.vel_weight(t_embed)
        weights = einops.rearrange(weights, '... (K dim) -> ... K dim', K=self.K)
        vmap = torch.einsum('...k,...ki->...i', deform_seg, weights)
        return vmap

    def get_basis(self, xt):
        x, y, z = xt[..., 0], xt[..., 1], xt[..., 2]
        zeros = xt[..., -1] * 0.
        ones = zeros + 1.
        b1 = torch.stack([ones, zeros, zeros], dim=-1)
        b2 = torch.stack([zeros, ones, zeros], dim=-1)
        b3 = torch.stack([zeros, zeros, ones], dim=-1)
        b4 = torch.stack([zeros, z, -y], dim=-1)
        b5 = torch.stack([-z, zeros, x], dim=-1)
        b6 = torch.stack([y, -x, zeros], dim=-1)

        a4 = torch.stack([zeros, -y, -z], dim=-1)
        a5 = torch.stack([-x, zeros, -z], dim=-1)
        a6 = torch.stack([-x, -y, zeros], dim=-1)
        return torch.stack([b1, b2, b3, b4, b5, b6], dim=-2), torch.stack([b1, b2, b3, a4, a5, a6], dim=-2)

class PositionEncoder(nn.Module):

    def __init__(self, encode_dim, log_sampling=True):
        super(PositionEncoder, self).__init__()

        self.encode_dim = encode_dim
        if log_sampling:
            frequency_bands = 2.0 ** torch.linspace(
                0.0,
                self.encode_dim - 1,
                self.encode_dim,
                dtype=torch.float32
            )
        else:
            frequency_bands = torch.linspace(
                2.0 ** 0.0,
                2.0 ** (self.encode_dim - 1),
                self.encode_dim,
                dtype=torch.float32
            )
        self.register_buffer('frequency_bands', frequency_bands)

    def forward(self, x):

        encoding = [x]

        for freq in self.frequency_bands:
            encoding.append(torch.sin(x * freq))
            encoding.append(torch.cos(x * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)

class WeightNet_zeropadding(nn.Module):
    def __init__(self, in_dim, hidden_dim, layers, K):
        super(WeightNet_zeropadding, self).__init__()
        self.K = K -1 
        self.layers = layers
        self.hidden_dim=hidden_dim
        # Define the first layer
        self.weight_net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU())
        
        # Define the hidden layers
        for _ in range(layers - 1):
            self.weight_net.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
        
        # Define the final layer
        self.weight_net.append(nn.Sequential(nn.Linear(hidden_dim, 6 * self.K)))

    def forward(self, x):
        out = self.weight_net(x)
        if len(out.shape) == 2:
            # Use device/dtype from input to avoid .cuda() call during graph capture
            zero = torch.zeros(out.shape[0], 6, device=out.device, dtype=out.dtype)
            out = torch.cat([out, zero], dim=-1)
        elif len(out.shape) == 1:
            zero = torch.zeros(6, device=out.device, dtype=out.dtype)
            out = torch.cat([out, zero], dim=-1)
        return out

class ControlNodes(nn.Module):
    def __init__(self, K=3):
        super().__init__()
        self.K = K
        self.nodes = None
        self.node_num = None
        self._node_radius = None
        self._node_weights = None
        self._node_indices = None
        return
    
    def upsample(
        self,
        node_dxyz: torch.Tensor,
        node_drot: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nn_weight = self._node_weights.detach() # GT : ([118640, 3])
        nn_idx = self._node_indices.detach() # GT ： ([118640, 3])
        translate = (node_dxyz[nn_idx] * nn_weight[..., None]).sum(dim=1)
        rotation = (node_drot[nn_idx] * nn_weight[..., None]).sum(dim=1)
        rotation = rotation / torch.norm(rotation, dim=-1, keepdim=True)
        return translate, rotation
    
    def init(self, xyz: torch.Tensor) -> None:
        scene_range = xyz.max() - xyz.min()
        self.nodes = nn.Parameter(xyz)
        self.node_num = xyz.shape[0]
        self._node_radius = nn.Parameter(torch.log(.1 * scene_range + 1e-7) * torch.ones([self.node_num]).float())
        return
    
    def update(self, xyzs: torch.Tensor) -> None:
        nodes = self.nodes
        K = self.K
        nn_dist, nn_idxs, _ = pytorch3d.ops.knn_points(xyzs[None], nodes[None], None, None, K=K)  # N, K
        nn_dist, nn_idxs = nn_dist[0], nn_idxs[0]  # N, K
        nn_radius = self._node_radius[nn_idxs]  # N, K
        nn_weight = torch.exp(- nn_dist / (2 * nn_radius ** 2))  # N, K
        nn_weight = nn_weight + 1e-7
        nn_weight = nn_weight / nn_weight.sum(dim=-1, keepdim=True)  # N, K
        self._node_weights = nn_weight
        self._node_indices = nn_idxs
        return


class Embedder(nn.Module):
    def __init__(self, input_dims, include_input, max_freq_log2, num_freqs, log_sampling):
        super().__init__()
        d = input_dims
        out_dim = 0
        
        # 保存配置信息
        self.include_input = include_input
        self.input_dims = input_dims
        # 硬编码周期函数为 sin 和 cos
        self.periodic_fns = [torch.sin, torch.cos]
        
        if include_input:
            out_dim += d

        max_freq = max_freq_log2
        N_freqs = num_freqs

        if log_sampling:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs, dtype=torch.float32, requires_grad=False)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs, dtype=torch.float32, requires_grad=False)
        self.register_buffer('freq_bands', freq_bands)

        # 计算输出维度：每个频率有 2 个周期函数（sin 和 cos）
        out_dim += 8 * 2 * d
        self.out_dim = out_dim

    def forward(self, inputs):
        """
        显式展开所有编码逻辑，避免 lambda 函数捕获问题，提升 CUDA graph 兼容性
        硬编码展开所有循环，提升性能
        """
        # inputs 已经在 FreeGavePhysicsNetwork.forward 中克隆过了
        # 为了避免 CUDA Graph 中多次使用同一张量的问题，我们为每个频率创建独立的中间变量
        # 硬编码展开：包含原始输入（include_input=True）
        # 然后对每个频率显式调用 sin 和 cos
        # 原始输入
        encoded = [inputs]
        
        # freq_bands = 2. ** torch.linspace(0., self.max_freq, steps=self.N_freqs, dtype=torch.float32, device=inputs.device)
        freq_bands = self.freq_bands
        # 频率 0
        freq0 = inputs * freq_bands[0]
        encoded.append(torch.sin(freq0))
        encoded.append(torch.cos(freq0))
        
        # 频率 1
        freq1 = inputs * freq_bands[1]
        encoded.append(torch.sin(freq1))
        encoded.append(torch.cos(freq1))
        
        # 频率 2
        freq2 = inputs * freq_bands[2]
        encoded.append(torch.sin(freq2))
        encoded.append(torch.cos(freq2))
        
        # 频率 3
        freq3 = inputs * freq_bands[3]
        encoded.append(torch.sin(freq3))
        encoded.append(torch.cos(freq3))
        
        # 频率 4
        freq4 = inputs * freq_bands[4]
        encoded.append(torch.sin(freq4))
        encoded.append(torch.cos(freq4))
        
        # 频率 5
        freq5 = inputs * freq_bands[5]
        encoded.append(torch.sin(freq5))
        encoded.append(torch.cos(freq5))
        
        # 频率 6
        freq6 = inputs * freq_bands[6]
        encoded.append(torch.sin(freq6))
        encoded.append(torch.cos(freq6))
        
        # 频率 7
        freq7 = inputs * freq_bands[7]
        encoded.append(torch.sin(freq7))
        encoded.append(torch.cos(freq7))
        
        return torch.cat(encoded, dim=-1)

def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim