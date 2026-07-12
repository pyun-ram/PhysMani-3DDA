import math
import torch
import einops
import numpy as np
from typing import Dict
from torch.autograd import Variable
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import torch.nn.functional as F

import random
from tqdm import tqdm
from torch import nn
from simple_knn._C import distCUDA2
from . import common_utils, metric_utils
from .general_utils import strip_symmetric, build_scaling_rotation, inverse_sigmoid, get_expon_lr_func, build_rotation
from gsplat.rendering import rasterization as gsplat_rasterization

# 如果 torch 有 _dynamo 属性，则禁用 gsplat_rasterization
if hasattr(torch, "_dynamo"):
    gsplat_rasterization = torch._dynamo.disable(gsplat_rasterization)

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)

def normalize_quaternion(q):
    return q / q.norm(dim=-1, keepdim=True)

def normalize_quaternion_np(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)

def rotation_to_quaternion_wxyz(R: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    将形状为(K,3,3)的旋转矩阵转换为形状为(K,4)的四元数,顺序为wxyz。
    
    参数:
        R (torch.Tensor): 输入旋转矩阵，形状为(K,3,3)
        eps (float): 用于数值稳定性的小量,默认为1e-8
    
    返回:
        torch.Tensor: 转换后的四元数，形状为(K,4),顺序为wxyz
    """
    K = R.shape[0]

    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    # 1. Calculate for tr > 0 case
    S_tr = torch.sqrt(tr + 1.0 + eps) * 2.0
    qw_tr = 0.25 * S_tr
    qx_tr = (R[:, 2, 1] - R[:, 1, 2]) / (S_tr + eps)
    qy_tr = (R[:, 0, 2] - R[:, 2, 0]) / (S_tr + eps)
    qz_tr = (R[:, 1, 0] - R[:, 0, 1]) / (S_tr + eps)

    # 2. Calculate for tr <= 0 cases
    # Find max diagonal element index
    diag = R.diagonal(dim1=1, dim2=2)
    _, max_diag_idx = torch.max(diag, dim=1)

    # Case 0: max_diag_idx == 0
    val0 = 1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]
    S0 = torch.sqrt(torch.clamp(val0, min=eps)) * 2.0
    S0_safe = S0 + eps
    qw_case0 = (R[:, 2, 1] - R[:, 1, 2]) / S0_safe
    qx_case0 = 0.25 * S0
    qy_case0 = (R[:, 0, 1] + R[:, 1, 0]) / S0_safe
    qz_case0 = (R[:, 0, 2] + R[:, 2, 0]) / S0_safe

    # Case 1: max_diag_idx == 1
    val1 = 1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]
    S1 = torch.sqrt(torch.clamp(val1, min=eps)) * 2.0
    S1_safe = S1 + eps
    qw_case1 = (R[:, 0, 2] - R[:, 2, 0]) / S1_safe
    qx_case1 = (R[:, 0, 1] + R[:, 1, 0]) / S1_safe
    qy_case1 = 0.25 * S1
    qz_case1 = (R[:, 1, 2] + R[:, 2, 1]) / S1_safe

    # Case 2: max_diag_idx == 2
    val2 = 1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]
    S2 = torch.sqrt(torch.clamp(val2, min=eps)) * 2.0
    S2_safe = S2 + eps
    qw_case2 = (R[:, 1, 0] - R[:, 0, 1]) / S2_safe
    qx_case2 = (R[:, 0, 2] + R[:, 2, 0]) / S2_safe
    qy_case2 = (R[:, 1, 2] + R[:, 2, 1]) / S2_safe
    qz_case2 = 0.25 * S2

    # 3. Combine results using masks (no explicit branching)
    mask_tr = tr > 0
    mask_case0 = max_diag_idx == 0
    mask_case1 = max_diag_idx == 1
    # mask_case2 is implied

    # Selection for tr <= 0
    qw_not_tr = torch.where(mask_case0, qw_case0, torch.where(mask_case1, qw_case1, qw_case2))
    qx_not_tr = torch.where(mask_case0, qx_case0, torch.where(mask_case1, qx_case1, qx_case2))
    qy_not_tr = torch.where(mask_case0, qy_case0, torch.where(mask_case1, qy_case1, qy_case2))
    qz_not_tr = torch.where(mask_case0, qz_case0, torch.where(mask_case1, qz_case1, qz_case2))

    # Final selection
    qw = torch.where(mask_tr, qw_tr, qw_not_tr)
    qx = torch.where(mask_tr, qx_tr, qx_not_tr)
    qy = torch.where(mask_tr, qy_tr, qy_not_tr)
    qz = torch.where(mask_tr, qz_tr, qz_not_tr)

    # 4. Stack and normalize
    quat = torch.stack([qw, qx, qy, qz], dim=1)
    quat = quat / torch.norm(quat, dim=1, keepdim=True)

    return quat

def rotation_to_quaternion_xyzw(R):
    """
    Args:
        R: torch.Tensor, shape (K, 3, 3)
    Returns:
        q: torch.Tensor, shape (K, 4,), (x, y, z, w)
    """
    return rotation_to_quaternion_wxyz(R)[:, [1, 2, 3, 0]]

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

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

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

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def compute_local_rigidity_loss_sampled(
    orig_xyz_sampled: torch.Tensor,
    pred_xyz_sampled: torch.Tensor,
    k: int = 20,
) -> torch.Tensor:
    """
    Local Rigidity Loss (DefGS-style) with pre-sampled inputs.
    This version accepts already-sampled tensors to avoid CUDAGraphs issues in torch.compile.
    
    Args:
        orig_xyz_sampled: (M, 3) sampled canonical positions (already detached)
        pred_xyz_sampled: (M, 3) sampled deformed positions (should carry gradients)
        k: number of neighbors
    Returns:
        scalar tensor on same device as pred_xyz_sampled
    """
    if orig_xyz_sampled is None or pred_xyz_sampled is None:
        device = pred_xyz_sampled.device if pred_xyz_sampled is not None else orig_xyz_sampled.device
        return torch.zeros((), device=device)
    if orig_xyz_sampled.numel() == 0 or pred_xyz_sampled.numel() == 0:
        return torch.zeros((), device=pred_xyz_sampled.device)
    if orig_xyz_sampled.shape[-1] != 3 or pred_xyz_sampled.shape[-1] != 3:
        raise ValueError(f"Expected (M,3) xyz, got orig {tuple(orig_xyz_sampled.shape)}, pred {tuple(pred_xyz_sampled.shape)}")
    if orig_xyz_sampled.shape[0] != pred_xyz_sampled.shape[0]:
        raise ValueError(f"orig/pred must have same M, got {orig_xyz_sampled.shape[0]} vs {pred_xyz_sampled.shape[0]}")
    
    device = pred_xyz_sampled.device
    M = orig_xyz_sampled.shape[0]
    if M <= 1:
        return torch.zeros((), device=device)
    
    # k must be <= M-1 (exclude self)
    k_eff = int(min(max(k, 1), M - 1))
    
    # Pairwise distances in canonical space (M,M)
    dist_o = torch.cdist(orig_xyz_sampled, orig_xyz_sampled)
    knn_val, knn_idx = torch.topk(dist_o, k=k_eff + 1, dim=1, largest=False)
    knn_idx = knn_idx[:, 1:]     # drop self
    knn_dist_o = knn_val[:, 1:]  # (M,k)
    
    # Distances in deformed space for same neighbor pairs
    neigh_p = pred_xyz_sampled[knn_idx]  # (M,k,3)
    center_p = pred_xyz_sampled.unsqueeze(1)  # (M,1,3)
    dist_p = torch.norm(center_p - neigh_p, dim=2)  # (M,k)
    
    return F.l1_loss(dist_p, knn_dist_o)

def gaussian(window_size, sigma, device=None, dtype=None):
    """创建高斯窗口，支持指定设备和数据类型，避免 CPU-GPU 传输"""
    gauss = torch.tensor(
        [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)],
        device=device,
        dtype=dtype
    )
    return gauss / gauss.sum()
def create_window(window_size, channel, device=None, dtype=None):
    """创建 SSIM 窗口，在指定设备上创建以避免 CPU-GPU 传输"""
    # 如果指定了设备，直接在设备上创建，避免 CPU->GPU 传输
    if device is not None:
        dtype = dtype if dtype is not None else torch.float32
        _1D_window = gaussian(window_size, 1.5, device=device, dtype=dtype).unsqueeze(1)
    else:
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    
    # 如果之前没有指定设备，现在移动
    if device is not None and window.device != device:
        window = window.to(device=device, dtype=dtype)
    elif dtype is not None and window.dtype != dtype:
        window = window.to(dtype=dtype)
    
    return window

# Cache for SSIM windows to avoid repeated CPU->GPU transfers
_ssim_window_cache = {}

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    device = img1.device
    dtype = img1.dtype
    
    # Use cached window if available
    cache_key = (window_size, channel, device, dtype)
    if cache_key not in _ssim_window_cache:
        _ssim_window_cache[cache_key] = create_window(window_size, channel, device, dtype)
    window = _ssim_window_cache[cache_key]

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map

def filter_gaussians(camera, xyz=None):

    valid_depth = torch.ones(
        (xyz.shape[0]), device=xyz.device, dtype=torch.bool)

    if 'R' in camera:
        # transform points to camera space
        R = torch.tensor(camera['R'], device=xyz.device, dtype=torch.float32)
        T = torch.tensor(camera['T'], device=xyz.device, dtype=torch.float32)
        # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
        xyz_cam = xyz @ R + T[None, :]

        xyz_to_cam = torch.norm(xyz_cam, dim=1)

        # project to screen space
        valid_depth = xyz_cam[:, 2] > max(camera['znear'], 0.02)
    else:
        assert False

    return valid_depth

def render_image_gsplat(
        gaussians,
        viewmats,  # [B, 4, 4] batched view matrices
        Ks,        # [B, 3, 3] batched intrinsics
        device:torch.device,
        bg_color:torch.Tensor=None,
        dxyz=None,
        drot=None,
        sh_override=None,
        gs_dict_add=None,
        height: int = None,  # image height (required, assumed same for all views)
        width: int = None,   # image width (required, assumed same for all views)
        colors_precomputed=None,  # (N, D) 预计算的 RGB+Weight 等，D >= 3
):
    """
    使用 gsplat.rasterization 进行 batched 渲染。

    Args:
        gaussians: GaussianModel
        viewmats: [B, 4, 4] batched world-to-camera view matrices
        Ks: [B, 3, 3] batched camera intrinsics
        device: torch.device
        bg_color: [3] background color tensor
        dxyz: optional [N, 3] displacement
        drot: optional [N, 4] rotation delta
        gs_dict_add: optional additional gaussians dict
        height: image height (required if not in viewpoint_camera)
        width: image width (required if not in viewpoint_camera)
        colors_precomputed: optional [N, D] precomputed colors (RGB+Weight), D >= 3

    Returns:
        {
            "render": [B, 3, H, W],
            "depth": [B, 1, H, W],
            "weight": [B, 1, H, W] (if colors_precomputed is provided and D >= 4),
        }
    """
    # 1. 背景色 & screenspace_points
    bg_color = torch.tensor(
        [1, 1, 1],
        dtype=torch.float32,
        device=device) if bg_color is None else bg_color

    screenspace_points = torch.zeros_like(
        gaussians.get_xyz,
        dtype=gaussians.get_xyz.dtype,
        requires_grad=True,
        device=device,
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # 2. 检查输入形状
    B = viewmats.shape[0]
    assert Ks.shape[0] == B, f"Ks batch size {Ks.shape[0]} != viewmats batch size {B}"
    assert viewmats.shape == (B, 4, 4), f"viewmats shape {viewmats.shape} != (B, 4, 4)"
    assert Ks.shape == (B, 3, 3), f"Ks shape {Ks.shape} != (B, 3, 3)"
    assert height is not None and width is not None, "height and width must be provided"
    height = int(height)
    width = int(width)

    # 3. 从 GaussianModel 取出参数，并处理 dxyz / drot / gs_dict_add
    means3D = gaussians.get_xyz if dxyz is None else gaussians.get_xyz + dxyz
    means2D = screenspace_points

    # 与 render_image 一致：get_scaling / get_rotation / get_opacity / get_features
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation if drot is None \
        else quaternion_multiply(drot, gaussians.get_rotation)
    # gsplat.rasterization 期望 raw opacity [N]，然后内部 sigmoid，所以用 _opacity 并手动 sigmoid
    opac_raw = gaussians._opacity  # [N, 1]
    
    # 如果提供了预计算的颜色，使用它并禁用 SH
    if colors_precomputed is not None:
        shs = colors_precomputed  # (N, D)
        sh_degree_to_use = None
    else:
        shs = gaussians.get_features if sh_override is None else sh_override
        sh_degree_to_use = gaussians.active_sh_degree

    # 注意：在 batched 情况下，我们不对每个相机分别做可见性过滤
    # 而是让 gsplat.rasterization 内部处理，或者使用所有相机的并集
    # 这里先简化：不过滤，让所有 Gaussians 参与渲染
    # 如果需要过滤，可以在调用前对每个相机分别过滤并取并集
    if gs_dict_add is not None:
        means3D = torch.cat([means3D, gs_dict_add['means3D']], dim=0)
        means2D = torch.cat([means2D, gs_dict_add['means2D']], dim=0)
        shs = torch.cat([shs, gs_dict_add['sh']], dim=0)
        opac_raw = torch.cat([opac_raw, gs_dict_add['opacity']], dim=0)
        scales = torch.cat([scales, gs_dict_add['scales']], dim=0)
        rotations = torch.cat([rotations, gs_dict_add['rotations']], dim=0)

    # 4. 调用 gsplat.rasterization（batched）
    renders, alphas, _ = gsplat_rasterization(
        means=means3D,
        quats=rotations,
        scales=scales,
        opacities=torch.sigmoid(opac_raw.squeeze(-1)),  # [N, 1] -> [N] -> sigmoid
        colors=shs,
        sh_degree=sh_degree_to_use,
        viewmats=viewmats,  # [B, 4, 4]
        Ks=Ks,             # [B, 3, 3]
        width=width,
        height=height,
        render_mode="RGB+ED",
        packed=False,
        absgrad=False,
        sparse_grad=False,
        rasterize_mode="classic",
        distributed=False,
        camera_model="pinhole",
        with_ut=False,
        with_eval3d=False,
        near_plane=0.5,
        far_plane=10.0
    )
    
    # 根据 colors_precomputed 的通道数拆分结果
    # 在 RGB+ED 模式下，renders 的输出是 [B, H, W, C+1]，其中 C 是 colors 的通道数，+1 是 depth
    if colors_precomputed is not None and renders.shape[-1] == 5:
        num_color_channels = colors_precomputed.shape[-1]  # 应该是 4 (RGB + Weight)
        rgb = renders[..., :3]  # [B, H, W, 3]
        weight = renders[..., 3:4]  # [B, H, W, 1]
        # depth 在 RGB+ED 模式下，会在所有 colors 通道之后，即第 num_color_channels 个通道
        depth = renders[..., num_color_channels:num_color_channels+1]  # [B, H, W, 1]
    else:
        # 原有逻辑：renders: [B, H, W, 4] (RGB + Depth)
        rgb = renders[..., :3]  # [B, H, W, 3]
        depth = renders[..., 3:4]  # [B, H, W, 1]
        weight = None

    # 背景合成（只对 RGB 做背景合成，weight 和 depth 保持原样）
    bg = bg_color.view(1, 1, 1, 3)  # [1, 1, 1, 3]
    rgb = rgb * alphas + bg * (1.0 - alphas)

    # 转成 (B, C, H, W)
    image_chw = rgb.permute(0, 3, 1, 2)  # [B, 3, H, W]
    depth_chw = depth.permute(0, 3, 1, 2)  # [B, 1, H, W]
    weight_chw = weight.permute(0, 3, 1, 2) if weight is not None else None  # [B, 1, H, W]

    # 5. 可见性相关字段：为简单起见，认为所有高斯均可见，radii 为 0
    # N = gaussians.get_xyz.shape[0]
    # visibility_filter = torch.ones(N, dtype=torch.bool, device=device)
    # radii_full = torch.zeros(N, dtype=torch.int32, device=device)

    result = {
        "render": image_chw,                # (B, 3, H, W)
        # "viewspace_points": screenspace_points,
        # "visibility_filter": visibility_filter,
        # "depth_filter": visibility_filter,
        # "radii": radii_full,
        "depth": depth_chw,                 # (B, 1, H, W)
    }
    if weight_chw is not None:
        result["weight"] = weight_chw  # (B, 1, H, W)
    return result

def render_image(
        gaussians,
        viewpoint_camera,
        device:torch.device,
        bg_color:torch.Tensor=None,
        dxyz=None,
        drot=None,
        gs_dict_add=None,
):
    """
    Render the scene. 
    """
    bg_color = torch.tensor(
        [1, 1, 1],
        dtype=torch.float32,
        device=device) if bg_color is None else bg_color
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(
        gaussians.get_xyz,
        dtype=gaussians.get_xyz.dtype,
        requires_grad=True,
        device=device,
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera['FoVx'] * 0.5)
    tanfovy = math.tan(viewpoint_camera['FoVy'] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera['image_height']),
        image_width=int(viewpoint_camera['image_width']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera['world_view_transform'],
        projmatrix=viewpoint_camera['full_proj_transform'],
        sh_degree=gaussians.active_sh_degree,
        campos=viewpoint_camera['camera_center'],
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = gaussians.get_xyz if dxyz is None \
        else gaussians.get_xyz + dxyz
    means2D = screenspace_points
    opacity = gaussians.get_opacity

    # filter out invisible Gaussians
    valid = filter_gaussians(viewpoint_camera, xyz=means3D.detach())

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    cov3D_precomp = None
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation if drot is None \
        else quaternion_multiply(drot, gaussians.get_rotation)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    colors_precomp = None
    shs = gaussians.get_features
    means3D = means3D[valid]
    means2D = means2D[valid]
    shs = shs[valid]
    opacity = opacity[valid]
    scales = scales[valid]
    rotations = rotations[valid]
    # 若当前视角下没有任何可见高斯，则直接返回背景图与零深度，避免在空输入上调用渲染器并导致反传错误
    if means3D.shape[0] == 0:
        print("No visible Gaussians, returning background image and zero depth")
        H = int(viewpoint_camera['image_height'])
        W = int(viewpoint_camera['image_width'])
        # 创建有梯度的 tensor，通过 dxyz 或 gaussians.get_xyz 来保持梯度连接
        # 如果 dxyz 存在，使用它来创建有梯度的零 tensor；否则使用 gaussians.get_xyz
        if dxyz is not None and dxyz.numel() > 0:
            # 使用 dxyz 的第一个元素来创建有梯度的零 tensor
            zero_grad = dxyz[0:1, 0:1].sum() * 0.0  # 保持梯度连接但值为0
        else:
            # 使用 gaussians.get_xyz 来创建有梯度的零 tensor
            zero_grad = gaussians.get_xyz[0:1, 0:1].sum() * 0.0 if gaussians.get_xyz.numel() > 0 else torch.tensor(0.0, device=device, requires_grad=True)
        
        # 确保 rendered_image 和 depth 都有梯度连接
        # 使用 zero_grad 来创建有梯度的 tensor，然后加上背景色（背景色不需要梯度）
        rendered_image = (zero_grad.expand(3, H, W) + bg_color.reshape(3, 1, 1).expand(3, H, W)).contiguous()
        depth = zero_grad.expand(1, H, W).contiguous()
        visibility_filter = torch.zeros_like(valid, dtype=torch.bool, device=device)
        radii_full = torch.zeros_like(valid, dtype=torch.int, device=device)
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": visibility_filter,
            'depth_filter': valid,
            "radii": radii_full,
            "depth": depth,
        }
    if gs_dict_add is not None:
        means3D = torch.cat([means3D, gs_dict_add['means3D']], dim=0)
        means2D = torch.cat([means2D, gs_dict_add['means2D']], dim=0)
        shs = torch.cat([shs, gs_dict_add['sh']], dim=0)
        opacity = torch.cat([opacity, gs_dict_add['opacity']], dim=0)
        scales = torch.cat([scales, gs_dict_add['scales']], dim=0)
        rotations = torch.cat([rotations, gs_dict_add['rotations']], dim=0)
        valid = torch.cat([valid, gs_dict_add['valid']], dim=0)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    visibility_filter = torch.zeros_like(valid, dtype=torch.bool, device=device)
    try:
        visibility_filter[valid] = radii > 0
    except RuntimeError:
        print("Error in visibility filter")
        visibility_filter[valid] = 1
    radii_full = torch.zeros_like(valid, dtype=torch.int, device=device)
    radii_full[valid] = radii
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility_filter,
        'depth_filter': valid,
        "radii": radii_full,
        "depth": depth,
    }

def construct_init_gaussian(
    data_dict: Dict,
    num_gaussian: int,
    scale: float = 0.04
) -> Dict:
    '''
    Args:
        data_dict (Dict)
        {
            "images": torch.Tensor (M, 3, H, W) [0,1],
            "depths": torch.Tensor (M, 1, H, W) ,
            "pcds": torch.Tensor (M, 3, H, W) ,
            "views": List[Dict] (M, ),
            "frame_ids": torch.Tensor (M,),
            "cam_names": List[str] (M,),
        }
        num_gaussian (int): if -1, use all points
        scale (float): init scale value
    Return:
        gs_dict (Dict)
        {
            "xyzs": pc_all, np.array, (N, 3)
            "rgbs": rgb_all, np.array, (N, 3)
            "scales": scales, np.array, (N, 3)
            "rotations": rotations, np.array, (N, 4)
            "opacities": opacities, np.array, (N, 1)
        }
    '''
    rgbs = einops.rearrange(data_dict['images'], 'm c h w -> (m h w) c')
    xyzs = einops.rearrange(data_dict['pcds'], 'm c h w -> (m h w) c')
    if num_gaussian > 0:
        index = common_utils.sample_points(N=rgbs.shape[0], target_N=num_gaussian)
        rgbs = rgbs[index]
        xyzs = xyzs[index]
    scales = np.ones((num_gaussian, 3)) * scale
    rotations = np.zeros((num_gaussian, 4))
    rotations[:, -1] = 1
    opacities = np.ones((num_gaussian, 1))
    return {
            "xyzs": xyzs,
            "rgbs": rgbs,
            "scales": scales,
            "rotations": rotations,
            "opacities": opacities,
        }

def compute_render_loss(
    image: torch.Tensor,
    depth: torch.Tensor,
    gt_image: torch.Tensor,
    gt_depth: torch.Tensor,
    lambda_dssim: float,
    lambda_depth: float,
    weight_image: torch.Tensor = None,
    canon_means_sampled: torch.Tensor = None,
    deform_means_sampled: torch.Tensor = None,
    rigid_weights_sampled: torch.Tensor = None,
    lambda_rigid: float = 0.0,
    rigid_k: int = 20,
) -> torch.Tensor:
    '''
    Args:
        image: torch.Tensor, (B, C, H, W)
        depth: torch.Tensor, (B, C, H, W)
        gt_image: torch.Tensor, (B, C, H, W)
        gt_depth: torch.Tensor, (B, C, H, W)
        lambda_dssim: float,
        lambda_depth: float,
        canon_means_sampled: (M, 3) sampled canonical gaussian centers (t=0), pre-sampled outside torch.compile
        deform_means_sampled: (M, 3) sampled deformed gaussian centers (current t), pre-sampled outside torch.compile
        lambda_rigid: weight for local rigidity loss (0 disables)
        rigid_k: number of neighbors for local rigidity loss
    '''
    w = weight_image

    # RGB loss (optionally weighted)
    if w is None:
        Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))
    else:
        Ll1 = (image - gt_image).abs()
        ssim_val = 1 - ssim(image, gt_image, size_average=False)
        image_loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * ssim_val
        image_loss = w * image_loss
        image_loss = image_loss.mean()

    # Depth loss (optionally weighted)
    if w is None:
        Ll1 = l1_loss(depth, gt_depth)
        depth_loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(depth, gt_depth))
    else:
        Ll1 = (depth - gt_depth).abs()
        ssim_val = 1 - ssim(depth, gt_depth, size_average=False)
        depth_loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * ssim_val
        depth_loss = w * depth_loss
        depth_loss = depth_loss.mean()
    rigidity_loss = 0.0
    if lambda_rigid is not None and float(lambda_rigid) > 0.0:
        if canon_means_sampled is not None and deform_means_sampled is not None:
            rigidity_loss = compute_local_rigidity_loss_sampled(
                orig_xyz_sampled=canon_means_sampled,
                pred_xyz_sampled=deform_means_sampled,
                k=rigid_k,
            )
            
    loss = image_loss + lambda_depth * depth_loss + float(lambda_rigid) * rigidity_loss
    return loss
    
def train_canonical_gaussian(
    gs_dict: Dict,
    data_dict: Dict,
    max_iter: int,
    device: torch.device,
    sh_degree: int = 3,
    densify_from_iter: int = 500,
    densification_interval: int = 100,
    densify_until_iter: int = 3000,
    densify_grad_threshold: float = 0.0002,
    opacity_reset_interval: int = 1000,
    size_threshold: int = 20,
    cameras_extent: int = 1.38,
    lambda_dssim: float = 0.2,
    lambda_depth: float = 1.0,
    lambda_rigid: float = 0.0,
    rigid_k: int = 20,
    percent_dense: float = 0.01,
    position_lr_init: float = 0.00016,
    position_lr_final: float = 0.0000016,
    position_lr_delay_mult: float = 0.01,
    position_lr_max_steps: float = 30_000,
    feature_lr: float = 0.0025,
    opacity_lr: float = 0.05,
    scaling_lr: float = 0.001,
    rotation_lr: float = 0.001,

) -> Dict:
    '''
    Args:
        gs_dict (Dict)
        {
            "xyzs": pc_all, np.array, (N, 3)
            "rgbs": rgb_all, np.array, (N, 3)
            "scales": scales, np.array, (N, 3)
            "rotations": rotations, np.array, (N, 4)
            "opacities": opacities, np.array, (N, 1)
        }
        data_dict (Dict)
        {
            "images": torch.Tensor (M, 3, H, W) [0,1],
            "depths": torch.Tensor (M, 1, H, W) ,
            "pcds": torch.Tensor (M, 3, H, W) ,
            "views": List[Dict] (M, ),
            "frame_ids": torch.Tensor (M,),
            "cam_names": List[str] (M,),
        }
        num_gaussian (int): if -1, use all points
    Return:
        gs_dict
    '''
    gaussians = GaussianModel(sh_degree)
    gaussians.create_from_gs_dict(gs_dict, device)
    gaussians.training_setup(
        percent_dense=percent_dense,
        position_lr_init=position_lr_init,
        position_lr_final=position_lr_final,
        position_lr_delay_mult=position_lr_delay_mult,
        position_lr_max_steps=position_lr_max_steps,
        feature_lr=feature_lr,
        opacity_lr=opacity_lr,
        scaling_lr=scaling_lr,
        rotation_lr=rotation_lr,
    )
    for iteration in tqdm(range(1, max_iter+1), desc="train_canonical_gaussian"):
        # sample one data from all
        idx = random.choice(range(len(data_dict['views'])))
        render_dict = render_image(
            gaussians,
            data_dict['views'][idx],
            device=device,
        )
        viewspace_point_tensor = render_dict["viewspace_points"]
        visibility_filter = render_dict["visibility_filter"]
        radii = render_dict["radii"]
        
        loss = compute_render_loss(
            image=render_dict['render'],
            depth=render_dict['depth'],
            gt_image=data_dict['images'][idx],
            gt_depth=data_dict['depths'][idx],
            lambda_dssim=lambda_dssim,
            lambda_depth=lambda_depth,
            weight_image=None,  # No weight_image in canonical training
            canon_means_sampled=None,
            deform_means_sampled=None,
            rigid_weights_sampled=None,
            lambda_rigid=0.0,
            rigid_k=20,
        )
        loss.backward()
        with torch.no_grad():
            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter])
            # Densification
            if iteration < densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter])
                gaussians.add_densification_stats(
                    viewspace_point_tensor, 
                    visibility_filter)
                if iteration > densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = size_threshold if iteration > opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        densify_grad_threshold,
                        min_opacity = 0.005,
                        extent = cameras_extent,
                        max_screen_size = size_threshold,
                        )
                if iteration % opacity_reset_interval == 0 or (iteration == densify_from_iter):
                    gaussians.reset_opacity()
            gaussians.optimizer.step()
            gaussians.update_learning_rate(iteration)
            gaussians.optimizer.zero_grad(set_to_none=True)
    return gaussians

def evaluate_gaussian(
    gaussian: Dict,
    data_dict: Dict,
    device: torch.device,
) -> Dict:
    '''
    Args:
        gs_dict (Dict)
        {
            "xyzs": pc_all, np.array, (N, 3)
            "rgbs": rgb_all, np.array, (N, 3)
            "scales": scales, np.array, (N, 3)
            "rotations": rotations, np.array, (N, 4)
            "opacities": opacities, np.array, (N, 1)
        }
        data_dict (Dict)
        {
            "images": torch.Tensor (M, 3, H, W) [0,1],
            "depths": torch.Tensor (M, 1, H, W) ,
            "pcds": torch.Tensor (M, 3, H, W) ,
            "views": List[Dict] (M, ),
            "frame_ids": torch.Tensor (M,),
            "cam_names": List[str] (M,),
        }
        num_gaussian (int): if -1, use all points
    Return:
        eval_dict (Dict)
        {
            "psnr_list": torch.Tensor (M,)
            "image_list": torch.Tensor (M, 3, H, W) [0,1],
            "gs_image_list": torch.Tensor (M, 3, H, W) [0,1],
        }
    '''
    psnr_list, image_list, gt_image_list = [], [], []
    for idx_cam, cam_name in enumerate(data_dict['cam_names']):
        render_dict = render_image(
            gaussian,
            data_dict['views'][idx_cam],
            device=device,
        )
        image = render_dict['render'] # (3, h, w) [0,1] torch.Tensor
        gt_image = data_dict['images'][idx_cam] # (3, h, w) [0,1] torch.Tensor
        psnr = metric_utils.compute_psnr_ts(
            einops.rearrange(image,  'c h w -> 1 h w c'),
            einops.rearrange(gt_image,  'c h w -> 1 h w c'),
        )
        psnr_list.append(psnr)
        image_list.append(image)
        gt_image_list.append(gt_image)
    return {
        "psnr_list": psnr_list,
        "image_list": image_list,
        "gt_image_list": gt_image_list,
    }

def load_gaussian(ckpt_path: str):
    m = GaussianModel(sh_degree=0)
    m.load_ckpt(ckpt_path)
    return m

class GaussianModel:
    def __init__(self, sh_degree: int):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.spatial_lr_scale = 5
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        # self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        self.hyper_activation = lambda x: torch.nn.functional.softmax(x, dim=-1)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        # features_rest = self._features_rest
        # return torch.cat((features_dc, features_rest), dim=1)
        return features_dc

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def create_from_gs_dict(self, gs_dict: Dict, device: torch.device):
        fused_point_cloud = gs_dict['xyzs'].float()
        fused_color = RGB2SH(gs_dict['rgbs'].float())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), device=device, dtype=torch.float).float()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(gs_dict['xyzs'].float().to(device)), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=device)
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)
        return

    def create_from_xyz_rgb(self, xyz: torch.Tensor, rgb: torch.Tensor):
        device = xyz.device
        fused_point_cloud = xyz
        fused_color = RGB2SH(rgb)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(device)
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(xyz.to(device)), 0.0000001)
        # scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        scales = torch.ones((fused_point_cloud.shape[0], 3), device=device) * 0.02
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=device)
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)
        return

    def training_setup(
        self,
        percent_dense,
        position_lr_init,
        position_lr_final,
        position_lr_delay_mult,
        position_lr_max_steps,
        feature_lr,
        opacity_lr,
        scaling_lr,
        rotation_lr,
    ):
        self.percent_dense = percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        spatial_lr_scale = self.spatial_lr_scale
        l = [
            {'params': [self._xyz], 'lr': position_lr_init * spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': scaling_lr * spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=position_lr_init * spatial_lr_scale,
                                                    lr_final=position_lr_final * spatial_lr_scale,
                                                    lr_delay_mult=position_lr_delay_mult,
                                                    max_steps=position_lr_max_steps)
        return

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def save_ckpt(self, ckpt_path: str):
        weight_dict = {
            "xyzs": self._xyz,
            "opacities": self._opacity,
            "scales": self._scaling,
            "rotations": self._rotation,
            "features_dc": self._features_dc,
            # "features_rest": self._features_rest,
        }
        common_utils.write_pkl(
            weight_dict,
            ckpt_path,
        )
        return

    def load_ckpt(self, ckpt_path: str):
        weight_dict = common_utils.read_pkl(ckpt_path)
        self._xyz = weight_dict["xyzs"]
        self._opacity = weight_dict["opacities"]
        self._scaling = weight_dict["scales"]
        self._rotation = weight_dict["rotations"]
        self._features_dc = weight_dict["features_dc"]
        # self._features_rest = weight_dict["features_rest"]
        return

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        return

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        return

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
            }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        return

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        # new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, None, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        # new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, None, new_opacities, new_scaling,
                                   new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            # prune_mask = torch.logical_or(prune_mask, big_points_ws)
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1
