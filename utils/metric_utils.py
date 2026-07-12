import torch
import numpy as np
import time
from . import common_utils
from torch.utils.tensorboard import SummaryWriter
class MetricLogger:
    def __init__(self, log_dir):
        self._data = {}
        self._t0_dict = {}
        self._log_dir = log_dir
        self._tb_writer = SummaryWriter(log_dir)
        return
    
    def tik(self, name: str):
        assert name not in self._t0_dict
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0_dict[name] = time.time()
        return
    
    def tok(self, name: str):
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
    
    def log_tb(self, name: str, iter: int, val: float):
        self._tb_writer.add_scalar(name, val, iter)
        return
    
    def save(self, path: str=None):
        if path is None:
            path = Path(self.log_dir) / "metric_dict.json"
        common_utils.write_json(self._data, path)
    

def compute_psnr(est: np.ndarray, gt: np.ndarray) -> float:
    '''
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        est (np.ndarray): Estimated image [H, W, 3] with pixel values in range [0, 1].
        gt (np.ndarray): Ground truth image [H, W, 3] with pixel values in range [0, 1].

    Returns:
        float: PSNR value.
    '''
    # Convert images to float32 for precision
    est = est.astype(np.float32)
    gt = gt.astype(np.float32)
    
    # Compute Mean Squared Error (MSE)
    mse = np.mean((est - gt) ** 2)
    
    if mse == 0:
        return float('inf')  # Infinite PSNR for identical images
    
    # Maximum possible pixel value
    max_pixel_value = 1.0
    
    # Compute PSNR
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
    return psnr

def compute_psnr_ts(est: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    '''
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        est (torch.Tensor): Estimated image [B, 3, H, W] with pixel values in range [0, 1].
        gt (torch.Tensor): Ground truth image [B, 3, H, W] with pixel values in range [0, 1].
    Returns:
        torch.Tensor: PSNR values (B,).
    '''
    # Ensure the inputs are float tensors
    est = est.float()
    gt = gt.float()

    # Compute Mean Squared Error (MSE) for each batch
    mse = torch.mean((est - gt) ** 2, dim=[1, 2, 3])  # Reduction over H, W, and C

    # Avoid division by zero in PSNR calculation
    mse = torch.clamp(mse, min=1e-10)

    # Compute PSNR
    max_pixel_value = 1.0
    psnr = 10.0 * torch.log10((max_pixel_value ** 2) / mse)

    return psnr

def compute_l2error_ts(est: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    '''
    Compute l2 error between two depth images

    Args:
        est (torch.Tensor): Estimated image [B, 1, H, W] with depth values
        gt (torch.Tensor): Ground truth image [B, 1, H, W] with depth values
    Returns:
        torch.Tensor: (B,).
    '''
    # 计算 L2 误差
    l2_error = torch.norm(est - gt, p=2, dim=(1, 2, 3))  # 计算每个图像的 L2 范数
    return l2_error

def compute_log_rmse_ts(
    est: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-6,
    mask_invalid: bool = True,
) -> torch.Tensor:
    """
    Compute log RMSE between two depth images.

    Args:
        est: [B, 1, H, W] estimated depth
        gt:  [B, 1, H, W] ground-truth depth
        eps: numerical stability for log/clamp and division
        mask_invalid: if True, compute only where est>0 and gt>0

    Returns:
        Tensor of shape (B,) : log RMSE per image
    """
    assert est.shape == gt.shape, f"Shape mismatch: est {est.shape}, gt {gt.shape}"
    B = est.shape[0]

    est = est.float()
    gt = gt.float()

    if mask_invalid:
        valid = (gt > 0) & (est > 0)
    else:
        valid = torch.ones_like(gt, dtype=torch.bool)

    # clamp to avoid log(0) or log(negative)
    log_est = torch.log(torch.clamp(est, min=eps))
    log_gt  = torch.log(torch.clamp(gt,  min=eps))

    diff2 = (log_est - log_gt) ** 2  # [B,1,H,W]

    # mean over valid pixels per batch item
    valid_f = valid.float()
    denom = valid_f.sum(dim=(1, 2, 3)).clamp_min(1.0)  # avoid div by 0

    mse = (diff2 * valid_f).sum(dim=(1, 2, 3)) / denom
    rmse = torch.sqrt(mse + eps)  # (B,)
    return rmse