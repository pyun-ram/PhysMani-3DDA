import torch
from torch.autograd import Function
import fps_cuda

class FarthestPointSampling(Function):
    @staticmethod
    def forward(ctx, pos, npoint, start_idx=None):
        '''
        Args:
            pos: torch.Tensor, (B, N, C)
            npoint: int
            start_idx: torch.Tensor, (B, )
        Returns:
            idx: torch.Tensor, (B, npoint)
        '''
        # pos: (B, N, C)
        assert pos.is_cuda
        B, N, C = pos.shape
        idx = torch.zeros((B, npoint), dtype=torch.long, device=pos.device)
        if start_idx is not None:
            raise NotImplementedError
        else:
            fps_cuda.farthest_point_sampling(pos, idx, npoint, torch.empty(0, dtype=torch.int, device=pos.device))
        return idx

    @staticmethod
    def backward(ctx, grad_output):
        # No gradient for indices
        return None, None, None

def farthest_point_sampling(pos, npoint, start_idx=None):
    return FarthestPointSampling.apply(pos, npoint, start_idx)
class FarthestPointSamplingWithMask(Function):
    @staticmethod
    def forward(ctx, pos, npoint, mask, start_idx=None):
        '''
        Args:
            pos: torch.Tensor, (B, N, C)
            npoint: int
            mask: torch.Tensor, (B, N), True, if it is valid to sample.
            start_idx: torch.Tensor, (B, )
        Returns:
            idx: torch.Tensor, (B, npoint)
        '''
        # pos: (B, N, C)
        assert len(pos.shape) == 3
        assert len(mask.shape) == 2
        assert torch.all(mask.sum(dim=-1) > 0)
        assert pos.is_cuda
        B, N, C = pos.shape
        idx = torch.zeros((B, npoint), dtype=torch.long, device=pos.device)
        if start_idx is not None:
            raise NotImplementedError
        else:
            fps_cuda.farthest_point_sampling_with_mask(pos, idx, npoint, mask, torch.empty(0, dtype=torch.int, device=pos.device))
        return idx

    @staticmethod
    def backward(ctx, grad_output):
        # No gradient for indices
        return None, None, None

def farthest_point_sampling_with_mask(pos, npoint, mask, start_idx=None):
    return FarthestPointSamplingWithMask.apply(pos, npoint, mask, start_idx)