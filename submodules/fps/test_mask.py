import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))
import time
from fps import farthest_point_sampling_with_mask
import dgl.geometry as dgl_geo
from utils import common_utils

def test_fps_with_mask_match_dglgeo():
    torch.manual_seed(0)
    B, N, C = 24, 4096, 120
    npoint = 819
    pos = common_utils.read_pkl('submodules/fps/test_fps_data.pkl')['context_features'].cuda()
    # 生成mask，保证每个batch至少有npoint个True
    mask = torch.zeros((B, N), dtype=torch.bool, device='cuda')
    for b in range(B):
        idx = torch.randperm(N, device='cuda')[:npoint+10]
        mask[b, idx] = True

    # 你的采样
    t3 = time.time()
    torch.cuda.synchronize()
    my_idx = farthest_point_sampling_with_mask(pos, npoint, mask, start_idx=None)
    torch.cuda.synchronize()
    t4 = time.time()

    # DGL每个batch单独采样
    t1 = time.time()
    torch.cuda.synchronize()
    dgl_idx = torch.zeros((B, npoint), dtype=torch.long, device='cuda')
    for b in range(B):
        valid_idx = torch.where(mask[b])[0]
        valid_pos = pos[b][valid_idx].unsqueeze(0).to(torch.float64)  # (1, N_valid, C)
        if valid_pos.shape[1] < npoint:
            # 有效点不足，补齐
            repeat_times = (npoint + valid_pos.shape[1] - 1) // valid_pos.shape[1]
            valid_pos = valid_pos.repeat(1, repeat_times, 1)
            valid_idx = valid_idx.repeat(repeat_times)[:npoint+1]
        fps_idx = dgl_geo.farthest_point_sampler(valid_pos, npoint, 0).long().squeeze(0)
        dgl_idx[b] = valid_idx[fps_idx]
    torch.cuda.synchronize()
    t2 = time.time()

    # 比较采样结果集合是否一致（顺序无关）
    for b in range(B):
        my_set = set(my_idx[b].cpu().tolist())
        dgl_set = set(dgl_idx[b].cpu().tolist())
        assert my_set == dgl_set, f"Mismatch in batch {b}: {my_set ^ dgl_set}"
    print(f"DGL-GEO time: {t2 - t1}")
    print(f"My FPS time: {t4 - t3}")
    print("Test passed: Both FPS-with-mask results are identical for all batches.")

if __name__ == "__main__":
    test_fps_with_mask_match_dglgeo()
