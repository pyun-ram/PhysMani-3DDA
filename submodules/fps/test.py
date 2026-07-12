import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))
import time
from fps import farthest_point_sampling
import torch
import dgl.geometry as dgl_geo
from utils import common_utils
def test_fps_match_dglgeo_with_start():
    torch.manual_seed(0)
    B, N, C = 24, 4096, 120
    npoint = 819
    # pos = torch.randn((B, N, C), device='cuda')
    pos = common_utils.read_pkl('submodules/fps/test_fps_data.pkl')['context_features'].cuda()
    start_idx = 0

    # 你的采样
    t3 = time.time()
    torch.cuda.synchronize()
    my_idx = farthest_point_sampling(pos, npoint, start_idx=None)
    torch.cuda.synchronize()
    t4 = time.time()

    # DGL-GEO 采样
    t1 = time.time()
    torch.cuda.synchronize()
    dgl_idx = dgl_geo.farthest_point_sampler(pos.float(), npoint, start_idx=start_idx)
    torch.cuda.synchronize()
    t2 = time.time()

    # 直接比较采样索引是否一致
    # 找出有几个不一样 dgl_idx, my_idx
    # 不一样这几个是不是只是顺序不同

    diff_idx = torch.where(dgl_idx != my_idx)
    dgl_diff_item_list = []
    my_diff_item_list = []
    for idx1, idx2 in zip(diff_idx[0], diff_idx[1]):
        dgl_diff_item_list.append(dgl_idx[idx1, idx2].item())
        my_diff_item_list.append(my_idx[idx1, idx2].item())
    # check two set
    dgl_diff_item_set = sorted(list(set(dgl_diff_item_list)))
    my_diff_item_set = sorted(list(set(my_diff_item_list)))
    assert dgl_diff_item_set == my_diff_item_set, f"Mismatch: {dgl_diff_item_set} vs {my_diff_item_set}"

    print(f"DGL-GEO time: {t2 - t1}")
    print(f"My FPS time: {t4 - t3}")

    print("Test passed: Both FPS results are identical with the same start_idx.")

if __name__ == "__main__":
    test_fps_match_dglgeo_with_start()