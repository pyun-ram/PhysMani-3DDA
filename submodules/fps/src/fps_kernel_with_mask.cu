#include <cuda_runtime.h>
#include <torch/extension.h>

#define THREADS_PER_BLOCK 256

template <typename scalar_t>
__global__ void fps_kernel_with_mask(
    const scalar_t* __restrict__ pos,    // (B*N, C)
    int64_t* __restrict__ idx,           // (B*npoint)
    const bool* __restrict__ mask,       // (B*N)
    const int B, const int N, const int C, const int npoint, const int* __restrict__ start_idx)
{
    int batch_idx = blockIdx.x;
    if (batch_idx >= B) return;

    const scalar_t* batch_pos = pos + batch_idx * N * C;
    const bool* batch_mask = mask + batch_idx * N;
    int64_t* batch_idx_out = idx + batch_idx * npoint;

    // 选起始点
    int cur_idx = 0;
    if (start_idx) {
        cur_idx = start_idx[batch_idx];
    } else {
        // 默认mask第一个True
        for (int i = 0; i < N; ++i) {
            if (batch_mask[i]) {
                cur_idx = i;
                break;
            }
        }
    }
    batch_idx_out[0] = cur_idx;

    extern __shared__ float temp_dist[]; // (N)
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        temp_dist[i] = batch_mask[i] ? 1e10f : -1.0f; // mask==False点不参与采样
    }
    __syncthreads();

    for (int i = 1; i < npoint; ++i) {
        int last_idx = batch_idx_out[i - 1];

        for (int j = threadIdx.x; j < N; j += blockDim.x) {
            if (!batch_mask[j]) continue;
            float dist = 0;
            for (int k = 0; k < C; ++k) {
                float diff = batch_pos[last_idx * C + k] - batch_pos[j * C + k];
                dist += diff * diff;
            }
            if (dist < temp_dist[j]) temp_dist[j] = dist;
        }
        __syncthreads();

        // 找最远的合法点
        float max_dist = -1.0f;
        int max_idx = 0;
        for (int j = threadIdx.x; j < N; j += blockDim.x) {
            if (!batch_mask[j]) continue;
            if (temp_dist[j] > max_dist) {
                max_dist = temp_dist[j];
                max_idx = j;
            }
        }

        __shared__ float max_dists[THREADS_PER_BLOCK];
        __shared__ int max_indices[THREADS_PER_BLOCK];
        max_dists[threadIdx.x] = max_dist;
        max_indices[threadIdx.x] = max_idx;
        __syncthreads();

        // 并行归约
        int tid = threadIdx.x;
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && max_dists[tid + s] > max_dists[tid]) {
                max_dists[tid] = max_dists[tid + s];
                max_indices[tid] = max_indices[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            batch_idx_out[i] = max_indices[0];
        }
        __syncthreads();
    }
}

void fps_with_mask_launcher(
    at::Tensor pos, at::Tensor idx, int npoint, at::Tensor mask, at::Tensor start_idx)
{
    const int B = pos.size(0);
    const int N = pos.size(1);
    const int C = pos.size(2);

    // mask 必须 bool
    auto mask_bool = mask.to(torch::kBool);

    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "fps_kernel_with_mask", ([&] {
        fps_kernel_with_mask<scalar_t><<<B, THREADS_PER_BLOCK, N * sizeof(float)>>>(
            pos.data_ptr<scalar_t>(),
            idx.data_ptr<int64_t>(),
            mask_bool.data_ptr<bool>(),
            B, N, C, npoint,
            start_idx.defined() ? start_idx.data_ptr<int>() : nullptr
        );
    }));
}
