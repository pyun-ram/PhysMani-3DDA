#include "fps.h"

void fps_launcher(
    at::Tensor pos, at::Tensor idx, int npoint, at::Tensor start_idx);

void farthest_point_sampling(
    at::Tensor pos, at::Tensor idx, int npoint, at::Tensor start_idx) {
    fps_launcher(pos, idx, npoint, start_idx);
}

void fps_with_mask_launcher(
    at::Tensor pos, at::Tensor idx, int npoint, at::Tensor mask, at::Tensor start_idx);

void farthest_point_sampling_with_mask(
    at::Tensor pos, at::Tensor idx, int npoint, at::Tensor mask, at::Tensor start_idx) {
    fps_with_mask_launcher(pos, idx, npoint, mask, start_idx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("farthest_point_sampling", &farthest_point_sampling, "Farthest Point Sampling (CUDA)");
    m.def("farthest_point_sampling_with_mask", &farthest_point_sampling_with_mask, "FPS with Mask (CUDA)");
}
