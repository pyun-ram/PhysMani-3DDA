#pragma once
#include <torch/extension.h>

void fps_launcher(
    at::Tensor pos, at::Tensor idx, int npoint, at::Tensor start_idx);

void fps_with_mask_launcher(
    at::Tensor pos, at::Tensor idx, int npoint, at::Tensor mask, at::Tensor start_idx);