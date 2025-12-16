/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */
#ifndef CUDA_RASTERIZER_FORWARD_WITH_CONTRIB_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_WITH_CONTRIB_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <functional>

namespace CudaRasterizer {

struct Rasterizer {
    static int forward_with_contrib(
        std::function<char*(size_t)> geomFunc,
        std::function<char*(size_t)> binningFunc,
        std::function<char*(size_t)> imgFunc,
        const int P, const int D, const int M,
        const float* background,
        const int W, const int H,
        const float* means3D,
        const float* sh,
        const float* colors,
        const float* opacity,
        const float* scales,
        const float scale_modifier,
        const float* rotations,
        const float* cov3D_precomp,
        const float* viewmatrix,
        const float* projmatrix,
        const float* campos,
        const float tan_fovx,
        const float tan_fovy,
        const bool prefiltered,
        float* out_color,
        float* out_invdepth,
        const bool antialiasing,
        int* radii,
        const bool debug,
        int* pixel_to_gaussian_idx,
        float* pixel_to_gaussian_alpha,
        int* pixel_to_gaussian_count,
        const int max_contributors
    );
};

} // namespace CudaRasterizer

#endif
