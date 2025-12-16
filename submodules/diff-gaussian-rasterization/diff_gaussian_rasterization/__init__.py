# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch
import torch.nn as nn
from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


# -------------------- Original Rasterizer -------------------- #
def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3D, means2D, sh, colors_precomp, opacities,
                scales, rotations, cov3Ds_precomp, raster_settings):

        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug
        )

        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp,
                              radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, invdepths

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth):
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        args = (
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            grad_out_depth,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.antialiasing,
            raster_settings.debug
        )

        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = \
            _C.rasterize_gaussians_backward(*args)

        return (grad_means3D, grad_means2D, grad_sh, grad_colors_precomp, grad_opacities,
                grad_scales, grad_rotations, grad_cov3Ds_precomp, None)


# -------------------- Contrib Rasterizer (updated) -------------------- #
class _RasterizeGaussiansWithContrib(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3D, means2D, sh, colors_precomp, opacities,
                scales, rotations, cov3Ds_precomp, raster_settings, K):
        """
        Calls the C++/CUDA top-K rasterizer and returns the contrib arrays.

        The compiled native function `rasterize_gaussians_with_contrib` is expected
        to return (in this order):
          rendered (int),
          out_color [C,H,W],
          radii [P],
          geomBuffer (byte tensor),
          binningBuffer (byte tensor),
          imgBuffer (byte tensor),
          out_invdepth [1,H,W],
          pixel_to_gaussian_idx [H, W, K]   (int)
          pixel_to_gaussian_alpha [H, W, K] (float)
          pixel_to_gaussian_count [H, W]    (int)

        We only return the top-K arrays to Python here because color/radii/invdepth
        are already produced by the standard rasterize_gaussians() call.
        """

        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            K,
            raster_settings.debug
        )

        # Call the native (single-step) top-K rasterizer. It returns many things;
        # we unpack and return only the contributor outputs here.
        (
            rendered,
            out_color,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            out_invdepth,
            pixel_to_gaussian_idx,
            pixel_to_gaussian_alpha,
            pixel_to_gaussian_count
        ) = _C.rasterize_gaussians_with_contrib(*args)

        # No gradients for the top-K bookkeeping tensors (non-differentiable),
        # so we don't save any tensors in ctx. Return the raw top-K outputs.
        return pixel_to_gaussian_idx, pixel_to_gaussian_alpha, pixel_to_gaussian_count

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Top-K aggregation / bookkeeping is non-differentiable in our pipeline.
        # Return Nones for all inputs (we received 10 inputs to forward, so match).
        # Inputs to forward: means3D, means2D, sh, colors_precomp, opacities,
        # scales, rotations, cov3Ds_precomp, raster_settings, K
        return (None,)*10


# -------------------- Settings -------------------- #
class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    antialiasing: bool


# -------------------- Rasterizer Module -------------------- #
class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings: GaussianRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Marks points visible based on camera frustum
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix
            )
        return visible

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None,
                scales=None, rotations=None, cov3D_precomp=None, dc=None, contrib=False, K=8):

        raster_settings = self.raster_settings

        # Handle optional inputs
        shs = shs if shs is not None else torch.Tensor([])
        colors_precomp = colors_precomp if colors_precomp is not None else torch.Tensor([])
        scales = scales if scales is not None else torch.Tensor([])
        rotations = rotations if rotations is not None else torch.Tensor([])
        cov3D_precomp = cov3D_precomp if cov3D_precomp is not None else torch.Tensor([])

        if (shs.shape[0] != 0 and colors_precomp.shape[0] != 0) or (shs.shape[0] == 0 and colors_precomp.shape[0] == 0):
            raise Exception('Please provide exactly one of either SHs or precomputed colors!')

        if ((scales.shape[0] == 0 or rotations.shape[0] == 0) and cov3D_precomp.shape[0] == 0) or \
        ((scales.shape[0] != 0 or rotations.shape[0] != 0) and cov3D_precomp.shape[0] != 0):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        # 1) Standard rasterization
        color, radii, invdepth = rasterize_gaussians(
            means3D, means2D, shs, colors_precomp, opacities,
            scales, rotations, cov3D_precomp, raster_settings
        )

        if not contrib:
            return color, radii, invdepth

        # 2) Single-step top-K rasterizer: get raw top-K outputs directly
        pixel_to_gaussian_idx, pixel_to_gaussian_alpha, pixel_to_gaussian_count = _RasterizeGaussiansWithContrib.apply(
            means3D, means2D, shs, colors_precomp, opacities,
            scales, rotations, cov3D_precomp, raster_settings, K
        )

        # # DEBUG: check top-K processed outputs (keep for debugging)
        # print("DEBUG: Top-K contrib indices shape:", pixel_to_gaussian_idx.shape)
        # print("DEBUG: Top-K contrib opacities shape:", pixel_to_gaussian_alpha.shape)
        # print("DEBUG: Top-K contrib counts shape:", pixel_to_gaussian_count.shape)
        # print("DEBUG: Sample top-K contrib indices [0,0,:]:", pixel_to_gaussian_idx[0,0,:])
        # print("DEBUG: Sample top-K contrib opacities [0,0,:]:", pixel_to_gaussian_alpha[0,0,:])

        # SANITY CHECK: sum of top-K opacities per pixel
        pixel_opacity_sum = pixel_to_gaussian_alpha.sum(dim=-1)
        max_sum = pixel_opacity_sum.max().item()
        min_sum = pixel_opacity_sum.min().item()
        # print(f"DEBUG: Pixel opacity sum min/max: {min_sum:.4f}/{max_sum:.4f}")

        # if max_sum > 1.1:  # small tolerance
        #     print("WARNING: Some pixels have top-K opacity sum > 1.0. Check rasterization.")

        # Return exactly the five items your Python code expects:
        # color, radii, invdepth, indices, opacities
        return color, radii, invdepth, pixel_to_gaussian_idx, pixel_to_gaussian_alpha

