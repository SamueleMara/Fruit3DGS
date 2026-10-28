# import torch
# import diff_gaussian_rasterization._C as raster

# # Dummy inputs
# H, W, P, K = 4, 4, 5, 3
# background = torch.zeros(H, W, 3, device='cuda', dtype=torch.float32)
# means3D = torch.rand(P, 3, device='cuda', dtype=torch.float32)
# colors = torch.rand(P, 3, device='cuda', dtype=torch.float32)
# opacity = torch.rand(P, device='cuda', dtype=torch.float32)
# scales = torch.rand(P, device='cuda', dtype=torch.float32)
# rotations = torch.rand(P, device='cuda', dtype=torch.float32)
# cov3D_precomp = torch.rand(P, 3, 3, device='cuda', dtype=torch.float32)
# viewmatrix = torch.eye(4, device='cuda')
# projmatrix = torch.eye(4, device='cuda')
# sh = torch.zeros(P, 9, device='cuda')
# campos = torch.zeros(3, device='cuda')
# tan_fovx = 1.0
# tan_fovy = 1.0

# # --- Standard rasterizer ---
# rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth = raster.rasterize_gaussians(
#     background, means3D, colors, opacity, scales, rotations,
#     1.0, cov3D_precomp, viewmatrix, projmatrix,
#     tan_fovx, tan_fovy, H, W, sh, 2, campos, False, True, False
# )
# print("Standard rasterizer out_color:", out_color.shape)

# # --- Top-K contributor rasterizer ---
# rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth, topk_idx, topk_val, topk_count = raster.rasterize_gaussians_with_contrib(
#     background, means3D, colors, opacity, scales, rotations,
#     1.0, cov3D_precomp, viewmatrix, projmatrix,
#     tan_fovx, tan_fovy, H, W, sh, 2, campos, False, True, K, False
# )
# print("Top-K rasterizer topk_idx:", topk_idx.shape)
# print("Top-K rasterizer topk_val:", topk_val.shape)

from ._C import rasterize_gaussians as _RasterizeGaussians
from ._C import rasterize_gaussians_with_contrib as _RasterizeGaussiansWithContrib

# Expose both as Python-callable functions
rasterize_gaussians = _RasterizeGaussians
rasterize_gaussians_with_contrib = _RasterizeGaussiansWithContrib
