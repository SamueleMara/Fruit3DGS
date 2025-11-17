#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()
    
# -----------------------------
# Binary mask render loss (weighted)
# -----------------------------
def binary_mask_render_loss(gaussians_mask, contrib_indices, contrib_opacities, gt_mask, alpha_mask=None):
    """
    Compute differentiable binary mask loss for Gaussian contributions,
    optionally weighted by alpha_mask (e.g., mask-center weights).

    Args:
        gaussians_mask: [num_gaussians] learnable scalar per Gaussian (0..1)
        contrib_indices: [H, W, K] top-K Gaussian indices per pixel
        contrib_opacities: [H, W, K] alpha values of Gaussians per pixel
        gt_mask: [H, W] binary ground-truth mask (0 or 1)
        alpha_mask: optional [H, W] float mask to limit loss region

    Returns:
        scalar loss
    """
    H, W, K = contrib_indices.shape
    device = gaussians_mask.device

    valid_mask = (contrib_indices >= 0)
    safe_indices = torch.clamp(contrib_indices, min=0).long()

    f_i = gaussians_mask[safe_indices]  # [H, W, K]
    f_i = f_i * valid_mask.float()
    contrib_opacities = contrib_opacities * valid_mask.float()

    # Front-to-back alpha compositing
    alpha_prod = torch.cumprod(1.0 - contrib_opacities, dim=2)
    alpha_prod = torch.cat([torch.ones((H, W, 1), device=device), alpha_prod[:, :, :-1]], dim=2)
    F_rendered = (f_i * contrib_opacities * alpha_prod).sum(dim=2)  # [H, W]

    F_rendered = torch.clamp(F_rendered, 0.0, 1.0)

    if alpha_mask is not None:
        F_rendered = F_rendered * alpha_mask

    # Ensure same shape as gt_mask
    if F_rendered.shape != gt_mask.shape:
        F_rendered = F_rendered.squeeze(0)

    loss = F.binary_cross_entropy(F_rendered, gt_mask.float())
    return loss


# -----------------------------
# Compute mask-center weights
# -----------------------------
def compute_mask_center_weights(mask_tensor):
    """
    Given a mask [H, W] (binary), compute a weight map emphasizing center pixels.
    Returns float tensor [H, W] with values in [0,1].
    """
    from scipy.ndimage import distance_transform_edt
    import numpy as np

    mask_np = mask_tensor.cpu().numpy().astype(np.bool_)
    dist_map = distance_transform_edt(mask_np)
    if dist_map.max() > 0:
        dist_map = dist_map / dist_map.max()  # normalize to 0-1
    return torch.from_numpy(dist_map).float().to(mask_tensor.device)


# -----------------------------
# Pseudo-GT from top-K (weighted version)
# -----------------------------
def pseudo_gt_from_topK(gaussians, topK_contrib_indices, mask_images, mask_weights=None):
    """
    Compute per-Gaussian pseudo-ground-truth semantic values using
    precomputed top-K contributor pixel indices, optionally weighted by mask-center confidence.

    Args:
        gaussians (GaussianModel)
        topK_contrib_indices: [N, num_cameras, K] pixel indices
        mask_images: [num_cameras, H*W] normalized 0-1 masks
        mask_weights: [num_cameras, H*W] optional float weights per pixel

    Returns:
        tuple: (pseudo_gt, counts)
            pseudo_gt: [N] averaged mask values for each Gaussian
            counts: [N] sum of weights or valid contributions
    """
    if topK_contrib_indices is None or mask_images is None:
        raise ValueError("Both topK_contrib_indices and mask_images are required for pseudo-GT computation.")

    N, num_cameras, K = topK_contrib_indices.shape
    device = gaussians.semantic_mask.device

    pseudo_gt = torch.zeros(N, device=device, dtype=torch.float32)
    counts = torch.zeros(N, device=device, dtype=torch.float32)

    for cam_idx in range(num_cameras):
        pixels = topK_contrib_indices[:, cam_idx, :]  # [N, K]
        valid = pixels >= 0
        safe_pixels = pixels.clone()
        safe_pixels[~valid] = 0

        cam_mask = mask_images[cam_idx].view(-1)
        max_pix = cam_mask.numel()
        safe_pixels = torch.clamp(safe_pixels, 0, max_pix - 1)

        mask_vals = cam_mask[safe_pixels]  # [N,K]
        mask_vals[~valid] = 0.0

        if mask_weights is not None:
            weight_map = mask_weights[cam_idx].view(-1)
            weight_vals = weight_map[safe_pixels]
            weight_vals[~valid] = 0.0
        else:
            weight_vals = valid.float()

        pseudo_gt += (mask_vals * weight_vals).sum(dim=1)
        counts += weight_vals.sum(dim=1)

    valid_counts = counts > 0
    pseudo_gt[valid_counts] /= counts[valid_counts]

    return pseudo_gt, counts



# -----------------------------
# Semantic loss from pseudo-GT
# -----------------------------
def semantic_loss_from_topK(gaussians, pseudo_gt, counts, λ_sem=1.0):
    """
    Weighted MSE loss between predicted semantics and pseudo-GT mask values.

    Args:
        gaussians (GaussianModel): Contains .semantic_mask [N, 1]
        pseudo_gt (Tensor): [N] averaged semantic supervision
        counts (Tensor): [N] number of valid pixel contributions
        λ_sem (float): Weight of semantic loss

    Returns:
        Tensor: Weighted scalar loss
    """
    preds = torch.sigmoid(gaussians.semantic_mask.squeeze())  # [N]
    valid = counts > 0

    if valid.sum() == 0:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)

    weights = counts[valid] / (counts[valid].sum() + 1e-8)
    loss = ((preds[valid] - pseudo_gt[valid]) ** 2) * weights

    return λ_sem * loss.sum()


# -----------------------------
# Hybrid spatial-semantic loss
# -----------------------------
def hybrid_spatial_semantic_loss(positions, semantics, alpha=10.0, neighbor_radius=0.05, λ_sem=1.0, debug=False):
    """
    Compute a spatial + semantic smoothness loss among nearby Gaussians.

    Args:
        positions (Tensor): [N, 3] Gaussian centroids
        semantics (Tensor): [N] or [N, C] semantic values
        alpha (float): Relative weighting for the semantic part
        neighbor_radius (float): Maximum distance for neighbor consideration
        λ_sem (float): Weight scaling for semantic loss term
        debug (bool): If True, print loss components

    Returns:
        Tensor: Scalar total hybrid loss
    """
    N = positions.shape[0]
    if N <= 1:
        return torch.tensor(0.0, device=positions.device, requires_grad=True)

    pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [N, N, 3]
    dist = torch.norm(pos_diff, dim=-1)  # [N, N]

    spatial_mask = (dist < neighbor_radius).float()
    denom = max(1.0, spatial_mask.sum().item())

    spatial_loss = (dist * spatial_mask).sum() / denom

    if semantics.ndim == 1:
        semantic_diff = (semantics.unsqueeze(1) - semantics.unsqueeze(0)).abs()
    else:
        semantic_diff = torch.norm(semantics.unsqueeze(1) - semantics.unsqueeze(0), dim=-1)

    semantic_loss = (semantic_diff * spatial_mask).sum() / denom

    total_loss = spatial_loss + alpha * λ_sem * semantic_loss

    if debug:
        print(f"[DEBUG] hybrid_loss: total={total_loss.item():.6f}, "
              f"spatial={spatial_loss.item():.6f}, semantic={semantic_loss.item():.6f}, "
              f"N={N}, neighbors={int(spatial_mask.sum().item())}")

    return total_loss


# -----------------------------
# Total cluster loss
# -----------------------------
def total_cluster_loss(gaussians,
                       topK_contrib_indices=None,
                       mask_images=None,
                       mask_weights=None,  # new optional argument
                       alpha=10.0,
                       neighbor_radius=0.05,
                       λ_sem=1.0,
                       active_positions=None,
                       active_semantics=None,
                       debug=False):
    """
    Combined hybrid spatial + semantic loss for Gaussian clustering with optional mask-center weighting.

    Args:
        gaussians (GaussianModel): Gaussian model with xyz and semantics
        topK_contrib_indices (Tensor): Optional pixel indices for pseudo-GT
        mask_images (Tensor): Optional mask images for pseudo-GT supervision
        mask_weights (Tensor): Optional weights emphasizing mask centers [num_cameras, H*W]
        alpha (float): Weight for semantic term in hybrid loss
        neighbor_radius (float): Radius for spatial neighbor computations
        λ_sem (float): Weight for semantic supervision
        active_positions (Tensor): Optional subset of positions [N_active, 3]
        active_semantics (Tensor): Optional subset of semantics [N_active]
        debug (bool): If True, prints debug info

    Returns:
        Tuple[Tensor, Tensor, Tensor]: total_loss, hybrid_loss, semantic_loss
    """
    if active_positions is None:
        positions = gaussians.get_xyz
        semantics = torch.sigmoid(gaussians.semantic_mask.squeeze())
    else:
        positions = active_positions
        semantics = active_semantics

    # Base hybrid spatial-semantic loss
    hybrid_loss = hybrid_spatial_semantic_loss(
        positions, semantics, alpha=alpha, neighbor_radius=neighbor_radius, λ_sem=λ_sem, debug=debug
    )

    # Semantic supervision from pseudo-GT
    semantic_loss = torch.tensor(0.0, device=positions.device)
    if topK_contrib_indices is not None and mask_images is not None:
        pseudo_gt, counts = pseudo_gt_from_topK(
            gaussians, topK_contrib_indices, mask_images, mask_weights=mask_weights
        )
        semantic_loss = semantic_loss_from_topK(gaussians, pseudo_gt, counts, λ_sem=λ_sem)
        total_loss = hybrid_loss + semantic_loss
    else:
        total_loss = hybrid_loss

    if debug:
        print(f"[DEBUG] total_cluster_loss: total={total_loss.item():.6f}, "
              f"hybrid={hybrid_loss.item():.6f}, semantic={semantic_loss.item():.6f}, "
              f"num_points={positions.shape[0]}")

    return total_loss, hybrid_loss, semantic_loss


