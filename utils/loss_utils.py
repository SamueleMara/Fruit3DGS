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


# -----------------------------
# FULL TRAINING LOSSES
# -----------------------------

# -----------------------------
# Appareance rendering losses
# -----------------------------

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

# -------------------------------------
# Binary mask render loss (weighted)
# -------------------------------------
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
    # contrib_indices: [H,W,K] gives gaussian idx per pixel per depth-sorted layer
    H, W, K = contrib_indices.shape
    device = gaussians_mask.device

    # Validity mask for indices (-1 entries mean empty layer)
    valid_mask = (contrib_indices >= 0)

    # Replace invalid indices with zero (safe indexing)
    safe_indices = torch.clamp(contrib_indices, min=0).long()

    # Gather gaussian mask values per contributing gaussian
    f_i = gaussians_mask[safe_indices]      # [H,W,K]
    f_i = f_i * valid_mask.float()

    # Also mask opacities
    contrib_opacities = contrib_opacities * valid_mask.float()

    # Compute cumulative transparency product (alpha compositing)
    alpha_prod = torch.cumprod(1.0 - contrib_opacities, dim=2)

    # Shift alpha_prod to get transparency *before* each layer
    alpha_prod = torch.cat([torch.ones((H, W, 1), device=device),
                            alpha_prod[:, :, :-1]], dim=2)

    # Rendered foreground = Σ f_i * α_i * T_i
    F_rendered = (f_i * contrib_opacities * alpha_prod).sum(dim=2)
    F_rendered = torch.clamp(F_rendered, 0.0, 1.0)

    # Optional external mask
    if alpha_mask is not None:
        F_rendered = F_rendered * alpha_mask

    # Ensure shape matches gt
    if F_rendered.shape != gt_mask.shape:
        F_rendered = F_rendered.squeeze(0)

    # BCE loss against ground-truth binary instance mask
    loss = F.binary_cross_entropy(F_rendered, gt_mask.float())
    return loss


# -----------------------------
# CLUSTERING LOSSES
# -----------------------------

# -------------------------
# Utilities
# -------------------------
def safe_prob(p, eps=1e-8):
    return torch.clamp(p, eps, 1.0)

def softmax_logits(u, temperature=1.0):
    if temperature != 1.0:
        return F.softmax(u / temperature, dim=1)
    return F.softmax(u, dim=1)

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


# -------------------------
# Losses
# -------------------------
def loss_label_ce(p, y, mask, weight=None, eps=1e-8):
    # Convert logits to safe probs (avoid log(0))
    p = safe_prob(p, eps)

    # Standard cross-entropy: −Σ y * log(p) over classes
    ce = -(y * torch.log(p)).sum(dim=1)

    # Optional per-sample weighting
    if weight is not None:
        ce = ce * weight

    # Mask out invalid samples
    ce = ce * mask
    return ce.mean()


def loss_pairwise_symmetric_kl(p, pairs_j, pairs_k, weights=None, eps=1e-8):
    # Probabilities for paired indices
    p_j = safe_prob(p[pairs_j], eps)
    p_k = safe_prob(p[pairs_k], eps)

    # KL(p_j || p_k)
    kl_jk = (p_j * (torch.log(p_j) - torch.log(p_k))).sum(dim=1)

    # KL(p_k || p_j)
    kl_kj = (p_k * (torch.log(p_k) - torch.log(p_j))).sum(dim=1)

    # Symmetric KL = KL(j,k) + KL(k,j)
    loss = kl_jk + kl_kj

    # Optional pair weights
    if weights is not None:
        loss = loss * weights

    return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=p.device)


def loss_propagation(p, A):
    # (I - A) p enforces consistency with affinity graph
    I_minus_A = torch.eye(A.size(0), device=p.device) - A
    diff = I_minus_A @ p

    # L2 norm per node averaged
    return (diff * diff).sum(dim=1).mean()


def loss_smoothness(q, Kmat):
    # Pairwise differences: [G,G,K]
    diff = q.unsqueeze(1) - q.unsqueeze(0)

    # Squared L2 distance per pair
    dist2 = (diff * diff).sum(dim=2)

    # Weighted by kernel matrix Kmat
    return (Kmat * dist2).mean()


def loss_marginal_entropy(p, eps=1e-8):
    # Compute class marginals m_k = 1/N Σ p_ik
    m = safe_prob(p.mean(dim=0), eps)

    # Σ m log m (negative entropy)
    return (m * torch.log(m)).sum()



# ============================================================
#               Explicit Gradients w.r.t p_j
# ============================================================

def grad_label_ce(p, y, mask, weight=None, eps=1e-8):
    # Safe probabilities
    p = safe_prob(p, eps)

    # d/dp of CE = − y / p
    g = -(y / torch.clamp(p, min=1e-8))

    if weight is not None:
        g = g * weight.unsqueeze(1)

    g = g * mask.unsqueeze(1)
    return g


def grad_pairwise_symmetric_kl(p, pairs_j, pairs_k, weights=None, eps=1e-8):
    # Safe probability
    p = safe_prob(p, eps)
    N, K = p.shape

    g = torch.zeros_like(p)

    pj = torch.clamp(p[pairs_j], min=1e-8)
    pk = torch.clamp(p[pairs_k], min=1e-8)

    # Grad of KL(pj || pk) wrt pj and pk
    grad_j = torch.log(pj) - torch.log(pk) + 1 - pk / pj
    grad_k = torch.log(pk) - torch.log(pj) + 1 - pj / pk

    # Remove NaN / inf (rare but safe)
    grad_j = torch.nan_to_num(grad_j, nan=0.0, posinf=0.0, neginf=0.0)
    grad_k = torch.nan_to_num(grad_k, nan=0.0, posinf=0.0, neginf=0.0)

    if weights is not None:
        grad_j = grad_j * weights.unsqueeze(1)
        grad_k = grad_k * weights.unsqueeze(1)

    # Accumulate gradients for each index
    g.index_add_(0, pairs_j, grad_j)
    g.index_add_(0, pairs_k, grad_k)

    # Normalize for stability
    g = g / max(1.0, pairs_j.numel())
    return g


def grad_propagation(p, A):
    # Gradient of ||(I - A)p||^2
    I_minus_A = torch.eye(A.size(0), device=p.device) - A

    # 2 (I-A)^T (I-A) p / N
    g = 2 * (I_minus_A.T @ (I_minus_A @ p)) / p.size(0)
    return g


def grad_marginal_entropy(p, eps=1e-8):
    # m_k = marginal distribution over clusters
    N, K = p.shape
    m = safe_prob(p.mean(dim=0), eps)

    # d/dp_i = (log m + 1) / N
    gm = (torch.log(m) + 1.0) / N
    return gm.unsqueeze(0).repeat(N, 1)


def grad_smoothness(q, Kmat):
    G, K = q.shape

    # diff[g1,g2] = q1 - q2
    diff = q.unsqueeze(1) - q.unsqueeze(0)

    # Gradient wrt q[g]: Σ_{g2} K[g,g2] * 2(q[g] - q[g2])
    grad = 2.0 * (Kmat.unsqueeze(2) * diff).sum(dim=1) / max(1.0, G)
    return grad


# ============================================================
#   Aggregate point-level grads → Gaussian-level grads
# ============================================================

def aggregate_gaussian_grads(r_point_idx, r_gauss_idx, r_vals, g_point, N_seg):
    """
    Aggregate gradients from points to gaussian segments.
    r_gauss_idx MUST lie in [0, N_seg).
    """
    device = g_point.device
    K = g_point.size(1)

    # Select gradient for each contributing point
    g_j_selected = g_point[r_point_idx]        # [M,K]

    # Multiply by scalar contribution r_vals per mapping
    contrib = r_vals.unsqueeze(1) * g_j_selected

    # Accumulate into per-gaussian gradient
    G_accum = torch.zeros((N_seg, K), device=device)
    G_accum.index_add_(0, r_gauss_idx, contrib)
    return G_accum


def logits_grad_from_q_grads(u, G_agg, temperature=1.0):
    # Convert NaNs/Infs to zeros for safety
    G_agg = torch.nan_to_num(G_agg, nan=0.0, posinf=0.0, neginf=0.0)

    # q = softmax(u)
    q = softmax_logits(u, temperature)

    # inner = q ⋅ G_agg (vector)
    inner = (q * G_agg).sum(dim=1, keepdim=True)

    # Softmax-Jacobian product: q * (G_agg − inner)
    grad_u = q * (G_agg - inner)
    return grad_u


# ============================================================
#          Total cluster loss + gradients pipeline
# ============================================================

def total_cluster_loss(
    gaussians,
    r_point_idx,
    r_gauss_idx,
    r_vals,
    p_j,
    q_i,
    pair_j=None,
    pair_k=None,
    pair_weights=None,
    A=None,
    Kmat=None,
    gaussians_mask=None,
    contrib_indices=None,
    contrib_opacities=None,
    gt_mask=None,
    alpha_mask=None,
    use_label_ce=True,
    use_pair_kl=False,
    use_prop=False,
    use_smooth=False,
    use_marg=False,
    use_instance_render=False,
    debug=False
):
    """
    Compute total instance-field clustering loss and the gradients w.r.t
    Gaussian logits q_i.

    p_j: point-level cluster probabilities
    q_i: gaussian-level logits
    """

    device = p_j.device
    N_seg = gaussians.get_xyz.shape[0]
    K = q_i.shape[1]

    loss_vals = {}
    total_loss = torch.tensor(0.0, device=device)

    # Gradients for point assignments (p_j)
    grad_p = torch.zeros_like(p_j)

    # Gradients for gaussian logits (q_i)
    grad_q_gauss = torch.zeros_like(q_i)

    # -------------------------
    # Label Cross-Entropy
    # -------------------------
    if use_label_ce:
        L_label = loss_label_ce(p_j, p_j, torch.ones(p_j.shape[0], device=device))
        loss_vals['label_ce'] = L_label
        total_loss += L_label

        grad_p += grad_label_ce(p_j, p_j, torch.ones(p_j.shape[0], device=device))
    else:
        loss_vals['label_ce'] = torch.tensor(0.0, device=device)

    # -------------------------
    # Pairwise Symmetric KL
    # -------------------------
    if use_pair_kl and pair_j is not None and pair_k is not None:
        L_pair = loss_pairwise_symmetric_kl(p_j, pair_j, pair_k, pair_weights)
        loss_vals['pair_kl'] = L_pair
        total_loss += L_pair

        grad_p += grad_pairwise_symmetric_kl(p_j, pair_j, pair_k, pair_weights)
    else:
        loss_vals['pair_kl'] = torch.tensor(0.0, device=device)

    # -------------------------
    # Graph Propagation Loss
    # -------------------------
    if use_prop and A is not None:
        L_prop = loss_propagation(p_j, A)
        loss_vals['prop'] = L_prop
        total_loss += L_prop

        grad_p += grad_propagation(p_j, A)
    else:
        loss_vals['prop'] = torch.tensor(0.0, device=device)

    # -------------------------
    # Gaussian Logit Smoothness
    # -------------------------
    if use_smooth and Kmat is not None:
        L_smooth = loss_smoothness(q_i, Kmat)
        loss_vals['smooth'] = L_smooth
        total_loss += L_smooth

        grad_q_gauss += grad_smoothness(q_i, Kmat)
    else:
        loss_vals['smooth'] = torch.tensor(0.0, device=device)

    # -------------------------
    # Marginal Entropy Regularizer
    # -------------------------
    if use_marg:
        L_marg = loss_marginal_entropy(p_j)
        loss_vals['marg'] = L_marg
        total_loss += L_marg

        grad_p += grad_marginal_entropy(p_j)
    else:
        loss_vals['marg'] = torch.tensor(0.0, device=device)

    # -------------------------
    # Optional Rendering-based Loss
    # -------------------------
    if use_instance_render and gaussians_mask is not None:
        L_render = binary_mask_render_loss(
            gaussians_mask, contrib_indices, contrib_opacities, gt_mask, alpha_mask
        )
        loss_vals['instance_render'] = L_render
        total_loss += L_render

        # Rendering loss gradient contribution (placeholder = ones)
        grad_q_gauss += aggregate_gaussian_grads(
            r_point_idx, r_gauss_idx, r_vals,
            torch.ones_like(p_j),     # dummy grad
            N_seg
        )
    else:
        loss_vals['instance_render'] = torch.tensor(0.0, device=device)

    # ============================================================
    # Aggregate point-level gradients → gaussian-level gradients
    # ============================================================
    grad_q_from_p = aggregate_gaussian_grads(
        r_point_idx, r_gauss_idx, r_vals, grad_p, N_seg
    )

    # Convert aggregated q-gradients to logit-gradients through softmax jacobian
    grad_q_gauss += logits_grad_from_q_grads(q_i, grad_q_from_p)

    return total_loss, loss_vals, grad_q_gauss



# ===========================================================
#         Embedding losses
# ===========================================================

def normalize_embeddings(e, eps=1e-8):
    """
    L2-normalize embeddings along feature dimension.
    e: [N, D]
    """
    return e / (torch.norm(e, dim=1, keepdim=True) + eps)

def contrastive_loss(
    emb,
    idx_i,
    idx_j,
    labels,
    margin=1.0
):
    """
    emb:     [N, D] embedding matrix
    idx_i:   [M] indices
    idx_j:   [M] indices
    labels:  [M] (1 = positive, 0 = negative)
    """

    e_i = emb[idx_i]
    e_j = emb[idx_j]

    dist = torch.norm(e_i - e_j, dim=1)

    pos_loss = labels * dist.pow(2)
    neg_loss = (1.0 - labels) * torch.clamp(margin - dist, min=0).pow(2)

    return (pos_loss + neg_loss).mean()

def cosine_pair_loss(
    emb,
    idx_i,
    idx_j,
    labels
):
    """
    emb must be normalized
    """

    e_i = emb[idx_i]
    e_j = emb[idx_j]

    sim = (e_i * e_j).sum(dim=1)  # cosine similarity

    pos_loss = labels * (1.0 - sim)
    neg_loss = (1.0 - labels) * torch.clamp(sim, min=0.0)

    return (pos_loss + neg_loss).mean()

def spatial_smoothness_loss(
    emb,
    xyz,
    radius=0.05
):
    """
    emb: [N, D]
    xyz: [N, 3]
    """

    # Pairwise distances in 3D
    dist = torch.cdist(xyz, xyz)  # [N, N]

    # Neighborhood mask
    W = (dist < radius).float()

    diff = emb.unsqueeze(1) - emb.unsqueeze(0)  # [N,N,D]
    loss = (W.unsqueeze(2) * diff.pow(2)).sum(dim=2)

    # Avoid trivial self-pairs
    loss = loss * (1.0 - torch.eye(emb.size(0), device=emb.device))

    return loss.mean()

def info_nce_loss(
    emb,
    idx_anchor,
    idx_positive,
    temperature=0.1
):
    """
    emb: [N, D] normalized
    idx_anchor:   [M]
    idx_positive: [M]
    """

    e_a = emb[idx_anchor]    # [M,D]
    e_p = emb[idx_positive]  # [M,D]

    logits = e_a @ emb.T     # [M,N]
    logits = logits / temperature

    labels = idx_positive
    return F.cross_entropy(logits, labels)


def frame_pixel_contrastive_loss(
    emb_norm: torch.Tensor,     # [N,D] normalized
    g_at_pix: torch.Tensor,     # [P] gaussian id per pixel (top-1), -1 invalid
    pix_inst: torch.Tensor,     # [P] instance label per pixel, -1 background
    pixels_per_iter: int = 20000,
    neg_per_pos: int = 16,
    temperature: float = 0.07,
    bg_neg_weight: float = 6.0,
):
    """
    Same-frame pixel-supervised contrastive loss.

    Strategy:
      - sample a set of pixels from this frame
      - map each sampled pixel to its gaussian (anchor)
      - for FG anchors (inst>=0):
           pick one positive from SAME inst
           pick K negatives from DIFFERENT inst
      - for BG anchors (inst==-1):
           pick K negatives from ANY FG (strong)
    """

    device = emb_norm.device
    P = g_at_pix.numel()
    if P == 0:
        return emb_norm.new_tensor(0.0)

    # sample pixels
    M = min(pixels_per_iter, P)
    pix_idx = torch.randint(0, P, (M,), device=device)

    inst_a = pix_inst[pix_idx]                 # [M]
    ga = g_at_pix[pix_idx].long()              # [M]
    valid_anchor = ga >= 0
    if not valid_anchor.any():
        return emb_norm.new_tensor(0.0)

    pix_idx = pix_idx[valid_anchor]
    inst_a = inst_a[valid_anchor]
    ga = ga[valid_anchor]

    # useful masks
    fg_mask_all = (pix_inst >= 0) & (g_at_pix >= 0)
    if not fg_mask_all.any():
        # no fg at all -> nothing to learn
        return emb_norm.new_tensor(0.0)

    fg_pixels = fg_mask_all.nonzero(as_tuple=True)[0]  # indices of FG pixels

    # Build per-instance pixel lists (on GPU) for sampling positives fast
    # (local ids per frame are small)
    max_inst = int(pix_inst[pix_inst >= 0].max().item()) if (pix_inst >= 0).any() else -1
    inst_to_pixels = []
    if max_inst >= 0:
        for cid in range(max_inst + 1):
            idx = ((pix_inst == cid) & (g_at_pix >= 0)).nonzero(as_tuple=True)[0]
            inst_to_pixels.append(idx)

    # Precompute anchor embedding
    ea = emb_norm[ga]  # [A,D]

    # Split FG anchors vs BG anchors
    fg_anchor = inst_a >= 0
    bg_anchor = ~fg_anchor

    loss_terms = []

    # ----------------------------------------
    # FG anchors: (pos from same instance) vs (negs other instances)
    # ----------------------------------------
    if fg_anchor.any():
        ga_fg = ga[fg_anchor]
        inst_fg = inst_a[fg_anchor]
        ea_fg = emb_norm[ga_fg]

        # sample positives: pick a pixel from same instance
        pos_pix = torch.empty_like(inst_fg)
        for i in range(inst_fg.numel()):
            cid = int(inst_fg[i].item())
            pool = inst_to_pixels[cid]
            if pool.numel() == 0:
                # fallback: self
                pos_pix[i] = pix_idx[fg_anchor][i]
            else:
                pos_pix[i] = pool[torch.randint(0, pool.numel(), (1,), device=device)]

        gp = g_at_pix[pos_pix].long()
        ep = emb_norm[gp]

        # sample negatives: from fg_pixels but different inst
        # We'll do rejection sampling with a small number of tries (works fine in practice).
        K = int(neg_per_pos)
        neg_pix = torch.empty((inst_fg.numel(), K), dtype=torch.long, device=device)

        # To speed up: sample candidates then filter
        for i in range(inst_fg.numel()):
            cid = int(inst_fg[i].item())
            # sample a bit more than needed and take first K that differ
            for _ in range(5):
                cand = fg_pixels[torch.randint(0, fg_pixels.numel(), (K * 3,), device=device)]
                good = cand[pix_inst[cand] != cid]
                if good.numel() >= K:
                    neg_pix[i] = good[:K]
                    break
            else:
                # worst-case fallback: just use random fg (may include same inst sometimes)
                neg_pix[i] = fg_pixels[torch.randint(0, fg_pixels.numel(), (K,), device=device)]

        gn = g_at_pix[neg_pix].long()          # [Afg,K]
        en = emb_norm[gn]                      # [Afg,K,D]

        # logits
        pos_logit = (ea_fg * ep).sum(dim=1, keepdim=True) / temperature          # [Afg,1]
        neg_logit = (ea_fg.unsqueeze(1) * en).sum(dim=2) / temperature           # [Afg,K]
        logits = torch.cat([pos_logit, neg_logit], dim=1)                         # [Afg, 1+K]
        labels = torch.zeros((logits.shape[0],), dtype=torch.long, device=device) # pos=0

        loss_fg = F.cross_entropy(logits, labels)
        loss_terms.append(loss_fg)

    # ----------------------------------------
    # BG anchors: push away from FG strongly
    # ----------------------------------------
    if bg_anchor.any():
        ga_bg = ga[bg_anchor]
        ea_bg = emb_norm[ga_bg]

        K = int(neg_per_pos)
        neg_pix = fg_pixels[torch.randint(0, fg_pixels.numel(), (ga_bg.numel(), K), device=device)]
        gn = g_at_pix[neg_pix].long()
        en = emb_norm[gn]

        # we want similarities to be LOW; use softplus(sim) as penalty
        sim = (ea_bg.unsqueeze(1) * en).sum(dim=2) / temperature  # [Abg,K]
        loss_bg = F.softplus(sim).mean() * float(bg_neg_weight)
        loss_terms.append(loss_bg)

    if len(loss_terms) == 0:
        return emb_norm.new_tensor(0.0)

    return torch.stack(loss_terms).mean()


def appearance_contrastive_pair_loss(
    emb_nodes,          # [B,D]
    emb_nbr,            # [B,K,D]
    rgb_nodes,          # [B,3]
    rgb_nbr,            # [B,K,3]
    op_nodes,           # [B,1]
    op_nbr,             # [B,K,1]
    sem_w,              # [B,K]  (e.g. s_i*s_j)
    rgb_w=1.0,
    op_w=2.0,
    tau_pos=0.15,
    tau_neg=0.15,
    app_margin=0.25,    # appearance threshold where neg starts to activate
    emb_margin=0.6,     # desired minimum embedding distance for neg pairs
    neg_scale=1.0,
    eps=1e-6,
    ):
    """
    Returns scalar loss.
    """
    # embedding distance
    d2_emb = (emb_nodes[:, None, :] - emb_nbr).pow(2).sum(dim=2)          # [B,K]
    d_emb = torch.sqrt(d2_emb + eps)

    # appearance distance
    d_rgb = (rgb_nodes[:, None, :] - rgb_nbr).pow(2).sum(dim=2).sqrt()    # [B,K]
    d_op  = (op_nodes[:, None, :]  - op_nbr).abs().squeeze(-1)            # [B,K]
    d_app = rgb_w * d_rgb + op_w * d_op                                   # [B,K]

    # POS: similar appearance => stronger pull
    w_pos = torch.exp(-d_app / max(tau_pos, eps)) * sem_w                 # [B,K]
    pos_loss = (w_pos * d2_emb).sum() / (w_pos.sum() + eps)

    # NEG: dissimilar appearance => push apart (hinge)
    # gate where d_app > app_margin
    w_neg = torch.sigmoid((d_app - app_margin) / max(tau_neg, eps)) * sem_w
    neg_hinge = F.relu(emb_margin - d_emb)                                # [B,K]
    neg_loss = (w_neg * neg_hinge.pow(2)).sum() / (w_neg.sum() + eps)

    return pos_loss + neg_scale * neg_loss


def centroid_instance_pull_loss(
    emb_norm: torch.Tensor,   # [N,D] normalized gaussian embeddings
    topk0: torch.Tensor,      # [P] top-1 gaussian per pixel for this camera (frame-local)
    inst_pix: torch.Tensor,   # [Q] flattened pixels (frame-local)
    centroid_g: int,          # centroid gaussian id (global gaussian index)
    centroid_pixel: int,      # flat centroid pixel index (frame-local)
    W: int,
    sigma_px: float = 25.0,
    max_pix: int = 8000,
    min_valid: int = 32,
    eps: float = 1e-6,
):
    device = emb_norm.device

    if inst_pix is None or inst_pix.numel() < int(min_valid):
        return emb_norm.new_tensor(0.0)

    # sanity centroid gaussian
    if centroid_g is None:
        return emb_norm.new_tensor(0.0)
    centroid_g = int(centroid_g)
    if centroid_g < 0 or centroid_g >= emb_norm.shape[0]:
        return emb_norm.new_tensor(0.0)

    # subsample pixels (keep on device)
    if inst_pix.numel() > int(max_pix):
        sel = torch.randint(0, inst_pix.numel(), (int(max_pix),), device=device)
        inst_pix = inst_pix[sel]

    # ensure topk0 / inst_pix on device
    if topk0.device != device:
        topk0 = topk0.to(device)
    if inst_pix.device != device:
        inst_pix = inst_pix.to(device)

    P = int(topk0.shape[0])
    inst_pix = inst_pix[(inst_pix >= 0) & (inst_pix < P)]
    if inst_pix.numel() < int(min_valid):
        return emb_norm.new_tensor(0.0)

    # centroid pixel -> (cx, cy) in pixel coords
    centroid_pixel = int(centroid_pixel)
    if centroid_pixel < 0 or centroid_pixel >= P:
        return emb_norm.new_tensor(0.0)

    cx = float(centroid_pixel % int(W))
    cy = float(centroid_pixel // int(W))

    # pixel -> (x,y)
    y = (inst_pix // int(W)).float()
    x = (inst_pix %  int(W)).float()

    # use top-1 gaussian at each instance pixel
    g_p = topk0[inst_pix].long()
    valid = (g_p >= 0)
    if valid.sum().item() < int(min_valid):
        return emb_norm.new_tensor(0.0)

    g_p = g_p[valid]
    x   = x[valid]
    y   = y[valid]

    # weights stronger near centroid pixel
    d2 = (x - cx).pow(2) + (y - cy).pow(2)
    w = torch.exp(-d2 / (2.0 * float(sigma_px) * float(sigma_px)))
    wsum = w.sum()
    if float(wsum.item()) < 1e-6:
        return emb_norm.new_tensor(0.0)

    e_c = emb_norm[centroid_g]  # [D]
    e_p = emb_norm[g_p]         # [Qv,D]

    loss = (w[:, None] * (e_p - e_c[None, :]).pow(2)).sum() / (wsum + eps)
    return loss