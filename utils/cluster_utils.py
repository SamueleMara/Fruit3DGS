# cluster_utils.py
import os
import glob
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter, defaultdict

from tqdm import tqdm
from scene.gaussian_model import GaussianModel

import torch.nn.functional as F
import torch.nn as nn
from math import log, sqrt

from scipy.spatial import cKDTree
from sklearn.neighbors import radius_neighbors_graph
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import csr_matrix,lil_matrix

import hdbscan

# -----------------------------
# Build a knn Graph using scipy
# -----------------------------
def build_knn_graph_scipy(xyz_torch: torch.Tensor, k: int = 16) -> torch.Tensor:
    xyz = xyz_torch.detach().float().cpu().numpy()
    N = xyz.shape[0]
    if N == 0:
        return torch.empty((0, k), dtype=torch.long)

    tree = cKDTree(xyz)

    k_eff = min(k + 1, N)  # include self if possible
    dists, idxs = tree.query(xyz, k=k_eff, workers=-1)  # idxs: [N,k_eff]

    # Make idxs 2D even when k_eff==1
    if idxs.ndim == 1:
        idxs = idxs[:, None]

    rows = np.arange(N)[:, None]
    idxs = idxs[idxs != rows].reshape(N, -1)  # remove self robustly

    # If we removed too much (N small), pad with -1
    if idxs.shape[1] < k:
        pad = -np.ones((N, k - idxs.shape[1]), dtype=idxs.dtype)
        idxs = np.concatenate([idxs, pad], axis=1)
    else:
        idxs = idxs[:, :k]

    return torch.from_numpy(idxs).long()

class UnionFind:
    """
    Union-Find / Disjoint Set Union with:
      - path compression
      - union by rank
      - fast numpy storage
      - ability to extract components
    """

    __slots__ = ("parent", "rank", "n")

    def __init__(self, n: int):
        self.n = n
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]   # path compression
            x = p[x]
        return x

    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return

        # union by rank
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    # --------------------------------------------------
    # Component extraction (merged functionality)
    # --------------------------------------------------
    def components(self, min_size: int = 1):
        """
        Returns:
            comps: list[list[int]] of connected components
        """
        buckets = defaultdict(list)
        for i in range(self.n):
            buckets[self.find(i)].append(i)

        if min_size <= 1:
            return list(buckets.values())

        return [v for v in buckets.values() if len(v) >= min_size]

    def labels(self, min_size: int = 1):
        """
        Returns:
            labels: np.ndarray [n] with compact ids 0..C-1, -1 for small comps
        """
        roots = np.array([self.find(i) for i in range(self.n)], dtype=np.int32)
        uniq, inv, counts = np.unique(roots, return_inverse=True, return_counts=True)

        labels = np.full(self.n, -1, dtype=np.int32)
        next_id = 0
        for cid, cnt in enumerate(counts):
            if cnt >= min_size:
                labels[inv == cid] = next_id
                next_id += 1
        return labels

# -----------------------------
# Debug tensor helper
# -----------------------------
def debug_tensor(name, tensor):
    """
    Print detailed debug info about a tensor.

    Inputs:
        name: str        Name of the tensor
        tensor: torch.Tensor or None
    Outputs:
        Prints shape, dtype, device, min, max (or None)
    """
    if tensor is None:
        print(f"[DEBUG] {name}: None")
    else:
        print(
            f"[DEBUG] {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"device={tensor.device}, min={tensor.min().item()}, max={tensor.max().item()}"
        )

# ---------------------------------------------------------
#  Unified version for top K contributors computation
#  (pixel_best_gaussian chosen by max semantic among top-K)
# ---------------------------------------------------------
def compute_topK_contributors_and_responsibilities(
    scene,
    K=8,
    bg_color=(0, 0, 0),
    storage_device="cpu",
    semantic_use_sigmoid=True,
    use_gs_camera_keys: bool = True,
    skip_responsibilities: bool = True,

):
    """
    Merged, information-preserving version of compute_topK_contributors + topK_to_responsibilities.
    Render on GPU (where gaussians live), but store/process indices on storage_device (cpu recommended).

    Modification:
        pixel_best_gaussian is NOT contrib[...,0] anymore.
        Instead, it's the gaussian among the per-pixel top-K with the maximum semantic value.

    Returns:
        topK_contrib: dict with
            indices:              [N, C, K]  (Gaussian -> pixels)
            opacities:            [N, C, K]
            pixel_best_gaussian:  list[C] of [P] tensors
            pixel_topk_gaussians: list[C] of [P, K] tensors (per-pixel contributors)
        r_point_idx: [M]  pixel indices (flattened per camera)
        r_gauss_idx: [M]  gaussian indices
        r_vals:      [M]  normalized responsibility weights per pixel
    """
   
    from gaussian_renderer import render

    gaussians = scene.gaussians

    # ------------------------------------------------------------
    # Camera order MUST match scene.gs_cameras.keys()
    # ------------------------------------------------------------
    if use_gs_camera_keys:
        cam_names_in = list(scene.gs_cameras.keys())
        cameras = [scene.gs_cameras[n] for n in cam_names_in]
    else:
        cameras = scene.getTrainCameras()
        cam_names_in = [getattr(c, "image_name", str(i)) for i, c in enumerate(cameras)]

    N = gaussians.get_xyz.shape[0]
    C = len(cameras)

    render_device = gaussians.get_xyz.device
    store_device = torch.device(storage_device)

    # ---- semantic (bring once to storage_device for cheap gathers) ----
    sem_all = getattr(gaussians, "get_sem", None)
    if sem_all is None:
        raise ValueError(
            "This function requires gaussians.get_sem to exist (semantic logits/values per gaussian)."
        )
    sem_all = sem_all.detach().view(-1)
    if sem_all.numel() != N:
        raise ValueError(f"gaussians.get_sem has shape {tuple(sem_all.shape)} but expected [{N}].")
    sem_all = sem_all.to(store_device)

    if semantic_use_sigmoid:
        sem_all = torch.sigmoid(sem_all)

    # ---------------------------
    # Outputs (store on CPU by default)
    # ---------------------------
    topK_indices = -torch.ones((N, C, K), dtype=torch.long, device=store_device)
    topK_opacities = torch.zeros((N, C, K), dtype=torch.float32, device=store_device)
    pixel_best_gaussian = [None] * C
    pixel_topk_gaussians = [None] * C
    cam_hw = [None] * C
    cam_names = [None] * C

    # ------------------------------------------------------------
    # BIG RAM FIX: insert_pos is just a counter in [0..K]
    # ------------------------------------------------------------
    cnt_dtype = torch.uint8 if K <= 255 else torch.int16
    insert_pos = torch.zeros((C, N), dtype=cnt_dtype, device=store_device)

    # responsibilities storage (only if requested)
    if not skip_responsibilities:
        r_point_idx = []
        r_gauss_idx = []
        r_vals = []
    else:
        r_point_idx = r_gauss_idx = r_vals = None

    class _DummyPipe:
        debug = False
        antialiasing = False
        compute_cov3D_python = False
        convert_SHs_python = False

    dummy_pipe = _DummyPipe()
    bg_col = torch.tensor(bg_color, dtype=torch.float32, device=render_device)

    pixel_offset = 0

    for cam_idx, cam in enumerate(tqdm(cameras, desc=f"Computing top-{K} contributors (merged)")):
        frame_name = cam_names_in[cam_idx]
        cam_names[cam_idx] = frame_name

        with torch.no_grad():
            out = render(cam, gaussians, dummy_pipe, bg_col, contrib=True, K=int(K))

        if out is None or out.get("contrib_indices", None) is None:
            continue

        contrib = out["contrib_indices"]  # [H,W,K] (likely on GPU)
        weights = out.get("contrib_opacities", None)

        H, W, KK = contrib.shape
        P = H * W
        cam_hw[cam_idx] = (int(H), int(W))

        if weights is None:
            weights = torch.ones((H, W, KK), dtype=torch.float32, device=contrib.device)
        else:
            weights = weights.float()

        # Move to storage device
        contrib = contrib.to(store_device, non_blocking=False).long()   # [H,W,K]
        weights = weights.to(store_device, non_blocking=False).float()  # [H,W,K]

        # ------------------------------------------------------------
        # Pixel top-K per pixel SORTED by semantic (desc)
        # ------------------------------------------------------------
        flat_g = contrib.reshape(-1, K)          # [P,K]
        flat_w = weights.reshape(-1, K).float()  # [P,K]

        valid = flat_g >= 0

        sem = torch.full((P, K), float("-inf"), dtype=torch.float32, device=store_device)
        if valid.any():
            sem[valid] = sem_all[flat_g[valid]]

        score = sem + 1e-6 * (flat_w / (flat_w.max(dim=1, keepdim=True).values + 1e-12))
        order = torch.argsort(score, dim=1, descending=True)

        flat_g_sorted = torch.gather(flat_g, 1, order)
        flat_w_sorted = torch.gather(flat_w, 1, order)

        invalid_sorted = (torch.gather(valid, 1, order) == 0)
        flat_g_sorted[invalid_sorted] = -1
        flat_w_sorted[invalid_sorted] = 0.0

        pixel_topk_gaussians[cam_idx] = flat_g_sorted

        best = flat_g_sorted[:, 0].clone()
        best[best < 0] = -1
        pixel_best_gaussian[cam_idx] = best

        # ------------------------------------------------------------
        # Gaussian-centric fill (+ optional responsibilities)
        # Memory-friendly: avoid arange().expand()[mask]
        # ------------------------------------------------------------
        valid2 = flat_g_sorted >= 0  # [P,K]

        # Flatten-mask order matches the old boolean indexing order
        idx_flat = valid2.reshape(-1).nonzero(as_tuple=False).squeeze(1)  # [M]
        if idx_flat.numel() > 0:
            rows = torch.div(idx_flat, K, rounding_mode="floor")          # [M] pixel id (0..P-1)
            cols = idx_flat - rows * K                                    # [M] k-column

            g_ids = flat_g_sorted[rows, cols].long()                      # [M]
            w_vals = flat_w_sorted[rows, cols].float()                    # [M]

            # ---- optional: responsibilities (skip by default) ----
            if not skip_responsibilities:
                # normalize per pixel (equivalent to your global unique/scatter)
                denom = flat_w_sorted.sum(dim=1)                           # [P]
                w_norm = w_vals / torch.clamp(denom[rows], min=1e-12)      # [M]

                pix_ids_full = (rows + pixel_offset).to(torch.long)
                r_point_idx.append(pix_ids_full)
                r_gauss_idx.append(g_ids)
                r_vals.append(w_norm)

            # ---- gaussian-centric fill (same batch semantics as your original code) ----
            cam_insert = insert_pos[cam_idx]                               # [N] counter dtype
            pos = cam_insert[g_ids]                                        # [M] counter dtype
            keep = pos < K
            if keep.any():
                gid = g_ids[keep]
                ppos = pos[keep].to(torch.long).clamp(max=K - 1)

                # Pixel IDs: use int32 intermediate, cast to long only when storing (no output dtype change)
                pid_i32 = (rows[keep].to(torch.int32) + int(pixel_offset))  # [M_keep] int32
                topK_indices[gid, cam_idx, ppos] = pid_i32.to(torch.long)
                topK_opacities[gid, cam_idx, ppos] = w_vals[keep]

                # increment counters; clamp to K
                newv = cam_insert[gid].to(torch.int32) + 1
                cam_insert[gid] = torch.clamp(newv, max=K).to(cnt_dtype)
                insert_pos[cam_idx] = cam_insert

        pixel_offset += P

    # ---------------------------
    # Finalize responsibilities
    # ---------------------------
    if skip_responsibilities:
        r_point_out = torch.empty(0, dtype=torch.long, device=store_device)
        r_gauss_out = torch.empty(0, dtype=torch.long, device=store_device)
        r_vals_out  = torch.empty(0, dtype=torch.float32, device=store_device)
    else:
        if len(r_point_idx) == 0:
            r_point_out = torch.empty(0, dtype=torch.long, device=store_device)
            r_gauss_out = torch.empty(0, dtype=torch.long, device=store_device)
            r_vals_out  = torch.empty(0, dtype=torch.float32, device=store_device)
        else:
            r_point_out = torch.cat(r_point_idx)
            r_gauss_out = torch.cat(r_gauss_idx)
            r_vals_out  = torch.cat(r_vals)

    topK_contrib = {
        "indices": topK_indices,
        "opacities": topK_opacities,
        "pixel_best_gaussian": pixel_best_gaussian,
        "pixel_topk_gaussians": pixel_topk_gaussians,
        "cam_hw": cam_hw,
        "cam_names": cam_names,
    }
    return topK_contrib, r_point_out, r_gauss_out, r_vals_out



# -----------------------------
# Map full Top-K -> segmented Top-K (vectorized)
# -----------------------------
def map_full_topK_to_segmented(topK_full, full_to_seg, kept_full):
    """
    Map full-resolution top-K Gaussian contributors -> segmented top-K
    """
    device = topK_full["indices"].device
    topK_indices_full = topK_full["indices"][kept_full]      # [N_seg, C, K]
    topK_opac_full = topK_full["opacities"][kept_full]      # [N_seg, C, K]

    # Map indices to segmented
    mapped_indices = topK_indices_full.clone()
    valid_mask = (mapped_indices >= 0) & (mapped_indices < full_to_seg.numel())
    mapped_indices[valid_mask] = full_to_seg[mapped_indices[valid_mask]]
    mapped_indices[~valid_mask] = -1

    # Zero out invalid opacities
    topK_opac_full[mapped_indices < 0] = 0.0

    # Debug
    # print(f"[DEBUG] topK_seg['indices'] min={mapped_indices.min().item()}, max={mapped_indices.max().item()}")
    # print(f"[DEBUG] topK_seg['opacities'] min={topK_opac_full.min().item()}, max={topK_opac_full.max().item()}")
    valid_contribs = (mapped_indices >= 0).sum().item()
    # print(f"[DEBUG] Segmented Gaussians with ≥1 contribution: {valid_contribs} / {kept_full.numel()}")

    return {"indices": mapped_indices, "opacities": topK_opac_full}


# -----------------------------
# Convert Gaussians -> pixels (vectorized version)
# -----------------------------
def convert_gauss_to_pixel_map(topK_full):
    """
    Convert Gaussian->pixel top-K map to pixel->Gaussian mapping (vectorized).
    """
    indices = topK_full["indices"]
    opacities = topK_full["opacities"]
    device = indices.device
    N_gauss, C, K = indices.shape

    # Flatten
    flat_idx = indices.reshape(-1)
    flat_gauss = torch.arange(N_gauss, device=device)[:, None, None].expand(-1, C, K).reshape(-1)
    flat_op = opacities.reshape(-1)

    # Filter valid contributions
    mask = (flat_idx >= 0) & (flat_op > 0)
    flat_idx, flat_gauss = flat_idx[mask], flat_gauss[mask]

    # Build dictionary: pixel -> list of gaussians
    pixel_to_gauss = {}
    for pix, g in zip(flat_idx.tolist(), flat_gauss.tolist()):
        pixel_to_gauss.setdefault(pix, []).append(g)

    # Flatten to list of tuples (tensor of gaussians, mask_val=1.0)
    mappings = [(torch.tensor(glist, dtype=torch.long, device=device), 1.0)
                for glist in pixel_to_gauss.values()]

    print(f"[INFO] Converted {len(mappings)} pixels → Gaussians")
    return mappings


# -----------------------------
# Compute instance coherence
# -----------------------------
def compute_instance_coherence(logits, ids):
    """
    logits: [N, C] tensor of per-Gaussian instance logits
    ids:    [N] tensor of final instance IDs

    Returns:
        coherence: [N] tensor of per-Gaussian coherence scores
    """
    probs = torch.softmax(logits, dim=1)
    coherence = probs[torch.arange(len(ids)), ids]
    return coherence


# -----------------------------
# Filter Coherent Gaussians
# -----------------------------

def filter_coherent_gaussians(gaussian_model, threshold=0.6):
    """
    Filter a GaussianModel by instance coherence.

    Inputs:
        gaussian_model: GaussianModel instance (segmented)
        threshold: float, coherence threshold (0.6 by default)

    Returns:
        filtered_gaussian_model: new GaussianModel containing only coherent Gaussians
        mask: boolean mask of kept Gaussians
    """
    # Retrieve data from the model
    xyz = gaussian_model.get_xyz
    logits = gaussian_model.instance_logits
    ids = gaussian_model.instance_ids

    # Compute coherence
    coherence = compute_instance_coherence(logits, ids)

    # --- DEBUG: Inspect coherence distribution ---
    # print("[DEBUG] Coherence stats:",
    #       f"min={coherence.min().detach().item():.4f}, "
    #       f"max={coherence.max().detach().item():.4f}, "
    #       f"mean={coherence.mean().detach().item():.4f}, "
    #       f"median={coherence.median().detach().item():.4f}")

    mask = coherence >= threshold
    kept_count = mask.sum().item()
    total_count = xyz.shape[0]
    # print(f"[INFO] Keeping {kept_count} / {total_count} coherent Gaussians (TH={threshold})")

    # Create new GaussianModel with filtered attributes
    filtered_gs = GaussianModel(sh_degree=gaussian_model.active_sh_degree)
    filtered_gs._xyz = nn.Parameter(xyz[mask].clone().detach())
    filtered_gs._features_dc = nn.Parameter(gaussian_model._features_dc[mask].clone().detach())
    filtered_gs._features_rest = nn.Parameter(gaussian_model._features_rest[mask].clone().detach())
    filtered_gs._scaling = nn.Parameter(gaussian_model._scaling[mask].clone().detach())
    filtered_gs._rotation = nn.Parameter(gaussian_model._rotation[mask].clone().detach())
    filtered_gs._opacity = nn.Parameter(gaussian_model._opacity[mask].clone().detach())
    filtered_gs.semantic_mask = nn.Parameter(gaussian_model.semantic_mask[mask].clone().detach()) if gaussian_model.semantic_mask is not None else None

    # Copy instance info
    filtered_gs.instance_logits = nn.Parameter(logits[mask].clone().detach())
    filtered_gs.instance_ids = ids[mask].clone().detach()

    return filtered_gs, mask


# -----------------------------
# Compute Mask Centroids
# -----------------------------
def compute_mask_centroids(mask_dir):
    """
    Compute centroids for all mask instances inside a folder.
    Accepts ANY image extension (png, jpg, jpeg, bmp, tif, tiff, webp, …)

    Returns:
        dict: { (frame_name, instance_idx): (cx, cy) }
    """
    mask_dir = Path(mask_dir)

    # Allowed extensions
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"]

    # Collect files
    mask_files = []
    for ext in exts:
        mask_files.extend(mask_dir.glob(ext))

    centroids = {}

    for mask_path in mask_files:
        stem = mask_path.stem

        # Only accept files containing '_instance_X'
        if "_instance_" not in stem:
            continue

        try:
            frame_name, idx_str = stem.rsplit("_instance_", 1)
            midx = int(idx_str)
        except:
            continue

        # Load mask as numpy
        mask = np.array(Image.open(mask_path).convert("L"))
        ys, xs = np.where(mask > 128)  # foreground pixels

        if len(xs) == 0:
            continue

        cx = float(xs.mean())
        cy = float(ys.mean())

        centroids[(frame_name, midx)] = (cx, cy)

    return centroids


# --------------------------------------------------------------------
# Map dt_norm -> dt_real using dt_scale from centroid distances
# --------------------------------------------------------------------
def dt_norm_to_real(dt_norm, dt_scale):
    """
    Simple linear mapping: dt_norm in [0,1] maps to dt_real in pixels/units.
    """
    dt_real = dt_norm * dt_scale
    return dt_real

# --------------------------------------------------------------------
# Refine the clusters iteratively with a metrics
# --------------------------------------------------------------------
# def refine_clusters_with_metrics

#     return clusters


def build_gaussian_mask_mappings(mask_instances_dict,
                                 pixel_best_gaussian,
                                 scene,
                                 device="cpu"):
    """
    Builds mappings using pixel-centric strongest Gaussian per pixel.

    Inputs:
        mask_instances_dict:
            dict[frame_name -> dict[inst_id -> {"pixel_indices": list[int], ...}]]

        pixel_best_gaussian:
            list[Tensor], length C, each Tensor is [P] with gaussian index or -1

        scene:
            Scene object

    Outputs:
        gaussian_to_mask_ids, mask_id_to_pixel_idx, pixel_idx_to_gaussians,
        gaussian_to_pixels, cam_offset_dict, global_id_map
    """

    gaussian_to_mask_ids = defaultdict(set)
    mask_id_to_pixel_idx = defaultdict(list)
    pixel_idx_to_gaussians = defaultdict(set)
    gaussian_to_pixels = defaultdict(list)

    cam_names = list(scene.gs_cameras.keys())

    # --------------------------------------------------
    # Global mask ids
    # --------------------------------------------------
    global_id_map = {}
    next_gid = 1
    for frame_name, inst_dict in mask_instances_dict.items():
        for inst_id in inst_dict.keys():
            global_id_map[(frame_name, inst_id)] = next_gid
            next_gid += 1

    # --------------------------------------------------
    # Camera offsets
    # --------------------------------------------------
    cam_offset_dict = {}
    offset = 0
    for name in cam_names:
        cam = scene.gs_cameras[name]
        cam_offset_dict[name] = offset
        offset += cam.image_height * cam.image_width

    # --------------------------------------------------
    # Pixel → Gaussian → Mask (FAST)
    # --------------------------------------------------
    for cam_idx, frame_name in enumerate(tqdm(cam_names, desc="Building gaussian-mask mappings (per camera)", total=len(cam_names))):
        offset = cam_offset_dict[frame_name]
        best = pixel_best_gaussian[cam_idx]     # Tensor [P]

        inst_items = mask_instances_dict.get(frame_name, {}).items()
        for inst_id, inst_data in tqdm(inst_items, desc=f"  Camera {frame_name} instances", leave=False):
            gid = global_id_map[(frame_name, inst_id)]

            pix = torch.tensor(inst_data["pixel_indices"], device=best.device, dtype=torch.long)
            g = best[pix]                         # strongest gaussian for each pixel

            valid = g >= 0
            pix = pix[valid]
            g = g[valid]

            global_pix = pix + offset

            for gi, pi in zip(g.tolist(), global_pix.tolist()):
                gaussian_to_mask_ids[gi].add(gid)
                mask_id_to_pixel_idx[gid].append(pi)
                pixel_idx_to_gaussians[pi].add(gi)
                gaussian_to_pixels[gi].append(pi)

    return (
        gaussian_to_mask_ids,
        mask_id_to_pixel_idx,
        pixel_idx_to_gaussians,
        gaussian_to_pixels,
        cam_offset_dict,
        global_id_map,
    )

def assign_gaussian_mask_ids(
    mask_instances_dict,
    pixel_best_gaussian,
    cam_offset_dict,
    global_id_map,
    num_gaussians,
    device,
):
    """
    Assigns global mask-instance IDs to Gaussians using pixel-level top contributors.

    For each frame:
        for each mask instance:
            for each pixel in that instance:
                g = pixel_best_gaussian[(cam_idx, pix)]  (top contributing Gaussian for that pixel)
                assign Gaussian g the global mask id (gid)

    Inputs:
        mask_instances_dict:
            dict[frame_name -> dict[inst_id -> {"pixel_indices": list[int], ...}]]

        pixel_best_gaussian:
            Either:
              - dict[(cam_idx, pix) -> gaussian_idx]  (slow, Python dict)
            OR (preferred / fast):
              - list/tuple of length C where each entry is a LongTensor [H*W]
                giving best Gaussian per pixel (=-1 if none)

        cam_offset_dict:
            dict[frame_name -> starting global pixel index] (not required here, but kept for API consistency)

        global_id_map:
            dict[(frame_name, local_inst_id) -> global_inst_id]

        num_gaussians:
            int, total Gaussians

        device:
            torch.device

    Outputs:
        mask_instance_ids:
            LongTensor[num_gaussians] with a single assigned mask id per Gaussian.
            If a Gaussian is hit by multiple masks, this stores the FIRST seen id.

        shared_masks:
            dict[gaussian_idx -> set(global_mask_ids)] with ALL mask ids that touched that Gaussian
    """
    mask_instance_ids = -1 * torch.ones(num_gaussians, dtype=torch.long, device=device)
    shared_masks = defaultdict(set)

    cam_names = list(cam_offset_dict.keys())

    # ------------------------------------------------------------
    # Fast path: pixel_best_gaussian is per-camera tensor [H*W]
    # ------------------------------------------------------------
    if isinstance(pixel_best_gaussian, (list, tuple)) and len(pixel_best_gaussian) == len(cam_names):
        for cam_idx, frame_name in enumerate(tqdm(cam_names, desc="Assigning gaussian mask ids (per camera)", total=len(cam_names))):
            best_map = pixel_best_gaussian[cam_idx].to(device)  # [H*W], gaussian ids or -1

            inst_items = mask_instances_dict.get(frame_name, {}).items()
            for inst_id, inst_data in tqdm(inst_items, desc=f"  Camera {frame_name} instances", leave=False):
                gid = int(global_id_map[(frame_name, inst_id)])
                pix_list = inst_data.get("pixel_indices", [])
                if not pix_list:
                    continue

                pix = torch.tensor(pix_list, dtype=torch.long, device=device)
                g = best_map[pix]                      # [num_pix]
                g = g[g >= 0]                          # drop invalid pixels
                if g.numel() == 0:
                    continue

                # Record all mask ids that touch each gaussian
                for gi in g.tolist():
                    shared_masks[gi].add(gid)
                    if mask_instance_ids[gi] < 0:
                        mask_instance_ids[gi] = gid

        return mask_instance_ids, shared_masks

    # ------------------------------------------------------------
    # Slow path: pixel_best_gaussian is dict[(cam_idx,pix)->g]
    # ------------------------------------------------------------
    for cam_idx, frame_name in enumerate(tqdm(cam_names, desc="Assigning gaussian mask ids (slow path)", total=len(cam_names))):
        inst_items = mask_instances_dict.get(frame_name, {}).items()
        for inst_id, inst_data in tqdm(inst_items, desc=f"  Camera {frame_name} instances", leave=False):
            gid = int(global_id_map[(frame_name, inst_id)])
            pix_list = inst_data.get("pixel_indices", [])
            if not pix_list:
                continue

            for pix in pix_list:
                g = pixel_best_gaussian.get((cam_idx, pix), None)
                if g is None or g < 0:
                    continue

                shared_masks[g].add(gid)
                if mask_instance_ids[g] < 0:
                    mask_instance_ids[g] = gid

    return mask_instance_ids, shared_masks



@torch.no_grad()
def semantic_weighted_quality_score(
    labels_np: np.ndarray,
    xyz: torch.Tensor,
    emb: torch.Tensor,
    semantic_mask: torch.Tensor,
    lambda_emb: float = 1.0,
    microcluster_penalty: float = 0.5,
    blob_penalty: float = 0.3,
    size_floor: int = 5,

    sem_is_logits: bool = True,
    sem_denom_power: float = 2.0,
    sem_mass_power: float = 1.0,
    sem_eps: float = 1e-6,

    sem_scale_power: float = 0.0,
    sem_scale_eps: float = 1e-6,
    sem_add_penalty: float = 0.0,

    eps: float = 1e-12,
    debug: bool = False,
):
    """
    Lower is better. 
    - intra terms: each cluster's denominator is multiplied by (mean_semantic)^sem_denom_power => high mean semantic clusters look "denser" (lower intra)
    - inter weights / entropy masses also use the same semantic-mean shaping (sem_mass_power) => clustering structure is evaluated with emphasis on semantically reliable clusters
    """

    device = xyz.device
    labels = torch.as_tensor(labels_np, device=device, dtype=torch.long)

    # sem in (0,1) (no clamping)
    sem = semantic_mask.detach().view(-1).float().to(device)
    sem = torch.sigmoid(sem) if sem_is_logits else sem

    labeled = labels >= 0
    if labeled.sum().item() == 0:
        # only noise
        score = float(1e6)
        return score, {"K": 0, "K_eff": 0.0, "SemC": 0.0, "score": score}

    emb_n = F.normalize(emb.detach(), dim=1)

    cids = torch.unique(labels[labeled])
    cids = cids[cids >= 0]
    K_true = int(cids.numel())
    if K_true == 0:
        score = float(1e6)
        return score, {"K": 0, "K_eff": 0.0, "SemC": 0.0, "score": score}

    cent_xyz = []
    cent_emb = []

    w_cluster = []
    mu_sem_cluster = []
    mass_eff = []

    intra_xyz_num = 0.0
    intra_xyz_den = 0.0
    intra_emb_num = 0.0
    intra_emb_den = 0.0

    micro_count = 0

    for cid in cids.tolist():
        m = (labels == int(cid))
        n = int(m.sum().item())
        if n == 0:
            continue

        pts = xyz[m]
        e   = emb_n[m]
        s   = sem[m]  # per-point semantic

        # But ensure strictly positive weights for stability in centroids:
        s_eff = s + float(sem_eps)

        wsum = float(s_eff.sum().item())
        mu   = float(s.mean().item())  # mean semantic from raw s (not s_eff)

        denom_mult = float(max(mu, 0.0) + sem_eps) ** float(sem_denom_power)

        # centroids
        cx = (pts * s_eff[:, None]).sum(dim=0) / (s_eff.sum() + eps)
        ce = (e   * s_eff[:, None]).sum(dim=0) / (s_eff.sum() + eps)
        ce = ce / (ce.norm() + eps)

        cent_xyz.append(cx)
        cent_emb.append(ce)

        # masses
        w_cluster.append(wsum)
        mu_sem_cluster.append(mu)
        meff = wsum * (float(max(mu, 0.0) + sem_eps) ** float(sem_mass_power))
        mass_eff.append(meff)

        # intra xyz: weighted variance (singleton => 0, fine)
        d2 = ((pts - cx) ** 2).sum(dim=1)
        intra_xyz_num += float((s_eff * d2).sum().item())
        intra_xyz_den += float(wsum * denom_mult)

        # intra emb: weighted (1-cos to centroid) (singleton => 0, fine)
        cos_i = (e * ce[None, :]).sum(dim=1).clamp(-1, 1)
        intra_emb_num += float((s_eff * (1.0 - cos_i)).sum().item())
        intra_emb_den += float(wsum * denom_mult)

        if n < size_floor:
            micro_count += 1

    # If somehow we lost everything (shouldn’t happen unless NaNs), keep safe:
    if len(cent_xyz) == 0:
        score = float(1e6)
        return score, {"K": 0, "K_eff": 0.0, "SemC": 0.0, "score": score}

    # If only 1 cluster, inter terms are undefined; treat as blob-ish but still scoreable
    if len(cent_xyz) == 1:
        # intra terms
        intra_xyz = float(intra_xyz_num / (intra_xyz_den + eps))
        intra_emb = float(intra_emb_num / (intra_emb_den + eps))
        S_xyz = intra_xyz  # no inter normalization possible
        S_emb = intra_emb

        # penalties: blob is maxed-ish, fragmentation is low
        blob_pen = float(blob_penalty)
        micro_pen = microcluster_penalty * float(micro_count > 0)

        base = float(S_xyz + lambda_emb * S_emb + micro_pen + blob_pen)

        # semantic “quality”
        mass_eff_t = torch.tensor(mass_eff, device=device).float()
        mu_sem_t   = torch.tensor(mu_sem_cluster, device=device).float()
        SemC = float((mu_sem_t * mass_eff_t).sum().item() / (mass_eff_t.sum().item() + eps))

        SemC_eff = float(max(SemC, 0.0)) ** float(sem_scale_power)
        score = float(base / (SemC_eff + sem_scale_eps) + sem_add_penalty * (1.0 - SemC))

        info = {
            "K": 1,
            "K_true": K_true,
            "K_eff": 1.0,
            "SemC": float(SemC),
            "S_xyz": float(S_xyz),
            "S_emb": float(S_emb),
            "micro_pen": float(micro_pen),
            "blob_pen": float(blob_pen),
            "base": float(base),
            "score": float(score),
        }
        if debug:
            print("[Score]", info)
        return score, info

    # normal case: K >= 2
    cent_xyz = torch.stack(cent_xyz, dim=0)
    cent_emb = torch.stack(cent_emb, dim=0)
    mass_eff = torch.tensor(mass_eff, device=device).float()
    mu_sem_cluster = torch.tensor(mu_sem_cluster, device=device).float()

    K_score = int(cent_xyz.shape[0])

    SemC = float((mu_sem_cluster * mass_eff).sum().item() / (mass_eff.sum().item() + eps))

    idx_i, idx_j = torch.triu_indices(K_score, K_score, offset=1, device=device)
    wij = (mass_eff[idx_i] * mass_eff[idx_j])
    wij_sum = float(wij.sum().item()) + eps

    d_xyz = torch.linalg.norm(cent_xyz[idx_i] - cent_xyz[idx_j], dim=1)
    inter_xyz = float((wij * d_xyz).sum().item()) / wij_sum

    cos_c = (cent_emb[idx_i] * cent_emb[idx_j]).sum(dim=1).clamp(-1, 1)
    d_emb = (1.0 - cos_c)
    inter_emb = float((wij * d_emb).sum().item()) / wij_sum

    intra_xyz = float(intra_xyz_num / (intra_xyz_den + eps))
    intra_emb = float(intra_emb_num / (intra_emb_den + eps))

    S_xyz = intra_xyz / (inter_xyz + eps)
    S_emb = intra_emb / (inter_emb + eps)

    # fragmentation + blob penalties on effective mass distribution
    p = mass_eff / (mass_eff.sum() + eps)
    H = float(-(p * (p + eps).log()).sum().item())
    K_eff = float(torch.exp(torch.tensor(H, device=device)).item())

    N_lab = float(labeled.sum().item())
    micro_pen = microcluster_penalty * (np.log(1.0 + K_eff) / np.log(1.0 + max(2.0, N_lab)))
    micro_pen += microcluster_penalty * (float(micro_count) / max(1.0, float(K_score)))

    p_max = float(p.max().item())
    blob_pen = blob_penalty * max(0.0, p_max - 0.6) / 0.4

    # base = float(S_xyz + lambda_emb * S_emb + micro_pen + blob_pen)
    base = float(S_xyz + lambda_emb * S_emb)

    SemC_eff = float(max(SemC, 0.0)) ** float(sem_scale_power)
    score = float(base / (SemC_eff + sem_scale_eps) + sem_add_penalty * (1.0 - SemC))

    info = {
        "K_true": int(K_true),
        "K": int(K_score),
        "K_eff": float(K_eff),
        "micro_count": int(micro_count),
        "p_max": float(p_max),
        "SemC": float(SemC),
        "S_xyz": float(S_xyz),
        "S_emb": float(S_emb),
        "intra_xyz": float(intra_xyz),
        "inter_xyz": float(inter_xyz),
        "intra_emb": float(intra_emb),
        "inter_emb": float(inter_emb),
        "micro_pen": float(micro_pen),
        "blob_pen": float(blob_pen),
        "base": float(base),
        "score": float(score),
        "sem_denom_power": float(sem_denom_power),
        "sem_mass_power": float(sem_mass_power),
    }

    if debug:
        print("[Score]", info)

    return score, info

def restrict_knn_to_subset(nbr_idx_cpu: torch.Tensor, subset_idx: torch.Tensor) -> torch.Tensor:
    """
    nbr_idx_cpu: [N, K] on CPU, neighbors in FULL index space
    subset_idx:  [M] on GPU/CPU, indices into FULL space

    Returns:
      nbr_sub: [M, K] on CPU, neighbors in SUBSET-local indices, -1 if neighbor not in subset
    """
    nbr = nbr_idx_cpu.detach().cpu().long()              # [N,K]
    sub = subset_idx.detach().cpu().long()               # [M]

    N = nbr.shape[0]
    M = sub.shape[0]
    K = nbr.shape[1]

    # full->sub mapping
    full_to_sub = torch.full((N,), -1, dtype=torch.long)
    full_to_sub[sub] = torch.arange(M, dtype=torch.long)

    # map neighbors: full idx -> sub idx (or -1)
    nbr_full = nbr[sub]                                  # [M,K] in full space
    nbr_sub = full_to_sub[nbr_full]                      # [M,K] in subset space (or -1)
    return nbr_sub

def build_embedded_knn_distance_graph(
    xyz: torch.Tensor,
    emb: torch.Tensor,
    sem: torch.Tensor,
    nbr_sub_cpu: torch.Tensor,

    # base xyz+emb
    alpha: float = 1.0,
    beta: float = None,

    # NEW: appearance (rgb + opacity) knobs
    rgb: torch.Tensor = None,          # [M,3] in [0,1]
    op: torch.Tensor = None,           # [M] or [M,1] in [0,1]
    gamma_rgb: float = 0.05,           # weight for rgb distance term  (<< tune)
    gamma_op: float = 0.005,           # weight for opacity distance   (<< smaller than rgb)
    rgb_pow: float = 1.5,              # amplify rgb differences (>1)
    op_pow: float = 0.5,               # compress opacity differences (<1)
    clamp_rgb: bool = True,

    # semantic knobs
    sem_gamma: float = 4.0,
    sem_pow: float = 2.0,
    sem_floor: float = 0.0,
    sem_beta: float = None,
    sem_mode: str = "diff",

    # cosine clip
    mu_cos: float = None,
    clip_cos: float = 3.0,

    symmetrize: bool = True,
    debug: bool = False,
):
    """
    Build sparse CSR distance matrix D for clustering using only KNN edges.

    Base:
      d0 = alpha*||xi-xj||2 + beta*(1 - cos(ei,ej))

    Optional appearance term (if rgb/op provided):
      d_app = gamma_rgb * ||rgb_i-rgb_j||_2^rgb_pow + gamma_op * |op_i-op_j|^op_pow
      d0 += d_app

    Semantic modifications (optional):
      - scale: d0 *= exp(gamma*(1-si)) * exp(gamma*(1-sj))
      - diff : d += sem_beta * |si - sj|^sem_pow
    """

    xyz_ = xyz.detach().float().cpu()                      # [M,3]
    emb_ = F.normalize(emb.detach().float(), dim=1).cpu()  # [M,D]
    nbr = nbr_sub_cpu.detach().cpu().long()                # [M,K]
    M, K = nbr.shape

    # Edge lists
    rows = torch.arange(M).unsqueeze(1).expand(M, K).reshape(-1)
    cols = nbr.reshape(-1)
    valid = cols >= 0
    rows = rows[valid]
    cols = cols[valid]

    if rows.numel() == 0:
        D = csr_matrix((M, M), dtype=np.float32)
        if debug:
            print("[Graph][HDB] No valid edges after subset restriction (rows=0).")
        return D


    # xyz distance (RAW)
    dx = xyz_[rows] - xyz_[cols]
    d_xyz_raw = torch.linalg.norm(dx, dim=1)                   # [E]
    
    # --------------------------------------------------
    # robust xyz normalization to remove scene scale
    # --------------------------------------------------
    with torch.no_grad():
        med_xyz = float(torch.median(d_xyz_raw).item())
        if (not np.isfinite(med_xyz)) or (med_xyz <= 1e-12):
            med_xyz = 1.0
    xyz_scale = med_xyz

    # normalized xyz distance used in d0
    d_xyz = d_xyz_raw / (xyz_scale + 1e-12)

    # cosine sim
    cos = (emb_[rows] * emb_[cols]).sum(dim=1).clamp(-1.0, 1.0)  # [E]

    # center + robust clip
    mu = float(cos.mean().item()) if mu_cos is None else float(mu_cos)
    std = float(cos.std().item() + 1e-6)
    z = ((cos - mu) / std).clamp(-clip_cos, clip_cos)
    cos_c = mu + z * std

    beta_val = 1.0 if beta is None else float(beta)

    # base distance (now scale-free wrt xyz units)
    d0 = float(alpha) * d_xyz + beta_val * (1.0 - cos_c)

    # --------------------------------------------------
    # NEW: appearance additions (rgb + opacity)
    # --------------------------------------------------
    d_rgb_term = None
    d_op_term = None
    d_app = None

    has_rgb = (rgb is not None) and (gamma_rgb is not None) and (float(gamma_rgb) > 0.0)
    has_op  = (op  is not None) and (gamma_op  is not None) and (float(gamma_op)  > 0.0)

    if has_rgb or has_op:
        if has_rgb:
            rgb_ = rgb.detach().float().cpu()
            if clamp_rgb:
                rgb_ = rgb_.clamp(0.0, 1.0)
            drgb = rgb_[rows] - rgb_[cols]                  # [E,3]
            d_rgb = torch.linalg.norm(drgb, dim=1)          # [E]
            if rgb_pow is not None and float(rgb_pow) != 1.0:
                d_rgb = d_rgb.clamp_min(1e-12).pow(float(rgb_pow))
            d_rgb_term = float(gamma_rgb) * d_rgb
        else:
            d_rgb_term = 0.0

        if has_op:
            op_ = op.detach().float().cpu()
            if op_.ndim == 2:
                op_ = op_.squeeze(1)
            op_ = op_.clamp(0.0, 1.0)
            dop = (op_[rows] - op_[cols]).abs()             # [E]
            if op_pow is not None and float(op_pow) != 1.0:
                dop = dop.clamp_min(1e-12).pow(float(op_pow))
            d_op_term = float(gamma_op) * dop
        else:
            d_op_term = 0.0

        d_app = d_rgb_term + d_op_term
        d0 = d0 + d_app

    # --------------------------------------------------
    # semantic modifications
    # --------------------------------------------------
    scale = None
    edge_scale = None
    d_sem = None

    if sem is not None and sem_mode != "none":
        sem_ = sem.detach().float().cpu().view(-1)
        
        # if sem are logits, uncomment:
        # sem_ = torch.sigmoid(sem_)

        if sem_floor is not None:
            sem_ = sem_.clamp(min=float(sem_floor))
        sem_ = sem_.clamp(max=1.0)

        si = sem_[rows]
        sj = sem_[cols]

        if sem_mode in ("scale", "scale+diff"):
            scale = torch.exp(float(sem_gamma) * (1.0 - sem_))   # [M]
            edge_scale = scale[rows] * scale[cols]               # [E]
            d = d0 * edge_scale
        else:
            d = d0

        if sem_mode in ("diff", "scale+diff"):
            diff = (si - sj).abs()
            d_sem_raw = diff.pow(float(sem_pow))
            sem_beta_val = float(torch.median(d0).item()) if sem_beta is None else float(sem_beta)
            d_sem = float(sem_beta_val) * d_sem_raw
            d = d + d_sem
    else:
        d = d0

    # Build sparse matrix efficiently: LIL -> CSR
    r = rows.numpy().astype(np.int32)
    c = cols.numpy().astype(np.int32)
    data = d.numpy().astype(np.float32)

    D_lil = lil_matrix((M, M), dtype=np.float32)
    D_lil[r, c] = data
    if symmetrize:
        D_lil = D_lil.maximum(D_lil.T)

    D = D_lil.tocsr()
    D.setdiag(0.0)
    D.eliminate_zeros()

    # ----------------------------
    # DEBUG
    # ----------------------------
    if debug:
        def _p(x, q): return float(torch.quantile(x, q).item())

        print("\n[Graph][HDB][DEBUG] ----- Additive embedded distance diagnostics -----")
        print(f"[Graph][HDB][DEBUG] M={M} K={K} edges(E)={int(rows.numel())} symmetrize={symmetrize}")
        print(f"[Graph][HDB][DEBUG] alpha={float(alpha):.4f} beta={beta_val:.6f} mu_cos={mu:.6f} std_cos={std:.6f} clip_z={clip_cos:.2f}")
        print(f"[Graph][HDB][DEBUG] sem_mode={sem_mode} sem_gamma={sem_gamma} sem_pow={sem_pow} sem_beta={sem_beta if sem_beta is not None else 'auto(med d0)'}")

        print("[Graph][HDB][DEBUG] Cosine sim over edges (raw):")
        print(f"  min={float(cos.min().item()):.4f}  p05={_p(cos,0.05):.4f}  p50={_p(cos,0.50):.4f}  p95={_p(cos,0.95):.4f}  max={float(cos.max().item()):.4f}")

        print("[Graph][HDB][DEBUG] xyz distances (RAW d_xyz) over edges:")
        print(f"  min={float(d_xyz_raw.min().item()):.6f}  p50={_p(d_xyz_raw,0.50):.6f}  p95={_p(d_xyz_raw,0.95):.6f}  max={float(d_xyz_raw.max().item()):.6f}")
        print(f"[Graph][HDB][DEBUG] xyz normalization scale (median raw): {xyz_scale:.6f}")
        print("[Graph][HDB][DEBUG] xyz distances (NORMALIZED d_xyz) over edges:")
        print(f"  min={float(d_xyz.min().item()):.6f}  p50={_p(d_xyz,0.50):.6f}  p95={_p(d_xyz,0.95):.6f}  max={float(d_xyz.max().item()):.6f}")

        emb_term = beta_val * (1.0 - cos_c)
        print("[Graph][HDB][DEBUG] embedding term beta*(1-cos_c):")
        print(f"  min={float(emb_term.min().item()):.6f}  p50={_p(emb_term,0.50):.6f}  p95={_p(emb_term,0.95):.6f}  max={float(emb_term.max().item()):.6f}")

        if (has_rgb or has_op) and (d_app is not None):
            print("[Graph][HDB][DEBUG] appearance term (rgb/op) added to d0:")
            if has_rgb:
                dr = d_rgb_term if torch.is_tensor(d_rgb_term) else None
                print(f"  rgb: gamma_rgb={gamma_rgb} rgb_pow={rgb_pow}")
                print(f"    min={float(d_rgb_term.min().item()):.6f}  p50={_p(d_rgb_term,0.50):.6f}  p95={_p(d_rgb_term,0.95):.6f}  max={float(d_rgb_term.max().item()):.6f}")
            if has_op:
                print(f"  op : gamma_op ={gamma_op} op_pow ={op_pow}")
                print(f"    min={float(d_op_term.min().item()):.6f}  p50={_p(d_op_term,0.50):.6f}  p95={_p(d_op_term,0.95):.6f}  max={float(d_op_term.max().item()):.6f}")
            print("  total d_app:")
            print(f"    min={float(d_app.min().item()):.6f}  p50={_p(d_app,0.50):.6f}  p95={_p(d_app,0.95):.6f}  max={float(d_app.max().item()):.6f}")

        print("[Graph][HDB][DEBUG] base d0 (after app) = alpha*d_xyz + beta*(1-cos_c) + app:")
        print(f"  min={float(d0.min().item()):.6f}  p50={_p(d0,0.50):.6f}  p95={_p(d0,0.95):.6f}  max={float(d0.max().item()):.6f}")

        if sem is not None and sem_mode in ("scale", "scale+diff"):
            print("[Graph][HDB][DEBUG] semantic node scale stats:")
            print(f"  scale min={float(scale.min()):.3f}  p50={float(torch.quantile(scale,0.5)):.3f}  p95={float(torch.quantile(scale,0.95)):.3f}  max={float(scale.max()):.3f}")
            print("[Graph][HDB][DEBUG] edge scale multiplier stats:")
            print(f"  min={float(edge_scale.min()):.3f}  p50={float(torch.quantile(edge_scale,0.5)):.3f}  p95={float(torch.quantile(edge_scale,0.95)):.3f}  max={float(edge_scale.max()):.3f}")

        if sem is not None and sem_mode in ("diff", "scale+diff"):
            print("[Graph][HDB][DEBUG] semantic mismatch term d_sem stats:")
            print(f"  min={float(d_sem.min().item()):.6f}  p50={_p(d_sem,0.50):.6f}  p95={_p(d_sem,0.95):.6f}  max={float(d_sem.max().item()):.6f}")

        if D.nnz > 0:
            print(f"[Graph][HDB][DEBUG] CSR nnz={D.nnz} data_min={float(D.data.min()):.6f} data_max={float(D.data.max()):.6f} data_mean={float(D.data.mean()):.6f}")
        else:
            print("[Graph][HDB][DEBUG] CSR nnz=0 (empty).")
        print("[Graph][HDB][DEBUG] -----------------------------------------------------\n")

    return D


def run_hdbscan_precomputed(
    D_csr: csr_matrix,
    min_cluster_size: int,
    min_samples: int,
    eps: float = 0.0,                 # you can later map this to cluster_selection_epsilon if you want
    allow_single_cluster: bool = False,
    debug: bool = False,
):
    """
    Robust HDBSCAN on a *sparse precomputed* distance matrix by running per connected component.
    Returns labels for all nodes (size M), noise=-1.
    """

    if not isinstance(D_csr, csr_matrix):
        D_csr = D_csr.tocsr()

    M = D_csr.shape[0]
    if M == 0:
        return np.zeros((0,), dtype=np.int32)

    # adjacency for connectivity: any nonzero distance edge connects nodes
    A = D_csr.copy()
    A.data = np.ones_like(A.data, dtype=np.int8)

    n_comp, comp = connected_components(A, directed=False, return_labels=True)

    if debug:
        sizes = np.bincount(comp, minlength=n_comp)
        print(f"[HDB][CC] components={n_comp}  size_min={sizes.min()} size_med={np.median(sizes):.0f} size_max={sizes.max()}")
        if n_comp <= 30:
            print("[HDB][CC] sizes:", sizes.tolist())

    # Fast path: single component
    if n_comp == 1:
        clusterer = hdbscan.HDBSCAN(
            metric="precomputed",
            min_cluster_size=int(min_cluster_size),
            min_samples=int(min_samples),
            cluster_selection_epsilon=float(eps),
            allow_single_cluster=bool(allow_single_cluster),
            # core_dist_n_jobs=... (optional)
        )
        labels = clusterer.fit_predict(D_csr).astype(np.int32)
        return labels

    # Multi-component path
    labels_all = np.full((M,), -1, dtype=np.int32)
    next_label = 0

    for cid in range(n_comp):
        idx = np.where(comp == cid)[0]
        n = idx.size

        # Too small to form clusters -> keep noise
        if n < max(2, int(min_cluster_size)):
            continue

        # Submatrix (keep CSR)
        D_sub = D_csr[idx][:, idx]

        clusterer = hdbscan.HDBSCAN(
            metric="precomputed",
            min_cluster_size=int(min_cluster_size),
            min_samples=int(min_samples),
            cluster_selection_epsilon=float(eps),
            allow_single_cluster=bool(allow_single_cluster),
        )
        lab = clusterer.fit_predict(D_sub).astype(np.int32)

        # Remap local cluster ids to global ids (preserve -1 noise)
        ok = lab >= 0
        if ok.any():
            lab2 = lab.copy()
            lab2[ok] = lab2[ok] + next_label
            next_label = int(lab2[ok].max()) + 1
            labels_all[idx] = lab2
        else:
            labels_all[idx] = -1

        if debug:
            n_cl = int(lab.max()) + 1 if (lab >= 0).any() else 0
            n_noise = int((lab < 0).sum())
            print(f"[HDB][CC] cid={cid:03d} n={n:5d} clusters={n_cl:4d} noise={n_noise:5d}")

    return labels_all


def build_pixel_inst_cache(mask_instances_dict, cam_names, pixel_topk, device):
    """
    Build per-camera pixel -> instance label tensor.

    Returns:
        pixel_inst_list: list length C
            each entry is LongTensor [P] on device:
              -1 for background / unlabeled pixel
              >=0 for (frame, inst_key) unique id within that frame
    Notes:
        - We keep instance ids LOCAL per frame (0..num_inst-1), since training is same-frame only.
        - P is taken from pixel_topk[cam_idx].shape[0]
    """
    pixel_inst_list = []

    for cam_idx, frame_name in enumerate(cam_names):
        topk = pixel_topk[cam_idx]
        if topk is None:
            pixel_inst_list.append(None)
            continue

        P = topk.shape[0]
        pix_inst = torch.full((P,), -1, dtype=torch.long, device=device)

        inst_dict = mask_instances_dict.get(frame_name, {})
        if len(inst_dict) == 0:
            pixel_inst_list.append(pix_inst)
            continue

        # local compact ids per frame: inst_key -> 0..M-1
        inst_keys = list(inst_dict.keys())
        local_map = {k: i for i, k in enumerate(inst_keys)}

        for inst_key, data in inst_dict.items():
            pix = torch.as_tensor(data.get("pixel_indices", []), dtype=torch.long, device=device)
            if pix.numel() == 0:
                continue
            pix = pix[(pix >= 0) & (pix < P)]
            if pix.numel() == 0:
                continue
            pix_inst[pix] = local_map[inst_key]

        pixel_inst_list.append(pix_inst)

    return pixel_inst_list


def compute_hdbscan_bounds_from_mask(
    gaussian_to_mask_ids,   # dict[int -> set[int]]  (or shared_masks)
    fg_idx: torch.Tensor,   # [M] indices into full gaussians
    M: int,                 # len(fg_idx)
    factor_hi: float = 2.0,
    floor_min: int = 5,
    robust: bool = True,
    q_lo: float = 0.10,
    q_hi: float = 0.90,
    debug: bool = True,
):
    """
    Derive min_cluster_size bounds from mask usage:
      - count_mask[m] = number of FG gaussians that touch mask m
      - lower bound ~ min (or percentile) of count_mask
      - upper bound ~ max (or percentile) * factor_hi

    robust=True uses percentiles to ignore rare tiny/huge outliers.
    """

    fg_list = fg_idx.detach().cpu().tolist()

    # mask -> count of gaussians in FG touching it
    mask_count = defaultdict(int)
    for g in fg_list:
        masks = gaussian_to_mask_ids.get(int(g), None)
        if not masks:
            continue
        for m in masks:
            mask_count[int(m)] += 1

    if len(mask_count) == 0:
        # fallback if FG has no mask relations
        mcs_min = floor_min
        mcs_max = min(M, max(20, int(0.05 * M)))
        if debug:
            print("[BO][HDB] WARNING: no mask_count in FG; using fallback bounds.")
            print(f"[BO][HDB]   min_cluster_size: {mcs_min} → {mcs_max}")
        return mcs_min, mcs_max

    counts = np.array(list(mask_count.values()), dtype=np.float32)

    if robust and counts.size >= 10:
        lo = float(np.quantile(counts, q_lo))
        hi = float(np.quantile(counts, q_hi))
        min_seen = int(max(1, round(lo)))
        max_seen = int(max(1, round(hi)))
        tag = f"q{int(q_lo*100)}/q{int(q_hi*100)}"
    else:
        min_seen = int(counts.min())
        max_seen = int(counts.max())
        tag = "min/max"

    mcs_min = max(floor_min, min_seen)
    mcs_max = int(max_seen * float(factor_hi))

    # clamp
    mcs_min = int(np.clip(mcs_min, 2, M))
    mcs_max = int(np.clip(mcs_max, mcs_min, M))

    if debug:
        print(f"[BO][HDB] bounds from mask_usage(FG) [{tag}]: min_seen={min_seen} max_seen={max_seen} factor_hi={factor_hi}")
        print(f"[BO][HDB]   min_cluster_size: {mcs_min} → {mcs_max}  (clamped to M={M})")
        print(f"[BO][HDB]   masks_in_fg={len(mask_count)}  count_min={int(counts.min())}  count_med={int(np.median(counts))}  count_max={int(counts.max())}")

    return mcs_min, mcs_max


def compute_dbscan_minsamples_bounds_from_mask(
    gaussian_to_mask_ids,   # dict[int -> set[int]]
    fg_idx: torch.Tensor,   # [M] indices into full gaussians
    M: int,                 # len(fg_idx)
    floor_min: int = 5,
    factor_hi: float = 1.0,
    robust: bool = True,
    q_lo: float = 0.10,
    q_hi: float = 0.90,
    debug: bool = True,
):
    """
    Derive DBSCAN min_samples bounds from mask usage (same as your HDBSCAN bounds logic),
    but returning bounds for min_samples.

    Intuition: min_samples should be in the scale of typical per-instance gaussian support.

    Returns:
      ms_min, ms_max
    """
    fg_list = fg_idx.detach().cpu().tolist()

    mask_count = defaultdict(int)
    for g in fg_list:
        masks = gaussian_to_mask_ids.get(int(g), None)
        if not masks:
            continue
        for m in masks:
            mask_count[int(m)] += 1

    if len(mask_count) == 0:
        ms_min = floor_min
        ms_max = min(M, max(20, int(0.05 * M)))
        if debug:
            print("[BO][DBSCAN] WARNING: no mask_count in FG; using fallback bounds.")
            print(f"[BO][DBSCAN]   min_samples: {ms_min} → {ms_max}")
        return int(ms_min), int(ms_max)

    counts = np.array(list(mask_count.values()), dtype=np.float32)

    if robust and counts.size >= 10:
        lo = float(np.quantile(counts, q_lo))
        hi = float(np.quantile(counts, q_hi))
        min_seen = int(max(1, round(lo)))
        max_seen = int(max(1, round(hi)))
        tag = f"q{int(q_lo*100)}/q{int(q_hi*100)}"
    else:
        min_seen = int(counts.min())
        max_seen = int(counts.max())
        tag = "min/max"

    ms_min = max(int(floor_min), int(min_seen))
    ms_max = int(max_seen * float(factor_hi))

    # clamp
    ms_min = int(np.clip(ms_min, 1, M))
    ms_max = int(np.clip(ms_max, ms_min, M))

    if debug:
        print(f"[BO][DBSCAN] bounds from mask_usage(FG) [{tag}]: min_seen={min_seen} max_seen={max_seen} factor_hi={factor_hi}")
        print(f"[BO][DBSCAN]   min_samples: {ms_min} → {ms_max}  (clamped to M={M})")
        print(f"[BO][DBSCAN]   masks_in_fg={len(mask_count)}  count_min={int(counts.min())}  count_med={int(np.median(counts))}  count_max={int(counts.max())}")

    return int(ms_min), int(ms_max)

def run_dbscan_precomputed(D_csr: csr_matrix, eps: float, min_samples: int, seed_idx: int):
    """
    Returns:
      labels: np.int32 [M], where cluster containing seed is label 0, others noise (-1)
      seed_cluster_nodes: np.int64 indices of nodes in the seed cluster (local indices)
    Notes:
      - Works on sparse CSR distances (kNN graph), NOT full all-pairs.
      - DBSCAN semantics approximate standard DBSCAN but on your kNN graph support.
    """
    if not isinstance(D_csr, csr_matrix):
        D_csr = D_csr.tocsr()

    M = D_csr.shape[0]
    labels = -np.ones((M,), dtype=np.int32)
    if M == 0 or seed_idx < 0 or seed_idx >= M:
        return labels, np.zeros((0,), dtype=np.int64)

    # adjacency at eps
    indptr = D_csr.indptr
    indices = D_csr.indices
    data = D_csr.data

    # precompute eps-neighbors count (degree under eps)
    deg = np.zeros((M,), dtype=np.int32)
    for i in range(M):
        a, b = indptr[i], indptr[i+1]
        if a == b:
            continue
        deg[i] = int(np.sum(data[a:b] <= eps))

    is_core = deg >= int(min_samples)

    # if seed is not core, DBSCAN would label it noise unless it is within eps of a core point
    # we handle this by trying to expand from seed if it can reach any core; else -> noise
    # Step 1: find initial frontier: seed + its eps neighbors
    def eps_neighbors(i):
        a, b = indptr[i], indptr[i+1]
        if a == b:
            return np.zeros((0,), dtype=np.int64)
        mask = data[a:b] <= eps
        return indices[a:b][mask].astype(np.int64, copy=False)

    # if seed is core: start from it
    if is_core[seed_idx]:
        queue = [int(seed_idx)]
    else:
        # seed is border/noise: if it touches any core under eps, start from that core(s)
        nb = eps_neighbors(seed_idx)
        core_nbs = nb[is_core[nb]] if nb.size > 0 else np.zeros((0,), dtype=np.int64)
        if core_nbs.size == 0:
            return labels, np.zeros((0,), dtype=np.int64)
        queue = [int(x) for x in np.unique(core_nbs)]

    # BFS expansion: only core nodes expand, but cluster can include border nodes too
    visited = np.zeros((M,), dtype=np.uint8)
    in_cluster = np.zeros((M,), dtype=np.uint8)

    while queue:
        u = queue.pop()
        if visited[u]:
            continue
        visited[u] = 1
        in_cluster[u] = 1

        nb = eps_neighbors(u)
        if nb.size == 0:
            continue

        # all eps-neighbors are in cluster (border allowed)
        in_cluster[nb] = 1

        # expand only through core neighbors
        core_nb = nb[is_core[nb]]
        for v in core_nb.tolist():
            if not visited[v]:
                queue.append(int(v))

    nodes = np.where(in_cluster > 0)[0].astype(np.int64)
    labels[nodes] = 0
    return labels, nodes


@torch.no_grad()
def enforce_mask_instance_consistency(
    mask_instances_dict,
    cam_names,
    pixel_topk_gaussians,
    gaussian_instance_ids,
    ignore_label: int = -1,          # background/noise
    prefer_non_noise: bool = True,   # majority computed excluding ignore_label when possible
    device="cpu",
    debug: bool = True,
):
    """
    GLOBAL VOTE + mismatch diagnostics, BUT:
      - only gaussians with label != ignore_label (i.e., >=0) participate and can be modified
      - gaussians with label == ignore_label are never reassigned
    """
    if device is None:
        device = gaussian_instance_ids.device

    gauss_lab = gaussian_instance_ids.to(device)

    votes = defaultdict(lambda: defaultdict(int))

    total_instances = 0
    total_gauss_touched = 0

    per_frame_stats = defaultdict(lambda: {"instances": 0, "touched": 0, "voted_gauss": 0, "skipped_no_fg": 0})

    # mismatch diagnostics
    mismatch_global = {"num": 0, "den": 0}
    per_frame_mismatch = defaultdict(lambda: {"num": 0, "den": 0})
    worst_instances = []

    instances_skipped_no_fg = 0

    # -------------------------
    # Pass 1: collect votes (FG-only: label != -1)
    # -------------------------
    for cam_idx, frame_name in enumerate(cam_names):
        if cam_idx >= len(pixel_topk_gaussians):
            break

        topk = pixel_topk_gaussians[cam_idx]
        if topk is None:
            continue
        if topk.device != device:
            topk = topk.to(device)

        inst_dict = mask_instances_dict.get(frame_name, {})
        if not inst_dict:
            continue

        for inst_id, data in inst_dict.items():
            pix_list = data.get("pixel_indices", [])
            if not pix_list:
                continue

            total_instances += 1
            per_frame_stats[frame_name]["instances"] += 1

            pix = torch.as_tensor(pix_list, dtype=torch.long, device=device)
            P = topk.shape[0]
            pix = pix[(pix >= 0) & (pix < P)]
            if pix.numel() == 0:
                continue

            # gaussians that contribute to pixels of this instance
            g = topk[pix].reshape(-1)
            g = g[g >= 0]
            if g.numel() == 0:
                continue
            g = torch.unique(g)

            total_gauss_touched += int(g.numel())
            per_frame_stats[frame_name]["touched"] += int(g.numel())

            lab_all = gauss_lab[g]

            # -------------------------
            # FG gate: only consider gaussians whose current label != ignore_label
            # -------------------------
            fg_mask = (lab_all != ignore_label)
            if not fg_mask.any():
                instances_skipped_no_fg += 1
                per_frame_stats[frame_name]["skipped_no_fg"] += 1
                continue

            g_fg = g[fg_mask]
            lab = lab_all[fg_mask]  # all != ignore_label, i.e. >=0 in your convention

            # Determine target label (majority among FG)
            if prefer_non_noise:
                # ignore_label shouldn't appear here, but keep logic consistent
                lab_valid = lab[lab != ignore_label]
                if lab_valid.numel() > 0:
                    uniq, cnt = torch.unique(lab_valid, return_counts=True)
                else:
                    uniq, cnt = torch.unique(lab, return_counts=True)
            else:
                uniq, cnt = torch.unique(lab, return_counts=True)

            target = int(uniq[cnt.argmax()].item())

            # mismatch diagnostics for this (frame, inst) on FG subset
            lab_eval = lab
            den = int(lab_eval.numel())
            if den > 0:
                num = int((lab_eval != target).sum().item())
                mismatch_global["num"] += num
                mismatch_global["den"] += den
                per_frame_mismatch[frame_name]["num"] += num
                per_frame_mismatch[frame_name]["den"] += den

                ratio = num / max(1, den)
                worst_instances.append({
                    "frame": frame_name,
                    "inst": inst_id,
                    "target": target,
                    "num": num,
                    "den": den,
                    "ratio": ratio,
                    "uniq_labels": uniq.detach().cpu().tolist(),
                    "cnt_labels": cnt.detach().cpu().tolist(),
                })
                if len(worst_instances) > 40:
                    worst_instances.sort(key=lambda x: x["ratio"], reverse=True)
                    worst_instances = worst_instances[:20]

            # Vote: one vote per gaussian per instance
            for gi in g_fg.tolist():
                votes[gi][target] += 1
            per_frame_stats[frame_name]["voted_gauss"] += int(g_fg.numel())

    # -------------------------
    # Pass 2: apply assignment once (FG-only apply)
    # -------------------------
    changed = 0
    assigned = 0
    gaussians_skipped_bg_in_apply = 0

    for gi, vc in votes.items():
        if not vc:
            continue

        cur = int(gauss_lab[gi].item())
        if cur == ignore_label:
            gaussians_skipped_bg_in_apply += 1
            continue

        best_cnt = max(vc.values())
        best_labels = [lab for lab, c in vc.items() if c == best_cnt]

        if cur in best_labels:
            target = cur
        else:
            target = int(min(best_labels))  # deterministic tie-break

        if cur != target:
            gauss_lab[gi] = target
            changed += 1
        assigned += 1

    stats = {
        "instances_processed": int(total_instances),
        "instances_skipped_no_fg": int(instances_skipped_no_fg),
        "gaussians_touched_total": int(total_gauss_touched),
        "gaussians_with_votes": int(assigned),
        "gaussians_changed_total": int(changed),
        "gaussians_skipped_bg_in_apply": int(gaussians_skipped_bg_in_apply),
        "per_frame": dict(per_frame_stats),

        "mismatch_global": {
            "num": int(mismatch_global["num"]),
            "den": int(mismatch_global["den"]),
            "ratio": float(mismatch_global["num"] / max(1, mismatch_global["den"])),
        },
        "per_frame_mismatch": {
            fn: {
                "num": int(v["num"]),
                "den": int(v["den"]),
                "ratio": float(v["num"] / max(1, v["den"])),
            }
            for fn, v in per_frame_mismatch.items()
        },
        "worst_instances": sorted(worst_instances, key=lambda x: x["ratio"], reverse=True)[:10],
    }

    if debug:
        print("[Consistency-GV-FG] ignore_label:", ignore_label)
        print("[Consistency-GV-FG] instances_processed:", stats["instances_processed"])
        print("[Consistency-GV-FG] instances_skipped_no_fg:", stats["instances_skipped_no_fg"])
        print("[Consistency-GV-FG] gaussians_touched_total:", stats["gaussians_touched_total"])
        print("[Consistency-GV-FG] gaussians_with_votes:", stats["gaussians_with_votes"])
        print("[Consistency-GV-FG] gaussians_changed_total:", stats["gaussians_changed_total"])
        print("[Consistency-GV-FG] gaussians_skipped_bg_in_apply:", stats["gaussians_skipped_bg_in_apply"])

        mg = stats["mismatch_global"]
        print(f"[Consistency-GV-FG] mismatch_global(FG-only): {mg['num']}/{mg['den']} = {100.0*mg['ratio']:.2f}%")

        frame_rows = []
        for fn, v in stats["per_frame_mismatch"].items():
            if v["den"] >= 50:
                frame_rows.append((fn, v["ratio"], v["num"], v["den"]))
        frame_rows.sort(key=lambda x: x[1], reverse=True)

        if frame_rows:
            print("[Consistency-GV-FG] worst frames by mismatch (den>=50):")
            for fn, r, num, den in frame_rows[:5]:
                print(f"  {fn}: {num}/{den} = {100.0*r:.2f}%")

        wi = stats["worst_instances"]
        if wi:
            print("[Consistency-GV-FG] worst instances by mismatch:")
            for x in wi:
                print(
                    f"  {x['frame']} inst={x['inst']} "
                    f"mismatch={x['num']}/{x['den']}={100.0*x['ratio']:.2f}% "
                    f"target={x['target']} labels={list(zip(x['uniq_labels'], x['cnt_labels']))}"
                )

    return gauss_lab, stats


def compute_pixel_topk_for_frames(scene, K=10, bg_color=(0,0,0), storage_device="cpu",use_gs_camera_keys=True):
    """
    Render high-K contributors per pixel for each training camera.

    Returns dict:
      pixel_topk_gaussians: list[C] of LongTensor [P,K] on storage_device
      pixel_topk_weights  : list[C] of FloatTensor [P,K] on storage_device
      cam_hw              : list[C] of (H,W)
      cam_names           : list[C] of camera image_name strings
    """
    from gaussian_renderer import render

    if use_gs_camera_keys:
        cam_names_in = list(scene.gs_cameras.keys())
        cameras = [scene.gs_cameras[n] for n in cam_names_in]
    else:
        cameras = scene.getTrainCameras()
        cam_names_in = [getattr(c, "image_name", str(i)) for i, c in enumerate(cameras)]

    gaussians = scene.gaussians
    

    render_device = gaussians.get_xyz.device
    store_device = torch.device(storage_device)

    pixel_topk_gaussians = [None] * len(cameras)
    pixel_topk_weights   = [None] * len(cameras)
    cam_hw   = [None] * len(cameras)
    cam_KK   = [None] * len(cameras)

    class _DummyPipe:
        debug = False
        antialiasing = False
        compute_cov3D_python = False
        convert_SHs_python = False

    dummy_pipe = _DummyPipe()
    bg_col = torch.tensor(bg_color, dtype=torch.float32, device=render_device)

    cam_names = [None] * len(cameras)

    for cam_idx, cam in enumerate(tqdm(cameras, desc=f"[Centroids] Rendering top-{K} contributors")):
        cam_names[cam_idx] = cam_names_in[cam_idx]

        with torch.no_grad():
            out = render(cam, gaussians, dummy_pipe, bg_col, contrib=True, K=int(K))

        if out is None or out.get("contrib_indices", None) is None:
            continue

        contrib = out["contrib_indices"]            # [H,W,K] (GPU)
        weights = out.get("contrib_opacities", None)  # [H,W,K] (GPU) — "contrib weights" from renderer

        H, W, KK = contrib.shape
        cam_hw[cam_idx] = (int(H), int(W))
        cam_KK[cam_idx] = int(KK)

        if weights is None:
            weights = torch.ones((H, W, KK), dtype=torch.float32, device=contrib.device)
        else:
            weights = weights.float()

        # Move to storage_device for CPU indexing / low VRAM
        contrib = contrib.to(store_device, non_blocking=False).long().reshape(-1, KK)    # [P,KK]
        weights = weights.to(store_device, non_blocking=False).float().reshape(-1, KK)  # [P,KK]

        pixel_topk_gaussians[cam_idx] = contrib
        pixel_topk_weights[cam_idx]   = weights

    return {
        "pixel_topk_gaussians": pixel_topk_gaussians,
        "pixel_topk_weights": pixel_topk_weights,
        "cam_hw": cam_hw,
        "cam_KK": cam_KK,
        "cam_names": cam_names
    }


def pick_centroid_gaussian_from_topk(
    gaussians,
    topk_g,                       # [K] long
    topk_w=None,                  # [K] float (optional)
    method="max_semantic",     # "weighted_mean" | "weighted_medoid" | "max_semantic"
    dist_gate=None,               # float in scene units (optional)
    min_keep=3,
    semantic_use_sigmoid=True,
):
    """
    Returns: gaussian id (int) or -1
    """

    # ---- ensure tensors ----
    g = topk_g if torch.is_tensor(topk_g) else torch.as_tensor(topk_g)
    if g.numel() == 0:
        return -1

    # ---- filter invalid ids ----
    valid = (g >= 0)
    if valid.sum().item() == 0:
        return -1
    g = g[valid]

    # ---- weights aligned to g ----
    if topk_w is None:
        w = torch.ones((g.shape[0],), dtype=torch.float32, device=g.device)
    else:
        tw = topk_w if torch.is_tensor(topk_w) else torch.as_tensor(topk_w)
        w = tw[valid].float()
        w = torch.clamp(w, min=0.0)

    # ---- fetch xyz + semantic from gaussians ----
    xyz_all = gaussians.get_xyz.detach()                 # [N,3]
    sem_all = gaussians.get_sem.detach()    # [N] logits 
    if method == "max_semantic" and sem_all is None:
        raise ValueError("method='max_semantic' requires gaussians.semantic_mask to be not None.")

    if sem_all is not None:
        sem_all = sem_all.detach().view(-1)

    # ---- move indices/weights to xyz device ----
    if xyz_all.device != g.device:
        g_dev = g.to(xyz_all.device)
        w_dev = w.to(xyz_all.device)
    else:
        g_dev = g
        w_dev = w

    pts = xyz_all[g_dev]  # [M,3]

    # ---- optional distance gating (gate g_dev, w_dev, pts together) ----
    if dist_gate is not None and pts.shape[0] > 1:
        wsum = w_dev.sum() + 1e-12
        mu = (pts * (w_dev[:, None] / wsum)).sum(dim=0, keepdim=True)  # [1,3]
        d = torch.linalg.norm(pts - mu, dim=1)                          # [M]
        keep = d <= float(dist_gate)

        if keep.sum().item() >= int(min_keep):
            pts   = pts[keep]
            g_dev = g_dev[keep]
            w_dev = w_dev[keep]

    if pts.shape[0] == 0:
        return -1

    # ============================================================
    # METHODS
    # ============================================================
    if method == "weighted_mean":
        wsum = w_dev.sum() + 1e-12
        mu = (pts * (w_dev[:, None] / wsum)).sum(dim=0)  # [3]
        d = torch.linalg.norm(pts - mu[None, :], dim=1)
        idx = int(torch.argmin(d).item())
        return int(g_dev[idx].item())

    elif method == "weighted_medoid":
        diff = pts[:, None, :] - pts[None, :, :]      # [M,M,3]
        dist = torch.linalg.norm(diff, dim=2)         # [M,M]
        cost = (dist * w_dev[None, :]).sum(dim=1)     # [M]
        idx = int(torch.argmin(cost).item())
        return int(g_dev[idx].item())

    elif method == "max_semantic":
        # IMPORTANT: compute sem *after* gating so shape matches w_dev
        sem = sem_all.to(xyz_all.device)[g_dev].float()   # [M]
        if semantic_use_sigmoid:
            sem = torch.sigmoid(sem)

        # tie-break with weights, but only if w_dev non-empty
        denom = (w_dev.max() + 1e-12)
        score = sem + 1e-6 * (w_dev / denom)

        idx = int(torch.argmax(score).item())
        return int(g_dev[idx].item())

    else:
        raise ValueError(f"Unknown method: {method}")

def compute_instance_centroid_gaussians(
    scene,
    mask_instances_dict,
    K_centroid=64,
    storage_device="cpu",
    method="weighted_medoid",   # "weighted_mean" or "weighted_medoid"
    dist_gate=None,             # e.g. 0.05 or None
    min_keep=3,
    debug=True,
):
    """
    Uses mask_instances_dict[frame][inst]["centroid_pixel"] (flat index) directly.

    Pipeline:
      1) render top-K contributors per pixel for each camera
      2) for each mask instance in that frame:
           p = centroid_pixel
           candidates = topk[p], weights = wts[p]
           pick robust centroid gaussian via pick_centroid_gaussian_from_topk()

    Also computes (internally) an eps prior radius per instance:
      radius = || xyz[g_far] - xyz[g_cent] ||,
      where g_far is selected from the pixel (inside the instance) farthest from centroid_pixel in 2D.

    Returns (UNCHANGED):
      centroid_gauss_map: dict (frame_name, inst_id) -> gaussian_id (int)
      stats: dict counters
    """

    # 1) render high-K
    topk_pack = compute_pixel_topk_for_frames(scene, K=K_centroid, storage_device=storage_device)
    pixel_topk = topk_pack["pixel_topk_gaussians"]   # list[C] [P,K]
    pixel_w    = topk_pack["pixel_topk_weights"]     # list[C] [P,K]
    cam_hw     = topk_pack["cam_hw"]                 # list[C] (H,W)

    # IMPORTANT: you said alignment is fine, so we keep it simple:
    cam_names = topk_pack.get("cam_names", None)
    if cam_names is None:
        cam_names = list(scene.gs_cameras.keys())  # safe fallback

    # xyz for eps prior computation (stays on gaussians device)
    xyz_all = scene.gaussians.get_xyz.detach()       # [N,3]

    centroid_gauss_map = {}

    # Internal-only: per-instance eps prior radius
    eps_prior_radius = {}  # (frame, inst_id) -> float

    stats = {
        "frames_total": len(cam_names),
        "frames_used": 0,
        "instances_total": 0,
        "instances_skipped_no_frame": 0,
        "instances_skipped_no_topk": 0,
        "instances_skipped_missing_centroid_pixel": 0,
        "instances_skipped_invalid_centroid_pixel": 0,
        "instances_skipped_no_pixels": 0,
        "instances_skipped_no_gauss": 0,
        "centroids_found": 0,
    }

    for cam_idx, frame_name in enumerate(cam_names):
        inst_dict = mask_instances_dict.get(frame_name, None)
        if inst_dict is None:
            stats["instances_skipped_no_frame"] += 1
            continue

        topk = pixel_topk[cam_idx]
        wts  = pixel_w[cam_idx]
        hw   = cam_hw[cam_idx]

        if topk is None or wts is None or hw is None:
            stats["instances_skipped_no_topk"] += len(inst_dict)
            continue

        H, W = hw
        P = int(H * W)

        if topk.shape[0] != P or wts.shape[0] != P:
            stats["instances_skipped_no_topk"] += len(inst_dict)
            continue

        stats["frames_used"] += 1

        for inst_id, inst_data in inst_dict.items():
            stats["instances_total"] += 1

            p = inst_data.get("centroid_pixel", None)
            if p is None:
                stats["instances_skipped_missing_centroid_pixel"] += 1
                continue
            p = int(p)
            if p < 0 or p >= P:
                stats["instances_skipped_invalid_centroid_pixel"] += 1
                continue

            pix_list = inst_data.get("pixel_indices", None)
            if pix_list is None or len(pix_list) == 0:
                stats["instances_skipped_no_pixels"] += 1
                continue

            # --- centroid gaussian from centroid pixel ---
            gK = topk[p]   # [K]
            wK = wts[p]    # [K]

            g_cent = pick_centroid_gaussian_from_topk(
                gaussians=scene.gaussians,
                topk_g=gK,
                topk_w=wK,
                method=method,
                dist_gate=dist_gate,
                min_keep=min_keep,
            )

            if int(g_cent) < 0:
                stats["instances_skipped_no_gauss"] += 1
                continue

            centroid_gauss_map[(frame_name, inst_id)] = int(g_cent)
            stats["centroids_found"] += 1

            # ------------------------------------------------------------
            # INTERNAL eps prior radius:
            # find farthest pixel (in 2D) from centroid_pixel within instance,
            # then pick a gaussian from that pixel (same method),
            # radius = 3D distance between those gaussians.
            # ------------------------------------------------------------
            far_pix = farthest_pixel_from_centroid_pixel(
                pixel_indices=inst_data.get("pixel_indices", []),
                centroid_pixel=p,
                W=W,
            )
            if far_pix is None:
                continue

            g_far = pick_centroid_gaussian_from_topk(
                gaussians=scene.gaussians,
                topk_g=topk[far_pix],     
                topk_w=wts[far_pix],
                method="max_semantic",     # <-- exactly as you asked
                dist_gate=None,            # usually DON'T gate here
                min_keep=1,
            )

            if g_cent >= 0 and g_far >= 0:
                r = torch.linalg.norm(xyz_all[g_far] - xyz_all[g_cent]).item()
                eps_prior_radius[(frame_name, inst_id)] = float(r)

    # expose eps radii without changing function return signature
    stats["eps_prior_radius"] = eps_prior_radius

    if debug:
        print("\n[Centroids] ---------- centroid-gaussian stats ----------")
        for k, v in stats.items():
            if k != "eps_prior_radius":
                print(f"[Centroids] {k}: {v}")
        print("[Centroids] --------------------------------------------\n")

        if len(eps_prior_radius) > 0:
            rs = np.array(list(eps_prior_radius.values()), dtype=np.float32)
            print(
                f"[Centroids] eps_prior_radius: n={len(rs)} "
                f"min={rs.min():.6f} p50={np.median(rs):.6f} "
                f"p95={np.quantile(rs,0.95):.6f} max={rs.max():.6f}\n"
            )

    return centroid_gauss_map, stats

def farthest_pixel_from_centroid_pixel(pixel_indices, centroid_pixel, W):
    """
    pixel_indices: list/array of flat pixel indices belonging to the instance
    centroid_pixel: flat index
    W: image width
    Returns: farthest flat pixel index (int), or None if empty
    """
    if pixel_indices is None or len(pixel_indices) == 0:
        return None

    p0 = int(centroid_pixel)
    cy, cx = divmod(p0, int(W))

    pix = np.asarray(pixel_indices, dtype=np.int64)
    pix = pix[pix >= 0]
    if pix.size == 0:
        return None

    ys = pix // int(W)
    xs = pix % int(W)
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2

    return int(pix[int(np.argmax(d2))])


# ============================================================
#  Seeded DBSCAN-like microclusters + IoU merge (BO over ms, IoU)
#  (Reuses: compute_instance_pairwise_iou_sparse + merge_instances_by_iou_threshold + UnionFind)
# ============================================================

def map_centroid_gaussians_to_fg_seeds(centroid_gauss_map, fg_idx, N_total):
    """
    centroid_gauss_map: dict[(frame, inst)-> global_gauss_id]
    fg_idx: torch.Tensor [M] global ids of FG gaussians
    N_total: int total gaussians in full model

    Returns:
      seed_fg: np.int32 [S] FG-local indices
    """
    fg_idx_cpu = fg_idx.detach().cpu().long().numpy()
    M = int(fg_idx_cpu.size)

    g2l = -np.ones((int(N_total),), dtype=np.int32)
    g2l[fg_idx_cpu] = np.arange(M, dtype=np.int32)

    seeds = []
    seen = set()
    for _, g in centroid_gauss_map.items():
        g = int(g)
        if g < 0 or g >= int(N_total):
            continue
        gl = int(g2l[g])
        if gl < 0 or gl in seen:
            continue
        seen.add(gl)
        seeds.append(gl)

    return np.asarray(seeds, dtype=np.int32)


def precompute_seed_sorted_neighbors(D_csr: csr_matrix, seed_fg: np.ndarray, max_take: int):
    """
    For each seed, store its neighbors sorted by distance (ascending).
    This avoids sorting again inside every BO iteration.

    Returns:
      neigh_dict: dict[int seed -> np.ndarray neighbors_sorted] (FG-local neighbor ids, excludes self)
    """
    if not isinstance(D_csr, csr_matrix):
        D_csr = D_csr.tocsr()

    seed_fg = np.asarray(seed_fg, dtype=np.int32)
    indptr, indices, data = D_csr.indptr, D_csr.indices, D_csr.data

    neigh_dict = {}
    for s in seed_fg.tolist():
        a, b = int(indptr[s]), int(indptr[s + 1])
        nbr = indices[a:b].astype(np.int32, copy=False)
        dst = data[a:b].astype(np.float32, copy=False)

        if nbr.size == 0:
            neigh_dict[int(s)] = np.zeros((0,), dtype=np.int32)
            continue

        # drop self if present
        mself = (nbr != int(s))
        nbr = nbr[mself]
        dst = dst[mself]

        if nbr.size == 0:
            neigh_dict[int(s)] = np.zeros((0,), dtype=np.int32)
            continue

        order = np.argsort(dst, kind="stable")
        nbr_sorted = nbr[order]
        if max_take is not None and int(max_take) > 0:
            nbr_sorted = nbr_sorted[: int(max_take)]
        neigh_dict[int(s)] = nbr_sorted

    return neigh_dict


def build_seed_microclusters_from_precomputed(seed_fg, neigh_dict, min_samples: int, require_full: bool = True):
    """
    Microcluster around each seed = {seed} U closest (min_samples-1) neighbors from neigh_dict.
    """
    seed_fg = np.asarray(seed_fg, dtype=np.int32)
    ms = int(max(2, min_samples))  # at least 2

    clusters = []
    kept = 0
    dropped = 0

    need = ms - 1
    for s in seed_fg.tolist():
        nbr = neigh_dict.get(int(s), None)
        if nbr is None:
            dropped += 1
            continue

        if nbr.size < need and require_full:
            dropped += 1
            continue

        take = min(need, int(nbr.size))
        if take > 0:
            arr = np.concatenate([np.asarray([s], np.int32), nbr[:take]])
        else:
            arr = np.asarray([s], np.int32)

        arr = np.unique(arr)
        arr.sort()
        clusters.append(arr)
        kept += 1

    return clusters, kept, dropped


def _build_inverted_index_for_sets(sets_list, M: int):
    """
    sets_list: list[np.ndarray] (each sorted unique)
    Returns:
      inv: list[list[int]] length M, inv[g] -> list of set indices containing g
    """
    inv = [[] for _ in range(int(M))]
    for si, arr in enumerate(sets_list):
        for g in arr.tolist():
            if 0 <= int(g) < int(M):
                inv[int(g)].append(int(si))
    return inv


def _union_sets_by_component(sets_list, comp_id: np.ndarray):
    """
    comp_id: [n_sets] -> component label 0..K-1
    Returns:
      merged_sets: list[np.ndarray] length K, each sorted unique
    """
    comp_id = np.asarray(comp_id, dtype=np.int32)
    K = int(comp_id.max()) + 1 if comp_id.size > 0 else 0
    buckets = [[] for _ in range(K)]
    for i, c in enumerate(comp_id.tolist()):
        buckets[int(c)].append(i)

    merged = []
    for ids in buckets:
        if len(ids) == 1:
            merged.append(sets_list[ids[0]])
        else:
            all_pts = np.concatenate([sets_list[j] for j in ids]).astype(np.int32, copy=False)
            all_pts = np.unique(all_pts)
            all_pts.sort()
            merged.append(all_pts)
    return merged


def labels_from_merged_sets(
    merged_sets,
    M: int,
    min_cluster_size: int = 1,
    conflict_policy: str = "largest",  # "largest" | "noise"
):
    """
    Assign a single label per node. Anything not assigned -> -1.
    If overlaps exist:
      - largest: keep assignment from largest cluster first
      - noise  : overlapping nodes become -1
    """
    labels = -np.ones((int(M),), dtype=np.int32)
    if not merged_sets:
        return labels

    # sort clusters by size desc so deterministic
    order = np.argsort([-int(s.size) for s in merged_sets]).tolist()

    assigned = np.zeros((int(M),), dtype=np.uint8)
    overlap = np.zeros((int(M),), dtype=np.uint8)

    cid_out = 0
    for k in order:
        arr = merged_sets[k]
        if int(arr.size) < int(min_cluster_size):
            continue

        arr = arr.astype(np.int32, copy=False)

        if conflict_policy == "largest":
            keep = assigned[arr] == 0
            labels[arr[keep]] = int(cid_out)
            assigned[arr] = 1
            cid_out += 1
        else:
            # label first, then mark overlaps as noise later
            overlap[arr[assigned[arr] == 1]] = 1
            labels[arr[assigned[arr] == 0]] = int(cid_out)
            assigned[arr] = 1
            cid_out += 1

    if conflict_policy == "noise":
        labels[overlap == 1] = -1

    return labels

def merge_microclusters_by_iou_unionfind(microclusters, M: int, iou_thresh: float, min_intersection: int = 2):
    """
    microclusters: list[np.ndarray] (sorted unique FG-local node ids)
    Returns:
      merged_sets: list[np.ndarray]
    """

    n_sets = len(microclusters)
    if n_sets == 0:
        return []

    sizes = np.array([int(s.size) for s in microclusters], dtype=np.int32)

    # inverted index: node -> list of microcluster ids containing it
    inv = _build_inverted_index_for_sets(microclusters, M=int(M))

    inter_counts = defaultdict(int)
    for ids in inv:
        L = len(ids)
        if L < 2:
            continue
        ids = sorted(ids)
        for a in range(L):
            ia = ids[a]
            for b in range(a + 1, L):
                ib = ids[b]
                inter_counts[(ia, ib)] += 1

    uf = UnionFind(n_sets)

    thr = float(iou_thresh)
    for (ia, ib), inter in inter_counts.items():
        if inter < int(min_intersection):
            continue
        union = int(sizes[ia] + sizes[ib] - inter)
        if union <= 0:
            continue
        iou = float(inter) / float(union)
        if iou >= thr:
            uf.union(int(ia), int(ib))

    roots = np.array([uf.find(i) for i in range(n_sets)], dtype=np.int32)
    _, comp_id = np.unique(roots, return_inverse=True)
    comp_id = comp_id.astype(np.int32)

    merged_sets = _union_sets_by_component(microclusters, comp_id)
    return merged_sets