import os
import glob
import torch
import numpy as np
from PIL import Image

import torch.nn.functional as F

# -----------------------------
# Initialize clusters from COLMAP points (with known cluster mapping)
# -----------------------------
def initialize_clusters_from_colmap(gaussians_xyz, colmap_points, colmap_cluster_ids, batch_size=20000):
    """
    Assign each Gaussian to one of the known clusters derived from COLMAP points,
    WITHOUT moving the Gaussian positions.

    Args:
        gaussians_xyz:      [N, 3] tensor of Gaussian centers (on GPU)
        colmap_points:      [M, 3] tensor of COLMAP seed centers (on GPU)
        colmap_cluster_ids: [M] LongTensor giving known cluster ID (e.g., 0, 1, 2) for each COLMAP point
        batch_size:         number of Gaussians processed per batch to avoid OOM

    Returns:
        cluster_ids:        [N] LongTensor (nearest COLMAP cluster for each Gaussian)
        cluster_centroids:  [K, 3] Tensor (mean position of COLMAP points in each cluster)
    """
    device = gaussians_xyz.device
    dtype = gaussians_xyz.dtype

    colmap_points = colmap_points.to(device=device, dtype=dtype)
    colmap_cluster_ids = colmap_cluster_ids.to(device=device)

    # Compute cluster centroids (for assignment only)
    unique_clusters = torch.unique(colmap_cluster_ids, sorted=True)
    cluster_centroids = torch.stack([
        colmap_points[colmap_cluster_ids == cid].mean(dim=0)
        for cid in unique_clusters
    ], dim=0)  # [K, 3]

    N = gaussians_xyz.shape[0]
    cluster_ids = torch.empty(N, dtype=torch.long, device=device)

    # Assign each Gaussian to nearest COLMAP cluster centroid
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = gaussians_xyz[start:end]  # keep original geometry
        dist = torch.cdist(batch, cluster_centroids, p=2)
        cluster_ids[start:end] = torch.argmin(dist, dim=1)

    return cluster_ids, cluster_centroids

# -----------------------------
# Update cluster centroids and semantic averages
# -----------------------------
def compute_cluster_stats(gaussians_xyz, gaussians_sem, cluster_ids, num_clusters):
    cluster_centroids = []
    cluster_semantics = []
    cluster_sizes = []
    
    for cid in range(num_clusters):
        mask = cluster_ids == cid
        if mask.sum() == 0:
            cluster_centroids.append(torch.zeros(3, device=gaussians_xyz.device))
            cluster_semantics.append(torch.tensor(0.0, device=gaussians_sem.device))
            cluster_sizes.append(torch.tensor(0, device=gaussians_xyz.device))
            continue
        cluster_centroids.append(gaussians_xyz[mask].mean(dim=0))
        cluster_semantics.append(gaussians_sem[mask].mean())
        cluster_sizes.append(mask.sum())
    
    cluster_centroids = torch.stack(cluster_centroids, dim=0)
    cluster_semantics = torch.stack(cluster_semantics, dim=0)
    cluster_sizes = torch.tensor(cluster_sizes, device=gaussians_xyz.device)
    return cluster_centroids, cluster_semantics, cluster_sizes


# -----------------------------
# Merge clusters if too close
# -----------------------------
def merge_clusters(
    cluster_centroids,
    cluster_semantics,
    cluster_ids,
    merge_distance=0.05,
    merge_semantic_diff=0.1,
    A=None,                     
    device="cuda"
):
    """
    Merge clusters that are spatially close and semantically similar.
    If A is provided (sparse Gaussians × MaskInstances), clusters that share any mask instance
    will be force-merged as well.

    Fully preserves original API and return format.

    Args:
        cluster_centroids: [C, 3]
        cluster_semantics: [C] or [C, F]
        cluster_ids: [N]
        merge_distance: float
        merge_semantic_diff: float
        A: optional sparse adjacency matrix [N, M]
    
    Returns:
        updated_cluster_ids: [N]
        num_clusters: int
    """

    device = cluster_centroids.device
    C = cluster_centroids.shape[0]

    if C <= 1:
        return cluster_ids, C

    # ============================================================
    # 1) ORIGINAL SPATIAL MERGE LOGIC (unchanged)
    # ============================================================

    diff = cluster_centroids.unsqueeze(1) - cluster_centroids.unsqueeze(0)
    dists = torch.norm(diff, dim=2)

    if cluster_semantics.ndim == 1:
        sem_diff = torch.abs(
            cluster_semantics.unsqueeze(1) - cluster_semantics.unsqueeze(0)
        )
    else:
        sem_diff = torch.norm(
            cluster_semantics.unsqueeze(1) - cluster_semantics.unsqueeze(0),
            dim=2
        )

    merge_mask = (dists < merge_distance) & (sem_diff < merge_semantic_diff)
    merge_mask.fill_diagonal_(False)

    # ============================================================
    # 2) OPTIONAL MASK-INSTANCE MERGE (vectorized)
    # ============================================================
    if A is not None:
        # A: sparse [N, M] gaussian × mask-instance
        A = A.coalesce()
        row = A.indices()[0]        # gaussian id
        col = A.indices()[1]        # mask instance id
        M = A.size(1)

        # Map cluster_ids to gaussian rows
        old_ids_clamped = torch.clamp(cluster_ids, 0, C - 1)
        gaussian_to_cluster = old_ids_clamped

        # Build cluster × mask sparse matrix directly
        # Convert A (N x M) -> cluster_mask (C x M) using scatter
        cluster_mask = torch.zeros((C, M), dtype=torch.bool, device=device)
        # For each non-zero gaussian×mask, mark the corresponding cluster
        cluster_mask.index_put_(
            (gaussian_to_cluster[row], col), 
            torch.ones_like(col, dtype=torch.bool, device=device),
            accumulate=True
        )

        # clusters share at least one mask instance?
        mask_overlap = (cluster_mask.unsqueeze(1) & cluster_mask.unsqueeze(0)).any(dim=2)

        # Add mask-based merging
        merge_mask |= mask_overlap
        merge_mask.fill_diagonal_(False)

    # ============================================================
    # 3) UNION-FIND (unchanged)
    # ============================================================
    parent = torch.arange(C, device=device)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in merge_mask.nonzero(as_tuple=False):
        ri = find(i.item())
        rj = find(j.item())
        if ri != rj:
            parent[rj] = ri

    # ============================================================
    # 4) REMAP TO SEQUENTIAL IDS (unchanged)
    # ============================================================
    for i in range(C):
        find(i)

    unique_roots, inverse = torch.unique(parent, return_inverse=True)
    old_to_new = inverse[parent]

    old_ids_clamped = torch.clamp(cluster_ids, 0, C - 1)
    updated_cluster_ids = old_to_new[old_ids_clamped]

    return updated_cluster_ids, int(unique_roots.numel())

# -----------------------------
# Split clusters if too dispersed
# -----------------------------
def split_cluster(gaussians_xyz, gaussians_sem, cluster_ids, max_dispersion=0.05, max_semantic_std=0.1, max_clusters=None):
    """
    Split clusters with high spatial or semantic variance.
    Returns: new_cluster_ids (same shape), new_num_clusters (int)
    If max_clusters is provided, never exceed it: instead pick the largest-variance cluster splits up to capacity.
    """
    device = gaussians_xyz.device
    cur_ids = cluster_ids.clone()
    num_clusters = int(cur_ids.max().item()) + 1
    new_ids = cur_ids.clone()

    next_id = num_clusters
    potential_splits = []

    for cid in range(num_clusters):
        mask = cur_ids == cid
        if mask.sum() < 2:
            continue
        xyz_c = gaussians_xyz[mask]
        sem_c = gaussians_sem[mask]
        spatial_std = float(xyz_c.std(dim=0).mean().item())
        sem_std = float(sem_c.std().item())

        if spatial_std > max_dispersion or sem_std > max_semantic_std:
            potential_splits.append((cid, spatial_std + sem_std, mask.nonzero(as_tuple=True)[0]))

    # sort splits by descending variance so we perform most-important splits first
    potential_splits.sort(key=lambda x: x[1], reverse=True)

    for cid, score, idxs in potential_splits:
        # respect max_clusters if given
        if max_clusters is not None and next_id >= max_clusters:
            break

        mask = (cur_ids == cid)
        xyz_c = gaussians_xyz[mask]
        # compute principal axis
        cov = torch.cov(xyz_c.T)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        principal_axis = eigvecs[:, -1]
        proj = (xyz_c - xyz_c.mean(dim=0)) @ principal_axis
        med = proj.median()
        split_mask_local = proj > med

        # global indices of elements in this cluster
        global_idx = mask.nonzero(as_tuple=True)[0]
        to_split_idx = global_idx[split_mask_local]
        if to_split_idx.numel() == 0:
            continue

        new_ids[to_split_idx] = next_id
        next_id += 1

    # Finally reindex sequentially to avoid sparse id space
    unique, inv = torch.unique(new_ids, return_inverse=True)
    new_num = unique.numel()
    new_ids[:] = inv

    # If max_clusters provided: cap new_num <= max_clusters (we re-map extra clusters to nearest existing)
    if max_clusters is not None and new_num > max_clusters:
        # map extra ids to nearest cluster centroid (simple heuristic). For simplicity, reassign highest ids to nearest lower ones.
        # Build centroids for new_ids
        centroids = []
        for cid in range(new_num):
            m = new_ids == cid
            if m.sum() == 0:
                centroids.append(torch.zeros(3, device=device))
            else:
                centroids.append(gaussians_xyz[m].mean(0))
        centroids = torch.stack(centroids, dim=0)
        # compute distances between centroids and collapse highest ids into nearest lower ids until <= max_clusters
        while new_num > max_clusters:
            # collapse last id into nearest other
            last = new_num - 1
            d = torch.norm(centroids - centroids[last].unsqueeze(0), dim=1)
            d[last] = float('inf')
            target = int(torch.argmin(d).item())
            new_ids[new_ids == last] = target
            # recompute unique/inv
            unique, inv = torch.unique(new_ids, return_inverse=True)
            new_ids[:] = inv
            new_num = unique.numel()
            # recompute centroids (cheap because new_num small)
            centroids = []
            for cid in range(new_num):
                m = new_ids == cid
                if m.sum() == 0:
                    centroids.append(torch.zeros(3, device=device))
                else:
                    centroids.append(gaussians_xyz[m].mean(0))
            centroids = torch.stack(centroids, dim=0)

    return new_ids, int(new_num)


def select_top_gaussians_by_distance(segmented_gaussians, cluster_centroids, n_active):
    """
    Select the top `n_active` Gaussians closest to each cluster centroid.

    Args:
        segmented_gaussians (GaussianModel): GaussianModel object with get_xyz [N, 3].
        cluster_centroids (Tensor): Tensor of cluster centroids [num_clusters, 3].
        n_active (int): Number of Gaussians to select per cluster.

    Returns:
        LongTensor: Indices of selected Gaussians.
    """
    xyz = segmented_gaussians.get_xyz  # [N, 3]
    num_clusters = cluster_centroids.shape[0]

    selected_indices = []

    # Compute distances and pick closest n_active per cluster
    for i in range(num_clusters):
        centroid = cluster_centroids[i].unsqueeze(0)  # [1,3]
        dists = torch.norm(xyz - centroid, dim=1)     # [N]
        closest = torch.topk(dists, k=min(n_active, xyz.shape[0]), largest=False).indices
        selected_indices.append(closest)

    # Flatten and unique
    selected_indices = torch.unique(torch.cat(selected_indices))
    return selected_indices


def safe_index_tensor(index_tensor, max_index):
    """
    Clamp invalid indices to 0 and return a boolean mask of valid entries.
    index_tensor: LongTensor of arbitrary shape
    max_index: exclusive upper bound (int)
    returns: (safe_index_tensor, valid_mask) where valid_mask = index_tensor >=0 & < max_index
    """
    valid_mask = (index_tensor >= 0) & (index_tensor < max_index)
    safe = index_tensor.clone()
    safe[~valid_mask] = 0
    return safe.long(), valid_mask


def load_instance_masks(mask_dir, basename, device="cuda"):
    """
    Load all instance masks for a given RGB image basename.
    Returns:
        instance_masks_list: list of (instance_id, mask_tensor) tuples, mask_tensor [H, W]
        masks_tensor: FloatTensor [num_instances, H, W], 0-1
        num_instances: int
    """
    instance_masks_list = []
    mask_tensors = []

    pattern = os.path.join(mask_dir, f"{basename}_instance_*.png")
    files = sorted(glob.glob(pattern))

    for i, fpath in enumerate(files):
        mask = Image.open(fpath).convert("L")  # grayscale
        mask_np = np.array(mask, dtype=np.float32) / 255.0  # [H, W], float in [0,1]
        mask_tensor = torch.from_numpy(mask_np).to(device)
        instance_masks_list.append((i, mask_tensor))
        mask_tensors.append(mask_tensor)

    if mask_tensors:
        masks_tensor = torch.stack(mask_tensors, dim=0)  # [num_instances, H, W]
    else:
        masks_tensor = torch.zeros((0, 0, 0), device=device)

    num_instances = len(instance_masks_list)

    return instance_masks_list, masks_tensor, num_instances


def enforce_multicam_consistency_instance(cluster_ids, xyz, cameras, mask_inst_dir,
                                          pixel_tolerance=2.0, min_view_support=2, device="cuda"):
    """
    Enforce multi-camera consistency by merging clusters whose projections overlap
    in at least `min_view_support` cameras *and* correspond to the same instance ID.

    Args:
        cluster_ids (Tensor): [N] cluster assignment per Gaussian
        xyz (Tensor): [N, 3] world-space Gaussian centers
        cameras (list): list of Camera objects
        mask_inst_dir (str): path to instance mask folder
        pixel_tolerance (float)
        min_view_support (int)
    """
    merged_ids = cluster_ids.clone().to(device)

    # Cache per-camera projections and instance masks
    reproj_dict = {}
    instance_dict = {}

    for cam_idx, cam in enumerate(cameras):
        basename = os.path.splitext(os.path.basename(cam.image_name))[0]
        instance_dict[cam_idx] = load_instance_masks(mask_inst_dir, basename, device=device)

        # Build proper intrinsic matrix 3x3 from FoV
        width, height = cam.image_width, cam.image_height
        fx = 0.5 * width / torch.tan(torch.tensor(cam.FoVx / 2))
        fy = 0.5 * height / torch.tan(torch.tensor(cam.FoVy / 2))
        cx = width / 2
        cy = height / 2
        K = torch.tensor([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], device=device, dtype=torch.float32)

        # Project points
        pts_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)       # [N,4]
        cam_pts = (cam.world_view_transform @ pts_h.T).T                     # [N,4]
        cam_pts = cam_pts[:, :3] / cam_pts[:, 3:4]                           # [N,3]
        uv = (K @ cam_pts.T)[:2, :].T                                        # [N,2] pixel coordinates
        reproj_dict[cam_idx] = uv

    # Compare clusters pairwise
    unique_clusters = torch.unique(merged_ids)
    for i, cid_a in enumerate(unique_clusters):
        for cid_b in unique_clusters[i + 1:]:
            mask_a = (merged_ids == cid_a)
            mask_b = (merged_ids == cid_b)
            if not mask_a.any() or not mask_b.any():
                continue

            overlap_views = 0
            consistent_instance = 0

            for cam_idx, uv in reproj_dict.items():
                inst_masks = instance_dict[cam_idx]

                ua = uv[mask_a]
                ub = uv[mask_b]
                if ua.numel() == 0 or ub.numel() == 0:
                    continue

                dist = torch.cdist(ua, ub)
                if not (dist < pixel_tolerance).any():
                    continue  # no geometric overlap

                overlap_views += 1

                # Check instance masks
                for inst_id, mask_tensor in inst_masks:
                    h, w = mask_tensor.shape
                    # Convert uv -> pixel coords
                    pa = (ua * torch.tensor([w, h], device=device)).long()
                    pb = (ub * torch.tensor([w, h], device=device)).long()

                    # Clamp each axis separately
                    pa[:, 0] = pa[:, 0].clamp(0, w - 1)
                    pa[:, 1] = pa[:, 1].clamp(0, h - 1)
                    pb[:, 0] = pb[:, 0].clamp(0, w - 1)
                    pb[:, 1] = pb[:, 1].clamp(0, h - 1)
                    if (mask_tensor[pa[:, 1], pa[:, 0]].mean() > 0.5 and
                        mask_tensor[pb[:, 1], pb[:, 0]].mean() > 0.5):
                        consistent_instance += 1
                        break

            if overlap_views >= min_view_support and consistent_instance > 0:
                merged_ids[mask_b] = cid_a

    # Reindex to contiguous IDs
    unique_ids = torch.unique(merged_ids)
    id_map = {old.item(): new for new, old in enumerate(unique_ids)}
    merged_ids = torch.tensor([id_map[i.item()] for i in merged_ids], device=device, dtype=torch.long)

    return merged_ids


    def merge_clusters_sparse(A: torch.sparse_coo_tensor, device="cuda"):
        """
        Merge Gaussians via sparse adjacency to mask instances.
        
        Args:
            A: torch.sparse_coo_tensor [G, M], values=1
            device: cuda/cpu
            
        Returns:
            merged_cluster_ids: torch.Tensor [G]
        """
        # COO format indices
        indices = A.coalesce().indices()  # [2, nnz], row=gaussian, col=mask_instance
        row, col = indices[0], indices[1]

        num_gaussians = A.size(0)

        # Union-Find
        parent = torch.arange(num_gaussians, device=device)

        def find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != x:
                nxt = parent[x]
                parent[x] = root
                x = nxt
            return root

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x

        # Iterate over mask instances
        unique_masks = torch.unique(col)
        for m in unique_masks:
            gaussians = row[col == m]
            if len(gaussians) > 1:
                base = gaussians[0]
                for g in gaussians[1:]:
                    union(base.item(), g.item())

        # Assign final cluster IDs
        roots = torch.tensor([find(i) for i in range(num_gaussians)], device=device)
        _, merged_cluster_ids = torch.unique(roots, return_inverse=True)

        return merged_cluster_ids


    def compute_mask_center_weights(A, instance_centroids, gaussians_xyz, sigma=0.1):
        """
        Compute reliability weights per Gaussian based on distance to its assigned mask-instance centroid.

        Args:
            A (torch.sparse_coo_tensor): sparse adjacency matrix [N_gaussians, M_mask_instances]
            instance_centroids (torch.Tensor): [M,3] tensor of mask-instance centroids
            gaussians_xyz (torch.Tensor): [N,3] tensor of Gaussian positions
            sigma (float): controls decay of weight with distance

        Returns:
            torch.Tensor: [N] weights in [0,1]
        """
        device = gaussians_xyz.device
        A = A.coalesce()
        gaussian_ids, mask_ids = A.indices()  # [nnz], [nnz]

        # Get coordinates
        pts_gauss = gaussians_xyz[gaussian_ids]       # [nnz, 3]
        pts_mask  = instance_centroids[mask_ids]     # [nnz, 3]

        # Compute distance and convert to weight
        dists = torch.norm(pts_gauss - pts_mask, dim=1)           # [nnz]
        weights_per_edge = torch.exp(-dists / sigma)              # Gaussian decay

        # Accumulate per Gaussian
        weights = torch.zeros(gaussians_xyz.shape[0], device=device)
        counts  = torch.zeros_like(weights)
        weights.index_add_(0, gaussian_ids, weights_per_edge)
        counts.index_add_(0, gaussian_ids, torch.ones_like(weights_per_edge))

        # Avoid division by zero
        counts = torch.clamp(counts, min=1.0)
        weights = weights / counts

        return weights



