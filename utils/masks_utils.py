# masks_utils.py
import json
import re
from pathlib import Path
from functools import lru_cache
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import itertools
import csv
from collections import defaultdict, deque, Counter
import cv2
import torch
import glob
import imageio
import psutil
import random

from .graphics_utils import getWorld2View2, getProjectionMatrix
from .read_write_model import qvec2rotmat

# ---------------- Mask Loading & Caching ---------------- #

@lru_cache(maxsize=1024)
def load_mask_cpu(mask_path, downsample=1):
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8)
    if downsample > 1:
        mask = mask[::downsample, ::downsample]
    return mask

def list_masks_for_frame(mask_dir, frame_name, log=print):
    mask_dir = Path(mask_dir)
    mask_files = sorted(
        mask_dir.glob(f"{frame_name}_instance_*.png"),
        key=lambda p: int(re.search(r'_instance_(\d+)', p.stem).group(1))
    )
    if not mask_files:
        log(f"[WARN] No mask instances found for frame {frame_name} in {mask_dir}")
    return mask_files

def compute_mask_instances_json(mask_dir, downsample=1, save_path=None, log=print):
    mask_dir = Path(mask_dir)
    mask_files = sorted(mask_dir.glob("*_instance_*.png"))
    mask_instances = {}

    for mask_path in tqdm(mask_files, desc="[mask_utils] Extracting mask_instances"):
        stem = mask_path.stem
        frame_name, midx_str = stem.rsplit("_instance_", 1)
        midx = int(midx_str)

        mask = load_mask_cpu(str(mask_path), downsample)
        if mask.sum() == 0:
            continue

        ys, xs = np.nonzero(mask)
        cx, cy = float(xs.mean()), float(ys.mean())
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        area = int(mask.sum())

        if frame_name not in mask_instances:
            mask_instances[frame_name] = {}
        mask_instances[frame_name][midx] = {
            "centroid": [cx, cy],
            "bbox": [xmin, ymin, xmax, ymax],
            "area": area
        }

    if save_path:
        with open(save_path, "w") as f:
            json.dump(mask_instances, f, indent=2)
        log(f"[INFO] Saved mask_instances JSON to {save_path}")

    return mask_instances

def load_or_create_mask_instances(mask_dir, downsample=1, log=print):
    mask_dir = Path(mask_dir)
    json_path = mask_dir / "mask_instances.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            mask_instances = json.load(f)
        log(f"[INFO] Loaded mask_instances from {json_path}")
    else:
        log("[INFO] No mask_instances.json found, computing...")
        mask_instances = compute_mask_instances_json(mask_dir, downsample, save_path=json_path, log=log)
    return mask_instances

def get_mask_info(mask_instances, frame_name, midx):
    """
    Retrieve mask instance info given frame_name and midx.
    Returns dict with keys: centroid, bbox, area, or None if missing.
    """
    frame_data = mask_instances.get(frame_name, {})
    # Try integer midx first, fallback to string key
    return frame_data.get(midx) or frame_data.get(str(midx))

def mask_centroid_and_bbox(mask_instances, frame_name, midx):
    info = get_mask_info(mask_instances, frame_name, midx)
    if info is None:
        return None, None
    cx, cy = info["centroid"]
    xmin, ymin, xmax, ymax = info["bbox"]
    return (cx, cy), (xmin, ymin, xmax, ymax)

def mask_area(mask_instances, frame_name, midx):
    info = get_mask_info(mask_instances, frame_name, midx)
    return info["area"] if info else 0

# # ---------------- 3D -> Mask Mapping ---------------- #
# def compute_full_point_to_mask_instance_mapping(points3D, images, mask_dir, downsample=1, save_path=None, device="cpu", log=print, batch_size=100000):
#     """
#     Compute full mapping from 3D points to mask instances, including all mask info, in one pass.

#     Returns:
#         mask_instances: dict[frame_name][midx] -> {centroid, bbox, area, mask_path}
#         point_to_masks: dict[pid] -> list of (frame_name, midx)
#         mask_to_points: dict[(frame_name, midx)] -> list of pids
#     """

#     device = torch.device(device)
#     mask_dir = Path(mask_dir)
#     mask_instances = {}
#     sparse_masks = {}
#     mask_shapes = {}

#     # --- Step 1: Collect mask files with extension-insensitive check ---
#     exts = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
#     exts += [e.upper() for e in exts]
#     mask_files = []
#     for f in mask_dir.iterdir():
#         for ext in exts:
#             if f.suffix.lower() == f".{ext.lower()}" and "_instance_" in f.stem:
#                 mask_files.append(f)
#                 break
#     mask_files = sorted(mask_files)

#     # --- Step 2: Precompute sparse masks and bounding boxes ---
#     for mask_path in tqdm(mask_files, desc="[mask_utils] Precomputing mask instances"):
#         frame_name, midx_str = mask_path.stem.rsplit("_instance_", 1)
#         midx = int(midx_str)

#         mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
#         if downsample != 1:
#             H, W = mask_img.shape
#             mask_img = cv2.resize(mask_img, (W // downsample, H // downsample), interpolation=cv2.INTER_NEAREST)

#         mask_bool = torch.tensor(mask_img > 0, dtype=torch.bool)
#         if mask_bool.sum() == 0:
#             continue

#         ys, xs = torch.nonzero(mask_bool, as_tuple=True)
#         cx, cy = float(xs.float().mean()), float(ys.float().mean())
#         xmin, xmax = int(xs.min()), int(xs.max())
#         ymin, ymax = int(ys.min()), int(ys.max())
#         area = int(mask_bool.sum())

#         mask_instances.setdefault(frame_name, {})[midx] = {
#             "centroid": (cx, cy),
#             "bbox": (xmin, ymin, xmax, ymax),
#             "area": area,
#             "mask_path": str(mask_path)
#         }

#         sparse_masks.setdefault(frame_name, {})[midx] = (xs + mask_bool.shape[1]*ys).to(device)
#         mask_shapes[frame_name] = mask_bool.shape

#     log(f"[INFO] Precomputed mask instances for {len(mask_instances)} frames")

#     # --- Step 3: Organize points per frame ---
#     point_img_map = defaultdict(list)
#     pid_map = {}
#     for pid, p in points3D.items():
#         if not hasattr(p, 'image_ids') or not hasattr(p, 'point2D_idxs'):
#             continue
#         for img_id, pt2d_idx in zip(p.image_ids, p.point2D_idxs):
#             if img_id not in images:
#                 continue
#             frame_name = Path(images[img_id].name).stem
#             xys = np.array(getattr(images[img_id], "xys", []), dtype=float)
#             if len(xys) == 0 or pt2d_idx >= len(xys):
#                 continue
#             u, v = xys[pt2d_idx]
#             u_ds, v_ds = int(u / downsample), int(v / downsample)
#             point_img_map[frame_name].append((pid, u_ds, v_ds))
#             pid_map[pid] = None

#     point_to_masks = defaultdict(list)
#     mask_to_points = defaultdict(list)

#     # --- Step 4: Mega-batch mapping per frame ---
#     for frame_name, pts in tqdm(point_img_map.items(),
#                                 desc="[mask_utils] Mapping points to masks (mega-batch)"):

#         if frame_name not in sparse_masks:
#             continue

#         H, W = mask_shapes[frame_name]

#         # Flattened uv for all points
#         uvs = torch.tensor([u + W * v for _, u, v in pts], device=device)
#         pids = torch.tensor([pid for pid, _, _ in pts], device=device)

#         # Collect all mask indices for this frame
#         mask_ids = list(sparse_masks[frame_name].keys())
#         all_mask_indices = torch.cat([sparse_masks[frame_name][mid] for mid in mask_ids])

#         # Offset table to map flattened position → mask id
#         mask_offsets = torch.cumsum(
#             torch.tensor([0] + [len(sparse_masks[frame_name][mid]) for mid in mask_ids[:-1]],
#                          device=device),
#             dim=0
#         )

#         # Process in batches to avoid OOM
#         for i in range(0, len(uvs), batch_size):
#             batch_uvs = uvs[i:i + batch_size]
#             batch_pids = pids[i:i + batch_size]

#             # membership test using broadcasting
#             mask_matrix = batch_uvs[:, None] == all_mask_indices[None, :]
#             hits = mask_matrix.any(dim=1)

#             if not hits.any():
#                 continue

#             hit_idx = torch.nonzero(hits, as_tuple=False).flatten()

#             for pid_i in hit_idx:
#                 pid_val = batch_pids[pid_i].item()

#                 # All matches for this point
#                 hit_positions = torch.nonzero(
#                     batch_uvs[pid_i] == all_mask_indices,
#                     as_tuple=False
#                 ).flatten()

#                 if hit_positions.numel() == 0:
#                     continue  # shouldn't happen but safe

#                 # use ONLY the first match (all belong to same mask)
#                 mask_pos = hit_positions[0].item()

#                 # map mask_pos → mask index
#                 midx_idx = torch.searchsorted(mask_offsets, mask_pos, right=True) - 1
#                 midx = mask_ids[midx_idx]

#                 point_to_masks[pid_val].append((frame_name, midx))
#                 mask_to_points[(frame_name, midx)].append(pid_val)

#     log(f"[INFO] Mapped {len(points3D)} points to {len(mask_to_points)} mask instances")

#     # --- Step 5: Optional save ---
#     if save_path:
#         save_path = Path(save_path)
#         os.makedirs(save_path.parent, exist_ok=True)
#         serializable = {
#             "mask_instances": mask_instances,
#             "point_to_masks": {str(pid): [(f, midx) for f, midx in masks] for pid, masks in point_to_masks.items()},
#             "mask_to_points": {f"{f}_{midx}": pids for (f, midx), pids in mask_to_points.items()}
#         }
#         with open(save_path, "w") as f:
#             json.dump(serializable, f, indent=2)
#         log(f"[INFO] Full mapping JSON saved to {save_path}")

#     return mask_instances, point_to_masks, mask_to_points

def load_full_mask_point_mapping(json_path, log=print):
    with open(json_path, "r") as f:
        data = json.load(f)

    mask_instances = data.get("mask_instances", {})
    point_to_masks = defaultdict(list)
    for pid_str, masks in data.get("point_to_masks", {}).items():
        point_to_masks[int(pid_str)] = [(f, int(midx)) for f, midx in masks]

    mask_to_points = defaultdict(list)
    for key, pids in data.get("mask_to_points", {}).items():
        frame, midx = key.rsplit("_", 1)
        mask_to_points[(frame, int(midx))] = [int(pid) for pid in pids]

    log(f"[INFO] Loaded full mask-point mapping from {json_path}")
    return mask_instances, point_to_masks, mask_to_points


def parse_mapping_from_file(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    # mask_instances: nested dict {frame: {midx: {...}}}
    mask_instances = data.get("mask_instances", {})
    # point_to_masks: keys are strings of pid -> list of [frame, midx]
    raw_pt2m = data.get("point_to_masks", {})
    point_to_masks = {int(pid): [(f, int(midx)) for f, midx in masks] for pid, masks in raw_pt2m.items()}
    # mask_to_points: keys are "frame_midx" strings -> list of pids
    raw_m2p = data.get("mask_to_points", {})
    mask_to_points = {}
    for key, pids in raw_m2p.items():
        # key may look like "frame_00030_4" or "frame_00030_4" (frame can have underscores)
        frame, midx = key.rsplit("_", 1)
        mask_to_points[(frame, int(midx))] = [int(p) for p in pids]
    return mask_instances, point_to_masks, mask_to_points

def analyze_full_mapping(json_path=None, mask_instances=None, point_to_masks=None, mask_to_points=None, total_points=None, top_n=10, log=print):
    """
    Compute statistics and print a short report summarizing how points and mask_instances relate.
    Provide either json_path or the three mapping dicts.
    total_points: optional, total #3D points in COLMAP (to count unseen points)
    """
    if json_path is not None:
        mask_instances, point_to_masks, mask_to_points = _parse_mapping_from_file(json_path)
    assert mask_instances is not None and point_to_masks is not None and mask_to_points is not None

    # Statistics: points
    pts_with_masks = len(point_to_masks)
    pts_mask_counts = [len(v) for v in point_to_masks.values()] if pts_with_masks else []
    pts_multi_mask = sum(1 for c in pts_mask_counts if c > 1)
    pts_single_mask = sum(1 for c in pts_mask_counts if c == 1)

    if total_points is not None:
        pts_unseen = total_points - pts_with_masks
    else:
        pts_unseen = None

    # Statistics: masks
    total_mask_instances = sum(len(v) for v in mask_instances.values())
    mask_points_counts = []
    for (f, midx), props in mask_instances.items():  # iterate frames then midx keys inconsistent types -> ignore here
        pass
    # Build counts from mask_to_points (this only includes masks that saw points)
    mask_sizes = {k: len(v) for k, v in mask_to_points.items()}
    masks_with_points = len(mask_sizes)
    masks_without_points = total_mask_instances - masks_with_points

    # Summaries
    report = {}
    report['total_mask_instances'] = total_mask_instances
    report['masks_with_points'] = masks_with_points
    report['masks_without_points'] = masks_without_points
    report['points_with_masks'] = pts_with_masks
    report['points_seen_by_multiple_masks'] = pts_multi_mask
    report['points_seen_by_single_mask'] = pts_single_mask
    report['points_unseen'] = pts_unseen

    if pts_mask_counts:
        report['pts_mask_counts_mean'] = float(np.mean(pts_mask_counts))
        report['pts_mask_counts_median'] = float(np.median(pts_mask_counts))
        report['pts_mask_counts_std'] = float(np.std(pts_mask_counts))
        report['pts_mask_counts_p90'] = float(np.percentile(pts_mask_counts, 90))
    else:
        report.update({'pts_mask_counts_mean':0,'pts_mask_counts_median':0,'pts_mask_counts_std':0,'pts_mask_counts_p90':0})

    if mask_sizes:
        mask_size_vals = list(mask_sizes.values())
        report['mask_points_mean'] = float(np.mean(mask_size_vals))
        report['mask_points_median'] = float(np.median(mask_size_vals))
        report['mask_points_std'] = float(np.std(mask_size_vals))
        report['mask_points_p90'] = float(np.percentile(mask_size_vals, 90))
    else:
        report.update({'mask_points_mean':0,'mask_points_median':0,'mask_points_std':0,'mask_points_p90':0})

    # Top examples
    report['top_points_by_mask_count'] = sorted(((pid, len(v)) for pid, v in point_to_masks.items()), key=lambda x: -x[1])[:top_n]
    report['top_masks_by_point_count'] = sorted(((k, len(v)) for k, v in mask_to_points.items()), key=lambda x: -x[1])[:top_n]

    # Print readable summary
    log("=== Full mapping analysis ===")
    log(f"Mask instances total (from mask_instances.json): {total_mask_instances}")
    log(f"Masks with points: {masks_with_points}    Masks without points: {masks_without_points}")
    log(f"3D points with mask assignments: {pts_with_masks}    Points seen by >1 mask: {pts_multi_mask}")
    if pts_unseen is not None:
        log(f"3D points not seen by any mask: {pts_unseen}")
    log(f"Points -> masks: mean={report['pts_mask_counts_mean']:.2f}, median={report['pts_mask_counts_median']:.2f}, p90={report['pts_mask_counts_p90']:.2f}")
    log(f"Masks -> points: mean={report['mask_points_mean']:.2f}, median={report['mask_points_median']:.2f}, p90={report['mask_points_p90']:.2f}")
    log("Top points by number of mask instances (pid, #masks):")
    for pid, c in report['top_points_by_mask_count']:
        log(f"  {pid}: {c}")
    log("Top mask instances by number of points ((frame,midx), #points):")
    for (frame, midx), c in report['top_masks_by_point_count']:
        log(f"  {(frame, midx)}: {c}")

    return report

def compute_mask_overlaps(point_to_masks, mask_to_points, min_shared=1, top_k=200, log=print):
    """
    Efficiently compute overlapping counts between mask pairs using point->mask lists.
    Returns list of (maskA, maskB, shared_count, jaccard, sizeA, sizeB) sorted by shared_count desc.
    mask keys are tuples (frame, midx).
    """
    # Build map of mask -> size
    mask_size = {mask: len(pids) for mask, pids in mask_to_points.items()}

    # use co-occurrence counting: for each point, increment counter for all combinations of masks seeing that point
    overlap_counts = defaultdict(int)
    for pid, masks in point_to_masks.items():
        # masks is list of [frame, midx] -> convert to tuple keys
        mask_keys = [tuple(m) for m in masks]
        if len(mask_keys) < 2:
            continue
        for a, b in itertools.combinations(sorted(mask_keys), 2):
            overlap_counts[(a, b)] += 1

    # compute jaccard
    results = []
    for (a, b), shared in overlap_counts.items():
        sizeA = mask_size.get(a, 0)
        sizeB = mask_size.get(b, 0)
        union = sizeA + sizeB - shared if (sizeA + sizeB - shared) > 0 else 1
        jaccard = shared / union
        if shared >= min_shared:
            results.append((a, b, shared, jaccard, sizeA, sizeB))

    results = sorted(results, key=lambda x: (-x[2], -x[3]))  # sort by shared_count desc then jaccard
    log(f"[INFO] Computed overlaps for {len(results)} mask-pairs (min_shared={min_shared})")
    return results[:top_k]

# -------------------------
# Merge masks by Jaccard
# -------------------------
def merge_masks_by_jaccard(point_to_masks, mask_to_points, jaccard_threshold=0.5, min_shared=1):
    """
    Merge mask instances based on Jaccard similarity.

    Returns:
        merged_groups : list[list]
    """
    overlaps = compute_mask_overlaps(point_to_masks, mask_to_points, min_shared=min_shared, top_k=10**9, log=lambda *a, **k: None)

    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, shared, jaccard, sA, sB in overlaps:
        if jaccard >= jaccard_threshold:
            union(a, b)

    groups = defaultdict(list)
    for m in mask_to_points.keys():
        groups[find(m)].append(m)

    merged_groups = [sorted(g) for g in groups.values() if len(g) > 1]
    return merged_groups

def bipartite_connected_components(point_to_masks, mask_to_points, log=print):
    """
    Compute connected components of bipartite graph (mask nodes and point nodes).
    Returns list of components each as {'masks': set(...), 'points': set(...)}.
    """
    # Build adjacency: nodes as 'p:{pid}' and 'm:frame_midx'
    adj = defaultdict(set)
    for pid, masks in point_to_masks.items():
        pid_node = f"p:{pid}"
        for f, midx in masks:
            m_node = f"m:{f}_{midx}"
            adj[pid_node].add(m_node)
            adj[m_node].add(pid_node)

    visited = set()
    components = []
    for node in adj.keys():
        if node in visited:
            continue
        q = deque([node])
        comp_nodes = set()
        while q:
            n = q.popleft()
            if n in visited:
                continue
            visited.add(n)
            comp_nodes.add(n)
            for nb in adj[n]:
                if nb not in visited:
                    q.append(nb)
        masks = {tuple(n.split("m:")[1].rsplit("_", 1)) if n.startswith("m:") else None for n in comp_nodes}
        # better parsing:
        comp_masks = set()
        comp_points = set()
        for n in comp_nodes:
            if n.startswith("m:"):
                rest = n[2:]
                frame, midx = rest.rsplit("_", 1)
                comp_masks.add((frame, int(midx)))
            elif n.startswith("p:"):
                comp_points.add(int(n[2:]))
        components.append({'masks': comp_masks, 'points': comp_points})
    log(f"[INFO] Found {len(components)} bipartite connected components")
    return components

def recommend_merge_parameter(point_to_masks, mask_to_points, GT=None, log=print):
    """
    Heuristic recommendation:
      - compute distribution of jaccard values and shared counts,
      - identify elbow (percentiles) and recommend candidate thresholds to try.
    If GT is provided, it also outputs closest thresholds that make #objects ~= GT.
    Returns a dict with candidate thresholds and simple guidance.
    """
    overlaps = compute_mask_overlaps(point_to_masks, mask_to_points, min_shared=1, top_k=10**6, log=log)
    if not overlaps:
        log("[WARN] No mask overlaps found")
        return {}

    jaccards = [j for (_, _, _, j, _, _) in overlaps]
    shareds = [s for (_, _, s, _, _, _) in overlaps]

   
    p50_j = float(np.percentile(jaccards, 50))
    p75_j = float(np.percentile(jaccards, 75))
    p90_j = float(np.percentile(jaccards, 90))
    p95_j = float(np.percentile(jaccards, 95))
    p50_s = int(np.percentile(shareds, 50))
    p75_s = int(np.percentile(shareds, 75))

    candidates = {
        'jaccard_candidates': [p75_j, p90_j, p95_j],
        'shared_count_candidates': [max(1, p50_s), p75_s]
    }
    log("Recommended Jaccard thresholds (try these): " + ", ".join(f"{x:.3f}" for x in candidates['jaccard_candidates']))
    log("Recommended shared-point counts (try these): " + ", ".join(str(x) for x in candidates['shared_count_candidates']))

    # quick mapping from threshold -> resulting number of merged groups (coarse)
    # (try few jaccard thresholds and compute resulting merged groups count)
    summary = {}
    for thr in candidates['jaccard_candidates']:
        groups = mask_merge_candidates_by_jaccard(point_to_masks, mask_to_points, jaccard_threshold=thr, min_shared=1, log=lambda *a, **k: None)
        # If we merge masks in each group into one mask and then recompute number of connected components by points->merged_mask,
        # the number of "objects" will be roughly: #components of merged bipartite graph. For speed, approximate by:
        merged_mask_count = len(mask_to_points) - sum(len(g)-1 for g in groups)  # naive approximate
        summary[f"jaccard_{thr:.3f}"] = {'groups': len(groups), 'approx_masks_after_merge': merged_mask_count}
    candidates['summary'] = summary
    return candidates


def plot_mask_instance_points(points3D, images, mask_instances, mask_dir, output_dir="debug_masks"):
    """
    For each mask instance in each frame:
        - Load the binary mask from mask_dir
        - Project all COLMAP 2D points on that frame
        - Highlight the points falling inside each mask instance
        - Save a debug visualization

    Args:
        points3D: COLMAP 3D points dictionary
        images: COLMAP image dictionary (contains xys, point2D_idxs)
        mask_instances: dict {frame_name: {instance_id: {...}}}
        mask_dir: directory containing files "*_instance_XXX.png"
        output_dir: directory where debug images will be saved
    """

    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # print(f"[DEBUG] Saving mask-instance visualizations in: {output_dir}")

    # ----------------------------------------------------
    # 1. Build mapping: frame_name → list of (pid, u, v)
    # ----------------------------------------------------
    frame_to_points = {}

    for pid, p in points3D.items():
        if not hasattr(p, 'image_ids'):
            continue

        for img_id, pt2d_idx in zip(p.image_ids, p.point2D_idxs):

            if img_id not in images:
                continue
            img = images[img_id]

            xys = np.array(getattr(img, "xys", []))
            if len(xys) == 0 or pt2d_idx >= len(xys):
                continue

            u, v = xys[pt2d_idx]
            frame_name = Path(img.name).stem

            frame_to_points.setdefault(frame_name, []).append((pid, int(u), int(v)))

    # ----------------------------------------------------
    # 2. Plot for each mask instance
    # ----------------------------------------------------
    for frame_name, inst_dict in mask_instances.items():

        for inst_id, props in inst_dict.items():

            # Reconstruct mask filename: {frame}_instance_{id}.png
            mask_path = mask_dir / f"{frame_name}_instance_{inst_id}.png"

            if not mask_path.exists():
                print(f"[WARN] Missing mask file: {mask_path}")
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[WARN] Failed to read mask: {mask_path}")
                continue

            h, w = mask.shape

            # green mask visualization
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            vis[:, :, 1] = mask

            # Add COLMAP points for this frame
            pts = frame_to_points.get(frame_name, [])
            for pid, u, v in pts:
                if 0 <= u < w and 0 <= v < h:
                    if mask[v, u] > 0:
                        vis = cv2.circle(vis, (u, v), 3, (0, 0, 255), -1)

            # Save image
            out_path = output_dir / f"{frame_name}_instance_{inst_id}_debug.png"
            cv2.imwrite(str(out_path), vis)

            print(f"[OK] Saved: {out_path}")

# def compute_and_propagate_masks(points3D, images, mask_dir, gs_cameras,
#                                 downsample=1, batch_size=100000,
#                                 device="cpu", save_path=None, log=print,
#                                 subset_frames=None):
#     """
#     Full pipeline with caching:
#       1) Check if JSON with previous computation exists in mask_dir → load and return
#       2) Otherwise, compute masks, point_to_masks, mask_to_points, mapping
#       3) Save JSON automatically in mask_dir

#     Returns:
#         mask_instances, point_to_masks, mask_to_points, mapping
#     """
#     device = torch.device(device)
#     mask_dir = Path(mask_dir)
    
#     if save_path is None:
#         json_path = mask_dir / "mask_mapping.json"
#     else:
#         json_path = Path(save_path)

#     # ----------------- 0) Load cached JSON -----------------
#     if json_path.exists():
#         log(f"[INFO] Found cached JSON at {json_path}, loading...")
#         with open(json_path, "r") as f:
#             data = json.load(f)
#         mask_instances = data.get("mask_instances", {})
#         point_to_masks = {int(k): v for k, v in data.get("point_to_masks", {}).items()}
#         mask_to_points = {}
#         for k, v in data.get("mask_to_points", {}).items():
#             f_name, m_id = k.rsplit("_", 1)
#             mask_to_points[(f_name, int(m_id))] = v
#         mapping = {}
#         for k, v in data.get("mapping", {}).items():
#             f_name, m_id = k.rsplit("_", 1)
#             mapping[(f_name, int(m_id))] = v
#         log("[INFO] Loaded cached masks and mapping.")
#         return mask_instances, point_to_masks, mask_to_points, mapping

#     # ----------------- 1) Load masks -----------------
#     mask_instances = {}
#     frame_masks = {}
#     frame_mids = {}
#     mask_shapes = {}

#     exts = {"png","jpg","jpeg","bmp","tif","tiff"}
#     exts |= {e.upper() for e in exts}
#     mask_files = [f for f in mask_dir.iterdir() if "_instance_" in f.stem and f.suffix[1:] in exts]
#     if len(mask_files) == 0:
#         log("[WARN] No mask files found")
#         return {}, defaultdict(list), defaultdict(list), defaultdict(list)

#     frame_to_list = defaultdict(list)
#     for f in mask_files:
#         try:
#             frame_name, midx_str = f.stem.rsplit("_instance_", 1)
#             midx = int(midx_str)
#         except:
#             continue
#         frame_to_list[frame_name].append((midx, f))

#     # Limit subset if needed
#     frame_names = list(frame_to_list.keys())
#     if subset_frames is not None:
#         if isinstance(subset_frames, int):
#             frame_names = random.sample(frame_names, min(subset_frames, len(frame_names)))
#         elif isinstance(subset_frames, str):
#             frame_names = [subset_frames]
#         else:
#             frame_names = [f for f in frame_names if f in subset_frames]

#     frame_to_list = {k: v for k, v in frame_to_list.items() if k in frame_names}

#     # Load masks into CPU tensors (bool)
#     for frame_name, mids_paths in tqdm(frame_to_list.items(), desc="[mask_utils] Loading masks"):
#         imgs = []
#         mids_order = []
#         H = W = None
#         for midx, fpath in mids_paths:
#             img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
#             if img is None: 
#                 continue
#             if H is None:
#                 H, W = img.shape
#             else:
#                 img = cv2.resize(img, (W,H), interpolation=cv2.INTER_NEAREST)
#             imgs.append(img)
#             mids_order.append(midx)

#             ys, xs = np.nonzero(img>0)
#             if ys.size>0:
#                 mask_instances.setdefault(frame_name, {})[midx] = {
#                     "centroid": (float(xs.mean()), float(ys.mean())),
#                     "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
#                     "area": int(ys.size),
#                     "mask_path": str(fpath)
#                 }

#         if len(imgs)==0:
#             continue

#         mask_stack = torch.from_numpy(np.stack(imgs,0)>0)  # (M,H,W) bool
#         frame_masks[frame_name] = mask_stack
#         frame_mids[frame_name] = mids_order
#         mask_shapes[frame_name] = (H,W)

#     # ----------------- 2) Build point→image map -----------------
#     point_img_map = defaultdict(list)
#     pids = list(points3D.keys())
#     for pid, p in points3D.items():
#         if not hasattr(p,"image_ids") or not hasattr(p,"point2D_idxs"):
#             continue
#         for img_id, pt2d_idx in zip(p.image_ids, p.point2D_idxs):
#             if img_id not in images:
#                 continue
#             img = images[img_id]
#             frame_name = Path(img.name).stem
#             xys = getattr(img,"xys", None)
#             if xys is None or pt2d_idx>=len(xys):
#                 continue
#             u,v = xys[pt2d_idx]
#             # Full resolution → scale=1
#             u_scaled = int(round(u))
#             v_scaled = int(round(v))
#             H,W = mask_shapes[frame_name]
#             # clamp to mask bounds
#             u_scaled = min(max(u_scaled,0),W-1)
#             v_scaled = min(max(v_scaled,0),H-1)
#             point_img_map[frame_name].append((pid,u_scaled,v_scaled))

#     # ----------------- 3) GPU gather: point→mask hits -----------------
#     point_to_masks = defaultdict(list)
#     mask_to_points = defaultdict(list)

#     for frame_name, pts in tqdm(point_img_map.items(), desc="[mask_utils] GPU gather mapping"):
#         if frame_name not in frame_masks: 
#             continue
#         mask_stack = frame_masks[frame_name].to(device)
#         mids_order = frame_mids[frame_name]
#         H,W = mask_shapes[frame_name]

#         pids_np = np.array([p for p,_,_ in pts],dtype=np.int64)
#         us_np = np.array([u for _,u,_ in pts],dtype=np.int64)
#         vs_np = np.array([v for _,_,v in pts],dtype=np.int64)
#         N = len(pids_np)

#         for start in range(0,N,batch_size):
#             end = min(N,start+batch_size)
#             u_batch = torch.from_numpy(us_np[start:end]).to(device)
#             v_batch = torch.from_numpy(vs_np[start:end]).to(device)
#             p_batch = pids_np[start:end]

#             hits = mask_stack[:, v_batch, u_batch]  # (M,B)
#             nz = torch.nonzero(hits, as_tuple=False)
#             if nz.numel()==0: 
#                 continue
#             for mask_idx, batch_idx in nz.cpu().numpy():
#                 pid_val = int(p_batch[batch_idx])
#                 mid_val = mids_order[mask_idx]
#                 point_to_masks[pid_val].append((frame_name, mid_val))
#                 mask_to_points[(frame_name, mid_val)].append(pid_val)

#         del mask_stack
#         torch.cuda.empty_cache()

#     log(f"[INFO] Points mapped: {len(point_to_masks)}, masks: {len(mask_to_points)}")

#     # ----------------- 4) Propagate points to other frames -----------------
#     pid_to_idx = {pid:i for i,pid in enumerate(pids)}
#     X_all = torch.tensor(np.stack([np.append(points3D[pid].xyz,1.0) for pid in pids]),dtype=torch.float32,device=device).T  # 4xN

#     full_proj = {}
#     for img in images.values():
#         frame = Path(img.name).stem
#         if frame not in gs_cameras:
#             continue
#         cam = gs_cameras[frame]
#         full_proj[frame] = (cam.projection_matrix.to(device) @ cam.world_view_transform.to(device))

#     points_in_masks = defaultdict(list)
#     frame_point_idx_map = defaultdict(list)
#     for pid, p in points3D.items():
#         if not hasattr(p,"image_ids"): continue
#         for img_id in p.image_ids:
#             if img_id not in images: continue
#             frame = Path(images[img_id].name).stem
#             frame_point_idx_map[frame].append(pid_to_idx[pid])

#     mapping = defaultdict(list)
#     for frame_name, mask_stack_cpu in tqdm(frame_masks.items(), desc="[propagate] mapping"):
#         if frame_name not in full_proj:
#             continue
#         mask_stack_gpu = mask_stack_cpu.to(device)
#         H,W = mask_shapes[frame_name]
#         idx_list = frame_point_idx_map.get(frame_name, [])
#         if len(idx_list)==0: 
#             continue
#         idx_arr = np.array(idx_list,dtype=np.int64)
#         for start in range(0,len(idx_arr),batch_size):
#             chunk_idx = idx_arr[start:start+batch_size].tolist()
#             X_sel = X_all[:, chunk_idx]
#             proj = full_proj[frame_name] @ X_sel
#             z = proj[2]
#             u_t = torch.clamp((proj[0]/proj[2]).long(),0,W-1)
#             v_t = torch.clamp((proj[1]/proj[2]).long(),0,H-1)
#             valid = z>0
#             inb = valid & (u_t>=0) & (u_t<W) & (v_t>=0) & (v_t<H)
#             if not inb.any(): 
#                 continue
#             u_valid = u_t[inb]
#             v_valid = v_t[inb]
#             sel_idx = np.array(chunk_idx)[inb.cpu().numpy()]
#             hits = mask_stack_gpu[:, v_valid, u_valid]
#             nz = torch.nonzero(hits, as_tuple=False)
#             for mask_idx, local_idx in nz.cpu().numpy():
#                 pid_val = pids[sel_idx[local_idx]]
#                 mid_val = frame_mids[frame_name][mask_idx]
#                 points_in_masks[(frame_name, mid_val)].append(pid_val)
#         del mask_stack_gpu
#         torch.cuda.empty_cache()

#     # ----------------- 5) Propagate to all destination frames -----------------
#     src_items = list(points_in_masks.items())
#     for (src_frame, src_mid), pid_list in tqdm(src_items, desc="[propagate] to dest frames"):
#         if len(pid_list)==0: 
#             continue
#         indices = np.array([pid_to_idx[pid] for pid in pid_list],dtype=np.int64)
#         for start in range(0,len(indices),batch_size):
#             chunk = indices[start:start+batch_size].tolist()
#             X_sel = X_all[:, chunk]
#             for dst_frame, mask_stack_cpu in frame_masks.items():
#                 if dst_frame not in full_proj: continue
#                 H,W = mask_shapes[dst_frame]
#                 mask_stack_gpu = mask_stack_cpu.to(device)
#                 proj = full_proj[dst_frame] @ X_sel
#                 u_t = torch.clamp((proj[0]/proj[2]).long(),0,W-1)
#                 v_t = torch.clamp((proj[1]/proj[2]).long(),0,H-1)
#                 valid = proj[2]>0
#                 inb = valid & (u_t>=0) & (u_t<W) & (v_t>=0) & (v_t<H)
#                 if not inb.any(): 
#                     del mask_stack_gpu
#                     continue
#                 u_valid = u_t[inb]
#                 v_valid = v_t[inb]
#                 sel_idx = np.array(chunk)[inb.cpu().numpy()]
#                 hits = mask_stack_gpu[:, v_valid, u_valid]
#                 nz = torch.nonzero(hits, as_tuple=False)
#                 for mask_idx, local_idx in nz.cpu().numpy():
#                     pid_val = pids[sel_idx[local_idx]]
#                     mid_val = frame_mids[dst_frame][mask_idx]
#                     mapping[(src_frame, src_mid)].append((dst_frame, mid_val))
#                 del mask_stack_gpu
#                 torch.cuda.empty_cache()

#     # ----------------- 6) Save JSON -----------------
#     os.makedirs(json_path.parent, exist_ok=True)
#     serializable = {
#         "mask_instances": mask_instances,
#         "point_to_masks": {str(pid): lst for pid,lst in point_to_masks.items()},
#         "mask_to_points": {f"{f}_{m}": pids for (f,m),pids in mask_to_points.items()},
#         "mapping": {f"{f}_{m}": dsts for (f,m),dsts in mapping.items()}
#     }
#     with open(json_path, "w") as f:
#         json.dump(serializable,f,indent=2)
#     log(f"[INFO] Saved JSON → {json_path}")

#     return mask_instances, point_to_masks, mask_to_points, mapping

def compute_and_propagate_masks(points3D, images, mask_dir, gs_cameras,
                                point_to_pixels=None, downsample=1, batch_size=100000,
                                device="cpu", save_path=None, log=print,
                                subset_frames=None):
    """
    Vectorized & OOM-safe mask mapping using precomputed point_to_pixels.

    Args:
        points3D: dict of COLMAP points
        images: dict of Image objects with `xys`
        mask_dir: directory containing instance masks
        point_to_pixels: optional precomputed mapping from build_point_pixel_mapping()
        downsample: unused (placeholder)
        batch_size: GPU batch size for mapping
        device: "cuda" or "cpu"
        save_path: optional JSON path to save masks/mapping
        subset_frames: optional subset of frames to process

    Returns:
        mask_instances, point_to_masks, mask_to_points, mapping
    """
    device = torch.device(device)
    mask_dir = Path(mask_dir)
    
    if save_path is None:
        json_path = mask_dir / "mask_mapping.json"
    else:
        json_path = Path(save_path)

    # ----------------- 0) Load cached JSON -----------------
    if json_path.exists():
        log(f"[INFO] Found cached JSON at {json_path}, loading...")
        with open(json_path, "r") as f:
            data = json.load(f)
        mask_instances = data.get("mask_instances", {})
        point_to_masks = {int(k): v for k, v in data.get("point_to_masks", {}).items()}
        mask_to_points = {}
        for k, v in data.get("mask_to_points", {}).items():
            f_name, m_id = k.rsplit("_", 1)
            mask_to_points[(f_name, int(m_id))] = v
        mapping = {}
        for k, v in data.get("mapping", {}).items():
            f_name, m_id = k.rsplit("_", 1)
            mapping[(f_name, int(m_id))] = v
        log("[INFO] Loaded cached masks and mapping.")
        return mask_instances, point_to_masks, mask_to_points, mapping

    # ----------------- 1) Load masks -----------------
    mask_instances = {}
    frame_masks = {}
    frame_mids = {}
    mask_shapes = {}

    exts = {"png","jpg","jpeg","bmp","tif","tiff"}
    exts |= {e.upper() for e in exts}
    mask_files = [f for f in mask_dir.iterdir() if "_instance_" in f.stem and f.suffix[1:] in exts]
    if len(mask_files) == 0:
        log("[WARN] No mask files found")
        return {}, defaultdict(list), defaultdict(list), defaultdict(list)

    frame_to_list = defaultdict(list)
    for f in mask_files:
        try:
            frame_name, midx_str = f.stem.rsplit("_instance_", 1)
            midx = int(midx_str)
        except:
            continue
        frame_to_list[frame_name].append((midx, f))

    # Limit subset if needed
    frame_names = list(frame_to_list.keys())
    if subset_frames is not None:
        if isinstance(subset_frames, int):
            frame_names = random.sample(frame_names, min(subset_frames, len(frame_names)))
        elif isinstance(subset_frames, str):
            frame_names = [subset_frames]
        else:
            frame_names = [f for f in frame_names if f in subset_frames]
    frame_to_list = {k: v for k, v in frame_to_list.items() if k in frame_names}

    # Load masks into CPU tensors (bool)
    for frame_name, mids_paths in tqdm(frame_to_list.items(), desc="[mask_utils] Loading masks"):
        imgs = []
        mids_order = []
        H = W = None
        for midx, fpath in mids_paths:
            img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
            if img is None: 
                continue
            if H is None:
                H, W = img.shape
            else:
                img = cv2.resize(img, (W,H), interpolation=cv2.INTER_NEAREST)
            imgs.append(img)
            mids_order.append(midx)

            ys, xs = np.nonzero(img>0)
            if ys.size>0:
                mask_instances.setdefault(frame_name, {})[midx] = {
                    "centroid": (float(xs.mean()), float(ys.mean())),
                    "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                    "area": int(ys.size),
                    "mask_path": str(fpath)
                }

        if len(imgs)==0:
            continue

        mask_stack = torch.from_numpy(np.stack(imgs,0)>0)  # (M,H,W) bool
        frame_masks[frame_name] = mask_stack
        frame_mids[frame_name] = mids_order
        mask_shapes[frame_name] = (H,W)

    # ----------------- 2) Build point→image map using point_to_pixels -----------------
    if point_to_pixels is None:
        raise RuntimeError("point_to_pixels must be provided or precomputed in Scene")


    point_img_map = defaultdict(list)
    for pid, obs_list in point_to_pixels.items():
        for obs in obs_list:
            frame_name = Path(obs["image_name"]).stem
            if frame_name not in mask_shapes:  # skip frames without masks
                continue
            u,v = obs["xy"]
            H,W = mask_shapes[frame_name]
            u_scaled = min(max(int(round(u)),0), W-1)
            v_scaled = min(max(int(round(v)),0), H-1)
            point_img_map[frame_name].append((pid, u_scaled, v_scaled))

    # ----------------- 3) GPU gather: point→mask hits -----------------
    point_to_masks = defaultdict(list)
    mask_to_points = defaultdict(list)

    for frame_name, pts in tqdm(point_img_map.items(), desc="[mask_utils] GPU gather mapping"):
        if frame_name not in frame_masks: 
            continue
        mask_stack = frame_masks[frame_name].to(device)
        mids_order = frame_mids[frame_name]
        H,W = mask_shapes[frame_name]

        pids_np = np.array([p for p,_,_ in pts],dtype=np.int64)
        us_np = np.array([u for _,u,_ in pts],dtype=np.int64)
        vs_np = np.array([v for _,_,v in pts],dtype=np.int64)
        N = len(pids_np)

        for start in range(0,N,batch_size):
            end = min(N,start+batch_size)
            u_batch = torch.from_numpy(us_np[start:end]).to(device)
            v_batch = torch.from_numpy(vs_np[start:end]).to(device)
            p_batch = pids_np[start:end]

            hits = mask_stack[:, v_batch, u_batch]  # (M,B)
            nz = torch.nonzero(hits, as_tuple=False)
            if nz.numel()==0: 
                continue
            for mask_idx, batch_idx in nz.cpu().numpy():
                pid_val = int(p_batch[batch_idx])
                mid_val = mids_order[mask_idx]
                point_to_masks[pid_val].append((frame_name, mid_val))
                mask_to_points[(frame_name, mid_val)].append(pid_val)

        del mask_stack
        torch.cuda.empty_cache()

    log(f"[INFO] Points mapped: {len(point_to_masks)}, masks: {len(mask_to_points)}")

    # ----------------- 4) Propagate points to other frames (OOM-safe batches) -----------------
    pid_to_idx = {pid:i for i,pid in enumerate(points3D.keys())}
    pids = list(points3D.keys())
    X_all = torch.tensor(np.stack([np.append(points3D[pid].xyz,1.0) for pid in pids]),dtype=torch.float32,device=device).T  # 4xN

    full_proj = {frame: (cam.projection_matrix.to(device) @ cam.world_view_transform.to(device))
                 for frame, cam in gs_cameras.items()}

    # Frame-wise point indices
    frame_point_idx_map = defaultdict(list)
    for pid, obs_list in point_to_pixels.items():
        for obs in obs_list:
            frame = Path(obs["image_name"]).stem
            frame_point_idx_map[frame].append(pid_to_idx[pid])

    mapping = defaultdict(list)
    points_in_masks = defaultdict(list)

    for frame_name, mask_stack_cpu in tqdm(frame_masks.items(), desc="[propagate] mapping"):
        if frame_name not in full_proj: continue
        mask_stack_gpu = mask_stack_cpu.to(device)
        H,W = mask_shapes[frame_name]
        idx_list = frame_point_idx_map.get(frame_name, [])
        if len(idx_list)==0: 
            continue
        idx_arr = np.array(idx_list,dtype=np.int64)
        for start in range(0,len(idx_arr),batch_size):
            chunk_idx = idx_arr[start:start+batch_size].tolist()
            X_sel = X_all[:, chunk_idx]
            proj = full_proj[frame_name] @ X_sel
            u_t = torch.clamp((proj[0]/proj[2]).long(),0,W-1)
            v_t = torch.clamp((proj[1]/proj[2]).long(),0,H-1)
            valid = proj[2]>0
            inb = valid & (u_t>=0) & (u_t<W) & (v_t>=0) & (v_t<H)
            if not inb.any(): continue
            u_valid = u_t[inb]
            v_valid = v_t[inb]
            sel_idx = np.array(chunk_idx)[inb.cpu().numpy()]
            hits = mask_stack_gpu[:, v_valid, u_valid]
            nz = torch.nonzero(hits, as_tuple=False)
            for mask_idx, local_idx in nz.cpu().numpy():
                pid_val = pids[sel_idx[local_idx]]
                mid_val = frame_mids[frame_name][mask_idx]
                points_in_masks[(frame_name, mid_val)].append(pid_val)
        del mask_stack_gpu
        torch.cuda.empty_cache()

    # ----------------- 5) Propagate to all destination frames -----------------
    src_items = list(points_in_masks.items())
    for (src_frame, src_mid), pid_list in tqdm(src_items, desc="[propagate] to dest frames"):
        if len(pid_list)==0: continue
        indices = np.array([pid_to_idx[pid] for pid in pid_list],dtype=np.int64)
        for start in range(0,len(indices),batch_size):
            chunk = indices[start:start+batch_size].tolist()
            X_sel = X_all[:, chunk]
            for dst_frame, mask_stack_cpu in frame_masks.items():
                if dst_frame not in full_proj: continue
                H,W = mask_shapes[dst_frame]
                mask_stack_gpu = mask_stack_cpu.to(device)
                proj = full_proj[dst_frame] @ X_sel
                u_t = torch.clamp((proj[0]/proj[2]).long(),0,W-1)
                v_t = torch.clamp((proj[1]/proj[2]).long(),0,H-1)
                valid = proj[2]>0
                inb = valid & (u_t>=0) & (u_t<W) & (v_t>=0) & (v_t<H)
                if not inb.any(): 
                    del mask_stack_gpu
                    continue
                u_valid = u_t[inb]
                v_valid = v_t[inb]
                sel_idx = np.array(chunk)[inb.cpu().numpy()]
                hits = mask_stack_gpu[:, v_valid, u_valid]
                nz = torch.nonzero(hits, as_tuple=False)
                for mask_idx, local_idx in nz.cpu().numpy():
                    pid_val = pids[sel_idx[local_idx]]
                    mid_val = frame_mids[dst_frame][mask_idx]
                    mapping[(src_frame, src_mid)].append((dst_frame, mid_val))
                del mask_stack_gpu
                torch.cuda.empty_cache()

    # ----------------- 6) Save JSON -----------------
    os.makedirs(json_path.parent, exist_ok=True)
    serializable = {
        "mask_instances": mask_instances,
        "point_to_masks": {str(pid): lst for pid,lst in point_to_masks.items()},
        "mask_to_points": {f"{f}_{m}": pids for (f,m),pids in mask_to_points.items()},
        "mapping": {f"{f}_{m}": dsts for (f,m),dsts in mapping.items()}
    }
    with open(json_path, "w") as f:
        json.dump(serializable,f,indent=2)
    log(f"[INFO] Saved JSON → {json_path}")

    return mask_instances, point_to_masks, mask_to_points, mapping
