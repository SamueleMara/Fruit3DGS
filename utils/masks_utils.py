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

# ---------------- 3D -> Mask Mapping ---------------- #
def compute_full_point_to_mask_instance_mapping(points3D, images, mask_dir, downsample=1, save_path=None, log=print):
    """
    Compute full mapping from 3D points to mask instances, including all mask info, in one pass.

    Returns:
        mask_instances: dict[frame_name][midx] -> {centroid, bbox, area, mask_path}
        point_to_masks: dict[pid] -> list of (frame_name, midx)
        mask_to_points: dict[(frame_name, midx)] -> list of pids
    """
    mask_dir = Path(mask_dir)
    mask_instances = {}
    mask_to_points = defaultdict(list)
    point_to_masks = defaultdict(list)

    # 1. Precompute mask instances info
    mask_files = sorted(mask_dir.glob("*_instance_*.png"))
    for mask_path in tqdm(mask_files, desc="[mask_utils] Computing mask instances"):
        frame_name, midx_str = mask_path.stem.rsplit("_instance_", 1)
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
            "centroid": (cx, cy),
            "bbox": (xmin, ymin, xmax, ymax),
            "area": area,
            "mask_path": str(mask_path)  # <-- REQUIRED for GPU propagation
        }

    log(f"[INFO] Precomputed mask instances for {len(mask_instances)} frames")

    # 2. Assign 3D points to mask instances
    for pid, point_data in tqdm(points3D.items(), desc="[mask_utils] Mapping points to mask instances"):
        if not hasattr(point_data, 'image_ids') or not hasattr(point_data, 'point2D_idxs'):
            continue

        for img_id, pt2d_idx in zip(point_data.image_ids, point_data.point2D_idxs):
            if img_id not in images:
                continue

            frame_name = Path(images[img_id].name).stem
            xys = np.array(getattr(images[img_id], "xys", []))
            if len(xys) == 0 or pt2d_idx >= len(xys):
                continue
            u, v = xys[pt2d_idx]

            # Check which mask instance contains this point
            frame_masks = mask_instances.get(frame_name, {})
            for midx, props in frame_masks.items():
                xmin, ymin, xmax, ymax = props['bbox']
                if xmin <= u <= xmax and ymin <= v <= ymax:
                    mask_to_points[(frame_name, midx)].append(pid)
                    point_to_masks[pid].append((frame_name, midx))
                    break  # assign to first mask containing point

    log(f"[INFO] Mapped {len(points3D)} points to {len(mask_to_points)} mask instances")

    # Optional: save to JSON in convenient format
    if save_path:
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        serializable = {
            "mask_instances": mask_instances,
            "point_to_masks": {str(pid): [(f, midx) for f, midx in masks] for pid, masks in point_to_masks.items()},
            "mask_to_points": {f"{f}_{midx}": pids for (f, midx), pids in mask_to_points.items()}
        }
        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2)
        log(f"[INFO] Full mapping JSON saved to {save_path}")

    return mask_instances, point_to_masks, mask_to_points

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


def propagate_all_masks_gpu(points3D, images, mask_instances, gs_cameras, device="cuda"):
    """
    Propagate mask-instance assignments from all source frames to all other frames
    using Gaussian Splatting cameras (GS cameras).

    Args:
        points3D: dict of COLMAP Point3D objects {pid: Point3D}
        images: dict of COLMAP Image objects {image_id: Image}
        mask_instances: dict {frame_name: {instance_id: props}}
        gs_cameras: dict {frame_name: GS Camera object}
        device: str, torch device

    Returns:
        mapping: dict {(src_frame, src_mask_id): [(dst_frame, dst_mask_id), ...]}
    """
    device = torch.device(device)
    mapping = defaultdict(list)

    # --- Preload masks as GPU tensors ---
    mask_tensors = {}
    mask_sizes = {}
    for frame_name, instances in mask_instances.items():
        mask_tensors[frame_name] = {}
        for mask_id, props in instances.items():
            mask_img = cv2.imread(props["mask_path"], cv2.IMREAD_GRAYSCALE)
            mask_tensors[frame_name][mask_id] = torch.tensor(mask_img, dtype=torch.bool, device=device)
            mask_sizes[frame_name] = mask_img.shape

    # --- Stack 3D points ---
    pids = []
    X = []
    for pid, p in points3D.items():
        pids.append(pid)
        X.append(np.append(p.xyz, 1.0))
    X = torch.tensor(np.stack(X), dtype=torch.float32, device=device).T  # 4 x N

    # --- Precompute full projection matrices for all images ---
    full_proj_matrices = {}
    image_sizes = {}
    for img in images.values():
        frame_name = Path(img.name).stem
        cam = gs_cameras[frame_name]
        world2view = cam.world_view_transform.to(device)
        proj = cam.projection_matrix.clone().detach().to(device)
        full_proj_matrices[frame_name] = proj @ world2view
        image_sizes[frame_name] = (cam.image_width, cam.image_height)

    # --- For each source frame, find points inside source masks ---
    points_in_masks = defaultdict(list)
    for src_frame, masks in mask_tensors.items():
        proj = full_proj_matrices[src_frame] @ X
        u = (proj[0] / proj[2]).long()
        v = (proj[1] / proj[2]).long()
        valid = proj[2] > 0
        H, W = mask_sizes[src_frame]

        for mask_id, mask in masks.items():
            inside = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            indices = torch.nonzero(inside, as_tuple=False).flatten()
            mask_vals = mask[v[indices], u[indices]]
            for idx, val in zip(indices, mask_vals):
                if val:
                    points_in_masks[(src_frame, mask_id)].append(pids[idx.item()])

    # --- Project these points to all neighbor frames ---
    for src_key, pid_list in points_in_masks.items():
        if len(pid_list) == 0:
            continue
        pid_indices = [pids.index(pid) for pid in pid_list]
        X_sel = X[:, pid_indices]

        for dst_frame, masks in mask_tensors.items():
            proj = full_proj_matrices[dst_frame] @ X_sel
            u = (proj[0] / proj[2]).long()
            v = (proj[1] / proj[2]).long()
            valid = proj[2] > 0
            H, W = mask_sizes[dst_frame]

            for dst_mask_id, mask in masks.items():
                inside = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                indices = torch.nonzero(inside, as_tuple=False).flatten()
                mask_vals = mask[v[indices], u[indices]]
                for idx, val in zip(indices, mask_vals):
                    if val:
                        pid = pid_list[idx.item()]
                        mapping[src_key].append((dst_frame, dst_mask_id))

    return mapping
