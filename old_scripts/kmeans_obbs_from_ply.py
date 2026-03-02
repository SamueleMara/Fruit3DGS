#!/usr/bin/env python3
"""
Load a PLY point cloud, flip Z (default), run KMeans (default K=3), compute per-cluster 3D OBBs,
and save to a JSON with the requested format.

Example:
  python kmeans_obbs_from_ply.py --ply /path/to/semantic_colormap_cut.ply --k 3 --method pca --viz

Output JSON default:
  <ply_folder>/<ply_stem>_kmeans_obbs.json

Dependencies:
  pip install numpy open3d scikit-learn
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

try:
    import open3d as o3d
except Exception as e:
    raise ImportError("open3d is required: pip install open3d") from e


# -----------------------------
# Quaternion helper (COLMAP-like): [w, x, y, z], enforce w >= 0
# -----------------------------
def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = float(np.trace(R))

    if t > 0.0:
        r = math.sqrt(1.0 + t)
        w = 0.5 * r
        s = 0.5 / r
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            r = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            x = 0.5 * r
            s = 0.5 / r
            y = (R[0, 1] + R[1, 0]) * s
            z = (R[0, 2] + R[2, 0]) * s
            w = (R[2, 1] - R[1, 2]) * s
        elif i == 1:
            r = math.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2])
            y = 0.5 * r
            s = 0.5 / r
            x = (R[0, 1] + R[1, 0]) * s
            z = (R[1, 2] + R[2, 1]) * s
            w = (R[0, 2] - R[2, 0]) * s
        else:
            r = math.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2])
            z = 0.5 * r
            s = 0.5 / r
            x = (R[0, 2] + R[2, 0]) * s
            y = (R[1, 2] + R[2, 1]) * s
            w = (R[1, 0] - R[0, 1]) * s

    q = np.array([w, x, y, z], dtype=np.float64)
    q /= (np.linalg.norm(q) + 1e-12)
    if q[0] < 0:
        q *= -1.0
    return q


# -----------------------------
# Compute + save OBBs JSON
# -----------------------------
def save_cluster_obbs_json(
    xyz: np.ndarray,
    cluster_ids: np.ndarray,
    json_out_path: str,
    *,
    method: str = "pca",          # "pca" or "minimal"
    sort_axes_by_extent: bool = True,
    save_rotation_matrix: bool = False,
) -> dict:
    xyz = np.asarray(xyz, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids).astype(np.int32).reshape(-1)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be (N,3). Got %s." % (xyz.shape,))
    if cluster_ids.shape[0] != xyz.shape[0]:
        raise ValueError("cluster_ids must be (N,). Got %s for xyz %s."
                         % (cluster_ids.shape, xyz.shape))

    unique_clusters = np.unique(cluster_ids)
    boxes = []

    for cid in unique_clusters:
        pts = xyz[cluster_ids == cid]
        if pts.shape[0] == 0:
            continue

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

        if method == "pca":
            obb = pcd.get_oriented_bounding_box()
        elif method == "minimal":
            if hasattr(pcd, "get_minimal_oriented_bounding_box"):
                obb = pcd.get_minimal_oriented_bounding_box()
            else:
                print("[WARN] Open3D build lacks get_minimal_oriented_bounding_box(); using PCA OBB instead.")
                obb = pcd.get_oriented_bounding_box()
        else:
            raise ValueError("method must be 'pca' or 'minimal'.")

        center = np.asarray(obb.center, dtype=np.float64)
        extent = np.asarray(obb.extent, dtype=np.float64)   # full lengths
        R = np.asarray(obb.R, dtype=np.float64)             # 3x3

        if sort_axes_by_extent:
            order = np.argsort(-extent)
            extent = extent[order]
            R = R[:, order]

        if np.linalg.det(R) < 0:
            R[:, 2] *= -1.0

        qvec = rotmat2qvec(R).astype(np.float64)

        box = {
            "cluster_id": int(cid),
            "center": center.tolist(),
            "extent": extent.tolist(),
            "qvec": qvec.tolist(),
        }
        if save_rotation_matrix:
            box["rotation_matrix"] = R.tolist()

        boxes.append(box)

    out = {
        "extent_definition": "full_lengths_along_local_axes",
        "method": method,
        "boxes": boxes,
    }

    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


# -----------------------------
# KMeans (fit on sample, assign all)
# -----------------------------
def _kmeans_fit_centers(x_fit: np.ndarray, k: int, seed: int) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans  # type: ignore
    except Exception as e:
        raise ImportError("scikit-learn is required for KMeans: pip install scikit-learn") from e

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(x_fit)
    return np.asarray(km.cluster_centers_, dtype=np.float64)


def assign_to_centers(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - centers[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    return np.argmin(d2, axis=1).astype(np.int32)


# -----------------------------
# Visualization (optional)
# -----------------------------
def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    i = int(h * 6.0)
    f = (h * 6.0) - i
    i %= 6
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    return (v, p, q)


def _color_for_cluster(cid: int) -> Tuple[float, float, float]:
    golden = 0.618033988749895
    h = (cid * golden) % 1.0
    return _hsv_to_rgb(h, 0.85, 0.95)


def visualize_clusters_and_obbs(xyz: np.ndarray, labels: np.ndarray, method: str) -> None:
    geoms: List[o3d.geometry.Geometry] = []

    colors = np.zeros((xyz.shape[0], 3), dtype=np.float64)
    for cid in np.unique(labels):
        col = np.array(_color_for_cluster(int(cid)), dtype=np.float64)
        colors[labels == cid] = col

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    geoms.append(pcd)

    for cid in np.unique(labels):
        pts = xyz[labels == cid]
        if pts.shape[0] < 10:
            continue
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        if method == "minimal" and hasattr(p, "get_minimal_oriented_bounding_box"):
            obb = p.get_minimal_oriented_bounding_box()
        else:
            obb = p.get_oriented_bounding_box()

        col = _color_for_cluster(int(cid))
        ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
        ls.colors = o3d.utility.Vector3dVector([col for _ in range(len(ls.lines))])
        geoms.append(ls)

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))
    o3d.visualization.draw_geometries(geoms, window_name="KMeans clusters + OBBs", width=1280, height=720)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=Path, required=True, help="Input PLY point cloud.")
    ap.add_argument("--k", type=int, default=3, help="Number of clusters/objects (default 3).")
    ap.add_argument("--method", choices=["pca", "minimal"], default="pca", help="OBB method.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--no_flip_z", action="store_true", help="Disable initial z-flip.")
    ap.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size (0 disables).")
    ap.add_argument("--max_points", type=int, default=200000, help="Max points used for KMeans fitting.")
    ap.add_argument("--sor_nb", type=int, default=0, help="SOR nb_neighbors (0 disables).")
    ap.add_argument("--sor_std", type=float, default=2.0, help="SOR std_ratio.")

    ap.add_argument("--out_json", type=Path, default=None, help="Output JSON path.")
    ap.add_argument("--save_rotation_matrix", action="store_true")
    ap.add_argument("--no_sort_axes", action="store_true", help="Disable sorting OBB axes by extent.")
    ap.add_argument("--viz", action="store_true", help="Visualize clusters + OBBs in Open3D.")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(str(args.ply))
    if pcd.is_empty():
        raise ValueError("Loaded empty point cloud from: %s" % str(args.ply))

    # --- Flip Z FIRST (default) ---
    if not args.no_flip_z:
        pts = np.asarray(pcd.points, dtype=np.float64).copy()
        pts[:, 2] *= -1.0
        pcd.points = o3d.utility.Vector3dVector(pts)

    # Optional: statistical outlier removal
    if args.sor_nb and args.sor_nb > 0:
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=int(args.sor_nb), std_ratio=float(args.sor_std))
        pcd = pcd.select_by_index(ind)

    # Optional: voxel downsample
    if args.voxel and args.voxel > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size=float(args.voxel))

    xyz = np.asarray(pcd.points, dtype=np.float64)
    N = xyz.shape[0]
    if N < args.k:
        raise ValueError("Not enough points (%d) for k=%d" % (N, args.k))

    rng = np.random.RandomState(args.seed)
    if N > args.max_points:
        idx = rng.choice(N, size=int(args.max_points), replace=False)
        xyz_fit = xyz[idx]
    else:
        xyz_fit = xyz

    centers = _kmeans_fit_centers(xyz_fit, args.k, args.seed)
    labels = assign_to_centers(xyz, centers)

    if args.out_json is None:
        out_json = args.ply.parent / (args.ply.stem + "_kmeans_obbs.json")
    else:
        out_json = args.out_json

    save_cluster_obbs_json(
        xyz=xyz,
        cluster_ids=labels,
        json_out_path=str(out_json),
        method=args.method,
        sort_axes_by_extent=(not args.no_sort_axes),
        save_rotation_matrix=args.save_rotation_matrix,
    )

    print("Wrote JSON:", str(out_json))

    if args.viz:
        visualize_clusters_and_obbs(xyz, labels, args.method)


if __name__ == "__main__":
    main()