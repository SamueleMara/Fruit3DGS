#!/usr/bin/env python3
"""
Compare OBB centers (JSON) vs GT positions (TXT), match closest pairs even if counts differ,
compute distance errors, visualize in Open3D (with optional overlapped point cloud),
and save CSV next to the JSON.
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class OBB:
    cluster_id: int
    center: np.ndarray


@dataclass(frozen=True)
class Match:
    gt_i: int
    obb_i: int
    gt_pos: np.ndarray
    obb_pos: np.ndarray
    dist: float


# ================================
# Loading
# ================================
def load_obbs(json_path: Path) -> List[OBB]:
    data = json.loads(json_path.read_text())
    boxes = data["boxes"] if isinstance(data, dict) and "boxes" in data else data

    obbs = []
    for i, b in enumerate(boxes):
        cid = int(b.get("cluster_id", i))
        c = np.array(b["center"], dtype=np.float64).reshape(3)
        obbs.append(OBB(cluster_id=cid, center=c))
    return obbs


def load_gt_positions(gt_txt_path: Path) -> Tuple[List[int], np.ndarray]:
    idx_re = re.compile(r"^\s*idx\s+(\d+)\s*$")
    pos_re = re.compile(
        r"^\s*position:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
        r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
        r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
    )

    indices = []
    positions = []
    current_idx = None
    have_position = False

    for line in gt_txt_path.read_text().splitlines():
        m_idx = idx_re.match(line)
        if m_idx:
            current_idx = int(m_idx.group(1))
            have_position = False
            continue

        m_pos = pos_re.match(line)
        if m_pos and current_idx is not None and not have_position:
            x, y, z = map(float, m_pos.groups())
            indices.append(current_idx)
            positions.append(np.array([x, y, z], dtype=np.float64))
            have_position = True

    return indices, np.stack(positions, axis=0)


# ================================
# Matching
# ================================
def pairwise_dist_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    diff = A[:, None, :] - B[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def match_hungarian(gt_positions, obb_centers):
    from scipy.optimize import linear_sum_assignment
    D = pairwise_dist_matrix(gt_positions, obb_centers)
    row_ind, col_ind = linear_sum_assignment(D)
    return list(zip(row_ind.tolist(), col_ind.tolist()))


def match_greedy_global(gt_positions, obb_centers):
    D = pairwise_dist_matrix(gt_positions, obb_centers)
    N, M = D.shape
    K = min(N, M)

    pairs = [(i, j, float(D[i, j])) for i in range(N) for j in range(M)]
    pairs.sort(key=lambda t: t[2])

    used_gt, used_obb = set(), set()
    assignment = []

    for i, j, _ in pairs:
        if i in used_gt or j in used_obb:
            continue
        assignment.append((i, j))
        used_gt.add(i)
        used_obb.add(j)
        if len(assignment) == K:
            break

    return assignment


# ================================
# Stats
# ================================
def summarize(distances: np.ndarray) -> str:
    if distances.size == 0:
        return "No matches."

    rmse = float(np.sqrt(np.mean(distances ** 2)))
    mean = float(np.mean(distances))
    median = float(np.median(distances))
    mx = float(np.max(distances))
    mn = float(np.min(distances))
    return (
        "=== Error statistics ===\n"
        f"count={len(distances)}\n"
        f"mean={mean:.6f} m\n"
        f"median={median:.6f} m\n"
        f"rmse={rmse:.6f} m\n"
        f"min={mn:.6f} m\n"
        f"max={mx:.6f} m\n"
    )


# ================================
# Visualization
# ================================
def hsv_to_rgb(h, s, v):
    i = int(h * 6.0)
    f = (h * 6.0) - i
    i %= 6
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    return (v, p, q)


def color_for_cluster(cid):
    golden = 0.618033988749895
    h = (cid * golden) % 1.0
    return hsv_to_rgb(h, 0.85, 0.95)


def visualize_open3d(matches, obbs, overlay_ply, marker_size, show_lines):
    import open3d as o3d

    geoms = []

    # ---- Load overlay point cloud if provided ----
    if overlay_ply is not None:
        import open3d as o3d
        from plyfile import PlyData

        ply = PlyData.read(str(overlay_ply))
        v = ply["vertex"].data

        if "cluster_id" in v.dtype.names:
            xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
            cids = v["cluster_id"].astype(np.int32)

            colors = np.zeros((xyz.shape[0], 3), dtype=np.float64)

            for cid in np.unique(cids):
                col = np.array(color_for_cluster(int(cid)))
                colors[cids == cid] = col

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            geoms.append(pcd)
        else:
            # fallback if no cluster_id present
            pcd = o3d.io.read_point_cloud(str(overlay_ply))
            geoms.append(pcd)

    if not matches:
        print("[WARN] No matches to visualize.")
        return

    all_pts = np.concatenate(
        [np.stack([m.gt_pos for m in matches]),
         np.stack([m.obb_pos for m in matches])],
        axis=0
    )

    diag = float(np.linalg.norm(all_pts.max(0) - all_pts.min(0)))
    if marker_size is None:
        marker_size = max(diag * 0.01, 1e-4)

    # Bigger reference axis
    axis_size = marker_size * 18.0   # increase multiplier as you like
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size))

    for m in matches:
        cluster_id = obbs[m.obb_i].cluster_id
        line_color = color_for_cluster(cluster_id)

        # --- OBB center ring (BIGGER + BLACK) ---
        ring = o3d.geometry.TriangleMesh.create_torus(
            torus_radius=marker_size * 0.9,     # bigger
            tube_radius=marker_size * 0.25
        )
        ring.compute_vertex_normals()
        ring.paint_uniform_color([0.0, 0.0, 0.0])  # black
        ring.translate(m.obb_pos)
        geoms.append(ring)

        # --- GT cube (BIGGER + BLACK) ---
        cube_size = marker_size * 1.4
        cube = o3d.geometry.TriangleMesh.create_box(
            width=cube_size,
            height=cube_size,
            depth=cube_size,
        )
        cube.compute_vertex_normals()
        cube.paint_uniform_color([0.0, 0.0, 0.0])  # black
        cube.translate(m.gt_pos - np.array([0.5, 0.5, 0.5]) * cube_size)
        geoms.append(cube)

        # --- Connecting line (keep cluster color) ---
        if show_lines:
            line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([m.gt_pos, m.obb_pos]),
                lines=o3d.utility.Vector2iVector([[0, 1]])
            )
            line.colors = o3d.utility.Vector3dVector([line_color])
            geoms.append(line)

    o3d.visualization.draw_geometries(
        geoms,
        window_name="OBB vs GT (overlay)",
        width=1280,
        height=720
    )


# ================================
# Main
# ================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obb_json", type=Path, required=True)
    ap.add_argument("--gt_txt", type=Path, required=True)
    ap.add_argument("--overlay_ply", type=Path, default=None,
                    help="Optional point cloud to render underneath.")
    ap.add_argument("--method", choices=["hungarian", "greedy"], default="hungarian")
    ap.add_argument("--out_csv", type=Path, default=None)
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--marker_size", type=float, default=None)
    ap.add_argument("--no_lines", action="store_true")
    args = ap.parse_args()

    obbs = load_obbs(args.obb_json)
    gt_indices, gt_positions = load_gt_positions(args.gt_txt)
    obb_centers = np.stack([o.center for o in obbs], axis=0)

    if args.method == "hungarian":
        try:
            assignment = match_hungarian(gt_positions, obb_centers)
        except:
            assignment = match_greedy_global(gt_positions, obb_centers)
    else:
        assignment = match_greedy_global(gt_positions, obb_centers)

    matches = []
    for gt_i, obb_i in assignment:
        d = float(np.linalg.norm(gt_positions[gt_i] - obb_centers[obb_i]))
        matches.append(Match(gt_i, obb_i,
                             gt_positions[gt_i],
                             obb_centers[obb_i],
                             d))

    distances = np.array([m.dist for m in matches])
    print(summarize(distances))

    if args.viz:
        visualize_open3d(
            matches,
            obbs,
            args.overlay_ply,
            marker_size=args.marker_size,
            show_lines=(not args.no_lines)
        )


if __name__ == "__main__":
    main()






# #!/usr/bin/env python3
# """
# Compare OBB centers (JSON) vs GT positions (TXT), match closest pairs even if counts differ,
# compute distance errors, visualize in Open3D, and save CSV next to the JSON.

# Matching when counts differ:
#   K = min(num_gt, num_obb)
#   - hungarian: optimal one-to-one assignment of size K (SciPy)
#   - greedy: global nearest-pairs, one-to-one, until K pairs

# CSV default path:
#   <json_folder>/<json_stem>_matches.csv
# """

# import argparse
# import csv
# import json
# import re
# from dataclasses import dataclass
# from pathlib import Path
# from typing import List, Tuple, Optional

# import numpy as np


# @dataclass(frozen=True)
# class OBB:
#     cluster_id: int
#     center: np.ndarray  # (3,)


# @dataclass(frozen=True)
# class Match:
#     gt_i: int
#     obb_i: int
#     gt_pos: np.ndarray   # (3,)
#     obb_pos: np.ndarray  # (3,)
#     dist: float          # meters


# def load_obbs(json_path: Path) -> List[OBB]:
#     data = json.loads(json_path.read_text())
#     boxes = data["boxes"] if isinstance(data, dict) and "boxes" in data else data
#     if not isinstance(boxes, list):
#         raise ValueError("Unexpected OBB JSON format (expected dict with 'boxes' list or list).")

#     obbs: List[OBB] = []
#     for i, b in enumerate(boxes):
#         if "center" not in b:
#             raise ValueError("Each box must have a 'center' field.")
#         cid = int(b.get("cluster_id", i))
#         c = np.array(b["center"], dtype=np.float64).reshape(3)
#         obbs.append(OBB(cluster_id=cid, center=c))
#     return obbs


# def load_gt_positions(gt_txt_path: Path) -> Tuple[List[int], np.ndarray]:
#     idx_re = re.compile(r"^\s*idx\s+(\d+)\s*$")
#     pos_re = re.compile(
#         r"^\s*position:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
#         r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
#         r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
#     )

#     indices: List[int] = []
#     positions: List[np.ndarray] = []

#     current_idx: Optional[int] = None
#     have_position_for_current = False

#     for line in gt_txt_path.read_text().splitlines():
#         m_idx = idx_re.match(line)
#         if m_idx:
#             current_idx = int(m_idx.group(1))
#             have_position_for_current = False
#             continue

#         m_pos = pos_re.match(line)
#         if m_pos and current_idx is not None and not have_position_for_current:
#             x, y, z = map(float, m_pos.groups())
#             indices.append(current_idx)
#             positions.append(np.array([x, y, z], dtype=np.float64))
#             have_position_for_current = True

#     if not positions:
#         raise ValueError("No GT positions parsed from GT file.")

#     return indices, np.stack(positions, axis=0)


# def pairwise_dist_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
#     # A: Nx3, B: Mx3 -> NxM
#     diff = A[:, None, :] - B[None, :, :]
#     return np.linalg.norm(diff, axis=-1)


# def match_hungarian(gt_positions: np.ndarray, obb_centers: np.ndarray) -> List[Tuple[int, int]]:
#     """
#     Rectangular assignment supported:
#       returns K = min(N, M) pairs, one-to-one, minimizing total distance.
#     """
#     from scipy.optimize import linear_sum_assignment  # type: ignore

#     D = pairwise_dist_matrix(gt_positions, obb_centers)
#     row_ind, col_ind = linear_sum_assignment(D)
#     # row_ind/col_ind length is min(N, M)
#     return list(zip(row_ind.tolist(), col_ind.tolist()))


# def match_greedy_global(gt_positions: np.ndarray, obb_centers: np.ndarray) -> List[Tuple[int, int]]:
#     """
#     Rectangular greedy matching:
#       - compute all pair distances
#       - sort by distance
#       - pick smallest available pair without reusing GT or OBB
#       - stop after K=min(N,M) pairs
#     """
#     D = pairwise_dist_matrix(gt_positions, obb_centers)
#     N, M = D.shape
#     K = min(N, M)

#     pairs = [(i, j, float(D[i, j])) for i in range(N) for j in range(M)]
#     pairs.sort(key=lambda t: t[2])

#     used_gt = set()
#     used_obb = set()
#     assignment: List[Tuple[int, int]] = []

#     for i, j, _d in pairs:
#         if i in used_gt or j in used_obb:
#             continue
#         assignment.append((i, j))
#         used_gt.add(i)
#         used_obb.add(j)
#         if len(assignment) == K:
#             break

#     return assignment


# def summarize(distances: np.ndarray) -> str:
#     if distances.size == 0:
#         return "No matches."

#     rmse = float(np.sqrt(np.mean(distances ** 2)))
#     mean = float(np.mean(distances))
#     median = float(np.median(distances))
#     mx = float(np.max(distances))
#     mn = float(np.min(distances))
#     return (
#         "=== Error statistics (OBB center vs GT position) ===\n"
#         f"count={len(distances)}\n"
#         f"mean={mean:.6f} m  ({mean*100:.3f} cm)\n"
#         f"median={median:.6f} m ({median*100:.3f} cm)\n"
#         f"rmse={rmse:.6f} m  ({rmse*100:.3f} cm)\n"
#         f"min={mn:.6f} m   ({mn*100:.3f} cm)\n"
#         f"max={mx:.6f} m   ({mx*100:.3f} cm)\n"
#         f"<=5mm:  {(distances <= 0.005).mean()*100:.1f}%\n"
#         f"<=1cm:  {(distances <= 0.010).mean()*100:.1f}%\n"
#         f"<=2cm:  {(distances <= 0.020).mean()*100:.1f}%\n"
#     )


# def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
#     i = int(h * 6.0)
#     f = (h * 6.0) - i
#     i %= 6
#     p = v * (1.0 - s)
#     q = v * (1.0 - f * s)
#     t = v * (1.0 - (1.0 - f) * s)
#     if i == 0: return (v, t, p)
#     if i == 1: return (q, v, p)
#     if i == 2: return (p, v, t)
#     if i == 3: return (p, q, v)
#     if i == 4: return (t, p, v)
#     return (v, p, q)


# def color_for_pair(i: int) -> Tuple[float, float, float]:
#     golden = 0.618033988749895
#     h = (i * golden) % 1.0
#     return hsv_to_rgb(h, 0.85, 0.95)


# def resolve_csv_path(obb_json_path: Path, out_csv_arg: Optional[Path]) -> Path:
#     base_dir = obb_json_path.parent
#     if out_csv_arg is None:
#         return base_dir / (obb_json_path.stem + "_matches.csv")
#     out_csv_arg = Path(out_csv_arg)
#     if out_csv_arg.is_absolute():
#         return out_csv_arg
#     return base_dir / out_csv_arg


# def visualize_open3d(matches: List[Match], marker_size: Optional[float], show_lines: bool) -> None:
#     import open3d as o3d

#     if not matches:
#         print("[WARN] No matches to visualize.")
#         return

#     gt_pts = np.stack([m.gt_pos for m in matches], axis=0)
#     obb_pts = np.stack([m.obb_pos for m in matches], axis=0)
#     all_pts = np.concatenate([gt_pts, obb_pts], axis=0)

#     bb_min = all_pts.min(axis=0)
#     bb_max = all_pts.max(axis=0)
#     diag = float(np.linalg.norm(bb_max - bb_min))

#     if marker_size is None:
#         marker_size = max(diag * 0.01, 1e-4)

#     geoms: List[o3d.geometry.Geometry] = []
#     geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=marker_size * 3.0))

#     for k, m in enumerate(matches):
#         col = color_for_pair(k)

#         # OBB center: ring (torus)
#         ring = o3d.geometry.TriangleMesh.create_torus(
#             torus_radius=marker_size * 0.55,
#             tube_radius=marker_size * 0.15,
#             radial_resolution=24,
#             tubular_resolution=18,
#         )
#         ring.compute_vertex_normals()
#         ring.paint_uniform_color(col)
#         ring.translate(m.obb_pos)
#         geoms.append(ring)

#         # GT: cube
#         cube = o3d.geometry.TriangleMesh.create_box(
#             width=marker_size * 0.9,
#             height=marker_size * 0.9,
#             depth=marker_size * 0.9,
#         )
#         cube.compute_vertex_normals()
#         cube.paint_uniform_color(col)
#         cube.translate(m.gt_pos - np.array([0.45, 0.45, 0.45]) * (marker_size * 0.9))
#         geoms.append(cube)

#         if show_lines:
#             ls = o3d.geometry.LineSet(
#                 points=o3d.utility.Vector3dVector([m.gt_pos, m.obb_pos]),
#                 lines=o3d.utility.Vector2iVector([[0, 1]]),
#             )
#             ls.colors = o3d.utility.Vector3dVector([col])
#             geoms.append(ls)

#     o3d.visualization.draw_geometries(
#         geoms,
#         window_name="OBB centers (rings) vs GT (cubes) - matched pairs colored",
#         width=1280,
#         height=720,
#     )


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--obb_json", type=Path, required=True)
#     ap.add_argument("--gt_txt", type=Path, required=True)
#     ap.add_argument("--method", choices=["hungarian", "greedy"], default="hungarian")
#     ap.add_argument("--out_csv", type=Path, default=None,
#                     help="If relative, it is saved next to the JSON. Default: <json_stem>_matches.csv")
#     ap.add_argument("--viz", action="store_true")
#     ap.add_argument("--marker_size", type=float, default=None)
#     ap.add_argument("--no_lines", action="store_true")
#     args = ap.parse_args()

#     obbs = load_obbs(args.obb_json)
#     gt_indices, gt_positions = load_gt_positions(args.gt_txt)

#     obb_centers = np.stack([o.center for o in obbs], axis=0)
#     K = min(gt_positions.shape[0], len(obbs))

#     print("=== Counts ===")
#     print(f"GT positions: {gt_positions.shape[0]}")
#     print(f"OBBs:         {len(obbs)}")
#     if gt_positions.shape[0] != len(obbs):
#         print(f"[WARN] Count mismatch -> keeping K=min(...)={K} matched pairs. Extras will be ignored.")

#     # Compute assignment
#     if args.method == "hungarian":
#         try:
#             assignment = match_hungarian(gt_positions, obb_centers)
#         except Exception as e:
#             print(f"[WARN] Hungarian failed ({e}); falling back to greedy.")
#             assignment = match_greedy_global(gt_positions, obb_centers)
#     else:
#         assignment = match_greedy_global(gt_positions, obb_centers)

#     # Ensure we keep exactly K (should already be true)
#     if len(assignment) > K:
#         assignment = assignment[:K]

#     # Build matches
#     matches: List[Match] = []
#     for gt_i, obb_i in assignment:
#         d = float(np.linalg.norm(gt_positions[gt_i] - obb_centers[obb_i]))
#         matches.append(Match(gt_i=gt_i, obb_i=obb_i, gt_pos=gt_positions[gt_i], obb_pos=obb_centers[obb_i], dist=d))

#     # Report unmatched
#     used_gt = set([m.gt_i for m in matches])
#     used_obb = set([m.obb_i for m in matches])
#     gt_unmatched = [i for i in range(gt_positions.shape[0]) if i not in used_gt]
#     obb_unmatched = [j for j in range(len(obbs)) if j not in used_obb]
#     print(f"Matched pairs: {len(matches)}")
#     print(f"Unmatched GT:  {len(gt_unmatched)}")
#     print(f"Unmatched OBB: {len(obb_unmatched)}")

#     # Stats
#     distances = np.array([m.dist for m in matches], dtype=np.float64)
#     print()
#     print(summarize(distances))

#     # Save CSV next to JSON
#     out_csv_path = resolve_csv_path(args.obb_json, args.out_csv)
#     out_csv_path.parent.mkdir(parents=True, exist_ok=True)
#     with out_csv_path.open("w", newline="") as f:
#         w = csv.writer(f)
#         w.writerow([
#             "gt_row", "gt_idx",
#             "obb_index", "obb_cluster_id",
#             "gt_x", "gt_y", "gt_z",
#             "obb_x", "obb_y", "obb_z",
#             "error_m", "error_cm",
#         ])
#         for m in sorted(matches, key=lambda mm: mm.gt_i):
#             w.writerow([
#                 m.gt_i,
#                 gt_indices[m.gt_i] if m.gt_i < len(gt_indices) else m.gt_i,
#                 m.obb_i,
#                 obbs[m.obb_i].cluster_id,
#                 float(m.gt_pos[0]), float(m.gt_pos[1]), float(m.gt_pos[2]),
#                 float(m.obb_pos[0]), float(m.obb_pos[1]), float(m.obb_pos[2]),
#                 float(m.dist),
#                 float(m.dist * 100.0),
#             ])
#     print(f"\nWrote CSV: {out_csv_path}")

#     # Viz
#     if args.viz:
#         visualize_open3d(matches, marker_size=args.marker_size, show_lines=(not args.no_lines))


# if __name__ == "__main__":
#     main()