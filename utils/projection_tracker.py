import os
import numpy as np
import cupy as cp
from pathlib import Path
from collections import defaultdict, deque
from tqdm import tqdm
import re
import json
from itertools import combinations
from collections import deque, defaultdict

from masks_utils import (parse_mapping_from_file, analyze_full_mapping, 
                        compute_mask_overlaps, mask_merge_candidates_by_jaccard, recommend_merge_parameter)

from masks_utils import (
    get_mask_info,
    load_mask_cpu,
    list_masks_for_frame,
    compute_full_point_to_mask_instance_mapping,
    load_full_mask_point_mapping
)
from masks_utils import load_mask_cpu as util_load_mask_cpu

HAVE_CUML = False
try:
    from cuml.cluster import DBSCAN as cuDBSCAN
    HAVE_CUML = True
except Exception:
    HAVE_CUML = False

try:
    from scipy.spatial import cKDTree as KDTree
    HAVE_KDTREE = True
except Exception:
    KDTree = None
    HAVE_KDTREE = False


class ProjectionTracker:
    def __init__(self, colmap_model_dir, mask_dir, dist_thresh=0.2, downsample=1, cache_size=512):
        self.colmap_model_dir = Path(colmap_model_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.dist_thresh = float(dist_thresh)
        self.downsample = int(downsample)
        self.cameras, self.images, self.points3D = {}, {}, {}

        # mapping structures
        self.point_to_masks = defaultdict(list)
        self.mask_to_points = defaultdict(list)

        # results
        self.real_objects = {}
        self.real_object_masks = {}

        # helpers
        self._cache_size = int(cache_size)
        self.current_pipeline = "3D2D"
        self.camera_graph = None
        self.camera_centers = {}
        self._masks_cache = {}
        self._mask_list_cache_size = 1024

        # mask instances cache
        self.mask_instances = None

    def log(self, msg):
        print(f"[ProjectionTracker] {msg}")

    # ---------------- Mask/Instance Utilities ---------------- #
    def load_or_create_mapping(self, output_folder, points3D=None, images=None, downsample=1, log=print):
        """
        Load or compute full mapping JSON:
        - mask_instances
        - point_to_masks
        - mask_to_points
        """
        mapping_path = Path(output_folder) / "full_mask_point_mapping.json"
        if mapping_path.exists():
            self.log(f"[INFO] Loading mapping from {mapping_path}")
            self.mask_instances, self.point_to_masks, self.mask_to_points = load_full_mask_point_mapping(mapping_path, log=log)
        else:
            if points3D is None or images is None:
                raise ValueError("points3D and images must be provided to compute mapping")
            self.log("[INFO] Computing full mapping JSON...")
            result = compute_full_point_to_mask_instance_mapping(
                points3D=points3D,
                images=images,
                mask_dir=self.mask_dir,
                downsample=downsample,
                save_path=mapping_path,
                log=log
            )
            self.mask_instances = result['mask_instances']
            self.point_to_masks = {int(k): v for k, v in result['point_to_masks'].items()}
            self.mask_to_points = {(f, int(midx)): v for f_midx, v in result['mask_to_points'].items()
                                for f, midx in [f_midx.rsplit("_", 1)]}
            self.log(f"[INFO] Full mapping saved to {mapping_path}")
        return mapping_path

    def load_or_create_mask_instance_mapping(self, mask_dir, save_path=None, downsample=1):
        """
        Load existing JSON mapping or compute it once and save for future use.
        """
        save_path = save_path or Path(mask_dir) / "full_point_to_mask_instance.json"
        if save_path.exists():
            with open(save_path, "r") as f:
                data = json.load(f)
            self.mask_instances = data["mask_instances"]
            self.point_to_masks = {int(k): v for k, v in data["point_to_masks"].items()}
            self.mask_to_points = {(f.split("_")[0], int(f.split("_")[1])): v
                                   for f, v in data["mask_to_points"].items()}
            self.log(f"[INFO] Loaded full mask-instance mapping from {save_path}")
        else:
            self.log("[INFO] Computing full mask-instance mapping...")
            data = compute_full_point_to_mask_instance_mapping(
                self.points3D, self.images, mask_dir, downsample=downsample, save_path=save_path, log=self.log
            )
            self.mask_instances = data["mask_instances"]
            self.point_to_masks = {int(k): v for k, v in data["point_to_masks"].items()}
            self.mask_to_points = {(f.split("_")[0], int(f.split("_")[1])): v
                                   for f, v in data["mask_to_points"].items()}
        
        mask_instances, point_to_masks, mask_to_points = parse_mapping_from_file(full_mapping_path)
        report = analyze_full_mapping(mask_instances=mask_instances, point_to_masks=point_to_masks, mask_to_points=mask_to_points, total_points=len(tracker.points3D))
        candidates = recommend_merge_parameter(point_to_masks, mask_to_points, GT=GT)

        return save_path

    def load_or_create_mask_instances(self):
        self.mask_instances = load_or_create_mask_instances(self.mask_dir, downsample=self.downsample, log=self.log)

    def get_mask_info(self, frame_name, midx):
        if self.mask_instances is None:
            self.load_or_create_mask_instances()
        return get_mask_info(self.mask_instances, frame_name, midx)

    def list_masks_for_frame(self, frame_name):
        return list_masks_for_frame(self.mask_dir, frame_name, log=self.log)

    def load_mask_list_for_frame(self, frame_name):
        return load_mask_list_for_frame(self.mask_dir, frame_name, downsample=self.downsample, log=self.log)

    def mask_centroid_and_bbox(self, frame_name, midx):
        return mask_centroid_and_bbox(self.mask_instances, frame_name, midx)

    def load_mask_cpu(self, mask_path, downsample=1):
        return util_load_mask_cpu(mask_path, downsample)    

    def compute_mask_centroids(self):
        self.mask_centroids = {}
        for mask_path in self.mask_dir.glob("*.png"):
            stem = mask_path.stem
            if "_instance_" not in stem:
                continue
            frame_name, midx = stem.rsplit("_instance_", 1)
            midx = int(midx)
            centroid, _ = self.mask_centroid_and_bbox(frame_name, midx)
            if centroid is not None:
                self.mask_centroids[(frame_name, midx)] = centroid

    # ---------------- COLMAP ---------------- #

    def load_colmap(self, read_model):
        self.log(f"Loading COLMAP model from {self.colmap_model_dir}")
        self.cameras, self.images, self.points3D = read_model(self.colmap_model_dir, ext='.txt')
        self.log(f"Loaded {len(self.cameras)} cameras, {len(self.images)} images, {len(self.points3D)} 3D points")
        self.camera_graph = None
        self.camera_centers = {}
        self._masks_cache.clear()

    # ---------------- Mapping 3D->2D ---------------- #

    def map_points_to_masks_streamed(self, downsample=None, progress=True):
        if downsample is None:
            downsample = self.downsample
        self.log("Mapping 3D points to mask instances (streamed)")
        self.point_to_masks, self.mask_to_points = map_points_to_masks_streamed(
            self.images, self.mask_dir, downsample=downsample, log=self.log
        )
   
    # ---------- Camera graph ----------
    @staticmethod
    def _qvec2rotmat(qvec):
        qw, qx, qy, qz = qvec
        n = np.linalg.norm(qvec)
        if n == 0:
            return np.eye(3)
        qw, qx, qy, qz = qvec / n
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ], dtype=np.float32)
        return R

    def build_camera_graph(self, k=5):
        if not self.images:
            self.camera_graph = {}
            self.camera_centers = {}
            return self.camera_graph
        ids, centers, frame_names = [], [], []
        for img_id, img in self.images.items():
            frame = Path(img.name).stem
            ids.append(img_id)
            frame_names.append(frame)
            C = np.zeros(3)
            try:
                R = self._qvec2rotmat(np.array(img.qvec, dtype=float))
                t = np.array(getattr(img, 'tvec', getattr(img, 'translation', np.zeros(3))), dtype=float)
                C = -R.T @ t
            except Exception:
                pass
            self.camera_centers[frame] = C
            centers.append(C)
        centers = np.vstack(centers)
        if HAVE_KDTREE:
            tree = KDTree(centers)
            neighs = tree.query(centers, k=min(k+1, len(centers)))[1]
            graph = {frame_names[i]: [frame_names[j] for j in nbr_idx if j != i] for i, nbr_idx in enumerate(neighs)}
        else:
            n = len(ids)
            graph = {frame_names[i]: frame_names[max(0, i - k//2):i] + frame_names[i+1:min(n, i+1 + k//2)] for i in range(n)}
        self.camera_graph = graph
        self.log(f"Camera graph built with {len(self.camera_graph)} nodes")
        return self.camera_graph


    # ---------- Utility: detection (mask) features ----------
    def mask_centroid_and_bbox(self, frame_name, midx):
        if not hasattr(self, "mask_instances"):
            self.load_or_create_mask_instances()
        info = self.get_mask_info(frame_name, midx)
        if info is None:
            return None, None
        cx, cy = info["centroid"]
        xmin, ymin, xmax, ymax = info["bbox"]
        return (cx, cy), (xmin, ymin, xmax, ymax)


    def compute_mask_centroids(self):
        """
        Precompute centroids for all mask instances and store in self.mask_centroids.
        Format: {(frame_name, midx): (cx, cy)}
        """
        self.mask_centroids = {}
        for mask_path in self.mask_dir.glob("*.png"):
            stem = mask_path.stem
            if "_instance_" not in stem:
                continue
            frame_name, midx = stem.rsplit("_instance_", 1)
            midx = int(midx)

            centroid, _ = self.mask_centroid_and_bbox(frame_name, midx)
            if centroid is not None:
                self.mask_centroids[(frame_name, midx)] = centroid


    def projection_overlap(self, proj_uv, mask):
        """
        Compute IoU between projected 3D points and detection mask.
        Projection is rasterized as points into an image mask.
        """
        h, w = mask.shape
        proj_mask = np.zeros_like(mask, dtype=np.uint8)
        for u, v in proj_uv.astype(int):
            if 0 <= v < h and 0 <= u < w:
                proj_mask[v, u] = 1

        inter = np.logical_and(proj_mask, mask).sum()
        union = np.logical_or(proj_mask, mask).sum()
        iou = inter / union if union > 0 else 0.0
        return iou    

    # ---------- Gather 2D detections (each mask instance treated as detection) ----------
    def gather_2d_detections(self):
        """
        Build a dictionary of detections across all images:
        det_id -> {'frame':frame_name, 'midx':mid, 'pids': set(...), 'centroid': (x,y), 'bbox': (...) }
        """
        detections = {}
        det_idx = 0
        for (frame, midx), pids in self.mask_to_points.items():
            pset = set(int(pid) for pid in pids) if pids else set()
            centroid, bbox = self.mask_centroid_and_bbox(frame, midx)
            detections[det_idx] = {
                'frame': frame,
                'midx': int(midx),
                'pids': pset,
                'centroid': centroid,
                'bbox': bbox
            }
            det_idx += 1
        self.log(f"Collected {len(detections)} 2D detections from masks")
        return detections

    # ---------- Association across neighboring frames ----------
    def associate_detections_across_images(self, detections, iou_threshold=0.05, k_neighbors=6):
        """
        Associate detections across images using overlap of 3D point sets.
        Only compare detections located in neighboring cameras (using camera graph).
        Returns groups as list of lists of detection indices.
        """
        # Build camera graph if missing
        if self.camera_graph is None:
            self.build_camera_graph(k=k_neighbors)

        # index detections by frame for quick lookup
        dets_by_frame = defaultdict(list)
        for det_id, det in detections.items():
            dets_by_frame[det['frame']].append(det_id)

        # union-find for detections
        parent = {d: d for d in detections}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # For each detection, compare with detections in neighbor image frames
        for det_id, det in detections.items():
            frame = det['frame']
            neighbors = self.camera_graph.get(frame, [])
            # also compare within same frame (different masks)
            neighbors_frames = [frame] + neighbors
            for nframe in neighbors_frames:
                for other_id in dets_by_frame.get(nframe, []):
                    if other_id == det_id:
                        continue
                    # compute IoU on 3D pid sets (fast set operations)
                    A = detections[det_id]['pids']
                    B = detections[other_id]['pids']
                    if not A or not B:
                        continue
                    inter = len(A & B)
                    union_size = len(A | B)
                    iou = inter / union_size if union_size > 0 else 0.0
                    if iou >= iou_threshold:
                        union(det_id, other_id)

        # gather groups
        groups = defaultdict(list)
        for det_id in detections:
            root = find(det_id)
            groups[root].append(det_id)

        groups_list = list(groups.values())
        self.log(f"Associated detections into {len(groups_list)} groups (iou_thresh={iou_threshold})")
        return groups_list

    # ---------- Convert detection groups to real objects (3D point clusters) ----------
    def detections_to_real_objects(self, detections, groups):
        """
        Given detection dict and groups (list of lists of det ids), produce real_objects mapping:
            cid -> numpy array of pids
        and real_object_masks mapping
        """
        self.real_objects.clear()
        self.real_object_masks.clear()
        for cid, group in enumerate(groups):
            # union all pid sets
            pid_set = set()
            masks_for_group = []
            for det_id in group:
                det = detections[det_id]
                pid_set.update(det['pids'])
                masks_for_group.append((det['frame'], det['midx']))
            if not pid_set:
                continue
            self.real_objects[cid] = np.array(sorted(pid_set), dtype=np.int32)
            self.real_object_masks[cid] = masks_for_group
        self.log(f"Built {len(self.real_objects)} real objects from detection groups")
        return self.real_objects

    # ---------- Compute starting dt ----------
    def compute_starting_dt(self, fraction=0.2):
        """
        Compute a starting dt for the current pipeline.
        For 3D2D: dt is now the strict Jaccard threshold (1.0 = perfect overlap).
        For 2D3D: keep normalized NN distance in 3D.
        """
        pipeline = self.current_pipeline.upper()

        if pipeline == "2D3D":
            dt_start = 0.0
            self.log(f"[INFO] Starting dt for {pipeline} pipeline: {dt_start:.2f} (normalized NN distance)")
            return dt_start

        elif pipeline == "3D2D":
            # Jaccard threshold: start from perfect overlap
            dt_start = 0
            self.log(f"[INFO] Starting dt for {pipeline} pipeline: {dt_start:.2f} (strict projection-mask overlap)")
            return dt_start

        else:
            self.log(f"[WARN] Unknown pipeline '{pipeline}'; using dt=0")
            return 0.0

    # ---------- expand_mask_point_assignments ----------
    def expand_mask_point_assignments(self, dt=None, adaptive=True):
        """
        Expands mask->point assignments by adding neighboring 3D points within radius dt.
        If adaptive==True, local dt is computed from local nearest neighbor distances.
        """
        dt = 0.0 if dt is None else float(dt)
        pids_all = np.array(list(self.points3D.keys()), dtype=int)
        if pids_all.size == 0:
            return

        coords = np.vstack([self.points3D[int(pid)].xyz for pid in pids_all]).astype(np.float32)

        if HAVE_KDTREE:
            tree = KDTree(coords)
            pid_to_idx = {int(pid): idx for idx, pid in enumerate(pids_all)}
            for mask_key, assigned_pids in list(self.mask_to_points.items()):
                if not assigned_pids:
                    continue
                seed_idxs = [pid_to_idx[int(pid)] for pid in assigned_pids if int(pid) in pid_to_idx]
                if not seed_idxs:
                    continue
                idxs = set()
                for s in seed_idxs:
                    if adaptive and dt > 0:
                        k = min(4, len(coords))
                        dists, _ = tree.query(coords[s:s+1], k=k)
                        local_dt = np.median(dists[0,1:]) if dists.shape[1] > 1 else dt
                    else:
                        local_dt = dt
                    neighbors = tree.query_ball_point(coords[s], r=local_dt)
                    idxs.update(neighbors)
                new_pids = set(pids_all[list(idxs)])
                merged = set(self.mask_to_points[mask_key]) | new_pids
                self.mask_to_points[mask_key] = list(merged)
        else:
            self.log("KDTree not available, skipping expansion")

    # ---------- PROJECTION helpers ----------
    def camera_intrinsics(self, cam):
        """
        Try to extract fx, fy, cx, cy from camera.params.
        Supports typical COLMAP pinhole param layouts.
        """
        params = getattr(cam, 'params', None)
        width = getattr(cam, 'width', None)
        height = getattr(cam, 'height', None)
        if params is None:
            return None
        try:
            if len(params) >= 4:
                fx = float(params[0])
                fy = float(params[1])
                cx = float(params[2])
                cy = float(params[3])
            elif len(params) == 3:
                fx = float(params[0])
                fy = fx
                cx = float(params[1])
                cy = float(params[2])
            elif len(params) == 2:
                fx = float(params[0])
                fy = float(params[0])
                cx = float(params[1])
                cy = float(params[1])
            else:
                return None
            return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'width': width, 'height': height}
        except Exception:
            return None

    def project_point(self, point, img_id):
        """
        Project a 3D point into image coordinates for a given image ID.
        Returns (u, v, in_frame). u,v are floats.
        """
        # image & camera existence
        if img_id not in self.images:
            return None, None, False
        img = self.images[img_id]
        cam_id = getattr(img, 'camera_id', getattr(img, 'camera', None))
        cam = self.cameras.get(cam_id, None) if cam_id is not None else None
        if cam is None:
            # try to infer intrinsics from image attributes
            cam = getattr(img, 'camera', None)
        intr = self.camera_intrinsics(cam) if cam is not None else None
        if intr is None:
            return None, None, False

        # rotation & translation
        if hasattr(img, 'qvec'):
            R = self._qvec2rotmat(np.array(img.qvec, dtype=float))
        else:
            # if no qvec assume identity
            R = np.eye(3, dtype=float)
        t = np.array(getattr(img, 'tvec', getattr(img, 'translation', np.zeros(3))), dtype=float).reshape(3)

        # world -> camera coords
        X = np.array(point.xyz, dtype=float).reshape(3)
        Xc = R @ X + t  # NOTE: depending on how read_model stores tvec, check sign; common COLMAP store uses t such that C = -R^T t; here we assume img.tvec is camera-to-world? If reprojections look flipped, change to R @ X + t or R @ (X - C) etc.

        # If point behind camera (z <= 0) or invalid depth
        if Xc[2] <= 1e-8:
            return None, None, False

        fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
        u = fx * (Xc[0] / Xc[2]) + cx
        v = fy * (Xc[1] / Xc[2]) + cy

        w = intr.get('width', None)
        h = intr.get('height', None)
        in_frame = True
        if w is not None and h is not None:
            if not (0 <= u < w and 0 <= v < h):
                in_frame = False
        return float(u), float(v), bool(in_frame)

    def compute_reproj_error(self, point, img_id, u_pred=None, v_pred=None):
        """
        Compute reprojection displacement (pixel Euclidean distance) for 3D point into img_id.
        If u_pred/v_pred are provided, uses them; else projects internally.
        """
        if u_pred is None or v_pred is None:
            u_pred, v_pred, ok = self.project_point(point, img_id)
            if not ok:
                return float('inf')
        # if image has stored xys for that 3D point, compare to closest observation - otherwise return distance to projected location
        # We'll return distance to projected coords (no ground truth 2D point available here)
        # If image.xys are available and mapping of pid->index exists, one could fetch ground truth
        return 0.0  # placeholder: by default zero because we lack direct 2D ground-truth here

    # ---------- NEW main pipeline: gather real objects via 2D-first association ----------
    def gather_3d2d_objects(self, dt=None):
        """
        Graph-aware 3D2D pipeline with projection-based aggregation.

        - Keep bipartite BFS connected components as initial objects.
        - For each 3D point of each component:
            * project that point into neighbor frames (neighbors defined by camera_graph)
            * find nearest mask instance (by centroid) in the projected frame
            * if pixel-distance <= dt, union the two components (link)
        - At the end, flatten union-find and return merged components.

        dt is expected to be in pixels (not normalized).
        """
        # sanity checks
        if self.points3D is None or not self.point_to_masks:
            self.log("[ERROR] COLMAP points or mappings not available!")
            return {}

        if dt is None:
            dt = self.compute_starting_dt()
            
        # -------------------------------
        # Step 1: initial bipartite BFS (unchanged)
        # -------------------------------
        visited = set()
        cid_counter = 0
        object_points = {}
        object_masks = {}

        def mask_node(frame, midx):
            return ("mask", frame, midx)

        for pid in self.point_to_masks.keys():
            if pid in visited:
                continue
            queue = deque([pid])
            visited.add(pid)
            pts = set()
            masks = []

            while queue:
                node = queue.popleft()
                if isinstance(node, int):  # point
                    pts.add(node)
                    for frame, midx in self.point_to_masks[node]:
                        mnode = mask_node(frame, midx)
                        if mnode not in visited:
                            visited.add(mnode)
                            queue.append(mnode)
                else:  # mask node
                    _, frame, midx = node
                    masks.append((frame, midx))
                    for pid2 in self.mask_to_points.get((frame, midx), []):
                        if pid2 not in visited:
                            visited.add(pid2)
                            queue.append(pid2)

            object_points[cid_counter] = set(pts)
            object_masks[cid_counter] = masks
            cid_counter += 1

        # self.log(f"[DEBUG] Initial bipartite components: {len(object_points)}")

        # -------------------------------
        # Prepare helpers for projection-based linking
        # -------------------------------
        # map each mask -> initial component id
        mask_to_component = {}
        for cid, masks in object_masks.items():
            for m in masks:
                mask_to_component[m] = cid

        # frame -> image ids (there can be multiple images mapping to same frame stem)
        frame_to_img_ids = defaultdict(list)
        for img_id, img in self.images.items():
            frame = Path(img.name).stem
            frame_to_img_ids[frame].append(img_id)

        # ensure camera graph exists
        if self.camera_graph is None:
            self.build_camera_graph()

        # ensure mask centroids are computed
        if not hasattr(self, "mask_centroids") or not self.mask_centroids:
            self.compute_mask_centroids()

        # build per-frame mask lists AND centroids aligned (skip masks without centroids)
        frame_mask_keys = defaultdict(list)
        frame_mask_centroids = defaultdict(list)
        for (frame, midx), pts in self.mask_to_points.items():
            cen = self.mask_centroids.get((frame, midx))
            if cen is None:
                cen, _ = self.mask_centroid_and_bbox(frame, midx)
            if cen is None:
                continue

            # convert single floats to proper [cx, cy]
            if isinstance(cen, float):
                # fallback: just skip if centroid is invalid
                continue
            elif isinstance(cen, (list, tuple)) and len(cen) == 2:
                pass
            else:
                # unexpected format: skip
                continue

            frame_mask_keys[frame].append((frame, midx))
            frame_mask_centroids[frame].append(cen)

        # build KDTree per frame (if available) for fast nearest neighbor search on centroids
        frame_kdtree = {}
        if HAVE_KDTREE:
            for frame, centers in frame_mask_centroids.items():
                if centers:
                    frame_kdtree[frame] = KDTree(np.array(centers, dtype=np.float32))

        # union-find over initial components
        parent = {cid: cid for cid in object_points.keys()}
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
                return True
            return False

        # projection cache: pid -> {frame: (u,v, img_id)}
        if not hasattr(self, "_proj_cache"):
            self._proj_cache = {}

        # -------------------------------
        # Step 2: projection-driven linking
        # -------------------------------
        unions_performed = 0
        total_checks = 0

        # iterate over initial components and their points
        for cid, pset in object_points.items():
            for pid in pset:
                total_checks += 1
                # fetch cached projections for this pid (if any)
                proj_map = self._proj_cache.get(pid)
                if proj_map is None:
                    proj_map = {}
                    # frames that already see this point (from mapping)
                    frames_seen = [f for f, _ in self.point_to_masks.get(pid, [])]
                    # add neighbors of those frames (camera_graph)
                    neighbor_frames = set(frames_seen)
                    for f in frames_seen:
                        neighbor_frames.update(self.camera_graph.get(f, []))

                    # For each neighbor frame, project the 3D point on ONE representative image of that frame
                    for frame in neighbor_frames:
                        img_ids = frame_to_img_ids.get(frame, [])
                        if not img_ids:
                            continue
                        # use the first image id for that frame (representative)
                        img_id = img_ids[0]
                        u, v, ok = self.project_point(self.points3D[int(pid)], img_id)
                        if not ok:
                            continue
                        # store projection by frame (we compare to mask centroids of frame)
                        proj_map[frame] = (float(u), float(v), img_id)
                    # cache it
                    self._proj_cache[pid] = proj_map

                # now, for every projected frame, find nearest mask centroid and check distance <= dt
                for frame, (u, v, img_id) in proj_map.items():
                    masks_in_frame = frame_mask_keys.get(frame)
                    if not masks_in_frame:
                        continue

                    # query KDTree if available
                    if frame in frame_kdtree:
                        projected_pixel = np.array([u, v], dtype=np.float32)
                        dist, idx = frame_kdtree[frame].query(projected_pixel, k=1)  # returns scalars
                        if dist <= dt:
                            nearest_mask = masks_in_frame[idx]
                            other_cid = mask_to_component.get(nearest_mask)
                            if other_cid is not None and union(cid, other_cid):
                                unions_performed += 1

                    else:
                        # brute force nearest
                        centers = frame_mask_centroids.get(frame, [])
                        if not centers:
                            continue
                        min_dist = float('inf')
                        min_idx = -1
                        for i, (cx, cy) in enumerate(centers):
                            d = ((cx - u) ** 2 + (cy - v) ** 2) ** 0.5
                            if d < min_dist:
                                min_dist = d
                                min_idx = i
                        if min_idx >= 0 and min_dist <= dt:
                            nearest_mask = masks_in_frame[min_idx]
                            other_cid = mask_to_component.get(nearest_mask)
                            if other_cid is not None and union(cid, other_cid):
                                unions_performed += 1

        # -------------------------------
        # Step 3: collect merged components
        # -------------------------------
        merged_map = defaultdict(set)
        merged_masks = defaultdict(list)
        for cid, pts in object_points.items():
            root = find(cid)
            merged_map[root].update(pts)
            merged_masks[root].extend(object_masks.get(cid, []))

        # reindex
        new_objects = {}
        new_masks = {}
        for new_cid, (root, pset) in enumerate(merged_map.items()):
            new_objects[new_cid] = np.array(sorted(pset), dtype=np.int32)
            # deduplicate masks and keep as list
            new_masks[new_cid] = list(dict.fromkeys(merged_masks[root]))

        self.real_objects = new_objects
        self.real_object_masks = new_masks

        # self.log(f"[INFO] Gathered {len(self.real_objects)} real objects (dt={dt:.3f}, projection-graph aggregation)")
        # self.log(f"[DEBUG] projection linking: total_checks={total_checks}, unions_performed={unions_performed}")

        return self.real_objects

    # ---------- Unified gather_real_objects ----------
    def gather_real_objects(self, dt=None, pipeline="3D2D", adaptive_dt=True,
                            iou_threshold=0.05, k_neighbors=6):
        """
        Unified entry point:
        - pipeline == "3D2D": optimized 3D->2D BFS propagation
        - pipeline == "2D3D": 2D-first association
        dt is in 'real' units (pixels for 3D2D, world units for 2D3D)
        This function updates self.real_objects and returns them.
        """
        self.current_pipeline = pipeline.upper()

        # Build camera graph once
        if self.camera_graph is None:
            self.build_camera_graph(k=k_neighbors)

        # If 3D2D, use optimized propagation
        if self.current_pipeline.startswith("3D2D"):
            if dt is None:
                dt = self.compute_starting_dt()
            return self.gather_3d2d_objects(dt=dt)

        elif self.current_pipeline == "2D3D":
            # 2D-first association (gather real objects by clustering 3D points that are close in 3D)
            # Implement a simple clustering by spatial proximity in 3D (dt is 3D radius).
            # We'll use KDTree to cluster points that are within dt of each other (single-linkage)
            if dt is None:
                dt = 0.0
            # Build an undirected graph where edge between p_i and p_j if dist <= dt
            pids = np.array(list(self.points3D.keys()), dtype=int)
            if pids.size == 0:
                self.real_objects.clear()
                self.real_object_masks.clear()
                return {}

            coords = np.vstack([self.points3D[int(pid)].xyz for pid in pids]).astype(np.float32)

            if dt <= 0 or not HAVE_KDTREE:
                # fallback: each point is its own object but only include those that have masks
                cid = 0
                for idx, pid in enumerate(pids):
                    masks_for_pid = self.point_to_masks.get(int(pid), [])
                    if not masks_for_pid:
                        continue
                    self.real_objects[cid] = np.array([int(pid)], dtype=np.int32)
                    self.real_object_masks[cid] = masks_for_pid.copy()
                    cid += 1
                self.log(f"Gathered {len(self.real_objects)} real objects using 2D3D (dt={dt})")
                return self.real_objects

            tree = KDTree(coords)
            # query neighbors within dt
            neighbors = tree.query_ball_tree(tree, r=dt)
            # union-find to cluster
            parent = {i: i for i in range(len(pids))}
            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
            for i, nbrs in enumerate(neighbors):
                for j in nbrs:
                    if i == j:
                        continue
                    union(i, j)
            clusters = defaultdict(list)
            for i in range(len(pids)):
                clusters[find(i)].append(pids[i])
            # save
            self.real_objects.clear()
            self.real_object_masks.clear()
            for cid, plist in enumerate(clusters.values()):
                self.real_objects[cid] = np.array(sorted(plist), dtype=np.int32)
                # collect mask instances for cluster (union of point_to_masks)
                masks_list = []
                for pid in plist:
                    masks_list.extend(self.point_to_masks.get(int(pid), []))
                self.real_object_masks[cid] = sorted(set(masks_list))
            self.log(f"Gathered {len(self.real_objects)} real objects using 2D3D (dt={dt})")
            return self.real_objects

        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")

    # ---------- Convert real objects to 2D tracks ----------
    def cluster_to_2d_tracks(self, clusters, dt=None):
        pipeline = getattr(self, "current_pipeline", "3D2D").upper()
        tracks_2d = {}

        if pipeline.startswith("3D2D"):
            # clusters is mapping cid -> pids; we map to masks that contain those pids
            # Build an inverted index: pid -> list of mask_keys
            pid_to_masks = defaultdict(list)
            for mask_key, pts in self.mask_to_points.items():
                for pid in pts:
                    pid_to_masks[int(pid)].append(mask_key)

            # Iterate over clusters
            for cid, pids in clusters.items():
                masks_set = set()
                for pid in pids:
                    masks_set.update(pid_to_masks.get(int(pid), []))

                # Smart ordering: BFS across camera graph starting from a frame present in masks_set
                if masks_set and self.camera_graph:
                    visited_frames = set()
                    ordered_masks = []
                    # choose start frame as the one closest to mean camera center if possible
                    frames_in_set = sorted({m[0] for m in masks_set})
                    start_frame = frames_in_set[0]
                    # find any img_id with this frame
                    start_img_ids = [iid for iid, im in self.images.items() if Path(im.name).stem == start_frame]
                    if start_img_ids:
                        start_img = start_img_ids[0]
                    else:
                        start_img = next(iter(self.images.keys()))

                    queue = deque([start_img])
                    frame_to_masks = defaultdict(list)
                    for frame, midx in masks_set:
                        frame_to_masks[frame].append((frame, midx))

                    while queue:
                        img_id = queue.popleft()
                        frame = Path(self.images[img_id].name).stem
                        if frame in visited_frames:
                            continue
                        visited_frames.add(frame)
                        ordered_masks.extend(frame_to_masks.get(frame, []))
                        for nbr in self.camera_graph.get(img_id, []):
                            if Path(self.images[nbr].name).stem not in visited_frames:
                                queue.append(nbr)
                    tracks_2d[cid] = ordered_masks
                else:
                    tracks_2d[cid] = list(masks_set)

            self.log(f"Created {len(tracks_2d)} 2D tracks from 3D clusters using projection (optimized)")
        elif pipeline == "2D3D":
            # clusters are already created from 2D groups -> just reformat
            for cid, pids in clusters.items():
                mask_list = self.real_object_masks.get(cid, [])
                tracks_2d[cid] = mask_list
            self.log(f"Adapted {len(tracks_2d)} 2D tracks from 2D->3D clusters")
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")

        return tracks_2d
