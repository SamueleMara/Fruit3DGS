import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.cluster import DBSCAN
from collections import defaultdict
import random
import json
import matplotlib.pyplot as plt
from read_write_model import read_model

PARAMS_FILE = "best_params.json"

# ------------------ Projection-based object tracker ------------------
class ProjectionTracker:
    def __init__(self, colmap_model_dir, mask_dir, dist_thresh=0.2):
        self.colmap_model_dir = Path(colmap_model_dir)
        self.mask_dir = Path(mask_dir)
        self.dist_thresh = dist_thresh

        self.cameras, self.images, self.points3D = {}, {}, {}
        self.mask_dict = {}
        self.point_to_masks = {}

    def load_colmap(self):
        print(f"[INFO] Loading COLMAP model from {self.colmap_model_dir}")
        self.cameras, self.images, self.points3D = read_model(self.colmap_model_dir, ext=".txt")
        print(f"[INFO] Loaded {len(self.cameras)} cameras, {len(self.images)} images, {len(self.points3D)} 3D points")

    def load_masks(self):
        print(f"[INFO] Loading mask instances from {self.mask_dir}")
        self.mask_dict = {}
        for mask_path in sorted(self.mask_dir.glob("*.png")):
            stem = mask_path.stem
            frame_name = stem.split("_instance")[0]
            mask = np.array(Image.open(mask_path))
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = (mask > 0).astype(np.uint8)
            self.mask_dict.setdefault(frame_name, []).append(mask)
        total_masks = sum(len(v) for v in self.mask_dict.values())
        print(f"[INFO] Loaded {total_masks} mask instances from {len(self.mask_dict)} frames")

    def map_points_to_masks(self):
        self.point_to_masks = {}
        for pid, pt in self.points3D.items():
            self.point_to_masks[pid] = []
            for img_id, pt2d_idx in zip(pt.image_ids, pt.point2D_idxs):
                img = self.images[img_id]
                frame_name = Path(img.name).stem
                if frame_name not in self.mask_dict:
                    continue
                x, y = img.xys[pt2d_idx]
                mask_h, mask_w = self.mask_dict[frame_name][0].shape
                px = np.clip(int(round(x)), 0, mask_w - 1)
                py = np.clip(int(round(y)), 0, mask_h - 1)
                for midx, mask in enumerate(self.mask_dict[frame_name]):
                    if mask[py, px] != 0:
                        self.point_to_masks[pid].append((frame_name, midx))
                        break

    def count_2d_objects(self):
        tracks = []
        assigned = {}
        for pid, masks in self.point_to_masks.items():
            for frame, midx in masks:
                key = (frame, midx)
                if key not in assigned:
                    assigned[key] = len(tracks)
                    tracks.append([key])
                else:
                    tracks[assigned[key]].append(key)
        return tracks

    def adapt_2d_tracks(self, clusters3d):
        tracks2d = self.count_2d_objects()
        split_tracks = []
        for track in tracks2d:
            cluster_to_keys = defaultdict(list)
            for frame, midx in track:
                for pid, masks in self.point_to_masks.items():
                    if (frame, midx) in masks:
                        for c_idx, cluster_pts in clusters3d.items():
                            if pid in cluster_pts:
                                cluster_to_keys[c_idx].append((frame, midx))
            for keys in cluster_to_keys.values():
                split_tracks.append(list(set(keys)))
        final_tracks = self.merge_and_cleanup_2d_tracks(split_tracks, clusters3d, adaptive=True)
        return final_tracks

    def merge_and_cleanup_2d_tracks(self, tracks2d, clusters3d, adaptive=True):
        all_lengths = [len(t) for t in tracks2d]
        min_track_len = max(1, int(np.median(all_lengths)//5)) if adaptive and all_lengths else 3
        cluster_to_tracks = defaultdict(list)
        for track in tracks2d:
            overlap_count = defaultdict(int)
            for frame, midx in track:
                for pid, masks in self.point_to_masks.items():
                    if (frame, midx) in masks:
                        for c_idx, pts in clusters3d.items():
                            if pid in pts:
                                overlap_count[c_idx] += 1
            if not overlap_count:
                continue
            best_cluster = max(overlap_count, key=overlap_count.get)
            cluster_to_tracks[best_cluster].append(track)

        final_tracks = []
        for cluster_tracks in cluster_to_tracks.values():
            main_tracks = [t for t in cluster_tracks if len(t) >= min_track_len]
            fragments = [t for t in cluster_tracks if len(t) < min_track_len]
            merged_main = []
            if main_tracks:
                merged_main = list(set(item for t in main_tracks for item in t))
            if fragments:
                merged_main += list(set(item for t in fragments for item in t))
            if merged_main:
                final_tracks.append(merged_main)
        return final_tracks

# ------------------ 3D clustering ------------------
def cluster_3d_points(points3D, eps=0.03, min_samples=1):
    coords = np.array([pt.xyz for pt in points3D.values()])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    clusters = defaultdict(list)
    for pid, lbl in zip(points3D.keys(), clustering.labels_):
        clusters[lbl].append(pid)
    return clusters

def compute_error(tracks2d, clusters3d):
    n2d, n3d = len(tracks2d), len(clusters3d)
    # Penalise number of clusters as well:
    return abs(n2d - n3d) + 0.5 * max(n2d, n3d)

# ------------------ Hybrid Gaussian + adaptive feedback + persistence ------------------
def hybrid_gaussian_feedback(tracker, max_iter=100):
    # Load previous best if exists
    if Path(PARAMS_FILE).exists():
        with open(PARAMS_FILE, "r") as f:
            saved = json.load(f)
        base_dt, base_eps = saved["dist_thresh"], saved["dbscan_eps"]
        print(f"[INFO] Warm-start from saved parameters dt={base_dt:.3f}, eps={base_eps:.3f}")
    else:
        base_dt, base_eps = tracker.dist_thresh, 0.03

    best_error = float("inf")
    best_params = (base_dt, base_eps)
    history = []

    for i in range(1, max_iter+1):
        std = 0.05 * (1 - i/max_iter)
        dt = np.clip(base_dt + random.gauss(0, std), 0.01, 0.5)
        eps = np.clip(base_eps + random.gauss(0, std), 0.01, 0.1)

        tracker.dist_thresh = dt
        tracker.map_points_to_masks()
        clusters3d = cluster_3d_points(tracker.points3D, eps=eps)
        tracks2d = tracker.adapt_2d_tracks(clusters3d)
        error = compute_error(tracks2d, clusters3d)
        history.append((dt, eps, error, len(tracks2d), len(clusters3d)))
        print(f"[INFO] Iteration {i}/{max_iter} -> dt={dt:.3f}, eps={eps:.3f}, error={error}, 2D={len(tracks2d)}, 3D={len(clusters3d)}")

        if error < best_error:
            best_error = error
            best_params = (dt, eps)
        if best_error == 0:
            print(f"[INFO] Converged after {i} iterations")
            break

    # Save best params for next run
    with open(PARAMS_FILE, "w") as f:
        json.dump({"dist_thresh": best_params[0], "dbscan_eps": best_params[1]}, f)

    print(f"[INFO] Best parameters: dist_thresh={best_params[0]:.3f}, dbscan_eps={best_params[1]:.3f}, min_error={best_error}")
    return best_params

# ------------------ Main ------------------
if __name__ == "__main__":
    colmap_model = "/home/samuelemara/colmap/samuele/lemons_only/manual_traj_masked_colmap_images/sparse/0"
    mask_dir = "/home/samuelemara/Grounded-Segment-Anything/GroundedSAM/Samuele/Lemon_only/mask_instances"

    tracker = ProjectionTracker(colmap_model, mask_dir)
    tracker.load_colmap()
    tracker.load_masks()
    tracker.map_points_to_masks()

    print(f"[INFO] Initial 2D object tracks: {len(tracker.count_2d_objects())}")

    best_dist_thresh, best_dbscan_eps = hybrid_gaussian_feedback(tracker, max_iter=100)
    clusters3d = cluster_3d_points(tracker.points3D, eps=best_dbscan_eps)
    final_tracks2d = tracker.adapt_2d_tracks(clusters3d)
    print(f"[INFO] Final 2D tracks: {len(final_tracks2d)}, Final 3D clusters: {len(clusters3d)}")
