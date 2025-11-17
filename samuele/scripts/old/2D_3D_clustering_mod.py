import numpy as np
import cupy as cp
from pathlib import Path
from PIL import Image
from sklearn.cluster import DBSCAN
from collections import defaultdict
import random, json, os, time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from read_write_model import read_model
from skopt.space import Real
from skopt import Optimizer
from tqdm import tqdm

PARAMS_FILE = "best_params.json"

# ------------------ Projection-based object tracker ------------------
class ProjectionTracker:
    def __init__(self, colmap_model_dir, mask_dir, dist_thresh=0.2):
        self.colmap_model_dir = Path(colmap_model_dir)
        self.mask_dir = Path(mask_dir)
        self.dist_thresh = dist_thresh
        self.cameras, self.images, self.points3D = {}, {}, {}
        self.point_to_masks = {}  # {point_id: [(frame_name, mask_idx), ...]}

    def load_colmap(self):
        print(f"[INFO] Loading COLMAP model from {self.colmap_model_dir}")
        self.cameras, self.images, self.points3D = read_model(self.colmap_model_dir, ext=".txt")
        print(f"[INFO] Loaded {len(self.cameras)} cameras, {len(self.images)} images, {len(self.points3D)} 3D points")

    def get_mask(self, frame_name, mask_idx, downsample=1):
        mask_files = sorted(self.mask_dir.glob(f"{frame_name}_instance*.png"))
        mask_path = mask_files[mask_idx]
        mask = Image.open(mask_path)
        if downsample > 1:
            w, h = mask.size
            mask = mask.resize((w // downsample, h // downsample), resample=Image.NEAREST)
        mask = np.array(mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        return (mask > 0).astype(np.uint8)

    def map_points_to_masks_gpu(self, downsample=1):
        """GPU-accelerated 3D point → 2D mask mapping with progress bar."""
        print("[INFO] Mapping 3D points to masks (GPU)")
        self.point_to_masks = defaultdict(list)

        for img_id, img in tqdm(self.images.items(), desc="Mapping 3D points"):
            frame_name = Path(img.name).stem
            mask_files = sorted(self.mask_dir.glob(f"{frame_name}_instance*.png"))
            if not mask_files or not hasattr(img, "point3D_ids") or len(img.point3D_ids) == 0:
                continue

            pts3D_ids = np.array(img.point3D_ids)
            xys = np.array([img.xys[idx] for idx in range(len(pts3D_ids))])
            xys_gpu = cp.asarray(xys) / downsample

            # Load one mask to get size
            mask_example = self.get_mask(frame_name, 0, downsample=downsample)
            mask_h, mask_w = mask_example.shape
            px = cp.clip(cp.round(xys_gpu[:, 0]), 0, mask_w - 1).astype(cp.int32)
            py = cp.clip(cp.round(xys_gpu[:, 1]), 0, mask_h - 1).astype(cp.int32)

            for midx, mask_file in enumerate(mask_files):
                mask = self.get_mask(frame_name, midx, downsample=downsample)
                mask_gpu = cp.asarray(mask, dtype=cp.uint8)
                hits = mask_gpu[py, px] > 0
                hits_cpu = cp.asnumpy(hits)

                for pt_id, hit in zip(pts3D_ids, hits_cpu):
                    if hit and pt_id != -1:
                        self.point_to_masks[pt_id].append((frame_name, midx))

    def count_2d_objects_gpu(self):
        """GPU-accelerated counting of 2D objects."""
        tracks, assigned = [], {}
        for pid, masks in self.point_to_masks.items():
            for frame, midx in masks:
                key = (frame, midx)
                if key not in assigned:
                    assigned[key] = len(tracks)
                    tracks.append([key])
                else:
                    tracks[assigned[key]].append(key)
        return tracks

    def adapt_2d_tracks_gpu(self, clusters3d):
        """GPU-accelerated adaptation of 2D tracks based on 3D clusters."""
        tracks2d = self.count_2d_objects_gpu()
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
        return self.merge_and_cleanup_2d_tracks(split_tracks, clusters3d, adaptive=True)

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
            merged_main = list(set(item for t in main_tracks for item in t)) if main_tracks else []
            if fragments:
                merged_main += list(set(item for t in fragments for item in t))
            if merged_main:
                final_tracks.append(merged_main)
        return final_tracks

    def save_colored_masks_gpu(self, clusters3d, output_dir="colored_masks", downsample=1):
        """GPU-accelerated coloring of 2D masks per cluster."""
        os.makedirs(output_dir, exist_ok=True)
        num_clusters = len(clusters3d)
        colors = plt.cm.get_cmap('tab20', num_clusters)
        mask_to_cluster_color = {}
        for c_idx, pids in clusters3d.items():
            color = (np.array(colors(c_idx)[:3]) * 255).astype(np.uint8)
            for pid in pids:
                for frame, midx in self.point_to_masks.get(pid, []):
                    mask_to_cluster_color.setdefault(frame, {})[midx] = color

        frame_names = set(f for f_dict in self.point_to_masks.values() for f, _ in f_dict)
        for i, frame_name in enumerate(tqdm(sorted(frame_names), desc="Saving colored masks"), start=1):
            mask_files = sorted(self.mask_dir.glob(f"{frame_name}_instance*.png"))
            if not mask_files:
                continue
            mask_example = self.get_mask(frame_name, 0, downsample=downsample)
            h, w = mask_example.shape
            colored_frame = cp.zeros((h, w, 3), dtype=cp.uint8)
            for midx, mask_file in enumerate(mask_files):
                mask = cp.asarray(self.get_mask(frame_name, midx, downsample=downsample), dtype=cp.uint8)
                if midx in mask_to_cluster_color.get(frame_name, {}):
                    color = cp.asarray(mask_to_cluster_color[frame_name][midx], dtype=cp.uint8)
                    colored_frame += mask[..., None] * color[None, None, :]
            Image.fromarray(cp.asnumpy(colored_frame)).save(os.path.join(output_dir, f"{i:03d}.png"))
        print(f"[INFO] Saved colored masks to {output_dir}")

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
    return abs(n2d - n3d) + 0.5 * max(n2d, n3d)

# ------------------ Main ------------------
if __name__ == "__main__":
    start_time = time.time()
    colmap_model = "/home/samuelemara/colmap/samuele/tree_01_masked_white_bckg/sparse/0"
    mask_dir = "/home/samuelemara/Grounded-SAM-2-autodistill/samuele/Fruit_Nerf/mask_instances"
    output_folder = "/home/samuelemara/gaussian-splatting/samuele/fruit_nerf"
    os.makedirs(output_folder, exist_ok=True)

    tracker = ProjectionTracker(colmap_model, mask_dir)
    tracker.load_colmap()
    tracker.map_points_to_masks_gpu(downsample=1)
    print(f"[INFO] Initial 2D object tracks: {len(tracker.count_2d_objects_gpu())}")

    clusters3d = cluster_3d_points(tracker.points3D, eps=0.03, min_samples=1)
    final_tracks2d = tracker.adapt_2d_tracks_gpu(clusters3d)
    print(f"[INFO] Final 2D tracks: {len(final_tracks2d)}, Final 3D clusters: {len(clusters3d)}")
    print(f"[INFO] Total computation time: {time.time()-start_time:.2f} seconds")

    tracker.save_colored_masks_gpu(clusters3d, output_dir=os.path.join(output_folder, "colored_masks"))
