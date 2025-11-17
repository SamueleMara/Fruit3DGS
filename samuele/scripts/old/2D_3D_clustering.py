import numpy as np
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

PARAMS_FILE = "best_params.json"  # JSON file to store the best hyperparameters (dist_thresh, dbscan_eps)

# ------------------ Projection-based object tracker ------------------
class ProjectionTracker:
    """
    Handles loading COLMAP model and 2D instance masks,
    projecting 3D points into 2D masks,
    and managing 2D-3D correspondences for object tracking.
    """
    def __init__(self, colmap_model_dir, mask_dir, dist_thresh=0.2):
        self.colmap_model_dir = Path(colmap_model_dir)
        self.mask_dir = Path(mask_dir)
        self.dist_thresh = dist_thresh  # threshold for assigning 3D points to masks
        self.cameras, self.images, self.points3D = {}, {}, {}
        self.mask_dict = {}      # {frame_name: [mask1, mask2, ...]}
        self.point_to_masks = {} # {point_id: [(frame_name, mask_idx), ...]}

    def load_colmap(self):
        """Load COLMAP cameras, images and 3D points from model directory."""
        print(f"[INFO] Loading COLMAP model from {self.colmap_model_dir}")
        self.cameras, self.images, self.points3D = read_model(self.colmap_model_dir, ext=".txt")
        print(f"[INFO] Loaded {len(self.cameras)} cameras, {len(self.images)} images, {len(self.points3D)} 3D points")

    def load_masks(self):
        """Load instance segmentation masks (PNG) for each frame."""
        print(f"[INFO] Loading mask instances from {self.mask_dir}")
        self.mask_dict = {}
        for mask_path in sorted(self.mask_dir.glob("*.png")):
            stem = mask_path.stem
            frame_name = stem.split("_instance")[0]  # extract frame name
            mask = np.array(Image.open(mask_path))
            if mask.ndim == 3:  # if mask is RGB, take one channel
                mask = mask[..., 0]
            mask = (mask > 0).astype(np.uint8)  # binarize
            self.mask_dict.setdefault(frame_name, []).append(mask)
        total_masks = sum(len(v) for v in self.mask_dict.values())
        print(f"[INFO] Loaded {total_masks} mask instances from {len(self.mask_dict)} frames")

    def map_points_to_masks(self):
        """
        For each 3D point, project its 2D coordinates into each image
        and find which mask it belongs to.
        Builds a mapping from 3D points -> (frame_name, mask_idx).
        """
        self.point_to_masks = {}
        for pid, pt in self.points3D.items():
            self.point_to_masks[pid] = []
            for img_id, pt2d_idx in zip(pt.image_ids, pt.point2D_idxs):
                img = self.images[img_id]
                frame_name = Path(img.name).stem
                if frame_name not in self.mask_dict:
                    continue
                x, y = img.xys[pt2d_idx]  # 2D projected coordinates
                mask_h, mask_w = self.mask_dict[frame_name][0].shape
                px = np.clip(int(round(x)), 0, mask_w - 1)
                py = np.clip(int(round(y)), 0, mask_h - 1)
                for midx, mask in enumerate(self.mask_dict[frame_name]):
                    if mask[py, px] != 0:  # point lies inside mask
                        self.point_to_masks[pid].append((frame_name, midx))
                        break

    def count_2d_objects(self):
        """Count unique 2D object tracks across frames from the masks."""
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

    def adapt_2d_tracks(self, clusters3d):
        """
        Split and merge 2D tracks based on 3D cluster assignments.
        Ensures 2D tracks correspond better to 3D clusters.
        """
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
        return self.merge_and_cleanup_2d_tracks(split_tracks, clusters3d, adaptive=True)

    def merge_and_cleanup_2d_tracks(self, tracks2d, clusters3d, adaptive=True):
        """
        Merge fragmented tracks into final 2D tracks per 3D cluster.
        Also removes very small fragments based on median track length.
        """
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

# ------------------ 3D clustering ------------------
def cluster_3d_points(points3D, eps=0.03, min_samples=1):
    """
    Cluster 3D points using DBSCAN.
    eps: distance threshold for clustering.
    """
    coords = np.array([pt.xyz for pt in points3D.values()])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    clusters = defaultdict(list)
    for pid, lbl in zip(points3D.keys(), clustering.labels_):
        clusters[lbl].append(pid)
    return clusters

def compute_error(tracks2d, clusters3d):
    """
    Error metric between number of 2D tracks and 3D clusters.
    Penalizes mismatch.
    """
    n2d, n3d = len(tracks2d), len(clusters3d)
    return abs(n2d - n3d) + 0.5 * max(n2d, n3d)

# ------------------ Hybrid Gaussian feedback ------------------
def hybrid_gaussian_feedback(tracker, patience=10, max_iter=100, min_samples_used=1):
    """
    A fallback optimizer:
    Randomly perturbs parameters around base values (Gaussian noise),
    keeps best result with early stopping.
    """
    history = []
    if Path(PARAMS_FILE).exists():
        with open(PARAMS_FILE, "r") as f:
            saved = json.load(f)
        base_dt, base_eps = saved["dist_thresh"], saved["dbscan_eps"]
        print(f"[INFO] Warm-start hybrid feedback dt={base_dt:.3f}, eps={base_eps:.3f}")
    else:
        base_dt, base_eps = tracker.dist_thresh, 0.03
        print(f"[INFO] Cold-start hybrid feedback")

    best_error, best_params = float("inf"), (base_dt, base_eps)
    no_improve_count = 0

    for i in range(1, max_iter+1):
        std = 0.05 * (1 - i/max_iter)  # gradually reduce noise
        dt = np.clip(base_dt + random.gauss(0, std), 0.01, 0.5)
        eps = np.clip(base_eps + random.gauss(0, std), 0.01, 0.1)

        tracker.dist_thresh = dt
        tracker.map_points_to_masks()
        clusters3d = cluster_3d_points(tracker.points3D, eps=eps, min_samples=min_samples_used)
        tracks2d = tracker.adapt_2d_tracks(clusters3d)
        error = compute_error(tracks2d, clusters3d)
        history.append((dt, eps, error))

        if error < best_error - 1e-6:
            best_error, best_params = error, (dt, eps)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"[INFO] Early stopping after {i+1} iterations (no improvement for {patience})")
                break
        if best_error == 0:
            break

    # Save best parameters to disk
    with open(PARAMS_FILE, "w") as f:
        json.dump({"dist_thresh": best_params[0], "dbscan_eps": best_params[1]}, f)

    print(f"[INFO] Hybrid feedback best dt={best_params[0]:.3f}, eps={best_params[1]:.3f}, error={best_error}")
    return best_params, history

# ------------------ GP Optimization with early stopping ------------------
def optimize_hyperparams(tracker, n_calls=50, warm_start=True, patience=10, min_samples_used=1):
    """
    Optimize hyperparameters dist_thresh and dbscan_eps using Bayesian Optimization (GP).
    Stops early if no improvement for `patience` iterations.
    """
    space = [Real(0.01, 0.5, name="dist_thresh"), Real(0.01, 0.1, name="dbscan_eps")]
    opt = Optimizer(space, random_state=42)
    history = []

    # Warm-start seed from saved params
    if warm_start and Path(PARAMS_FILE).exists():
        with open(PARAMS_FILE, "r") as f:
            saved = json.load(f)
        dt0 = np.clip(saved["dist_thresh"], 0.01, 0.5)
        eps0 = np.clip(saved["dbscan_eps"], 0.01, 0.1)
        print(f"[INFO] Warm-start GP from dt={dt0}, eps={eps0}")
        opt.tell([dt0, eps0], saved.get("error", 5.0))
        history.append((dt0, eps0, saved.get("error", 5.0)))
    else:
        print("[INFO] Cold start GP (no warm parameters)")

    best_error, best_dt, best_eps = float("inf"), None, None
    no_improve = 0

    # Optimization loop
    for i in range(n_calls):
        dt, eps = opt.ask()  # propose new parameters
        tracker.dist_thresh = dt
        tracker.map_points_to_masks()
        clusters3d = cluster_3d_points(tracker.points3D, eps=eps,min_samples=min_samples_used)
        tracks2d = tracker.adapt_2d_tracks(clusters3d)
        error = compute_error(tracks2d, clusters3d)
        history.append((dt, eps, error))
        print(f"[INFO] Iter {i+1}/{n_calls}: dt={dt:.3f}, eps={eps:.3f}, error={error}, 2D={len(tracks2d)}, 3D={len(clusters3d)}")

        opt.tell([dt, eps], error)

        if error < best_error - 1e-8:
            best_error, best_dt, best_eps = error, dt, eps
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[INFO] Early stopping after {i+1} iterations (no improvement for {patience})")
                break

    if best_dt is None:
        best_dt, best_eps, best_error = history[-1]

    return best_dt, best_eps, history

# ------------------ Visualization ------------------
def plot_gp_history(history, output_path=None):
    """Plot the GP exploration history and optionally save to file."""
    h = np.array(history)
    plt.figure(figsize=(8,5))
    sc = plt.scatter(h[:,0], h[:,1], c=h[:,2], cmap='viridis', s=80)
    plt.colorbar(sc, label="Error")
    plt.xlabel("dist_thresh"); plt.ylabel("dbscan_eps")
    plt.title("GP / Hybrid Exploration History")
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"[INFO] Saved GP history plot to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_3d_clusters(points3D, clusters3d, output_path=None):
    """Plot 3D clusters and optionally save to file."""
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab20(np.linspace(0,1,len(clusters3d)))
    for c_idx, color in zip(clusters3d.keys(), colors):
        pts = np.array([points3D[pid].xyz for pid in clusters3d[c_idx]])
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], color=color, label=f"Cluster {c_idx}", s=40)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.title("3D Point Cloud Clusters")
    plt.legend()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"[INFO] Saved 3D clusters plot to {output_path}")
    else:
        plt.show()
    plt.close()

# ------------------ Save colored 2D masks per cluster ------------------
def save_colored_masks(tracker, clusters3d, output_dir="colored_masks"):
    """
    For each frame, color all masks according to their assigned 3D cluster,
    and save as new images in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_clusters = len(clusters3d)
    colors = plt.cm.get_cmap('tab20', num_clusters)
    mask_to_cluster_color = {}
    # Build map from (frame, mask_idx) -> cluster color
    for c_idx, pids in clusters3d.items():
        color = (np.array(colors(c_idx)[:3]) * 255).astype(np.uint8)
        for pid in pids:
            for frame, midx in tracker.point_to_masks.get(pid, []):
                mask_to_cluster_color.setdefault(frame, {})[midx] = color

    # Create and save colored images
    for i, frame_name in enumerate(sorted(tracker.mask_dict.keys()), start=1):
        masks = tracker.mask_dict[frame_name]
        h, w = masks[0].shape
        colored_frame = np.zeros((h, w, 3), dtype=np.uint8)
        for midx, mask in enumerate(masks):
            if midx in mask_to_cluster_color.get(frame_name, {}):
                color = mask_to_cluster_color[frame_name][midx]
                for c in range(3):
                    colored_frame[..., c] += mask * color[c]
        Image.fromarray(colored_frame).save(os.path.join(output_dir, f"{i:03d}.png"))
    print(f"[INFO] Saved colored masks to {output_dir}")

# ------------------ Compute and save the important infos for each cluster ------------------
def compute_cluster_stats(points3D, clusters3d, dbscan_eps=None, dbscan_min_samples=None):
    """Compute center and radius for each 3D cluster, and store DBSCAN parameters."""
    stats = {}
    for c_idx, pids in clusters3d.items():
        pts = np.array([points3D[pid].xyz for pid in pids])
        center = pts.mean(axis=0)
        radius = np.max(np.linalg.norm(pts - center, axis=1))
        stats[str(int(c_idx))] = {
            "center": [float(x) for x in center],
            "radius": float(radius),
            "num_points": int(len(pids))
        }
    # Add DBSCAN parameters at the top level
    stats["_dbscan_params"] = {
    "eps": float(dbscan_eps) if dbscan_eps is not None else None,
    "min_samples": int(dbscan_min_samples) if dbscan_min_samples is not None else None
    }       
    return stats


# ------------------ Main ------------------
if __name__ == "__main__":
    start_time = time.time()  # start timer for total computation time

    # Paths to COLMAP model and masks
    colmap_model = "/home/samuelemara/colmap/samuele/lemons_only_sam2/manual_traj_masked_no_bckg/sparse/0"
    mask_dir = "/home/samuelemara/Grounded-SAM-2-autodistill/samuele/Lemon_only/mask_instances"
    output_folder = "/home/samuelemara/gaussian-splatting/samuele/grounded_sam2/output"
    
    # # Paths to COLMAP model and masks
    # colmap_model = "/home/samuelemara/colmap/samuele/tree_01_masked_white_bckg/sparse/0"
    # mask_dir = "/home/samuelemara/Grounded-SAM-2-autodistill/samuele/Fruit_Nerf/mask_instances"
    # output_folder = "/home/samuelemara/gaussian-splatting/samuele/fruit_nerf"
    
    
    os.makedirs(output_folder, exist_ok=True)

    # Warm-start flag
    warm_start_flag = True  # set False for cold start optimization

    # Initialize tracker and load data
    tracker = ProjectionTracker(colmap_model, mask_dir)
    tracker.load_colmap(); tracker.load_masks(); tracker.map_points_to_masks()
    print(f"[INFO] Initial 2D object tracks: {len(tracker.count_2d_objects())}")

    # Optimize hyperparameters (dist_thresh, dbscan_eps)
    best_dt, best_eps, history = optimize_hyperparams(tracker, n_calls=50, warm_start=warm_start_flag, patience=10)

    # Cluster 3D points with best parameters and adapt 2D tracks accordingly
    min_samples_used = 1  # whatever value you want to use
    clusters3d = cluster_3d_points(tracker.points3D, eps=best_eps, min_samples=min_samples_used)
    final_tracks2d = tracker.adapt_2d_tracks(clusters3d)
    print(f"[INFO] Final 2D tracks: {len(final_tracks2d)}, Final 3D clusters: {len(clusters3d)}")

    # Total runtime
    print(f"[INFO] Total computation time: {time.time()-start_time:.2f} seconds")

    # Plot results 
    plot_gp_history(history, output_path=os.path.join(output_folder, "gp_history.png"))
    plot_3d_clusters(tracker.points3D, clusters3d, output_path=os.path.join(output_folder, "3d_clusters.png"))

    # Compute and save cluster stats
    stats = compute_cluster_stats(
    tracker.points3D, clusters3d, 
    dbscan_eps=best_eps, 
    dbscan_min_samples=min_samples_used
    )
    stats_path = os.path.join(output_folder, "cluster_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"[INFO] Saved cluster stats with DBSCAN params to {stats_path}")

    # Save matched masks with clustered objects
    save_colored_masks(tracker, clusters3d, output_dir=os.path.join(output_folder, "colored_masks"))