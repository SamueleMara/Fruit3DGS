from pathlib import Path
import numpy as np
from PIL import Image
from read_write_model import read_model
import cv2
from scipy.optimize import linear_sum_assignment
import open3d as o3d
import shutil
import matplotlib.pyplot as plt
import time

# ------------------ Simple Kalman filter for 3D constant-velocity ------------------
class Kalman3D:
    """
    Simple Kalman filter for 3D constant velocity model.
    State: [x, y, z, vx, vy, vz]^T
    Measurement: [x, y, z]^T
    """
    def __init__(self, init_pos, dt=1.0, process_var=1e-2, meas_var=1e-1):
        # state vector
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.x[0:3, 0] = init_pos.reshape(3)
        self.x[3:6, 0] = 0.0

        # state covariance
        self.P = np.eye(6, dtype=np.float32) * 1.0

        # state transition
        self.dt = float(dt)
        self.F = np.eye(6, dtype=np.float32)
        for i in range(3):
            self.F[i, 3 + i] = self.dt

        # process noise Q (tunable)
        q_pos = process_var
        q_vel = process_var * 10.0
        Q = np.zeros((6, 6), dtype=np.float32)
        Q[0:3, 0:3] = np.eye(3) * q_pos
        Q[3:6, 3:6] = np.eye(3) * q_vel
        self.Q = Q

        # measurement matrix
        self.H = np.zeros((3, 6), dtype=np.float32)
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        # measurement noise R
        self.R = np.eye(3, dtype=np.float32) * meas_var

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0:3, 0].copy()

    def update(self, meas, meas_var=None):
        z = meas.reshape(3, 1).astype(np.float32)
        if meas_var is not None:
            self.R = np.eye(3, dtype=np.float32) * float(meas_var)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0], dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    def get_position(self):
        return self.x[0:3, 0].copy()

# ------------------ SORT-like track object ------------------
class Track:
    def __init__(self, track_id, init_centroid3d, hist, t_frame, kalman_params):
        self.id = track_id
        init_pos = np.array(init_centroid3d, dtype=np.float32)
        # Kalman parameters: dict with dt, process_var, meas_var
        self.kf = Kalman3D(init_pos, dt=kalman_params.get("dt", 1.0),
                           process_var=kalman_params.get("process_var", 1e-2),
                           meas_var=kalman_params.get("meas_var", 1e-1))
        self.hist = hist.copy() if hist is not None else None
        self.time_since_update = 0
        self.age = 0
        self.hits = 1
        self.last_frame = t_frame
        # store assignments as list of (frame_name, mask_idx)
        self.assigned_masks = []

    def predict(self):
        pred = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return pred

    def update(self, meas3d, hist, frame_name_masktuple):
        # measurement variance set small if measurement exists, large otherwise
        if meas3d is None:
            meas = np.array(self.kf.get_position())
            meas_var = 1.0  # big uncertainty
        else:
            meas = np.array(meas3d, dtype=np.float32)
            meas_var = 1e-2
        self.kf.update(meas, meas_var=meas_var)
        if hist is not None:
            self.hist = hist.copy()
        self.time_since_update = 0
        self.hits += 1
        self.last_frame = frame_name_masktuple[0]
        self.assigned_masks.append(frame_name_masktuple)

    def get_state(self):
        return self.kf.get_position()

# ------------------ Main SAM-based SORT-like tracker ------------------
class SAMSortTracker:
    def __init__(self,
                 colmap_model_dir,
                 mask_dir,
                 image_dir,
                 mask_ext=".png",
                 image_ext=".png",
                 hist_bins=(16,16,16),
                 appearance_weight=0.5,
                 spatial_weight=0.4,
                 depth_weight=0.1,
                 max_missing=3,
                 match_cost_threshold=0.7,
                 kalman_params=None):
        self.colmap_model_dir = Path(colmap_model_dir)
        self.mask_dir = Path(mask_dir)
        self.image_dir = Path(image_dir)
        self.mask_ext = mask_ext
        self.image_ext = image_ext

        self.hist_bins = hist_bins
        self.appearance_weight = appearance_weight
        self.spatial_weight = spatial_weight
        self.depth_weight = depth_weight

        self.max_missing = max_missing
        self.match_cost_threshold = match_cost_threshold

        if kalman_params is None:
            kalman_params = {"dt": 1.0, "process_var": 1e-2, "meas_var": 1e-1}
        self.kalman_params = kalman_params

        # data containers
        self.cameras = {}
        self.images = {}
        self.points3D = {}
        self.mask_dict = {}      # frame_name -> list of masks
        self.point_to_masks = {} # pid -> list of (frame_name, midx)

        # tracks
        self.tracks = {}
        self.next_id = 0

        # store sorting indices
        self.sorted_mask_indices = {}

    # ---------- COLMAP + masks ----------
    def load_colmap(self):
        print(f"[INFO] Loading COLMAP model from {self.colmap_model_dir}")
        self.cameras, self.images, self.points3D = read_model(self.colmap_model_dir, ext=".txt")
        print(f"[INFO] Loaded {len(self.cameras)} cameras, {len(self.images)} images, {len(self.points3D)} 3D points")

    def load_masks(self):
        print(f"[INFO] Loading mask instances from {self.mask_dir}")
        self.mask_dict = {}
        for mask_path in sorted(self.mask_dir.glob(f"*{self.mask_ext}")):
            stem = mask_path.stem
            frame_name = stem.split("_instance")[0]
            mask = np.array(Image.open(mask_path))
            if frame_name not in self.mask_dict:
                self.mask_dict[frame_name] = []
            self.mask_dict[frame_name].append(mask)
        total_masks = sum(len(v) for v in self.mask_dict.values())
        print(f"[INFO] Loaded {total_masks} mask instances from {len(self.mask_dict)} frames")

    def map_points_to_masks(self):
        print("[INFO] Mapping COLMAP 3D points to mask instances...")
        self.point_to_masks = {}
        for pid, pt in self.points3D.items():
            self.point_to_masks[pid] = []
            for img_id, pt2d_idx in zip(pt.image_ids, pt.point2D_idxs):
                img = self.images[img_id]
                image_name = Path(img.name).stem
                if image_name not in self.mask_dict:
                    continue
                x, y = img.xys[pt2d_idx]
                px, py = int(round(x)), int(round(y))
                for midx, mask in enumerate(self.mask_dict[image_name]):
                    if 0 <= py < mask.shape[0] and 0 <= px < mask.shape[1]:
                        if mask[py, px] != 0:
                            self.point_to_masks[pid].append((image_name, midx))
                            break
        print("[INFO] Mapping complete.")

    # ---------- Features ----------
    def compute_mask_features(self, frame_name, mask_idx, mask, image_bgr):
        ys, xs = np.where(mask != 0)
        if len(xs) == 0:
            return None
        cx2d, cy2d = float(xs.mean()), float(ys.mean())
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        # HSV hist
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask_uint8 = (mask > 0).astype('uint8') * 255
        hist = cv2.calcHist([hsv], [0,1,2], mask_uint8, self.hist_bins, [0,180,0,256,0,256])
        cv2.normalize(hist, hist)
        hist = hist.flatten().astype(np.float32)

        # 3D centroid of mask (if enough points)
        pts3d = []
        for pid, assoc in self.point_to_masks.items():
            if (frame_name, mask_idx) in assoc:
                pts3d.append(self.points3D[pid].xyz)
        centroid3d = np.mean(pts3d, axis=0) if len(pts3d) > 0 else None

        return {"centroid2d": (cx2d, cy2d), "bbox": bbox, "hist": hist, "centroid3d": centroid3d}

    # ---------- Association cost ----------
    def association_cost(self, pred_positions, detections):
        """
        pred_positions: (M,3) predicted 3D positions of tracks
        detections: list of (midx, feat) where feat contains centroid3d (or None) and hist
        returns cost matrix MxN
        """
        M = pred_positions.shape[0]
        N = len(detections)
        cost = np.full((M, N), 1e6, dtype=np.float32)

        # build det centroids and hist arrays
        det_cent3d = np.array([d[1]["centroid3d"] if d[1]["centroid3d"] is not None else [np.nan,np.nan,np.nan] for d in detections], dtype=np.float32)
        det_hists = [d[1]["hist"] for d in detections]

        # normalization for 3D distances
        # compute a reasonable scale using available predicted/detected positions
        all_pos = []
        for p in pred_positions:
            if not np.isnan(p).any():
                all_pos.append(p)
        for d in det_cent3d:
            if not np.isnan(d).any():
                all_pos.append(d)
        if all_pos:
            all_pos = np.array(all_pos)
            scale3d = np.max(np.linalg.norm(all_pos - all_pos.mean(axis=0), axis=1)) + 1e-6
        else:
            scale3d = 1.0

        # compute cost per pair
        for i in range(M):
            for j in range(N):
                # spatial (3D) distance
                pred = pred_positions[i]
                det_c = det_cent3d[j]
                if not np.isnan(pred).any() and not np.isnan(det_c).any():
                    d3d = np.linalg.norm(pred - det_c) / scale3d
                else:
                    d3d = 1.0

                # appearance: histogram correlation -> cost = (1 - sim)
                try:
                    # use track hist? we will pass track hist externally by reading tracks order
                    hist_det = det_hists[j]
                    # placeholder for track hist: will be provided by caller by precomputing track_hists in same order as pred_positions
                    # we set hist similarity externally, see match step
                    hist_cost = 0.0  # will be filled outside
                except Exception:
                    hist_cost = 1.0

                # combine with weights (hist part will be added externally)
                cost[i, j] = self.spatial_weight * d3d + self.depth_weight * 0.0  # appearance added later
        return cost, scale3d

    # ---------- Main SORT-like loop ----------
    def run(self):
        self.load_colmap()
        self.load_masks()
        self.map_points_to_masks()

        frame_names = sorted(self.mask_dict.keys())
        print(f"[INFO] Running tracker on {len(frame_names)} frames...")

        # Spatially sort masks per frame and store indices
        self.sorted_mask_indices = {}
        for frame_name in frame_names:
            centroids = []
            for mask in self.mask_dict[frame_name]:
                ys, xs = np.where(mask != 0)
                if len(xs) == 0:
                    centroids.append((0.0, 0.0))
                else:
                    centroids.append((float(xs.mean()), float(ys.mean())))
            sorted_indices = list(sorted(range(len(centroids)), key=lambda i: (centroids[i][1], centroids[i][0])))
            self.sorted_mask_indices[frame_name] = sorted_indices
            # reorder mask list in place
            self.mask_dict[frame_name] = [self.mask_dict[frame_name][int(i)] for i in sorted_indices]

        # iterate frames
        for t_frame, frame_name in enumerate(frame_names):
            image_path = self.image_dir / f"{frame_name}{self.image_ext}"
            img = cv2.imread(str(image_path))
            if img is None:
                img = np.zeros((1,1,3), dtype=np.uint8)

            # compute detections (midx, feat)
            detections = []
            for midx, mask in enumerate(self.mask_dict[frame_name]):
                feat = self.compute_mask_features(frame_name, midx, mask, img)
                if feat is not None:
                    detections.append((midx, feat))

            # 1) Predict step for all existing tracks
            track_ids = list(self.tracks.keys())
            preds = []
            track_hists = []
            for tid in track_ids:
                tr = self.tracks[tid]
                pred = tr.predict()  # updates kf internally and increments time_since_update
                preds.append(pred)
                track_hists.append(tr.hist if tr.hist is not None else None)
            preds = np.array(preds) if len(preds) > 0 else np.zeros((0,3), dtype=np.float32)

            # 2) Build cost matrix: spatial (3D) + appearance
            if len(track_ids) > 0 and len(detections) > 0:
                base_cost, scale3d = self.association_cost(preds, detections)
                # add appearance term and finalize cost matrix
                for i, tid in enumerate(track_ids):
                    tr_hist = track_hists[i]
                    for j, det in enumerate(detections):
                        det_hist = det[1]["hist"]
                        # appearance similarity via correlation [-1,1] -> convert to [0,1]
                        try:
                            if tr_hist is None:
                                hist_sim = 0.0
                            else:
                                hist_sim = cv2.compareHist(tr_hist.astype('float32'), det_hist.astype('float32'), cv2.HISTCMP_CORREL)
                                hist_sim = float(np.clip((hist_sim + 1.0) / 2.0, 0.0, 1.0))
                        except Exception:
                            hist_sim = 0.0
                        hist_cost = (1.0 - hist_sim) * self.appearance_weight
                        # keep the base spatial part (already scaled by spatial_weight)
                        # combine: base_cost already holds spatial_weight * d3d (and depth_weight placeholder), add hist_cost
                        cost_val = base_cost[i, j] + hist_cost
                        base_cost[i, j] = cost_val
                cost = base_cost
            else:
                cost = np.zeros((len(track_ids), len(detections)), dtype=np.float32)

            # 3) Solve assignment with Hungarian
            matches, unmatched_tracks, unmatched_detections = [], list(range(len(track_ids))), list(range(len(detections)))
            if cost.size > 0:
                row_idx, col_idx = linear_sum_assignment(cost)
                assigned_t = set()
                assigned_d = set()
                for r, c in zip(row_idx, col_idx):
                    if cost[r, c] <= self.match_cost_threshold:
                        matches.append((r, c))
                        assigned_t.add(r)
                        assigned_d.add(c)
                unmatched_tracks = [i for i in range(len(track_ids)) if i not in assigned_t]
                unmatched_detections = [j for j in range(len(detections)) if j not in assigned_d]

            # 4) Update matched tracks
            for (i, j) in matches:
                tid = track_ids[i]
                midx, feat = detections[j]
                meas3d = feat["centroid3d"]
                hist = feat["hist"]
                self.tracks[tid].update(meas3d if meas3d is not None else None, hist, (frame_name, midx))

            # 5) Create new tracks for unmatched detections
            for j in unmatched_detections:
                midx, feat = detections[j]
                meas3d = feat["centroid3d"]
                hist = feat["hist"]
                init_pos = np.array(meas3d if meas3d is not None else [0.0, 0.0, 0.0], dtype=np.float32)
                new_track = Track(self.next_id, init_pos, hist, t_frame, self.kalman_params)
                new_track.assigned_masks.append((frame_name, midx))
                self.tracks[self.next_id] = new_track
                self.next_id += 1

            # 6) Manage unmatched tracks (increase time_since_update and delete if necessary)
            to_delete = []
            for idx in unmatched_tracks:
                tid = track_ids[idx]
                tr = self.tracks[tid]
                tr.time_since_update += 1
                if tr.time_since_update > self.max_missing:
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]

        # After all frames, build mask_keys_full by using assigned_masks (already appended in update/create steps)
        for tid, tr in self.tracks.items():
            # assigned_masks already contains tuples (frame_name, midx) in order of updates
            tr.mask_keys_full = tr.assigned_masks

        print("[INFO] Tracking complete.")
        print(f"[INFO] Total unique tracks: {len(self.tracks)}")
        for tid, tr in self.tracks.items():
            print(f"Track {tid}: masks_assigned={len(getattr(tr, 'mask_keys_full', []))}")

    # ---------- plotting helpers ----------
    def plot_open3d_tracks(self):
        # build points array and mapping pid->index
        pids = list(self.points3D.keys())
        points = np.array([self.points3D[pid].xyz for pid in pids])
        n = points.shape[0]

        base_color = np.array([0.2, 0.2, 0.6])
        colors = np.tile(base_color, (n, 1))

        # assign colors per track with consistent colormap
        track_ids = sorted(self.tracks.keys())
        cmap = plt.get_cmap("tab20")
        track_colors = {tid: np.array(cmap(i % 20)[:3]) for i, tid in enumerate(track_ids)}

        pid_to_index = {pid: i for i, pid in enumerate(pids)}

        for tid in track_ids:
            tr = self.tracks[tid]
            for (frame_name, midx) in getattr(tr, "mask_keys_full", []):
                for pid, assoc in self.point_to_masks.items():
                    if (frame_name, midx) in assoc:
                        colors[pid_to_index[pid]] = track_colors[tid]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        print("[INFO] Visualizing tracks in Open3D...")
        o3d.visualization.draw_geometries([pcd, axis], window_name="COLMAP Tracks (SORT-like)")

        # legend (matplotlib)
        if len(track_ids) > 0:
            plt.figure(figsize=(max(6, len(track_ids)), 1.2))
            for tid in track_ids:
                c = track_colors[tid]
                plt.plot([], [], color=c, marker='o', linestyle='', label=f"Track {tid}")
            plt.legend(ncol=min(8, len(track_ids)), loc='center', frameon=False)
            plt.axis('off')
            plt.show()

    def plot_matplotlib_3d(self):
        pids = list(self.points3D.keys())
        points = np.array([self.points3D[pid].xyz for pid in pids])
        track_ids = sorted(self.tracks.keys())
        track_colors = {tid: np.random.rand(3) for tid in track_ids}
        colors = np.ones((points.shape[0], 3)) * 0.5
        pid_to_index = {pid: i for i, pid in enumerate(pids)}
        for tid, tr in self.tracks.items():
            for (frame_name, midx) in getattr(tr, "mask_keys_full", []):
                for pid, assoc in self.point_to_masks.items():
                    if (frame_name, midx) in assoc:
                        colors[pid_to_index[pid]] = track_colors[tid]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=12)
        ax.set_title("3D Tracks (matplotlib)")
        plt.show()

# ----------------------------- main -----------------------------
def main():
    colmap_model_dir = "/home/samuelemara/colmap/samuele/lemons_only/manual_traj_masked_colmap_images/sparse/0"
    mask_dir = "/home/samuelemara/Grounded-Segment-Anything/GroundedSAM/Samuele/Lemon_only/mask_instances"
    image_dir = "/home/samuelemara/Grounded-Segment-Anything/GroundedSAM/Samuele/Lemon_only/input"

    tracker = SAMSortTracker(
        colmap_model_dir=colmap_model_dir,
        mask_dir=mask_dir,
        image_dir=image_dir,
        mask_ext=".png",
        image_ext=".png",
        hist_bins=(16,16,16),
        appearance_weight=0.5,
        spatial_weight=0.4,
        depth_weight=0.1,
        max_missing=3,
        match_cost_threshold=0.7,
        kalman_params={"dt": 1.0, "process_var": 1e-2, "meas_var": 1e-1}
    )

    t0 = time.time()
    tracker.run()
    print(f"[INFO] Tracking done in {time.time()-t0:.2f}s")

    tracker.plot_open3d_tracks()
    tracker.plot_matplotlib_3d()

    # print details
    print("\nTracks detail (frame, mask_idx) per track:")
    for tid, tr in tracker.tracks.items():
        print(f"Track {tid}: {getattr(tr, 'mask_keys_full', tr.assigned_masks)}")


if __name__ == "__main__":
    main()
