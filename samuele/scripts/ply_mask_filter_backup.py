import os
import numpy as np
import cv2
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from read_write_model import read_model, write_model
from scipy.spatial.transform import Rotation as R
import glob


class GaussianPLYMaskFilter:
    def __init__(self, ply_path):
        self.ply = PlyData.read(ply_path)
        self.points = self.ply['vertex'].data
        self.properties = self.points.dtype.names
        self.initial_count = len(self.points)

    def get_xyz(self):
        return np.vstack([self.points['x'], self.points['y'], self.points['z']]).T

    def project_points(self, xyz, K, R, t):
        P_cam = (R @ xyz.T) + t.reshape(3, 1)
        z = P_cam[2, :]
        uv = (K @ (P_cam / z))[:2, :].T
        return uv, z

    def filter_with_mask(self, K, R, t, mask, view_name=None):
        H, W = mask.shape
        xyz = self.get_xyz()
        uv, z = self.project_points(xyz, K, R, t)
        uv_int = np.round(uv).astype(int)

        valid = (
            (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W) &
            (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H) &
            (z > 0)
        )

        affected = np.zeros(len(self.points), dtype=bool)
        valid_uv = uv_int[valid]

        for i, (u, v) in enumerate(valid_uv):
            if mask[v, u]:
                affected[np.where(valid)[0][i]] = True

        removed_count = affected.sum()
        self.points = self.points[~affected]

        if view_name:
            print(f"[View: {view_name}] Removed {removed_count} splats. Remaining: {len(self.points)}")
        return removed_count

    def filter_with_masks(self, Ks, Rs, ts, masks, names=None, min_percent=0.2):
        n_points = len(self.points)
        n_views = len(Ks)
        hit_count = np.zeros(n_points, dtype=int)

        for i, (K, R, t, mask) in enumerate(tqdm(zip(Ks, Rs, ts, masks), total=n_views, desc="Filtering views")):
            view_name = names[i] if names else None
            H, W = mask.shape
            xyz = self.get_xyz()
            uv, z = self.project_points(xyz, K, R, t)
            uv_int = np.round(uv).astype(int)

            valid = (
                (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W) &
                (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H) &
                (z > 0)
            )
            valid_idx = np.where(valid)[0]
            valid_uv = uv_int[valid]

            for j, (u, v) in enumerate(valid_uv):
                if mask[v, u]:  # True = remove
                    hit_count[valid_idx[j]] += 1

            if view_name:
                print(f"[View: {view_name}] Points with hits so far: {np.sum(hit_count>0)}")

        min_views = max(1, int(np.ceil(min_percent * n_views)))
        keep_mask = hit_count >= min_views
        total_removed = n_points - np.sum(keep_mask)
        self.points = self.points[keep_mask]

        print(f"\nFiltering complete.")
        print(f"Initial splats: {n_points}")
        print(f"Total removed: {total_removed}")
        print(f"Total remaining: {len(self.points)}")

    def save(self, out_path):
        el = PlyElement.describe(self.points, 'vertex')
        PlyData([el], text=False).write(out_path)

    @staticmethod
    def qvec2rotmat(qvec):
        q0, q1, q2, q3 = qvec
        return np.array([
            [1 - 2 * (q2**2 + q3**2),     2 * (q1*q2 - q0*q3),     2 * (q1*q3 + q0*q2)],
            [2 * (q1*q2 + q0*q3),     1 - 2 * (q1**2 + q3**2),     2 * (q2*q3 - q0*q1)],
            [2 * (q1*q3 - q0*q2),         2 * (q2*q3 + q0*q1), 1 - 2 * (q1**2 + q2**2)]
        ])

    @staticmethod
    def load_masks_from_dir(mask_dir, img_names, ext=".png.png"):
        masks = []
        valid_names = []
        for name in img_names:
            mask_path = os.path.join(mask_dir, os.path.splitext(name)[0] + ext)
            if not os.path.exists(mask_path):
                print(f"[WARNING] Mask not found for {name}, skipping")
                continue
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[WARNING] Mask not readable for {name}, skipping")
                continue
            masks.append(mask >= 127)
            valid_names.append(name)
        return masks, valid_names

    @classmethod
    def filter_ply_with_colmap(cls, colmap_txt_dir, ply_path, mask_dir, mask_ext=".png.png", out_path="filtered.ply"):
        print(f"[INFO] Loading COLMAP model from {colmap_txt_dir}")
        cameras, images, _ = read_model(str(colmap_txt_dir), ext=".txt")
        print(f"[INFO] Loaded {len(cameras)} cameras, {len(images)} images")

        Ks, Rs, ts, names = [], [], [], []
        for img in images.values():
            if img.camera_id not in cameras:
                print(f"[WARNING] Image {img.name} has invalid camera ID {img.camera_id}, skipping")
                continue
            cam = cameras[img.camera_id]
            if cam.model in ("PINHOLE", "SIMPLE_PINHOLE"):
                fx = cam.params[0]
                fy = cam.params[1] if cam.model == "PINHOLE" else cam.params[0]
                cx, cy = cam.params[-2], cam.params[-1]
            else:
                raise NotImplementedError(f"Camera model {cam.model} not supported")
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
            Ks.append(K)
            R = cls.qvec2rotmat(img.qvec)
            t = img.tvec
            Rs.append(R)
            ts.append(t)
            names.append(img.name)

        masks, valid_names = cls.load_masks_from_dir(mask_dir, names, ext=mask_ext)
        Ks_filtered, Rs_filtered, ts_filtered = [], [], []
        for K, R, t, name in zip(Ks, Rs, ts, names):
            if name in valid_names:
                Ks_filtered.append(K)
                Rs_filtered.append(R)
                ts_filtered.append(t)

        filter = cls(ply_path)
        filter.filter_with_masks(Ks_filtered, Rs_filtered, ts_filtered, masks, names=valid_names)
        filter.save(out_path)
        print(f"[INFO] Saved filtered PLY to {out_path}")
        return out_path


# NEW: a single callable wrapper
def filter_ply_with_masks(colmap_txt_dir, ply_path, mask_dir, mask_ext=".png", out_path="filtered.ply"):
    """
    High-level function to filter a PLY point cloud using COLMAP cameras and masks.
    """
    return GaussianPLYMaskFilter.filter_ply_with_colmap(
        colmap_txt_dir=colmap_txt_dir,
        ply_path=ply_path,
        mask_dir=mask_dir,
        mask_ext=mask_ext,
        out_path=out_path
    )

def filter_ply_with_strict_masks(colmap_txt_dir, ply_path, mask_dir,
                                 out_path="scene_mask_filtered_strict.ply",
                                 save_inmask_points=True,
                                 min_views=2, mask_threshold=128):
    """
    Project 3D points into COLMAP images and keep only points that fall inside at least `min_views` masks.
    Stricter multi-view mask filtering.
    Automatically detects mask file extensions in mask_dir.
    """
    
    cameras_txt = os.path.join(colmap_txt_dir, "cameras.txt")
    images_txt = os.path.join(colmap_txt_dir, "images.txt")

    if not os.path.exists(cameras_txt) or not os.path.exists(images_txt):
        raise FileNotFoundError(f"Missing COLMAP files in {colmap_txt_dir}")

    # --- Load PLY ---
    plydata = PlyData.read(ply_path)
    vertices = np.array(plydata["vertex"].data)
    xyz = np.vstack((vertices["x"], vertices["y"], vertices["z"])).T

    # --- Parse cameras.txt ---
    cameras = {}
    with open(cameras_txt, "r") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split()
            if len(parts) < 5: continue
            cam_id = int(parts[0])
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = np.array(list(map(float, parts[4:])))
            cameras[cam_id] = (model, width, height, params)

    # --- Parse images.txt ---
    images = []
    with open(images_txt, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    for i in range(0, len(lines), 2):
        if lines[i].startswith("#"): continue
        parts = lines[i].split()
        if len(parts) < 10: continue
        qvec = np.array(list(map(float, parts[1:5])))
        tvec = np.array(list(map(float, parts[5:8])))
        cam_id = int(parts[8])
        img_name = parts[9]
        images.append((img_name, qvec, tvec, cam_id))

    # --- Map image names to mask files automatically ---
    mask_files = glob.glob(os.path.join(mask_dir, "*"))
    mask_map = {}
    for mf in mask_files:
        base = os.path.splitext(os.path.basename(mf))[0]
        mask_map[base] = mf

    # --- Multi-view mask counter ---
    view_counts = np.zeros(len(xyz), dtype=int)

    for img_name, qvec, C, cam_id in tqdm(images, desc="[Strict Mask Filtering]"):
        base_name = os.path.splitext(img_name)[0]
        mask_path = mask_map.get(base_name, None)
        if mask_path is None or not os.path.exists(mask_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        model, width, height, params = cameras[cam_id]
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        if not model.startswith("PINHOLE"):
            continue

        fx, fy, cx, cy = params[:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Transform world points to camera coordinates
        qw, qx, qy, qz = qvec
        R_wc = R.from_quat([qx, qy, qz, qw]).as_matrix()
        xyz_cam = (R_wc @ (xyz - C).T).T

        in_front = xyz_cam[:, 2] > 0
        if not np.any(in_front):
            continue

        xyz_cam_front = xyz_cam[in_front]
        proj = (K @ xyz_cam_front.T).T
        u = np.clip(np.floor(proj[:, 0] / proj[:, 2]).astype(int), 0, width - 1)
        v = np.clip(np.floor(proj[:, 1] / proj[:, 2]).astype(int), 0, height - 1)

        hits = mask[v, u] > mask_threshold
        view_counts[np.where(in_front)[0][hits]] += 1

    # --- Keep points seen in >= min_views masks ---
    inmask_flags = view_counts >= min_views
    filtered_vertices = vertices[inmask_flags]

    PlyData([PlyElement.describe(filtered_vertices, "vertex")], text=False).write(out_path)
    print(f"[INFO] Points kept after strict multi-view filtering: {np.sum(inmask_flags)} / {len(vertices)}")
    print(f"[INFO] Saved filtered PLY: {out_path}")

    if save_inmask_points:
        mask_ply_path = os.path.join(os.path.dirname(out_path), "scene_mask_points_strict.ply")
        PlyData([PlyElement.describe(filtered_vertices, "vertex")], text=False).write(mask_ply_path)
        print(f"[INFO] Saved points-inside-mask PLY: {mask_ply_path}")

    return out_path