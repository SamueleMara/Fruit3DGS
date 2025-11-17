import os
import numpy as np
import cv2
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from read_write_model import read_model, write_model

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
        
        # Counter for how many times each point lands on a white pixel
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

            # Increment hit count if projected pixel is white
            for j, (u, v) in enumerate(valid_uv):
                if mask[v, u]:  # True = white
                    hit_count[valid_idx[j]] += 1

            if view_name:
                print(f"[View: {view_name}] Points with hits so far: {np.sum(hit_count>0)}")

        # Convert percentage to absolute number of views
        min_views = max(1, int(np.ceil(min_percent * n_views)))
        
        # Keep points that were white in at least min_views
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
            masks.append(mask >= 127)  # Black = True = remove, White = False = keep
            valid_names.append(name)
        return masks, valid_names

    @classmethod
    def filter_ply_with_colmap(cls, colmap_txt_dir, ply_path, mask_dir, mask_ext=".png.png", out_path="filtered.ply"):
        # Load COLMAP model
        print(f"[INFO] Loading COLMAP model from {colmap_txt_dir}")
        cameras, images, _ = read_model(str(colmap_txt_dir), ext=".txt")
        print(f"[INFO] Loaded {len(cameras)} cameras, {len(images)} images")

        # Prepare intrinsics, rotations, translations, names
        Ks, Rs, ts, names = [], [], [], []
        for img in images.values():
            if img.camera_id not in cameras:
                print(f"[WARNING] Image {img.name} has invalid camera ID {img.camera_id}, skipping")
                continue
            cam = cameras[img.camera_id]
            # Build K
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
            # Rotation and translation
            R = cls.qvec2rotmat(img.qvec)
            t = img.tvec
            Rs.append(R)
            ts.append(t)
            names.append(img.name)

        # Load masks
        masks, valid_names = cls.load_masks_from_dir(mask_dir, names, ext=mask_ext)
        Ks_filtered, Rs_filtered, ts_filtered = [], [], []
        for K, R, t, name in zip(Ks, Rs, ts, names):
            if name in valid_names:
                Ks_filtered.append(K)
                Rs_filtered.append(R)
                ts_filtered.append(t)

        # Apply mask filtering
        filter = cls(ply_path)
        filter.filter_with_masks(Ks_filtered, Rs_filtered, ts_filtered, masks, names=valid_names)
        filter.save(out_path)
        print(f"[INFO] Saved filtered PLY to {out_path}")
        return out_path


if __name__ == "__main__":

    # Define paths
    sparse_dir = '/home/samuelemara/colmap/samuele/lemons_only/manual_traj_masked_colmap_images/sparse/0'
    ply_folder = '/home/samuelemara/gaussian-splatting/output/b80f59f1-6/point_cloud/iteration_30000'
    ply_path = os.path.join(ply_folder, 'point_cloud.ply')
    mask_dir = '/home/samuelemara/Grounded-Segment-Anything/GroundedSAM/Samuele/Lemon_only/masks'
    out_ply_path = os.path.join(ply_folder, 'scene_mask_filtered.ply')

    # Apply 3DGS mask filter
    GaussianPLYMaskFilter.filter_ply_with_colmap(
        colmap_txt_dir=sparse_dir,
        ply_path=ply_path,
        mask_dir=mask_dir,
        mask_ext=".png.png",
        out_path=out_ply_path
    )
