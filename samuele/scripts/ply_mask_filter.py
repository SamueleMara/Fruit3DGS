import os
import glob
import numpy as np
import cv2
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from read_write_model import read_model


class GaussianPLYMaskProjector:
    def __init__(self, ply_path):
        self.ply = PlyData.read(ply_path)
        self.points = self.ply["vertex"].data
        self.xyz = np.vstack([self.points["x"], self.points["y"], self.points["z"]]).T
        self.initial_count = len(self.xyz)

    @staticmethod
    def qvec2rotmat(qvec):
        q0, q1, q2, q3 = qvec
        return np.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])

    @staticmethod
    def load_masks_from_dir(mask_dir, image_names):
        masks = {}
        valid_names = []
        for name in image_names:
            candidates = glob.glob(os.path.join(mask_dir, f"{os.path.splitext(name)[0]}.*"))
            if not candidates:
                continue
            mask_path = candidates[0]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[WARNING] Could not load mask {mask_path}")
                continue
            masks[name] = mask >= 127
            valid_names.append(name)
        return masks, valid_names

    def filter_with_forward_projection(self, Ks, Rs, ts, masks, image_sizes,
                                       z_buffer_tol=0.01, min_views=1):
        """
        Project each 3D point into camera masks and keep only those that are visible
        and lie inside the white region in at least `min_views` images.

        Args:
            z_buffer_tol: Depth tolerance (meters) for visibility check.
            min_views: Minimum number of mask-covered views to keep a point.
        """
        n_views = len(Ks)
        n_points = len(self.xyz)
        hit_count = np.zeros(n_points, dtype=int)

        for i, (K, R_wc, t_wc, mask, (H, W)) in enumerate(
                tqdm(zip(Ks, Rs, ts, masks, image_sizes),
                     total=n_views, desc="[Forward Projection Filtering]")):

            # Camera-to-world → world-to-camera transform
            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc.reshape(3, 1)

            # Transform points to camera coordinates
            pts_cam = (R_cw @ self.xyz.T + t_cw).T
            in_front = pts_cam[:, 2] > 0
            if np.count_nonzero(in_front) == 0:
                continue

            pts_cam = pts_cam[in_front]
            proj = (K @ pts_cam.T).T
            u = np.round(proj[:, 0] / proj[:, 2]).astype(int)
            v = np.round(proj[:, 1] / proj[:, 2]).astype(int)
            z = pts_cam[:, 2]

            valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if np.count_nonzero(valid) == 0:
                continue

            u, v, z = u[valid], v[valid], z[valid]
            idx_visible = np.where(in_front)[0][valid]

            # --- Visibility check (Z-buffer simulation) ---
            depth_map = np.full((H, W), np.inf, dtype=np.float32)
            np.minimum.at(depth_map, (v, u), z)
            visible = z <= depth_map[v, u] + z_buffer_tol

            # --- Mask check ---
            in_mask = mask[v, u]
            visible_in_mask = visible & in_mask

            # Update view hits
            hit_count[idx_visible[visible_in_mask]] += 1

        # Keep points visible in enough mask-covered views
        keep_mask = hit_count >= min_views
        removed = n_points - np.sum(keep_mask)
        self.points = self.points[keep_mask]
        print(f"[INFO] Forward-projection filtering complete. Removed {removed} / {n_points} points. Remaining: {len(self.points)}")

    def save(self, out_path):
        el = PlyElement.describe(self.points, "vertex")
        PlyData([el], text=False).write(out_path)
        print(f"[INFO] Saved filtered PLY: {out_path}")


# ------------------ TOP-LEVEL FUNCTION ------------------

def filter_ply_with_forward_projection(colmap_txt_dir, ply_path, mask_dir,
                                       out_path="scene_mask_filtered_forward.ply",
                                       z_buffer_tol=0.01,
                                       min_views=1):
    """Forward projection mask filtering for Gaussian Splatting PLYs."""
    cameras, images, _ = read_model(str(colmap_txt_dir), ext=".txt")

    Ks, Rs, ts, names, image_sizes = [], [], [], [], []
    for img in images.values():
        cam = cameras[img.camera_id]
        if cam.model not in ("PINHOLE", "SIMPLE_PINHOLE"):
            continue
        fx = cam.params[0]
        fy = cam.params[1] if cam.model == "PINHOLE" else fx
        cx, cy = cam.params[-2], cam.params[-1]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        Ks.append(K)
        Rs.append(GaussianPLYMaskProjector.qvec2rotmat(img.qvec))
        ts.append(img.tvec)
        names.append(img.name)
        image_sizes.append((int(cy * 2), int(cx * 2)))

    masks, valid_names = GaussianPLYMaskProjector.load_masks_from_dir(mask_dir, names)
    Ks_f, Rs_f, ts_f, masks_f, sizes_f = [], [], [], [], []
    for K, R, t, name, size in zip(Ks, Rs, ts, names, image_sizes):
        if name in valid_names:
            Ks_f.append(K)
            Rs_f.append(R)
            ts_f.append(t)
            masks_f.append(masks[name])
            sizes_f.append(masks[name].shape)

    filter_obj = GaussianPLYMaskProjector(ply_path)
    filter_obj.filter_with_forward_projection(Ks_f, Rs_f, ts_f, masks_f, sizes_f,
                                              z_buffer_tol=z_buffer_tol, min_views=min_views)
    filter_obj.save(out_path)
    return out_path
