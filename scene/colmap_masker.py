import os
import cv2
import torch
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from utils.read_write_model import read_model, write_model


class ColmapMaskFilter:
    """
    COLMAP Mask Filtering and Mapping (dual-mask version)
    -----------------------------------------------------
    - Supports separate directories for semantic and instance masks
    - Maps COLMAP 3D points to per-pixel mask instances in a single vectorized pass
    - Can optionally save JSON mappings

    Args:
        base_dir: COLMAP project directory (with 'sparse/0')
        mask_dir: path to directory containing semantic masks (optional)
        mask_instances_dir: path to directory containing instance masks
        output_base_dir: path where mapping JSON will be saved
        downsample: factor for image coordinate scaling
    """

    def __init__(self, base_dir, mask_dir, mask_instances_dir, output_base_dir=None, sparse_subdir="sparse/0", downsample=1):
        self.base_dir = Path(base_dir)
        self.mask_dir = Path(mask_dir)
        self.mask_instances_dir = Path(mask_instances_dir)
        self.output_base_dir = (
            Path(output_base_dir)
            if output_base_dir else self.base_dir.parent / "colmap_masked"
        )
        self.colmap_txt_dir = self.base_dir / sparse_subdir
        self.output_dir = self.output_base_dir 

        # COLMAP containers
        self.cameras = None
        self.images = None
        self.points3D = None
        self.filtered_points3D = None

        # Mappings
        self.point_to_masks = defaultdict(list)
        self.mask_to_points = defaultdict(list)

        self.downsample = downsample

    # ----------------------------------------------------------------------
    def load_colmap_model(self):
        """Load COLMAP model"""
        print(f"[INFO] Loading COLMAP model from {self.colmap_txt_dir}")
        self.cameras, self.images, self.points3D = read_model(str(self.colmap_txt_dir), ext=".txt")
        print(f"→ Loaded {len(self.cameras)} cameras, {len(self.images)} images, {len(self.points3D)} points")

    # ----------------------------------------------------------------------
    def map_points_to_masks_tensor(self, save_json=True, save_path=None, device="cuda", progress=True):
        """
        Vectorized mapping of COLMAP 3D points to semantic & instance masks.

        Workflow:
        1. Semantic mask (mask_dir) filters points: only points seen by white pixels are kept.
        2. Instance masks (mask_instances_dir) map kept points → instance masks for tensor export.

        Returns:
            save_path: Path to JSON (if saved)
            pixel_to_point_tensor: [num_pixels, max_points_per_pixel]
            pixel_mask_vals: 1 for each valid pixel
            valid_mask: boolean mask for padding
        """
        if self.images is None or self.points3D is None:
            raise ValueError("Call load_colmap_model() first.")

        self.point_to_masks.clear()
        self.mask_to_points.clear()

        pts_ids = np.array(list(self.points3D.keys()))
        sort_idx = np.argsort(pts_ids)
        pts_ids = pts_ids[sort_idx]

        pixel_point_lists = []
        pixel_vals = []

        img_iter = tqdm(self.images.values(), desc="Mapping 3D points → mask pixels") if progress else self.images.values()

        # Supported image extensions
        img_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']

        for img in img_iter:
            image_name = Path(img.name).name
            stem = Path(image_name).stem

            # Projected COLMAP points
            p3d_ids = img.point3D_ids
            valid_idx = np.where(p3d_ids != -1)[0]
            if len(valid_idx) == 0:
                continue
            p3d_ids_valid = p3d_ids[valid_idx]
            pts_idx_global = np.searchsorted(pts_ids, p3d_ids_valid)
            xy_coords = img.xys[valid_idx]
            x_int = np.round(xy_coords[:, 0] / self.downsample).astype(int)
            y_int = np.round(xy_coords[:, 1] / self.downsample).astype(int)

            # -----------------------
            # 1) Semantic mask filtering
            # -----------------------
            sem_mask_files = []
            for ext in img_exts:
                sem_mask_files.extend(self.mask_dir.glob(f"{stem}{ext}"))
            if sem_mask_files:
                sem_mask_path = sem_mask_files[0]  # take first match
                sem_mask = cv2.imread(str(sem_mask_path), cv2.IMREAD_GRAYSCALE)
                if sem_mask is None or np.max(sem_mask) == 0:
                    continue
                h, w = sem_mask.shape
                x_clip = np.clip(x_int, 0, w-1)
                y_clip = np.clip(y_int, 0, h-1)
                sem_hits = sem_mask[y_clip, x_clip] > 127
                if not np.any(sem_hits):
                    continue
                # Keep only points seen by white pixels
                p3d_ids_valid = p3d_ids_valid[sem_hits]
                pts_idx_global = pts_idx_global[sem_hits]
                x_int = x_int[sem_hits]
                y_int = y_int[sem_hits]
            # else: no semantic mask, keep all points

            # -----------------------
            # 2) Instance mask mapping
            # -----------------------
            instance_masks = []
            for ext in img_exts:
                instance_masks.extend(sorted(self.mask_instances_dir.glob(f"{stem}_instance_*{ext}")))
            if not instance_masks:
                continue

            valid_masks = []
            valid_midx = []
            for midx, mask_path in enumerate(instance_masks):
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None or np.max(mask) == 0:
                    continue
                valid_masks.append(mask)
                valid_midx.append(midx)
            if not valid_masks:
                continue

            masks_stack = np.stack(valid_masks, axis=-1)
            h, w, n_masks = masks_stack.shape
            x_clip = np.clip(x_int, 0, w-1)
            y_clip = np.clip(y_int, 0, h-1)

            mask_vals_all = masks_stack[y_clip, x_clip, :] > 0
            any_hits = np.any(mask_vals_all, axis=1)
            if not np.any(any_hits):
                continue

            for j, midx in enumerate(valid_midx):
                hits_for_mask = np.nonzero(mask_vals_all[:, j])[0]
                if len(hits_for_mask) == 0:
                    continue
                mask_pids = p3d_ids_valid[hits_for_mask]

                # Register associations
                self.mask_to_points[(image_name, midx)].extend(mask_pids.tolist())
                for pid in mask_pids:
                    self.point_to_masks[pid].append((image_name, midx))

                pixel_point_lists.append(pts_idx_global[hits_for_mask])
                pixel_vals.append(np.ones(len(hits_for_mask), dtype=np.float32))

        if len(pixel_point_lists) == 0:
            raise RuntimeError("No valid pixel→point associations found.")

        # -----------------------
        # Tensorization
        # -----------------------
        num_pix = len(pixel_point_lists)
        max_len = max(len(p) for p in pixel_point_lists)

        pixel_to_point_tensor = torch.full((num_pix, max_len), -1, dtype=torch.long, device=device)
        valid_mask = torch.zeros((num_pix, max_len), dtype=torch.bool, device=device)
        pixel_mask_vals = torch.ones(num_pix, dtype=torch.float32, device=device)

        for i, pts_idx in enumerate(pixel_point_lists):
            l = len(pts_idx)
            pixel_to_point_tensor[i, :l] = torch.tensor(pts_idx, dtype=torch.long, device=device)
            valid_mask[i, :l] = True

        # -----------------------
        # JSON output
        # -----------------------
        if save_json:
            save_path = Path(save_path) if save_path else self.output_dir / "point_mask_mapping.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump({
                    "point_to_masks": {str(pid): v for pid, v in self.point_to_masks.items()},
                    "mask_to_points": {f"{k[0]}_{k[1]}": v for k, v in self.mask_to_points.items()}
                }, f, indent=2)
            print(f"[INFO] Saved mapping JSON: {save_path}")
        else:
            save_path = None
            print("[INFO] Skipped JSON saving.")

        print(f"[INFO] Mapped {num_pix} pixels → 3D points | Tensor shape: {tuple(pixel_to_point_tensor.shape)}")

        return save_path, pixel_to_point_tensor, pixel_mask_vals, valid_mask


