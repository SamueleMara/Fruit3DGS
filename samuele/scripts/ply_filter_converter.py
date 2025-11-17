from plyfile import PlyData, PlyElement
import numpy as np
import os
import open3d as o3d
import matplotlib.colors as mcolors
# from ply_mask_filter import filter_ply_with_masks,filter_ply_with_strict_masks,GaussianPLYMaskFilter
from ply_mask_filter import filter_ply_with_forward_projection
import cv2
from read_write_model import read_model

class GaussianSplatFilter:
    def __init__(self, ply_folder_path):
        self.ply_folder_path = ply_folder_path
        self.input_ply = os.path.join(ply_folder_path, "scene_mask_filtered.ply")
        # self.input_ply = os.path.join(ply_folder_path, "point_cloud.ply")
        self.filtered_ply = os.path.join(ply_folder_path, "scene_filtered.ply")
        self.rgb_ply = os.path.join(ply_folder_path, "scene_rgb.ply")
        self.original_rgb_ply = os.path.join(ply_folder_path, "point_cloud_rgb.ply")

        # HSV caches
        self._hsv_cache = None         # stores (r_all, g_all, b_all, h, s, v)
        self._hsv_masks = {}

    # -----------------------------
    # SH <-> RGB conversion
    # -----------------------------
    @staticmethod
    def SH2RGB(sh):
        C0 = 0.28209479177387814
        return sh * C0 + 0.5

    # -----------------------------
    # Debug helpers
    # -----------------------------
    @staticmethod
    def debug_rgb_stats(self, arr, name="[RGB Stats]"):
        if arr.size == 0:
            print(f"{name}: array is empty, skipping stats")
            return
        print(f"  {name}: min {arr.min():.4f}, max {arr.max():.4f}, mean {arr.mean():.4f}")
        for name, arr in zip(["R", "G", "B"], [r, g, b]):
            print(f"  {name}: min {arr.min():.4f}, max {arr.max():.4f}, "
                  f"mean {arr.mean():.4f}, median {np.median(arr):.4f}")
        rgb_sum = r + g + b
        print(f"  RGB sum: min {rgb_sum.min():.4f}, max {rgb_sum.max():.4f}, "
              f"mean {rgb_sum.mean():.4f}, median {np.median(rgb_sum):.4f}")
        print(f"  Non-black fraction: {(np.sum(rgb_sum > 0.05)/len(rgb_sum)):.2%}")

    @staticmethod
    def debug_hsv_stats(h, s, v, s_trshd=0.05, v_trshd=0.1, label="[HSV Stats]"):
        print(f"\n{label}")
        print(f"  H: min {h.min():.4f}, max {h.max():.4f}, mean {h.mean():.4f}, median {np.median(h):.4f}")
        print(f"  S: min {s.min():.4f}, max {s.max():.4f}, mean {s.mean():.4f}, median {np.median(s):.4f}")
        print(f"  V: min {v.min():.4f}, max {v.max():.4f}, mean {v.mean():.4f}, median {np.median(v):.4f}")
        print(f"  Fraction with V>{v_trshd:.3f}: {(np.mean(v>v_trshd)*100):.2f}%")
        print(f"  Fraction with S>{s_trshd:.3f}: {(np.mean(s>s_trshd)*100):.2f}%")

    # -----------------------------
    # HSV caching
    # -----------------------------
    def _compute_hsv_cache(self):
        """Compute and cache r,g,b,h,s,v for the input ply."""
        if self._hsv_cache is not None:
            return self._hsv_cache

        plydata = PlyData.read(self.input_ply)
        vertices = np.array(plydata['vertex'].data)
        r_all = np.array([p['red'] for p in self.points if 'red' in p.dtype.names])
        g_all = np.array([p['green'] for p in self.points if 'green' in p.dtype.names])
        b_all = np.array([p['blue'] for p in self.points if 'blue' in p.dtype.names])

        if r_all.size == 0 or g_all.size == 0 or b_all.size == 0:
            print("[WARNING] No RGB channels found in filtered PLY, skipping preview generation")
            return

        rgb = np.stack([r_all, g_all, b_all], axis=-1)
        hsv = mcolors.rgb_to_hsv(rgb)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        self._hsv_cache = (r_all, g_all, b_all, h, s, v)
        return self._hsv_cache

    def _get_hsv_mask_cached(self, remove_white=False, v_threshold=None, s_threshold=None, verbose=True):
        key = (remove_white, float(v_threshold) if v_threshold is not None else None, float(s_threshold) if s_threshold is not None else None)
        if key in self._hsv_masks:
            return self._hsv_masks[key]

        r_all, g_all, b_all, h, s, v = self._compute_hsv_cache()

        if remove_white:
            v_thresh = v_threshold if v_threshold is not None else 0.85
            s_thresh = s_threshold if s_threshold is not None else 0.15
        else:
            v_thresh = v_threshold if v_threshold is not None else 0.15
            s_thresh = s_threshold if s_threshold is not None else 0.08

        mask = (v > v_thresh) & (s > s_thresh)

        if verbose:
            self.debug_hsv_stats(h, s, v, s_trshd=s_thresh, v_trshd=v_thresh,
                                 label=f"[HSV Filter Stats] remove_white={remove_white}")
            print(f"[HSV Filter] V>{v_thresh:.3f}, S>{s_thresh:.3f}, remove_white={remove_white} -> kept {np.sum(mask)} / {len(r_all)} points")

        self._hsv_masks[key] = mask
        return mask

    # -----------------------------
    # RGB filter
    # -----------------------------
    def apply_rgb_filter(self, r, g, b, threshold=0.1, verbose=True):
        rgb_sum = r + g + b
        mask_rgb = rgb_sum > threshold
        if verbose:
            print(f"[RGB Filter] threshold={threshold:.3f} -> kept {np.sum(mask_rgb)} / {len(r)} points")
        return mask_rgb

    # -----------------------------
    # Save original RGB preview
    # -----------------------------
    def save_original_rgb(self, rgb_threshold=0.1, hsv=False, verbose=True,
                      hsv_s_threshold=None, hsv_v_threshold=None,
                      out_path=None):
        """
        Save an RGB preview of the original ply.
        - If hsv=True, keeps only points with s >= s_min and v_min <= v <= v_max.
        Pass hsv_s_threshold as s_min and hsv_v_threshold as (v_min, v_max).
        - If hsv=False, simple RGB sum > threshold filtering.
        """
        out_path = out_path or self.original_rgb_ply
        plydata = PlyData.read(self.input_ply)
        vertices = np.array(plydata['vertex'].data)

        r_all = np.clip(self.SH2RGB(vertices['f_dc_0'].astype(np.float64)), 0.0, 1.0)
        g_all = np.clip(self.SH2RGB(vertices['f_dc_1'].astype(np.float64)), 0.0, 1.0)
        b_all = np.clip(self.SH2RGB(vertices['f_dc_2'].astype(np.float64)), 0.0, 1.0)

        if verbose:
            self.debug_rgb_stats(r_all, g_all, b_all, label="[Original RGB Stats]")

        if hsv:
            # explicit thresholds
            s_min = hsv_s_threshold if hsv_s_threshold is not None else 0.08
            if isinstance(hsv_v_threshold, (tuple, list)):
                v_min, v_max = hsv_v_threshold
            else:
                v_min = hsv_v_threshold if hsv_v_threshold is not None else 0.0
                v_max = 1.0

            rgb = np.stack([r_all, g_all, b_all], axis=-1)
            hsv_vals = mcolors.rgb_to_hsv(rgb)
            h, s, v = hsv_vals[..., 0], hsv_vals[..., 1], hsv_vals[..., 2]

            mask = (s >= s_min) & (v >= v_min) & (v <= v_max)
            if verbose:
                self.debug_hsv_stats(h, s, v, s_trshd=s_min, v_trshd=v_min,
                                    label=f"[HSV Preview Filter] s_min={s_min}, v_min={v_min}, v_max={v_max}")
                print(f"[HSV Preview] kept {np.sum(mask)} / {len(r_all)} points")
        else:
            mask = self.apply_rgb_filter(r_all, g_all, b_all, threshold=rgb_threshold, verbose=verbose)

        # Save preview
        rgb_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        rgb_data = np.empty(np.sum(mask), dtype=rgb_dtype)
        rgb_data['x'], rgb_data['y'], rgb_data['z'] = vertices['x'][mask], vertices['y'][mask], vertices['z'][mask]
        rgb_data['red'], rgb_data['green'], rgb_data['blue'] = (
            (r_all[mask]*255).astype(np.uint8),
            (g_all[mask]*255).astype(np.uint8),
            (b_all[mask]*255).astype(np.uint8)
        )
        PlyData([PlyElement.describe(rgb_data, 'vertex')], text=False).write(out_path)
        if verbose:
            print(f"[INFO] Original RGB preview saved to: {out_path}")
        return out_path


    # -----------------------------
    # Main filtering pipeline
    # -----------------------------
    def filter_and_save(
        self,
        top_percent_remove=None,
        max_scale=None,
        min_opacity=None,
        aspect_percentile=None,
        use_variability_filter=False,
        variability_percentile=2.0,
        variability_threshold=None,
        percentile_radius=None,
        radius_scale=None,
        rgb_threshold=None,
        use_hsv=False,
        remove_white=False,
        hsv_v_threshold=None,
        hsv_s_threshold=None,
        verbose=True,
        out_filtered=None,
        out_rgb=None
    ):
        out_filtered = out_filtered or self.filtered_ply
        out_rgb = out_rgb or self.rgb_ply

        plydata = PlyData.read(self.input_ply)
        vertices = np.array(plydata['vertex'].data)
        n_original = len(vertices)
        if verbose:
            print(f"\n[PIPELINE] Original splats: {n_original}")

        # --- Geometry filters ---
        scales = np.exp(np.vstack([vertices['scale_0'], vertices['scale_1'], vertices['scale_2']]).T)
        max_scales = np.max(scales, axis=1)
        min_scales = np.min(scales, axis=1)
        aspect_ratios = max_scales / np.clip(min_scales, 1e-8, None)
        opacities = vertices['opacity']

        size_threshold = np.inf if not top_percent_remove else np.percentile(max_scales, max(0.0, min(100.0, 100.0 - top_percent_remove)))
        max_scale_filter = np.inf if not max_scale else max_scale
        aspect_threshold = np.inf if not aspect_percentile else np.percentile(aspect_ratios, max(0.0, min(100.0, aspect_percentile)))
        opacity_threshold = -np.inf if not min_opacity else min_opacity

        mask_geom = (max_scales <= size_threshold) & (max_scales <= max_scale_filter) & \
                    (aspect_ratios <= aspect_threshold) & (opacities >= opacity_threshold)
        if verbose:
            print(f"[DEBUG] After geometry filter: {np.sum(mask_geom)} / {n_original}")

        # --- Variability filter ---
        if use_variability_filter:
            coeff_names = [n for n in vertices.dtype.names if n.startswith("f_")]
            dc_names = sorted([n for n in coeff_names if "f_dc" in n], key=lambda s: int("".join(filter(str.isdigit, s)) or -1))
            rest_names = sorted([n for n in coeff_names if "f_rest" in n], key=lambda s: int("".join(filter(str.isdigit, s)) or -1))
            f_dc = np.vstack([vertices[name].astype(np.float64) for name in dc_names]).T
            f_rest = np.vstack([vertices[name].astype(np.float64) for name in rest_names]).T if rest_names else None

            dc_energy = np.linalg.norm(f_dc, axis=1)
            rest_energy = np.linalg.norm(f_rest, axis=1) if f_rest is not None else np.zeros_like(dc_energy)
            variability_ratio = np.divide(rest_energy, dc_energy, out=np.zeros_like(rest_energy), where=(dc_energy > 1e-8))
            var_thresh = (np.percentile(variability_ratio, variability_percentile) if variability_threshold is None else variability_threshold)
            mask_variability = variability_ratio > var_thresh if f_rest is not None else np.ones_like(dc_energy, bool)
            if verbose:
                print(f"[DEBUG] After variability filter: {np.sum(mask_variability)} / {len(mask_variability)}")
        else:
            mask_variability = np.ones(len(vertices), bool)

        mask_combined_pre = mask_geom & mask_variability
        if verbose:
            print(f"[DEBUG] After geom+var combined: {np.sum(mask_combined_pre)} / {n_original}")

        filtered_vertices_pre = vertices[mask_combined_pre]
        if len(filtered_vertices_pre) == 0:
            PlyData([PlyElement.describe(filtered_vertices_pre, 'vertex')], text=False).write(out_filtered)
            if verbose:
                print(f"[INFO] No points remain after geom+var filters. Wrote empty {out_filtered}")
            return out_filtered, None

        # --- Outlier removal ---
        if percentile_radius and radius_scale and radius_scale > 0:
            xyz = np.vstack((filtered_vertices_pre['x'], filtered_vertices_pre['y'], filtered_vertices_pre['z'])).T
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            max_scales_pre = np.max(np.exp(np.vstack([filtered_vertices_pre['scale_0'],
                                                    filtered_vertices_pre['scale_1'],
                                                    filtered_vertices_pre['scale_2']]).T), axis=1)
            radius_value = radius_scale * np.percentile(max_scales_pre, float(percentile_radius))
            min_points_in_radius = max(3, int(len(filtered_vertices_pre) * 0.01))
            if verbose:
                print(f"[DEBUG] radius outlier -> radius_value={radius_value:.6f}, min_points_in_radius={min_points_in_radius}")
            _, ind = pcd.remove_radius_outlier(nb_points=min_points_in_radius, radius=radius_value)
        else:
            ind = np.arange(len(filtered_vertices_pre))
            if verbose:
                print("[INFO] Skipping radius-based outlier removal.")

        final_idx = np.nonzero(mask_combined_pre)[0][ind]
        if verbose:
            print(f"[DEBUG] After outlier removal: {len(final_idx)}")

        # --- RGB/HSV filter ---
        r_all = np.clip(self.SH2RGB(vertices['f_dc_0'].astype(np.float64)), 0.0, 1.0)
        g_all = np.clip(self.SH2RGB(vertices['f_dc_1'].astype(np.float64)), 0.0, 1.0)
        b_all = np.clip(self.SH2RGB(vertices['f_dc_2'].astype(np.float64)), 0.0, 1.0)

        if use_hsv:
            # new explicit thresholds
            s_min = hsv_s_threshold if hsv_s_threshold is not None else 0.08
            v_min = hsv_v_threshold[0] if isinstance(hsv_v_threshold, (tuple, list)) else (hsv_v_threshold if hsv_v_threshold is not None else 0.0)
            v_max = hsv_v_threshold[1] if isinstance(hsv_v_threshold, (tuple, list)) else 1.0

            rgb = np.stack([r_all, g_all, b_all], axis=-1)
            hsv = mcolors.rgb_to_hsv(rgb)
            h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

            # explicit mask
            full_mask = (s >= s_min) & (v >= v_min) & (v <= v_max)

            mask_color = full_mask[final_idx]
            if verbose:
                print(f"[DEBUG] After HSV filter: kept {np.sum(mask_color)} / {len(final_idx)} points "
                    f"(s_min={s_min:.3f}, v_min={v_min:.3f}, v_max={v_max:.3f})")
        else:
            # simple RGB sum filter if HSV off
            rgb_thresh = rgb_threshold if rgb_threshold is not None else 0.1
            rgb_sum = r_all + g_all + b_all
            mask_color = rgb_sum[final_idx] > rgb_thresh
            if verbose:
                print(f"[DEBUG] After RGB filter: kept {np.sum(mask_color)} / {len(final_idx)} points")

        final_idx = final_idx[mask_color]

        if verbose:
            print(f"[SUMMARY] Final points kept: {len(final_idx)} / {n_original}")


        # --- Save filtered PLY ---
        filtered_vertices = vertices[final_idx]
        PlyData([PlyElement.describe(filtered_vertices, 'vertex')], text=False).write(out_filtered)
        if verbose:
            print(f"[INFO] Filtered PLY saved to: {out_filtered}")

        # --- Save RGB preview ---
        rgb_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        rgb_data = np.empty(len(final_idx), dtype=rgb_dtype)
        rgb_data['x'], rgb_data['y'], rgb_data['z'] = vertices['x'][final_idx], vertices['y'][final_idx], vertices['z'][final_idx]
        rgb_data['red'], rgb_data['green'], rgb_data['blue'] = (
            (r_all[final_idx]*255).astype(np.uint8),
            (g_all[final_idx]*255).astype(np.uint8),
            (b_all[final_idx]*255).astype(np.uint8)
        )
        PlyData([PlyElement.describe(rgb_data, 'vertex')], text=False).write(out_rgb)
        if verbose:
            print(f"[INFO] Final RGB preview saved to: {out_rgb}")

        return out_filtered, out_rgb

    # -----------------------------
    # Visualization
    # -----------------------------
    @staticmethod
    def visualize_rgb(rgb_ply):
        pcd_rgb = o3d.io.read_point_cloud(rgb_ply)
        print(f"[INFO] Showing RGB point cloud: {rgb_ply}")
        o3d.visualization.draw_geometries([pcd_rgb])



# -----------------------------
# Standalone usage example
# -----------------------------
if __name__ == "__main__":
    import os

    # === Input paths ===
    sparse_dir = '/media/samuelemara/Elements/Colmap/tree_01_fruit_nerf_imported_colmap/sparse/0'
    out_ply_dir = '/media/samuelemara/Elements/Colmap/tree_01_fruit_nerf_imported_colmap/output'
    mask_dir = '/home/samuelemara/FruitNeRF/FruitNeRF_Real/FruitNeRF_Dataset/tree_01/semantics_sam'
    os.makedirs(out_ply_dir, exist_ok=True)

    # === Core file paths ===
    ply_path = os.path.join(out_ply_dir, "point_cloud.ply")
    out_ply_lenient = os.path.join(out_ply_dir, "scene_mask_filtered.ply")
    out_ply_strict = os.path.join(out_ply_dir, "scene_mask_filtered_strict.ply")

    print("[INFO] Running forward-projection mask filtering...")
    out_ply_forward = os.path.join(out_ply_dir, "scene_mask_filtered_forward.ply")
    filter_ply_with_forward_projection(
        colmap_txt_dir=sparse_dir,
        ply_path=ply_path,
        mask_dir=mask_dir,
        out_path=out_ply_forward,
        z_buffer_tol=0.01,
        min_views=2
    )
    print(f"[INFO] Forward-projection mask-filtered PLY saved to: {out_ply_forward}")

    # === Gaussian splat filtering ===
    filterer = GaussianSplatFilter(out_ply_dir)
    filterer.input_ply = out_ply_strict  # use strictly filtered PLY for further processing

    # --- Save HSV-based RGB previews ---
    orig_black_preview = os.path.join(out_ply_dir, "point_cloud_rgb_black_preview.ply")
    orig_white_preview = os.path.join(out_ply_dir, "point_cloud_rgb_white_preview.ply")

    print("[INFO] Generating HSV-based RGB previews...")
    filterer.save_original_rgb(
        hsv=True,
        hsv_s_threshold=0.1,
        hsv_v_threshold=(0.1, 1.0),
        out_path=orig_black_preview
    )
    filterer.save_original_rgb(
        hsv=True,
        hsv_s_threshold=0.1,
        hsv_v_threshold=(0.0, 0.9),
        out_path=orig_white_preview
    )

    filterer.visualize_rgb(orig_black_preview)
    filterer.visualize_rgb(orig_white_preview)

    # === Step 4: Full Gaussian splat filtering pipeline ===
    out_filtered_final = os.path.join(out_ply_dir, "scene_filtered_custom.ply")
    out_rgb_final = os.path.join(out_ply_dir, "scene_rgb_custom.ply")

    print("[INFO] Running final Gaussian splat filtering pipeline...")
    filtered_ply, rgb_ply = filterer.filter_and_save(
        aspect_percentile=90,
        use_hsv=True,
        hsv_s_threshold=0.1,
        hsv_v_threshold=(0.1, 0.9),
        verbose=True,
        out_filtered=out_filtered_final,
        out_rgb=out_rgb_final
    )

    print(f"[INFO] Final filtered PLY saved to: {filtered_ply}")
    print(f"[INFO] Final RGB PLY saved to: {rgb_ply}")
    filterer.visualize_rgb(rgb_ply)
