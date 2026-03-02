#!/usr/bin/env python3
import os
import re
import json
import argparse
import numpy as np
import torch
from argparse import Namespace
from PIL import Image, ImageDraw
from plyfile import PlyData

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render

from utils.graphics_utils import fov2focal
from utils.system_utils import mkdir_p

try:
    import open3d as o3d
    HAVE_O3D = True
except Exception:
    HAVE_O3D = False


# -----------------------------
# Small helpers
# -----------------------------
def _safe_name(s):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))


def _ensure_dir(p):
    if p:
        mkdir_p(p)


def _tensor_chw_to_u8_hwc(img_chw):
    """[3,H,W] float -> (H,W,3) uint8"""
    img = img_chw.detach().float().clamp(0, 1).cpu().numpy()
    img = (img * 255.0 + 0.5).astype(np.uint8)
    return np.transpose(img, (1, 2, 0))


def _save_u8_rgb(path, rgb_u8):
    _ensure_dir(os.path.dirname(path))
    Image.fromarray(rgb_u8, mode="RGB").save(path)


def _save_u8_rgba(path, rgba_u8):
    _ensure_dir(os.path.dirname(path))
    Image.fromarray(rgba_u8, mode="RGBA").save(path)


def _list_iterations(point_cloud_root):
    iters = []
    if os.path.isdir(point_cloud_root):
        for d in os.listdir(point_cloud_root):
            if d.startswith("iteration_"):
                try:
                    iters.append(int(d.split("_")[-1]))
                except Exception:
                    pass
    return sorted(iters)


def _resolve_iter_dir(model_dir, ply_iteration):
    pc_root = os.path.join(model_dir, "point_cloud")
    if ply_iteration < 0:
        iters = _list_iterations(pc_root)
        if not iters:
            raise FileNotFoundError("No iteration_* found under: %s" % pc_root)
        ply_iteration = iters[-1]
    iter_dir = os.path.join(model_dir, "point_cloud", "iteration_%d" % ply_iteration)
    if not os.path.isdir(iter_dir):
        raise FileNotFoundError("Iteration dir not found: %s" % iter_dir)
    return iter_dir, ply_iteration


def _move_gaussians_to_cuda(g):
    dev = torch.device("cuda")
    for a in [
        "_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation",
        "semantic_mask", "instance_ids", "cluster_ids", "cluster_id", "instance_embed",
    ]:
        if hasattr(g, a):
            t = getattr(g, a)
            if torch.is_tensor(t):
                setattr(g, a, t.to(dev))
    return g


def _load_ply_xyz_and_cluster(inst_ply_path, cluster_field="cluster_id"):
    ply = PlyData.read(inst_ply_path)
    v = ply["vertex"].data
    if cluster_field not in v.dtype.names:
        raise ValueError("[PLY] Missing '%s' in: %s" % (cluster_field, inst_ply_path))
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
    cid = v[cluster_field].astype(np.int32)
    return xyz, cid


def _set_background_exact(rgb_u8, fg_mask_hw, bg_rgb_u8=(255, 255, 255)):
    """Set background pixels to exact bg_rgb_u8 where fg_mask_hw is False."""
    out = rgb_u8.copy()
    out[~fg_mask_hw] = np.array(bg_rgb_u8, dtype=np.uint8)
    return out


def _force_pure_white(rgb_u8, thr=252):
    """Clamp near-white pixels to pure white."""
    rgb = rgb_u8.copy()
    m = (rgb[:, :, 0] >= thr) & (rgb[:, :, 1] >= thr) & (rgb[:, :, 2] >= thr)
    rgb[m] = 255
    return rgb


def _bgdiff_fgmask(img_pts_u8, img_bg_u8, tol=2):
    a = img_pts_u8.astype(np.int16)
    b = img_bg_u8.astype(np.int16)
    d = np.abs(a - b)
    return (d[:, :, 0] > tol) | (d[:, :, 1] > tol) | (d[:, :, 2] > tol)


# -----------------------------
# Feature override (render colors from arbitrary Nx3 RGB)
# -----------------------------
def _override_gaussian_colors(gaussians, rgb_n3):
    """
    Temporarily override SH features so render() outputs rgb_n3 as color.
    Returns a restore() closure.
    """
    assert rgb_n3.ndim == 2 and rgb_n3.shape[1] == 3
    device = gaussians.get_xyz.device
    rgb_n3 = rgb_n3.to(device=device, dtype=torch.float32).clamp(0, 1)

    if not hasattr(gaussians, "_features_dc") or gaussians._features_dc is None:
        raise RuntimeError("GaussianModel has no _features_dc to override.")

    old_dc = gaussians._features_dc
    old_rest = getattr(gaussians, "_features_rest", None)

    new_dc = old_dc.clone()
    if new_dc.ndim == 3:
        if new_dc.shape[1] == 3:
            new_dc[:, :, 0] = rgb_n3
        elif new_dc.shape[2] == 3:
            new_dc[:, 0, :] = rgb_n3
        else:
            raise RuntimeError("Unsupported _features_dc shape: %s" % (tuple(new_dc.shape),))
    elif new_dc.ndim == 2:
        if new_dc.shape[1] != 3:
            raise RuntimeError("Unsupported _features_dc shape: %s" % (tuple(new_dc.shape),))
        new_dc[:] = rgb_n3
    else:
        raise RuntimeError("Unsupported _features_dc ndim: %d" % new_dc.ndim)

    gaussians._features_dc = new_dc

    if old_rest is not None and torch.is_tensor(old_rest):
        gaussians._features_rest = torch.zeros_like(old_rest)

    def restore():
        gaussians._features_dc = old_dc
        if old_rest is not None:
            gaussians._features_rest = old_rest

    return restore


# -----------------------------
# Cluster colormap
# -----------------------------
def _make_cluster_cmap(cluster_ids_np, seed=42):
    rng = np.random.default_rng(seed)
    uniq = np.unique(cluster_ids_np).tolist()
    cmap = {}
    for cid in uniq:
        cid = int(cid)
        if cid < 0:
            cmap[cid] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            c = rng.random(3).astype(np.float32)
            c = np.clip(c, 0.20, 1.0)
            cmap[cid] = c
    return cmap


def _clusters_to_rgb_per_gaussian(cluster_ids_np, cmap):
    rgb = np.zeros((cluster_ids_np.shape[0], 3), dtype=np.float32)
    for i, cid in enumerate(cluster_ids_np):
        rgb[i] = cmap.get(int(cid), np.array([0.0, 0.0, 0.0], dtype=np.float32))
    return torch.from_numpy(rgb)


# -----------------------------
# Semantic: CloudCompare-like 256-step ramp (blue->green->yellow->red)
# -----------------------------
def _semantic_to_rgb_cloudcompare(sem_1d, gamma=1.0, intensity_power=1.0):
    s = sem_1d.clamp(0, 1)
    if float(gamma) != 1.0:
        s = s.pow(float(gamma)).clamp(0, 1)

    q = torch.round(s * 255.0).clamp(0, 255) / 255.0
    N = q.numel()
    rgb = torch.zeros((N, 3), device=q.device, dtype=torch.float32)

    t0 = 1.0 / 3.0
    t1 = 2.0 / 3.0

    m0 = (q <= t0)
    m1 = (q > t0) & (q <= t1)
    m2 = (q > t1)

    if m0.any():
        u = (q[m0] / t0).clamp(0, 1)
        rgb[m0, 1] = u
        rgb[m0, 2] = 1.0 - u

    if m1.any():
        u = ((q[m1] - t0) / (t1 - t0)).clamp(0, 1)
        rgb[m1, 0] = u
        rgb[m1, 1] = 1.0

    if m2.any():
        u = ((q[m2] - t1) / (1.0 - t1)).clamp(0, 1)
        rgb[m2, 0] = 1.0
        rgb[m2, 1] = 1.0 - u

    if float(intensity_power) != 1.0:
        inten = q.pow(float(intensity_power)).unsqueeze(1)
        rgb = rgb * inten

    return rgb.clamp(0.0, 1.0)


# -----------------------------
# Masks (3DGS)
# -----------------------------
def _extract_render_mask(out_dict, bg_rgb_u8):
    for k in ["accumulation", "alpha", "accumulated_alpha", "opacity"]:
        if k in out_dict and out_dict[k] is not None and torch.is_tensor(out_dict[k]):
            a = out_dict[k]
            if a.ndim == 3:
                a = a[0]
            if a.ndim == 2:
                return (a.detach().float().cpu().numpy() > 0.01)

    rgb = _tensor_chw_to_u8_hwc(out_dict["render"])
    bg = np.array(bg_rgb_u8, dtype=np.uint8).reshape(1, 1, 3)
    diff = np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).sum(axis=2)
    return diff > 2


# -----------------------------
# Open3D camera convention (DON'T CHANGE)
# -----------------------------
def _camera_to_o3d_intrinsic(cam):
    W = int(cam.image_width)
    H = int(cam.image_height)
    fx = float(fov2focal(float(cam.FoVx), W))
    fy = float(fov2focal(float(cam.FoVy), H))
    cx = 0.5 * float(W)
    cy = 0.5 * float(H)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    return intrinsic, W, H


def _camera_w2c_from_3dgs(cam):
    wvt = cam.world_view_transform
    if not torch.is_tensor(wvt):
        wvt = torch.as_tensor(wvt)
    return wvt.detach().cpu().numpy().astype(np.float64).T


def _build_o3d_pcd_from_cluster_ply(inst_ply_path, cmap):
    ply = PlyData.read(inst_ply_path)
    v = ply["vertex"].data
    if "cluster_id" not in v.dtype.names:
        raise ValueError("PLY has no 'cluster_id': %s" % inst_ply_path)

    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
    cids = v["cluster_id"].astype(np.int32)

    colors = np.zeros((xyz.shape[0], 3), dtype=np.float64)
    for i, cid in enumerate(cids):
        colors[i] = cmap.get(int(cid), np.array([0.0, 0.0, 0.0], dtype=np.float32)).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _as_u8_rgb(arr):
    """Accept uint8 or float images from O3D; output uint8 RGB."""
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    if a.dtype == np.uint8:
        return a
    a = np.clip(a, 0.0, 1.0)
    return (a * 255.0 + 0.5).astype(np.uint8)


def _as_f32_depth(arr):
    d = np.asarray(arr).astype(np.float32)
    if d.ndim == 3:
        d = d[:, :, 0]
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    return d


def _depth_bg_mask_mode(depth_f32, quant_frac=0.001):
    """
    Robust bg/fg split:
      - quantize depth
      - take mode bin as background depth (works even if bg depth is not 0/1)
    """
    d = depth_f32
    flat = d.reshape(-1)
    dmin = float(flat.min())
    dmax = float(flat.max())
    span = max(1e-6, dmax - dmin)

    if span < 1e-6:
        bg = np.ones_like(d, dtype=bool)
        return bg, ~bg

    step = span * float(quant_frac)
    qd = np.round((d - dmin) / step).astype(np.int64)

    qflat = qd.reshape(-1)
    qflat = qflat[(qflat >= 0) & (qflat < (1 << 31))]
    if qflat.size == 0:
        bg = np.ones_like(d, dtype=bool)
        return bg, ~bg

    mode_bin = int(np.bincount(qflat).argmax())
    bg_mask = (qd == mode_bin)
    fg_mask = ~bg_mask
    return bg_mask, fg_mask


def _recover_straight_rgba_from_white_black(img_white_u8, img_black_u8, alpha_eps=1e-6):
    """
    Halo-free recovery:
      Cw = a*Cf + (1-a)*1
      Cb = a*Cf
      => a = 1 - (Cw - Cb)
      => Cf = Cb / a
    """
    w = img_white_u8.astype(np.float32) / 255.0
    b = img_black_u8.astype(np.float32) / 255.0

    delta = np.clip(w - b, 0.0, 1.0)
    a = np.clip(1.0 - np.mean(delta, axis=2), 0.0, 1.0)

    a_safe = np.maximum(a, float(alpha_eps))
    cf = np.clip(b / a_safe[:, :, None], 0.0, 1.0)

    cf[a < (1.0 / 255.0)] = 0.0

    cf_u8 = (cf * 255.0 + 0.5).astype(np.uint8)
    a_u8 = (a * 255.0 + 0.5).astype(np.uint8)
    return cf_u8, a, a_u8


class _O3DOffscreenBank:
    def __init__(self, pcd, point_size=18.0):
        self.pcd = pcd
        self.point_size = float(point_size)
        self._renderers = {}

    def _get_renderer(self, W, H):
        key = (int(W), int(H))
        if key in self._renderers:
            return self._renderers[key]

        from open3d.visualization import rendering
        renderer = rendering.OffscreenRenderer(int(W), int(H))

        # Kill skybox/IBL/sun if present (prevents background vignette in many builds)
        try:
            renderer.scene.show_skybox(False)
        except Exception:
            pass
        try:
            renderer.scene.set_indirect_light(None)
        except Exception:
            pass
        try:
            renderer.scene.enable_sun_light(False)
        except Exception:
            pass

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = float(self.point_size)

        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("pcd", self.pcd, mat)

        self._renderers[key] = (renderer, mat)
        return renderer, mat

    def render_white_black_and_depth(self, cam):
        intrinsic, W, H = _camera_to_o3d_intrinsic(cam)
        W2C = _camera_w2c_from_3dgs(cam)

        renderer, _ = self._get_renderer(W, H)
        renderer.setup_camera(intrinsic, W2C)

        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        img_w = _as_u8_rgb(renderer.render_to_image())
        depth = _as_f32_depth(renderer.render_to_depth_image(z_in_view_space=True))

        renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
        img_b = _as_u8_rgb(renderer.render_to_image())

        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        return img_w, img_b, depth, W, H

    def close(self):
        for _, (renderer, _) in self._renderers.items():
            try:
                renderer.release_resources()
            except Exception:
                pass
        self._renderers.clear()


# -----------------------------
# Composite: RGB dim + Open3D points + centroid dots (+ optional OBBs)
# -----------------------------
def _compute_cluster_centroids(xyz_np, cid_np):
    uniq = np.unique(cid_np)
    uniq = [int(c) for c in uniq.tolist() if int(c) >= 0]
    uniq.sort()

    cent_xyz = []
    cent_cid = []
    for c in uniq:
        m = (cid_np == c)
        if not np.any(m):
            continue
        pts = xyz_np[m]
        if pts.shape[0] < 1:
            continue
        cent_xyz.append(pts.mean(axis=0))
        cent_cid.append(c)

    if len(cent_xyz) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.int32)

    return np.stack(cent_xyz, axis=0).astype(np.float64), np.array(cent_cid, dtype=np.int32)


def _project_world_to_pixel(pts_xyz, cam):
    W = int(cam.image_width)
    H = int(cam.image_height)
    fx = float(fov2focal(float(cam.FoVx), W))
    fy = float(fov2focal(float(cam.FoVy), H))
    cx = 0.5 * float(W)
    cy = 0.5 * float(H)

    W2C = _camera_w2c_from_3dgs(cam)

    M = pts_xyz.shape[0]
    pts_h = np.concatenate([pts_xyz.astype(np.float64), np.ones((M, 1), dtype=np.float64)], axis=1)
    cam_h = (W2C @ pts_h.T).T

    X = cam_h[:, 0]
    Y = cam_h[:, 1]
    Z = cam_h[:, 2]

    valid = (Z > 1e-6)
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    uv = np.stack([u, v], axis=1)
    valid = valid & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv, valid


def _rgba_from_rgb_with_alpha(rgb_u8, alpha_float):
    a = int(np.clip(alpha_float, 0.0, 1.0) * 255.0 + 0.5)
    H, W, _ = rgb_u8.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb_u8
    rgba[:, :, 3] = a
    return rgba


def _composite_rgb_o3d_centroids(
    base_rgb_u8,
    o3d_rgba_u8,
    cent_uv,
    cent_cid,
    cmap,
    base_alpha=0.25,
    centroid_radius_px=10,
):
    # DON'T TOUCH THIS (your request)
    H, W, _ = base_rgb_u8.shape

    canvas = Image.fromarray(
        np.full((H, W, 4), (255, 255, 255, 255), dtype=np.uint8),
        mode="RGBA"
    )

    base_rgba = _rgba_from_rgb_with_alpha(base_rgb_u8, base_alpha)
    canvas = Image.alpha_composite(canvas, Image.fromarray(base_rgba, mode="RGBA"))

    if o3d_rgba_u8 is not None:
        canvas = Image.alpha_composite(canvas, Image.fromarray(o3d_rgba_u8, mode="RGBA"))

    draw = ImageDraw.Draw(canvas, mode="RGBA")
    r = int(max(1, centroid_radius_px))
    t = max(1, int(round(0.15 * r)))
    border_col = (0, 0, 0, 255)

    for (u, v), cid in zip(cent_uv, cent_cid):
        cid = int(cid)
        c = cmap.get(cid, np.array([1.0, 1.0, 1.0], dtype=np.float32))
        fill_col = (int(c[0]*255+0.5), int(c[1]*255+0.5), int(c[2]*255+0.5), 255)

        x = int(u); y = int(v)
        draw.ellipse([x-(r+t), y-(r+t), x+(r+t), y+(r+t)], fill=border_col, outline=None)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=fill_col, outline=None)

    return np.array(canvas.convert("RGB"), dtype=np.uint8)


# -----------------------------
# Optional: draw 3D oriented bounding boxes
# -----------------------------
def _load_obbs_json(obbs_json_path):
    if obbs_json_path is None or (not os.path.exists(obbs_json_path)):
        return []
    with open(obbs_json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ["obbs", "boxes", "bboxes", "oriented_bboxes", "obb_list"]:
            if k in data and isinstance(data[k], list):
                return data[k]
        out = []
        for k, v in data.items():
            if isinstance(v, dict) and ("center" in v or "extent" in v):
                vv = dict(v)
                if "cluster_id" not in vv:
                    try:
                        vv["cluster_id"] = int(k)
                    except Exception:
                        pass
                out.append(vv)
        return out
    return []


def _as_mat3(R):
    if R is None:
        return np.eye(3, dtype=np.float64)
    R = np.array(R, dtype=np.float64)
    if R.shape == (3, 3):
        return R
    if R.size == 9:
        return R.reshape(3, 3)
    return np.eye(3, dtype=np.float64)


def _obb_corners(center, extent, R):
    c = np.array(center, dtype=np.float64).reshape(3)
    e = np.array(extent, dtype=np.float64).reshape(3)
    R = _as_mat3(R)
    he = 0.5 * e
    signs = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
    ], dtype=np.float64)
    local = signs * he.reshape(1, 3)
    world = (R @ local.T).T + c.reshape(1, 3)
    return world


def _draw_obbs_on_rgb(rgb_u8, cam, obbs, cmap, line_w=3, black_border=True):
    if obbs is None or len(obbs) == 0:
        return rgb_u8

    img = Image.fromarray(rgb_u8, mode="RGB")
    draw = ImageDraw.Draw(img, mode="RGB")

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for b in obbs:
        center = b.get("center", b.get("centroid", None))
        extent = b.get("extent", b.get("size", None))
        R = b.get("R", b.get("rotation", b.get("rot", None)))
        cid = b.get("cluster_id", b.get("id", -1))
        try:
            cid = int(cid)
        except Exception:
            cid = -1
        if center is None or extent is None:
            continue

        corners = _obb_corners(center, extent, R)
        uv, valid = _project_world_to_pixel(corners, cam)

        c = cmap.get(cid, np.array([1.0, 1.0, 1.0], dtype=np.float32))
        col = (int(c[0] * 255 + 0.5), int(c[1] * 255 + 0.5), int(c[2] * 255 + 0.5))

        for (i0, i1) in edges:
            if not (valid[i0] and valid[i1]):
                continue
            x0, y0 = int(uv[i0, 0]), int(uv[i0, 1])
            x1, y1 = int(uv[i1, 0]), int(uv[i1, 1])
            if black_border:
                draw.line([x0, y0, x1, y1], fill=(0, 0, 0), width=int(line_w + 2))
            draw.line([x0, y0, x1, y1], fill=col, width=int(line_w))

    return np.array(img, dtype=np.uint8)


# -----------------------------
# Scene init (cameras only)
# -----------------------------
def initialize_scene_for_cameras(colmap_dir, model_dir, sh_degree, load_iteration, load_filtered, images_folder):
    dataset = Namespace(
        sh_degree=int(sh_degree),
        source_path=os.path.abspath(colmap_dir),
        model_path=os.path.abspath(model_dir),
        images=images_folder,
        depths="",
        resolution=1.0,
        white_background=False,
        train_test_exp=False,
        data_device="cuda",
        eval=False,
    )

    dummy = GaussianModel(sh_degree=dataset.sh_degree)
    scene = Scene(
        dataset,
        dummy,
        load_iteration=load_iteration,
        shuffle=False,
        resolution_scales=[1.0],
        mask_dir=None,
        load_filtered=load_filtered,
    )
    return scene, dataset


class _DummyPipe:
    debug = False
    antialiasing = False
    compute_cov3D_python = False
    convert_SHs_python = False


def main():
    parser = argparse.ArgumentParser(
        "Export: RGB+SEM (black bg) + instances (white bg) + Open3D (true RGBA no-bg -> exact white) + composite (+centroids [+OBB])"
    )

    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--ply_iteration", type=int, default=-1)

    parser.add_argument("--images_folder", type=str, default="images")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--load_filtered", action="store_true")

    parser.add_argument("--base_ply_name", type=str, default="point_cloud.ply")
    parser.add_argument("--inst_ply_name", type=str, default="filtered_scene_clusters_cc_no_bkg.ply")

    parser.add_argument("--semantic_use_sigmoid", action="store_true")
    parser.add_argument("--semantic_scale", type=float, default=1.0)
    parser.add_argument("--semantic_gamma", type=float, default=1.0)
    parser.add_argument("--semantic_intensity_power", type=float, default=1.0)

    # -----------------------------
    # LEGACY CLI COMPAT
    # -----------------------------
    parser.add_argument("--overlay_alpha", type=float, default=None,
                        help="LEGACY alias for --composite_o3d_alpha (kept for CLI-compat).")
    parser.add_argument("--o3d_disable_post", action="store_true",
                        help="LEGACY: disable white/black recovery (still forces bg white via depth mask).")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--o3d_force_pure_white", action="store_true", default=True,
                   help="LEGACY: clamp near-white to pure white for Open3D whitebg output (default: ON).")
    g.add_argument("--o3d_no_force_pure_white", dest="o3d_force_pure_white", action="store_false",
                   help="Disable pure-white clamping.")
    parser.add_argument("--o3d_white_thr", type=int, default=252)

    # Your CLI uses this name:
    parser.add_argument("--o3d_black_bg_tol", type=int, default=None,
                        help="LEGACY alias for --o3d_bgdiff_tol (accepted).")
    parser.add_argument("--o3d_bgdiff_tol", type=int, default=2,
                        help="Tolerance for bg subtraction (used only in legacy bg-diff fallback).")

    # NEW (kept)
    parser.add_argument("--o3d_alpha_bg_thr", type=int, default=2,
                        help="Pixels with recovered alpha <= this (0..255) are treated as pure background (default: 2).")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--point_size", type=float, default=8.0, help="Open3D point size (MANDATORY for composite)")
    parser.add_argument("--out_subdir", type=str, default="output/paper_images_simple")

    # Open3D outputs
    parser.add_argument("--o3d_save_rgba", action="store_true",
                        help="Also save Open3D straight RGBA with transparent background (no halos).")

    # composite
    parser.add_argument("--composite_base_alpha", type=float, default=0.25)
    parser.add_argument("--composite_o3d_alpha", type=float, default=0.60)
    parser.add_argument("--centroid_radius_px", type=int, default=10)

    # OBB drawing
    parser.add_argument("--draw_obbs", action="store_true")
    parser.add_argument("--obbs_json", type=str, default=None)
    parser.add_argument("--obb_line_width", type=int, default=3)

    args = parser.parse_args()

    # legacy mapping
    if args.overlay_alpha is not None:
        args.composite_o3d_alpha = float(args.overlay_alpha)
    if args.o3d_black_bg_tol is not None:
        args.o3d_bgdiff_tol = int(args.o3d_black_bg_tol)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    iter_dir, it = _resolve_iter_dir(args.model_dir, args.ply_iteration)
    base_ply = os.path.join(iter_dir, args.base_ply_name)
    inst_ply = os.path.join(iter_dir, args.inst_ply_name)

    if not os.path.exists(base_ply):
        raise FileNotFoundError("Base PLY not found: %s" % base_ply)
    if not os.path.exists(inst_ply):
        raise FileNotFoundError("Instance PLY not found: %s" % inst_ply)

    print("[INFO] iteration=%d" % it)
    print("[INFO] base_ply: %s" % base_ply)
    print("[INFO] inst_ply: %s" % inst_ply)

    if args.obbs_json is None:
        guess = os.path.splitext(inst_ply)[0] + "_obbs.json"
        if os.path.exists(guess):
            args.obbs_json = guess
    obbs = _load_obbs_json(args.obbs_json) if args.draw_obbs else []

    out_root = os.path.join(args.model_dir, args.out_subdir)
    rgb_dir = os.path.join(out_root, "rgb_full_blackbg")
    sem_dir = os.path.join(out_root, "semantic_cloudcompare_blackbg")
    inst_real_dir = os.path.join(out_root, "instances_real_rgb_whitebg")
    inst_cluster_dir = os.path.join(out_root, "instances_cluster_colors_whitebg")
    o3d_dir = os.path.join(out_root, "open3d_instances_whitebg")
    o3d_rgba_dir = os.path.join(out_root, "open3d_instances_rgba")
    comp_dir = os.path.join(out_root, "composite_rgbdim_o3d_centroids")

    for d in [rgb_dir, sem_dir, inst_real_dir, inst_cluster_dir, o3d_dir, comp_dir]:
        mkdir_p(d)
    if args.o3d_save_rgba:
        mkdir_p(o3d_rgba_dir)

    print("[Step] Loading Scene (cameras)...")
    scene, dataset = initialize_scene_for_cameras(
        colmap_dir=args.colmap_dir,
        model_dir=args.model_dir,
        sh_degree=args.sh_degree,
        load_iteration=args.load_iteration,
        load_filtered=args.load_filtered,
        images_folder=args.images_folder,
    )

    print("[Step] Loading base GaussianModel...")
    base_g = GaussianModel(sh_degree=int(args.sh_degree))
    base_g.load_ply(base_ply, dataset.train_test_exp)
    base_g = _move_gaussians_to_cuda(base_g)

    # semantics source
    if hasattr(base_g, "semantic_mask") and base_g.semantic_mask is not None:
        sem_src = base_g.semantic_mask.detach().view(-1).float()
    else:
        fallback_sem = os.path.join(iter_dir, "scene_semantics_filtered.ply")
        if os.path.exists(fallback_sem):
            print("[WARN] base_ply has no semantic_mask; loading semantics from: %s" % fallback_sem)
            tmp = GaussianModel(sh_degree=int(args.sh_degree))
            tmp.load_ply(fallback_sem, dataset.train_test_exp)
            tmp = _move_gaussians_to_cuda(tmp)
            if hasattr(tmp, "semantic_mask") and tmp.semantic_mask is not None and tmp.semantic_mask.numel() == base_g.get_xyz.shape[0]:
                sem_src = tmp.semantic_mask.detach().view(-1).float()
            else:
                raise RuntimeError("Fallback semantic ply did not provide compatible semantic_mask.")
        else:
            raise RuntimeError("No semantic_mask found on base, and no scene_semantics_filtered.ply fallback.")

    if args.semantic_use_sigmoid:
        sem_src = torch.sigmoid(sem_src)
    sem_src = (sem_src * float(args.semantic_scale)).clamp(0, 1)

    sem_rgb = _semantic_to_rgb_cloudcompare(
        sem_src,
        gamma=float(args.semantic_gamma),
        intensity_power=float(args.semantic_intensity_power),
    )

    print("[Step] Loading instance GaussianModel (no_bkg)...")
    inst_g = GaussianModel(sh_degree=int(args.sh_degree))
    inst_g.load_ply(inst_ply, dataset.train_test_exp)
    inst_g = _move_gaussians_to_cuda(inst_g)

    inst_xyz_np, inst_cid_np = _load_ply_xyz_and_cluster(inst_ply, cluster_field="cluster_id")
    cmap = _make_cluster_cmap(inst_cid_np, seed=int(args.seed))
    inst_rgb_cluster = _clusters_to_rgb_per_gaussian(inst_cid_np, cmap=cmap)

    cent_xyz, cent_cid = _compute_cluster_centroids(inst_xyz_np, inst_cid_np)
    print("[INFO] clusters for centroid markers: %d" % cent_xyz.shape[0])

    # Open3D bank
    o3d_bank = None
    if HAVE_O3D:
        try:
            pcd = _build_o3d_pcd_from_cluster_ply(inst_ply, cmap=cmap)
            o3d_bank = _O3DOffscreenBank(pcd=pcd, point_size=float(args.point_size))
        except Exception as e:
            print("[WARN] Open3D setup failed (%s); skipping Open3D." % str(e))
    else:
        print("[WARN] open3d not importable; skipping Open3D.")

    pipe = _DummyPipe()
    bg_black = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    bg_white = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    bg_white_u8 = (255, 255, 255)

    cam_names = list(scene.gs_cameras.keys())
    print("[Step] Rendering %d views..." % len(cam_names))

    for view_key in cam_names:
        cam = scene.gs_cameras[view_key]
        fname = _safe_name(view_key)

        # --- Full RGB (BLACK BG) ---
        with torch.no_grad():
            out_rgb = render(cam, base_g, pipe, bg_black, contrib=False)
        rgb_u8 = _tensor_chw_to_u8_hwc(out_rgb["render"])
        _save_u8_rgb(os.path.join(rgb_dir, f"{fname}.png"), rgb_u8)

        # --- Full SEMANTIC (BLACK BG) ---
        restore_sem = _override_gaussian_colors(base_g, sem_rgb)
        try:
            with torch.no_grad():
                out_sem = render(cam, base_g, pipe, bg_black, contrib=False)
            sem_u8 = _tensor_chw_to_u8_hwc(out_sem["render"])
            _save_u8_rgb(os.path.join(sem_dir, f"{fname}_sem_cc.png"), sem_u8)
        finally:
            restore_sem()

        # --- Instances REAL COLORS (WHITE BG) ---
        with torch.no_grad():
            out_inst_real = render(cam, inst_g, pipe, bg_white, contrib=False)
        m_real = _extract_render_mask(out_inst_real, bg_rgb_u8=bg_white_u8)
        inst_real_u8 = _tensor_chw_to_u8_hwc(out_inst_real["render"])
        inst_real_u8 = _set_background_exact(inst_real_u8, m_real, bg_rgb_u8=bg_white_u8)
        _save_u8_rgb(os.path.join(inst_real_dir, f"{fname}_instances_real.png"), inst_real_u8)

        # --- Instances CLUSTER COLORS (WHITE BG) ---
        restore_inst = _override_gaussian_colors(inst_g, inst_rgb_cluster)
        try:
            with torch.no_grad():
                out_inst_cluster = render(cam, inst_g, pipe, bg_white, contrib=False)
            m_cl = _extract_render_mask(out_inst_cluster, bg_rgb_u8=bg_white_u8)
            inst_cluster_u8 = _tensor_chw_to_u8_hwc(out_inst_cluster["render"])
            inst_cluster_u8 = _set_background_exact(inst_cluster_u8, m_cl, bg_rgb_u8=bg_white_u8)
            _save_u8_rgb(os.path.join(inst_cluster_dir, f"{fname}_clusters.png"), inst_cluster_u8)
        finally:
            restore_inst()

        # --- Open3D ---
        o3d_rgba_for_composite = None
        if o3d_bank is not None:
            try:
                img_w_u8, img_b_u8, depth_f32, _, _ = o3d_bank.render_white_black_and_depth(cam)

                # Robust bg mask from depth (this is what kills the gray vignette)
                bg_mask, fg_mask = _depth_bg_mask_mode(depth_f32, quant_frac=0.001)

                if args.o3d_disable_post:
                    # Legacy path: no white/black recovery, but STILL enforce pure white background via depth mask
                    out_white_u8 = img_w_u8.copy()
                    out_white_u8[bg_mask] = 255

                    if args.o3d_force_pure_white:
                        out_white_u8 = _force_pure_white(out_white_u8, thr=int(args.o3d_white_thr))

                    _save_u8_rgb(os.path.join(o3d_dir, f"{fname}_o3d.png"), out_white_u8)

                    alpha_scale = float(np.clip(args.composite_o3d_alpha, 0.0, 1.0))
                    a_comp = int(alpha_scale * 255.0 + 0.5)

                    o3d_rgba_for_composite = np.zeros((out_white_u8.shape[0], out_white_u8.shape[1], 4), dtype=np.uint8)
                    o3d_rgba_for_composite[:, :, :3] = out_white_u8
                    o3d_rgba_for_composite[fg_mask, 3] = a_comp

                else:
                    # Full halo-free path + depth mask to force bg pure white
                    cf_u8, a_float, a_u8 = _recover_straight_rgba_from_white_black(img_w_u8, img_b_u8)

                    # snap tiny alpha (kept from your code)
                    bg_thr = int(np.clip(int(args.o3d_alpha_bg_thr), 0, 255))
                    a_u8[a_u8 <= bg_thr] = 0

                    # IMPORTANT: depth says bg -> force alpha 0 there (kills gray background)
                    a_u8[bg_mask] = 0
                    a_float = a_u8.astype(np.float32) / 255.0

                    if args.o3d_save_rgba:
                        rgba_save = np.zeros((cf_u8.shape[0], cf_u8.shape[1], 4), dtype=np.uint8)
                        rgba_save[:, :, :3] = cf_u8
                        rgba_save[:, :, 3] = a_u8
                        _save_u8_rgba(os.path.join(o3d_rgba_dir, f"{fname}_o3d_rgba.png"), rgba_save)

                    # exact white bg output
                    out_white_u8 = np.full_like(cf_u8, 255, dtype=np.uint8)
                    out_white_u8[fg_mask] = cf_u8[fg_mask]

                    if args.o3d_force_pure_white:
                        out_white_u8 = _force_pure_white(out_white_u8, thr=int(args.o3d_white_thr))

                    _save_u8_rgb(os.path.join(o3d_dir, f"{fname}_o3d.png"), out_white_u8)

                    # RGBA used for composite
                    alpha_scale = float(np.clip(args.composite_o3d_alpha, 0.0, 1.0))
                    a_comp_u8 = (a_float * alpha_scale * 255.0 + 0.5).astype(np.uint8)
                    a_comp_u8[bg_mask] = 0

                    o3d_rgba_for_composite = np.zeros((cf_u8.shape[0], cf_u8.shape[1], 4), dtype=np.uint8)
                    o3d_rgba_for_composite[:, :, :3] = cf_u8
                    o3d_rgba_for_composite[:, :, 3] = a_comp_u8

            except Exception as e:
                print("[WARN] Open3D render failed for %s: %s" % (view_key, str(e)))
                o3d_rgba_for_composite = None

        # --- Composite (dim RGB + O3D points + centroid dots [+ OBB]) ---
        if cent_xyz.shape[0] > 0:
            uv, valid = _project_world_to_pixel(cent_xyz, cam)
            uv_v = uv[valid]
            cid_v = cent_cid[valid]
        else:
            uv_v = np.zeros((0, 2), dtype=np.float64)
            cid_v = np.zeros((0,), dtype=np.int32)

        comp_u8 = _composite_rgb_o3d_centroids(
            base_rgb_u8=rgb_u8,
            o3d_rgba_u8=o3d_rgba_for_composite,
            cent_uv=uv_v,
            cent_cid=cid_v,
            cmap=cmap,
            base_alpha=float(args.composite_base_alpha),
            centroid_radius_px=int(args.centroid_radius_px),
        )

        if args.draw_obbs and len(obbs) > 0:
            comp_u8 = _draw_obbs_on_rgb(
                comp_u8, cam=cam, obbs=obbs, cmap=cmap,
                line_w=int(args.obb_line_width), black_border=True
            )

        _save_u8_rgb(os.path.join(comp_dir, f"{fname}_composite.png"), comp_u8)

    if o3d_bank is not None:
        o3d_bank.close()

    print("[OK] Done. Output written to: %s" % out_root)
    print("Folders:")
    print("  - %s" % rgb_dir)
    print("  - %s" % sem_dir)
    print("  - %s" % inst_real_dir)
    print("  - %s" % inst_cluster_dir)
    print("  - %s" % o3d_dir)
    if args.o3d_save_rgba:
        print("  - %s" % o3d_rgba_dir)
    print("  - %s" % comp_dir)


if __name__ == "__main__":
    main()