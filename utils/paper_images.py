import os
import re
import numpy as np
import torch
from PIL import Image
import open3d as o3d
from plyfile import PlyData

from gaussian_renderer import render
from utils.graphics_utils import fov2focal
from utils.system_utils import mkdir_p


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))


def _tensor_chw_to_pil_uint8(img_chw: torch.Tensor) -> Image.Image:
    # [3,H,W] float -> uint8 PIL
    img = img_chw.detach().float().clamp(0, 1).cpu().numpy()
    img = (img * 255.0 + 0.5).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))  # HWC
    return Image.fromarray(img)


def _make_cluster_color_map(cluster_ids: np.ndarray, seed: int = 42) -> dict:
    """
    cluster_id >= 0 -> random color (deterministic),
    cluster_id == -1 -> black
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(cluster_ids)
    uniq_pos = [int(x) for x in uniq.tolist() if int(x) >= 0]
    uniq_pos.sort()

    cmap = {-1: np.array([0.0, 0.0, 0.0], dtype=np.float32)}
    for cid in uniq_pos:
        cmap[cid] = rng.random(3).astype(np.float32)
    return cmap


def _colorize_cluster_labels(labels_hw: np.ndarray, cmap: dict) -> np.ndarray:
    """
    labels_hw: (H,W) int32
    returns: (H,W,3) uint8
    """
    H, W = labels_hw.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)

    # black background for -1 already default
    for cid, c in cmap.items():
        if cid < 0:
            continue
        m = (labels_hw == cid)
        if m.any():
            out[m] = (np.clip(c, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return out


def _build_cluster_pcd_from_ply_with_bg_black(ply_path: str, cmap: dict) -> o3d.geometry.PointCloud:
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data

    if "cluster_id" not in v.dtype.names:
        raise ValueError("PLY has no 'cluster_id' field. Did you save clustered PLY?")

    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
    cids = v["cluster_id"].astype(np.int32)

    colors = np.zeros((xyz.shape[0], 3), dtype=np.float64)
    for i, cid in enumerate(cids):
        c = cmap.get(int(cid), np.array([0.0, 0.0, 0.0], dtype=np.float32))
        colors[i] = c.astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _camera_to_o3d_params_3dgs(cam):
    """
    3DGS Camera -> Open3D pinhole params.

    cameras.py:
      self.world_view_transform = torch.tensor(getWorld2View2(...)).transpose(0,1).cuda()
    So actual W2C for Open3D (column-vector convention) is:
      W2C = cam.world_view_transform.T
    """
    W = int(cam.image_width)
    H = int(cam.image_height)

    fx = float(fov2focal(float(cam.FoVx), W))
    fy = float(fov2focal(float(cam.FoVy), H))
    cx = 0.5 * float(W)
    cy = 0.5 * float(H)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    wvt = cam.world_view_transform
    if not torch.is_tensor(wvt):
        wvt = torch.as_tensor(wvt)

    W2C = wvt.detach().cpu().numpy().astype(np.float64).T
    return intrinsic, W2C, W, H


def _capture_o3d_screenshot_pcd(
    pcd: o3d.geometry.PointCloud,
    out_path: str,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    extrinsic_w2c: np.ndarray,
    *,
    width: int,
    height: int,
    point_size: float = 2.0,
):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(width), height=int(height), visible=False)
    try:
        vis.add_geometry(pcd)

        opt = vis.get_render_option()
        opt.point_size = float(point_size)

        ctr = vis.get_view_control()
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic = intrinsic
        params.extrinsic = extrinsic_w2c
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

        vis.poll_events()
        vis.update_renderer()

        d = os.path.dirname(out_path)
        if d:
            mkdir_p(d)
        vis.capture_screen_image(out_path, do_render=True)
    finally:
        vis.destroy_window()


def export_paper_images_per_view(
    scene,
    gaussians,
    topK_full: dict,
    clustered_ply_path: str,
    model_dir: str,
    *,
    bg_color=(0, 0, 0),
    semantic_use_sigmoid: bool = True,
    point_size: float = 2.0,
    seed: int = 42,
):
    """
    Saves into: <model_dir>/output/paper_images/
      - rgb/<view>.png
      - semantic/<view>_sem.png
      - clusters_2d/<view>_clusters.png      (colored labels; -1 black)
      - clusters_o3d/<view>_o3d.png          (Open3D screenshot; -1 black)

    Camera ordering:
      Uses topK_full['cam_names'] (which your compute_topK... fills from scene.gs_cameras keys).
      Scene stores cameras in scene.gs_cameras_by_name and scene.gs_cameras = that dict.

    Cluster source for 2D:
      Uses gaussians.instance_ids (must exist; you set full_model.instance_ids = ids_final)
      projected via pixel_best_gaussian (topK_full).
    """
    # --------------------------
    # Output dirs
    # --------------------------
    out_root = os.path.join(model_dir, "output", "paper_images")
    rgb_dir = os.path.join(out_root, "rgb")
    sem_dir = os.path.join(out_root, "semantic")
    cl2d_dir = os.path.join(out_root, "clusters_2d")
    o3d_dir = os.path.join(out_root, "clusters_o3d")
    mkdir_p(rgb_dir); mkdir_p(sem_dir); mkdir_p(cl2d_dir); mkdir_p(o3d_dir)

    # --------------------------
    # Validate required topK_full fields
    # --------------------------
    cam_names = topK_full.get("cam_names", None)
    cam_hw = topK_full.get("cam_hw", None)
    pixel_best = topK_full.get("pixel_best_gaussian", None)
    if cam_names is None or pixel_best is None:
        raise KeyError("topK_full must contain at least: 'cam_names', 'pixel_best_gaussian'.")
    if cam_hw is None:
        cam_hw = [None] * len(cam_names)

    # --------------------------
    # Semantic per-gaussian
    # --------------------------
    if hasattr(gaussians, "get_sem") and gaussians.get_sem is not None:
        sem_all = gaussians.get_sem.detach().view(-1)
    elif hasattr(gaussians, "semantic_mask") and gaussians.semantic_mask is not None:
        sem_all = gaussians.semantic_mask.detach().view(-1)
    else:
        raise ValueError("No semantic field found on gaussians (expected get_sem or semantic_mask).")

    if semantic_use_sigmoid:
        sem_all = torch.sigmoid(sem_all)
    sem_all = sem_all.float().cpu()  # [N]

    # --------------------------
    # Cluster ids per gaussian (for 2D projection)
    # --------------------------
    if not hasattr(gaussians, "instance_ids") or gaussians.instance_ids is None:
        raise ValueError("gaussians.instance_ids not found. Set full_model.instance_ids before calling.")
    inst_ids = gaussians.instance_ids.detach().cpu().numpy().astype(np.int32)  # [N]

    # Colormap consistent across 2D and 3D
    cmap = _make_cluster_color_map(inst_ids, seed=seed)

    # Build Open3D point cloud once (uses PLY cluster_id; bg black)
    pcd = _build_cluster_pcd_from_ply_with_bg_black(clustered_ply_path, cmap=cmap)

    # --------------------------
    # Render pipe + bg
    # --------------------------
    class _DummyPipe:
        debug = False
        antialiasing = False
        compute_cov3D_python = False
        convert_SHs_python = False

    dummy_pipe = _DummyPipe()
    bg_col = torch.tensor(bg_color, dtype=torch.float32, device=gaussians.get_xyz.device)

    # --------------------------
    # Per-view export
    # --------------------------
    for cam_idx, view_key in enumerate(cam_names):
        # IMPORTANT: scene.gs_cameras keys are Path(cam.image_name).stem
        # and topK_full cam_names should match those keys if you used scene.gs_cameras.keys()
        if view_key not in scene.gs_cameras:
            # fallback: try stem extraction
            fallback = os.path.splitext(os.path.basename(str(view_key)))[0]
            if fallback in scene.gs_cameras:
                view_key = fallback
            else:
                print(f"[WARN] View '{view_key}' not found in scene.gs_cameras; skipping.")
                continue

        cam = scene.gs_cameras[view_key]
        fname = _safe_name(view_key)

        # --- RGB render ---
        with torch.no_grad():
            out = render(cam, gaussians, dummy_pipe, bg_col, contrib=False)

        if out is None or out.get("render", None) is None:
            print(f"[WARN] render() returned no image for {view_key}; skipping.")
            continue

        rgb = out["render"]  # [3,H,W] at cam.image_height/width
        rgb_path = os.path.join(rgb_dir, f"{fname}.png")
        _tensor_chw_to_pil_uint8(rgb).save(rgb_path)

        # Target resolution (same for all outputs)
        Ht, Wt = int(cam.image_height), int(cam.image_width)

        # Source resolution of pixel_best (from topK_full if present)
        hw = cam_hw[cam_idx]
        if hw is None:
            Hs, Ws = Ht, Wt
        else:
            Hs, Ws = int(hw[0]), int(hw[1])

        best_ids = pixel_best[cam_idx]
        if best_ids is None:
            print(f"[WARN] pixel_best_gaussian is None for {view_key}; skipping semantic/cluster2d.")
            continue

        best_ids = best_ids.detach().cpu().long().view(-1)  # [Hs*Ws] ideally

        # If pixel_best size mismatch, clamp and reshape best-effort
        P_expect = Hs * Ws
        if best_ids.numel() != P_expect:
            # try camera dims
            Hs, Ws = Ht, Wt
            P_expect = Hs * Ws
            if best_ids.numel() < P_expect:
                # pad with -1
                pad = -torch.ones((P_expect - best_ids.numel(),), dtype=torch.long)
                best_ids = torch.cat([best_ids, pad], dim=0)
            else:
                best_ids = best_ids[:P_expect]

        best_ids_hw = best_ids.view(Hs, Ws)

        # --- semantic render (grayscale) ---
        sem_src = torch.zeros((Hs, Ws), dtype=torch.float32)
        v = best_ids_hw >= 0
        if v.any():
            sem_src[v] = sem_all[best_ids_hw[v]]
        sem_u8 = (sem_src.numpy().clip(0, 1) * 255.0 + 0.5).astype(np.uint8)
        sem_img = Image.fromarray(sem_u8, mode="L")
        if (Hs, Ws) != (Ht, Wt):
            sem_img = sem_img.resize((Wt, Ht), resample=Image.NEAREST)
        sem_img.save(os.path.join(sem_dir, f"{fname}_sem.png"))

        # --- cluster instances render (colored labels; -1 black) ---
        labels = -np.ones((Hs, Ws), dtype=np.int32)
        v_np = v.numpy()
        if v_np.any():
            gidx = best_ids_hw.numpy()[v_np].astype(np.int64)
            gidx = np.clip(gidx, 0, inst_ids.shape[0] - 1)
            labels[v_np] = inst_ids[gidx]

        cl_rgb = _colorize_cluster_labels(labels, cmap=cmap)  # Hs,Ws,3 uint8
        cl_img = Image.fromarray(cl_rgb, mode="RGB")
        if (Hs, Ws) != (Ht, Wt):
            cl_img = cl_img.resize((Wt, Ht), resample=Image.NEAREST)
        cl_img.save(os.path.join(cl2d_dir, f"{fname}_clusters.png"))

        # --- Open3D screenshot from same camera pose at same resolution ---
        try:
            intrinsic, W2C, W, H = _camera_to_o3d_params_3dgs(cam)
            # ensure same resolution as camera image
            o3d_path = os.path.join(o3d_dir, f"{fname}_o3d.png")
            _capture_o3d_screenshot_pcd(
                pcd=pcd,
                out_path=o3d_path,
                intrinsic=intrinsic,
                extrinsic_w2c=W2C,
                width=W,
                height=H,
                point_size=float(point_size),
            )
        except Exception as e:
            print(f"[WARN] Open3D screenshot failed for {view_key}: {e}")

    print(f"[OK] Saved paper images to: {out_root}")
