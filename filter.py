import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from gaussian_renderer import render


def filter_and_save(scene, mask_dir, iteration, K=2, semantic_threshold=0.3):
    """
    Filter Gaussians based on:
      1. Contribution maps (which splats contribute to white mask pixels)
      2. Learned semantic field (gaussian.semantic_mask > threshold)
    
    Saves filtered point cloud and per-camera images showing:
      - Binary mask
      - Contribution overlay
    """
    gaussians = scene.gaussians
    all_xyz = gaussians.get_xyz
    N = all_xyz.shape[0]
    device = all_xyz.device
    keep_mask = torch.zeros(N, dtype=torch.bool, device=device)

    cameras = scene.getTrainCameras()
    output_dir = os.path.join(scene.model_path, "point_cloud", f"iteration_{iteration}")
    vis_dir = os.path.join(output_dir, "filter_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Dummy rendering pipeline
    class DummyPipe:
        debug = False
        antialiasing = False
        compute_cov3D_python = False
        convert_SHs_python = False

    dummy_pipe = DummyPipe()
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    # Semantic mask
    if hasattr(gaussians, "semantic_mask") and isinstance(gaussians.semantic_mask, torch.Tensor):
        semantic_mask = torch.sigmoid(gaussians.semantic_mask.squeeze())
        print(f"[INFO] Semantic mask found (range: {semantic_mask.min().item():.3f}–{semantic_mask.max().item():.3f})")
    else:
        print("[WARN] No semantic_mask field found in Gaussians — using only 2D contribution filtering.")
        semantic_mask = torch.zeros(N, device=device)

    # Iterate over cameras
    for cam in tqdm(cameras, desc="Filtering cameras"):
        render_out = render(
            cam,
            gaussians,
            dummy_pipe,
            bg_color,
            contrib=True,
            K=K
        )

        if not render_out or "contrib_indices" not in render_out or render_out["contrib_indices"] is None:
            print(f"[WARN] No contributor data for {cam.image_name}, skipping.")
            continue

        contrib_indices = render_out["contrib_indices"].detach().cpu().numpy().astype(np.int32)
        H, W, _ = contrib_indices.shape

        # Load mask
        image_basename = os.path.splitext(cam.image_name)[0]
        possible_exts = [".png", ".jpg", ".jpeg"]
        mask_path = None
        for ext in possible_exts:
            candidate = os.path.join(mask_dir, image_basename + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is None:
            print(f"[WARN] Mask not found for {cam.image_name}, skipping.")
            continue

        mask_img = np.array(Image.open(mask_path).convert("L").resize((W, H), Image.NEAREST))
        mask_white = mask_img > 127

        # Contribution map: which pixels contribute to kept Gaussians
        white_y, white_x = np.where(mask_white)
        if len(white_y) == 0:
            continue

        # Get Gaussian IDs that contributed to masked pixels
        gauss_ids = contrib_indices[white_y, white_x, :].reshape(-1)
        gauss_ids = gauss_ids[gauss_ids >= 0]
        if len(gauss_ids) == 0:
            continue

        unique_ids = np.unique(gauss_ids)
        keep_mask[unique_ids] = True

        # Save mask as before
        mask_save_path = os.path.join(vis_dir, f"{image_basename}_mask.png")
        Image.fromarray(mask_white.astype(np.uint8) * 255).save(mask_save_path)

        # Create a proper contribution map
        contrib_count = np.zeros((H, W), dtype=np.uint8)
        for k in range(contrib_indices.shape[2]):
            contrib_count[:, :, k] = (contrib_indices[:, :, k] >= 0).astype(np.uint8)

        contrib_map = contrib_count.sum(axis=2)
        if contrib_map.max() > 0:
            contrib_map = (contrib_map / contrib_map.max() * 255).astype(np.uint8)

        # Overlay mask on contribution map (mask in red)
        contrib_color = np.stack([contrib_map, contrib_map, contrib_map], axis=2)  # gray heatmap
        contrib_color[mask_white, 0] = 255    # red channel
        contrib_color[mask_white, 1] = 0      # green channel
        contrib_color[mask_white, 2] = 0      # blue channel

        contrib_save_path = os.path.join(vis_dir, f"{image_basename}_contrib_overlay.png")
        Image.fromarray(contrib_color).save(contrib_save_path)


    # Combine with semantic mask
    keep_mask &= (semantic_mask > semantic_threshold)

    kept_count = int(keep_mask.sum().item())
    print(f"[INFO] Keeping {kept_count}/{N} Gaussians after filtering")

    if kept_count == 0:
        print("[WARN] No Gaussians selected — saving original scene.")
        filtered_ply_path = os.path.join(output_dir, "scene_mask_filtered_renderer.ply")
        gaussians.save_ply(filtered_ply_path)
        return

    # Apply filtering to all per-Gaussian tensors
    for attr_name in list(vars(gaussians).keys()):
        attr = getattr(gaussians, attr_name)
        if isinstance(attr, torch.Tensor) and attr.shape[0] == keep_mask.shape[0]:
            setattr(gaussians, attr_name, attr[keep_mask])

    filtered_ply_path = os.path.join(output_dir, "scene_mask_filtered_renderer.ply")
    gaussians.save_ply(filtered_ply_path)
    print(f"[OK] Filtered scene saved to: {filtered_ply_path}")
    print(f"[OK] Saved mask and contribution images to: {vis_dir}")
