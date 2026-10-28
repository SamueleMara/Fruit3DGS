# output_filter_standalone.py
import os
import torch
import glob
from random import randint
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import torch.nn.functional as F


               
# -----------------------------
# Build the mask index
# -----------------------------
def build_mask_index(mask_dir):
    """
    Build an index: lower(basename_without_ext) -> full mask path.
    If duplicates exist, the first one found is used.
    """
    exts = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]
    # include upper-case variants
    exts += [e.upper() for e in exts]

    idx = {}
    for ext in exts:
        for p in glob.glob(os.path.join(mask_dir, f"*.{ext}")):
            base = os.path.splitext(os.path.basename(p))[0].lower()
            if base not in idx:
                idx[base] = p
    return idx
               
# -----------------------------
# Load the mask
# -----------------------------
def load_mask_as_bool(mask_path, target_hw=None):
    """
    Loads mask image and returns boolean mask (white=True).
    If target_hw=(H,W) is provided, resize with NEAREST to match.
    """
    m = Image.open(mask_path)

    # Convert to single channel
    # - if RGB, take first channel
    # - if L already, ok
    m_np = np.array(m)
    if m_np.ndim == 3:
        m_np = m_np[..., 0]

    # normalize to [0,1] then threshold
    if m_np.dtype != np.float32:
        m_np = m_np.astype(np.float32)

    # if it's 0..255 typical
    if m_np.max() > 1.5:
        m_np = m_np / 255.0

    if target_hw is not None:
        H, W = target_hw
        if (m_np.shape[0] != H) or (m_np.shape[1] != W):
            m_img = Image.fromarray((m_np * 255).astype(np.uint8))
            m_img = m_img.resize((W, H), resample=Image.NEAREST)
            m_np = np.array(m_img).astype(np.float32) / 255.0

    return (m_np > 0.5)


# -----------------------------
# Initialize the scene
# -----------------------------
def initialize_scene(colmap_dir, model_dir, load_iteration=-1,load_filtered=False):
    """
    Initialize a Scene and optionally load a trained segmented Gaussian model and COLMAP-seeded Gaussians.

    Inputs:
        colmap_dir (str): Path to COLMAP reconstruction directory.
        model_dir (str): Path to Gaussian splatting model directory.
        mask_inst_dir (str): Path to mask instances directory.
        load_iteration (int): Iteration index of the trained model to load (-1 = latest).

    Outputs:
        scene (Scene): Scene object containing all Gaussians and cameras.
        dataset (Namespace): Model parameter namespace.
        colmap_seed_gaussians (GaussianModel): COLMAP-seeded Gaussian model.
        trained_gs_seg (GaussianModel): Segmented Gaussian model (loaded if available).

    Debug helpers:
        Prints info about loaded Gaussian models and number of segmented Gaussians.
    """
    parser = ArgumentParser()
    model_args = ModelParams(parser)

    model_args._source_path = os.path.abspath(colmap_dir)
    model_args._model_path = os.path.abspath(model_dir)
    model_args._images = "images"
    model_args._depths = ""
    model_args._resolution = 1.0
    model_args._white_background = False
    model_args.train_test_exp = False
    model_args.data_device = "cpu"
    model_args.eval = False

    dataset = Namespace(
        sh_degree=model_args.sh_degree,
        source_path=model_args._source_path,
        model_path=model_args._model_path,
        images=model_args._images,
        depths=model_args._depths,
        resolution=model_args._resolution,
        white_background=model_args._white_background,
        train_test_exp=model_args.train_test_exp,
        data_device=model_args.data_device,
        eval=model_args.eval
    )

    gaussians = GaussianModel(sh_degree=dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, load_filtered=load_filtered, shuffle=False, resolution_scales=[1.0])
    
    return scene, dataset

def filter_and_save(
    scene,
    mask_dir,
    iteration,
    K=10,                  # renderer top-K
    K_keep=10,             # how many to keep per pixel after gating/selection
    semantic_threshold=0.7,
    semantic_use_sigmoid=False,
    storage_device="cpu",
    missing_mask_policy="all_white",  # "all_white" or "skip"
):
    """
    Hole-resistant filtering:
      1) semantic prefilter (global): semantic_score > semantic_threshold
      2) contrib vote:
           per pixel, from renderer top-K:
             - prefer semantic-passing contributors
             - select top K_keep by opacity within preferred set
             - fill remainder from non-semantic set by opacity (geometry-preserving)
      3) final keep = semantic_keep & contrib_keep

    missing_mask_policy:
      - "all_white": if a camera has no mask, treat mask as full foreground (keeps scene intact)
      - "skip": ignore that camera (can cause holes if many missing)
    """
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm

    gaussians = scene.gaussians
    all_xyz = gaussians.get_xyz
    N = all_xyz.shape[0]

    render_device = all_xyz.device
    store_device = torch.device(storage_device)

    cameras = scene.getTrainCameras()
    output_dir = os.path.join(scene.model_path, "point_cloud", f"iteration_{iteration}")
    vis_dir = os.path.join(output_dir, "filter_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    mask_index = build_mask_index(mask_dir)
    print(f"[INFO] Indexed {len(mask_index)} mask files from {mask_dir}")

    class DummyPipe:
        debug = False
        antialiasing = False
        compute_cov3D_python = False
        convert_SHs_python = False

    dummy_pipe = DummyPipe()
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=render_device)

    # ---- semantics -> store_device once ----
    if hasattr(gaussians, "semantic_mask") and isinstance(gaussians.semantic_mask, torch.Tensor):
        sem = gaussians.semantic_mask.squeeze().detach().view(-1)
        if sem.numel() != N:
            raise ValueError(f"semantic_mask has {sem.numel()} elements but expected {N}.")
        sem = sem.to(store_device)
        sem_score = torch.sigmoid(sem) if semantic_use_sigmoid else sem
        semantic_keep = sem_score > float(semantic_threshold)
        print(
            f"[INFO] Semantic prefilter: keep {int(semantic_keep.sum().item())}/{N} "
            f"(thr={semantic_threshold}, sigmoid={semantic_use_sigmoid}) | "
            f"range: {sem_score.min().item():.3f}–{sem_score.max().item():.3f}"
        )
    else:
        print("[WARN] No semantic_mask field found — semantic prefilter disabled.")
        sem_score = torch.ones(N, device=store_device)
        semantic_keep = torch.ones(N, dtype=torch.bool, device=store_device)

    contrib_keep = torch.zeros(N, dtype=torch.bool, device=store_device)

    K = int(K)
    K_keep = int(K_keep)
    if K_keep <= 0:
        raise ValueError("K_keep must be >= 1.")

    missing_masks = 0
    used_masks = 0

    for cam in tqdm(cameras, desc="Filtering cameras"):
        base = os.path.splitext(cam.image_name)[0]
        base_key = base.lower()

        mask_path = mask_index.get(base_key, None)
        if mask_path is None:
            mask_path = mask_index.get(base_key.strip(), None)

        # ---- Render on GPU ----
        with torch.no_grad():
            out = render(cam, gaussians, dummy_pipe, bg_color, contrib=True, K=K)

        if out is None or out.get("contrib_indices", None) is None:
            continue

        contrib_indices = out["contrib_indices"]  # [H,W,K] (GPU)
        contrib_opacities = out.get("contrib_opacities", None)

        H, W, Kk = contrib_indices.shape
        if Kk != K:
            K = int(Kk)

        Kk_keep = min(K_keep, K)

        # ---- Mask handling ----
        if mask_path is None:
            missing_masks += 1
            if missing_mask_policy == "skip":
                continue
            elif missing_mask_policy == "all_white":
                # keep everything for this camera
                mask_white = np.ones((H, W), dtype=bool)
            else:
                raise ValueError("missing_mask_policy must be 'all_white' or 'skip'")
        else:
            used_masks += 1
            mask_white = load_mask_as_bool(mask_path, target_hw=(H, W))

        white_y, white_x = np.where(mask_white)
        if len(white_y) == 0:
            continue

        # ---- Move topK data to CPU (or storage_device) ----
        g = contrib_indices.to(store_device, non_blocking=False).long().view(-1, K)  # [P,K]
        if contrib_opacities is None:
            w = torch.ones_like(g, dtype=torch.float32, device=store_device)
        else:
            w = contrib_opacities.to(store_device, non_blocking=False).float().view(-1, K)

        P = H * W
        valid = g >= 0

        # semantic pass mask within the topK list
        sem_pass = torch.zeros((P, K), dtype=torch.bool, device=store_device)
        if valid.any():
            sem_pass[valid] = (sem_score[g[valid]] > float(semantic_threshold))

        # We will select K_keep per pixel by:
        #   - prefer sem_pass contributors
        #   - within each group, pick highest opacity
        # This avoids "semantic winners" that have tiny opacity and cause holes.
        # Create a selection score:
        #   big bonus if sem_pass else 0, plus opacity for ranking.
        # invalid gets -inf so it never selected
        score = torch.full((P, K), float("-inf"), dtype=torch.float32, device=store_device)
        if valid.any():
            bonus = sem_pass.float() * 1e3  # huge bonus: semantic-pass always outranks non-pass
            score[valid] = bonus[valid] + w[valid]  # opacity decides within each group

        # top-K_keep indices per pixel
        # (torch.topk is faster than argsort for small K_keep)
        topv, topi = torch.topk(score, k=Kk_keep, dim=1, largest=True, sorted=False)  # [P,K_keep]
        top_g = torch.gather(g, 1, topi)  # [P,K_keep]

        # Remove invalid selections (can happen if too many -inf)
        top_g = top_g.reshape(-1)
        top_g = top_g[top_g >= 0]
        if top_g.numel() == 0:
            continue

        # Only consider mask-white pixels
        white_flat = torch.from_numpy((white_y * W + white_x).astype(np.int64)).to(store_device)
        # Need top_g for just white pixels:
        top_g_white = torch.gather(
            torch.gather(g, 1, topi),  # [P,K_keep]
            0,
            white_flat.unsqueeze(1).expand(-1, Kk_keep)
        ).reshape(-1)
        top_g_white = top_g_white[top_g_white >= 0]
        if top_g_white.numel() == 0:
            continue

        contrib_keep[torch.unique(top_g_white)] = True

        # ---- (optional) visualization ----
        if mask_path is not None:
            Image.fromarray(mask_white.astype(np.uint8) * 255).save(
                os.path.join(vis_dir, f"{base}_mask.png")
            )

    print(f"[INFO] Masks used: {used_masks}/{len(cameras)} | missing: {missing_masks} (policy={missing_mask_policy})")

    keep_mask = semantic_keep & contrib_keep
    kept_count = int(keep_mask.sum().item())
    print(f"[INFO] Keeping {kept_count}/{N} Gaussians after filtering")

    if kept_count == 0:
        print("[WARN] No Gaussians selected — saving original scene.")
        filtered_ply_path = os.path.join(output_dir, "scene_mask_filtered_renderer.ply")
        gaussians.save_ply(filtered_ply_path)
        return

    # slice model on render_device
    keep_mask_rd = keep_mask.to(render_device)

    for attr_name in list(vars(gaussians).keys()):
        attr = getattr(gaussians, attr_name)
        if isinstance(attr, torch.Tensor) and attr.shape[0] == keep_mask_rd.shape[0]:
            setattr(gaussians, attr_name, attr[keep_mask_rd])

    filtered_ply_path = os.path.join(output_dir, "scene_mask_filtered_renderer.ply")
    gaussians.save_ply(filtered_ply_path)
    print(f"[OK] Filtered scene saved to: {filtered_ply_path}")
    print(f"[OK] Saved visualizations to: {vis_dir}")


# -----------------------------
# Main
# -----------------------------
def main():
    
    parser = ArgumentParser(description="Gaussian Cluster Filtering Pipeline")

    # Required paths
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)


    # Optional outputs
    parser.add_argument("--ply_iteration", type=int, default=30000)

    # Training settings
    parser.add_argument("--contribs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--fg_thresh", type=float, default=0.7)

    args = parser.parse_args()

    # ----------------------------------------------------
    # Load scene
    # ----------------------------------------------------
    print("[Step] Loading scene...")
    scene, dataset = initialize_scene(
        args.colmap_dir,
        args.model_dir,
    )
    full_model = scene.gaussians
    print(f"[INFO] Scene initialized. {full_model.get_xyz.shape[0]} Gaussians.")

    # ----------------------------------------------------
    # Compute responsibilities
    # ----------------------------------------------------
    filter_and_save(
        scene,
        mask_dir=args.mask_dir,
        iteration=args.ply_iteration,
        K=args.contribs,
        K_keep=args.contribs,          # start with same as K
        semantic_threshold=args.fg_thresh,
        semantic_use_sigmoid=False,
        storage_device="cpu",
        missing_mask_policy="all_white",
    )

if __name__ == "__main__":
    main()
