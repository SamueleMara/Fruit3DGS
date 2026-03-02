# filter_standalone.py
import os
import time
import random
import torch
import torch.nn.functional as F
import numpy as np

from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from argparse import ArgumentParser, Namespace

from utils.loss_utils import frame_pixel_contrastive_loss, appearance_contrastive_pair_loss, centroid_instance_pull_loss
from utils.masks_utils import load_or_create_mask_instances,filter_mask_instances_by_scene_cameras
from utils.cluster_utils import build_gaussian_mask_mappings, assign_gaussian_mask_ids
from utils import cluster_utils
from utils.visualize_clusters import visualize_clusters_from_ply,plot_cosine_vs_index_distance, plot_cosine_histogram, plot_cosine_by_semantic,visualize_fg_clusters_with_centroid_gaussians
from gaussian_renderer import render

from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn.cluster import AgglomerativeClustering
from skopt import Optimizer
from skopt.space import Real, Integer
from scipy.sparse import csr_matrix


# -----------------------------
# Initialize the scene
# -----------------------------
def initialize_scene(colmap_dir, model_dir, mask_inst_dir, load_iteration=-1,load_filtered=True):
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

# -----------------------------------------------------------------
# Contrastive Lift-style training using top-K splat contributors
# -----------------------------------------------------------------
def train_instance_embeddings_graph(
    gaussians,
    topK_full,
    mask_instances_dict,
    cam_names,
    rgb_all, 
    op_all,
    centroid_gauss_map=None,
    iterations=10000,
    lr=5e-4,
    embed_dim=64,
    k_neighbors=16,

    pixel_weight=1.0,
    smooth_weight=1.0,
    app_weight=1.0,
    centroid_weight=1.0,

    app_rgb_w=10.0,
    app_op_w=0.4,
    app_tau_pos=0.15,
    app_tau_neg=0.15,
    app_margin=0.25,
    emb_margin=0.6,
    app_neg_scale=1.0,
    pixels_per_iter=20000,
    neg_per_pos=16,
    temperature=0.07,
    bg_neg_weight=6.0,

    smooth_nodes=4096,

    centroid_sigma_px=25.0,
    centroid_max_pix=8000,
    centroid_min_valid=32,

    seed=0,
    debug=False,
):
    torch.manual_seed(seed)
    device = gaussians.get_xyz.device
    xyz = gaussians.get_xyz
    N = xyz.shape[0]

    assert gaussians.semantic_mask is not None, "semantic_mask required"
    s = torch.sigmoid(gaussians.semantic_mask.detach().float())
    s = (s / (s.max() + 1e-6)).clamp(0, 1)

    # kNN graph (CPU)
    nbr_idx_cpu = cluster_utils.build_knn_graph_scipy(xyz, k=k_neighbors)

    # embeddings
    gaussians.instance_embed = torch.nn.Parameter(0.1 * torch.randn(N, embed_dim, device=device))
    optimizer = torch.optim.Adam([gaussians.instance_embed], lr=lr)

    pixel_topk = topK_full["pixel_topk_gaussians"]

    # cache pixel->instance label per camera (same-frame supervision)
    pixel_inst_list = cluster_utils.build_pixel_inst_cache(mask_instances_dict, cam_names, pixel_topk, device=device)

    # cache appearance once (rgb + opacity) 
    rgb_all = rgb_all.to(device)
    op_all  = op_all.to(device)

    if debug:
        progress_bar = tqdm(range(iterations), desc="Frame contrastive training")
        ema = {"loss": 0.0, "pix": 0.0, "smooth": 0.0, "app": 0.0, "cent": 0.0}
    else:
        progress_bar = range(iterations)

    for it in progress_bar:
        optimizer.zero_grad()
        emb = F.normalize(gaussians.instance_embed, dim=1)

        # ==================================================
        # 1) SAME-FRAME contrastive using pixel->gaussian map
        # ==================================================
        loss_pixel = emb.new_tensor(0.0)

        cam_idx = torch.randint(0, len(pixel_topk), (1,), device=device).item()
        frame_name = cam_names[cam_idx]
        topk = pixel_topk[cam_idx]
        pix_inst = pixel_inst_list[cam_idx]

        if topk is not None and pix_inst is not None:
            topk = topk.to(device)
            g_at_pix = topk[:, 0].long()   # strongest gaussian per pixel

            loss_pixel = frame_pixel_contrastive_loss(
                emb_norm=emb,
                g_at_pix=g_at_pix,
                pix_inst=pix_inst,
                pixels_per_iter=pixels_per_iter,
                neg_per_pos=neg_per_pos,
                temperature=temperature,
                bg_neg_weight=bg_neg_weight,
            )

        # ==================================================
        # 2) SEMANTIC-WEIGHTED SPATIAL SMOOTHNESS
        # ==================================================
        nodes = torch.randint(0, N, (min(smooth_nodes, N),), device=device)
        nbr = nbr_idx_cpu[nodes.cpu()].to(device)

        ei = emb[nodes].unsqueeze(1)
        ej = emb[nbr]

        w_ij = s[nodes].unsqueeze(1) * s[nbr]
        diff2 = (ei - ej).pow(2).sum(dim=2)
        loss_smooth = (w_ij * diff2).sum() / (w_ij.sum() + 1e-6)
        
        # ==================================================
        # 3) APPEARANCE-BASED CONTRASTIVE
        # ==================================================
        if app_weight > 0.0:
            rgb_i = rgb_all[nodes]      # [B,3]
            rgb_j = rgb_all[nbr]        # [B,K,3]
            op_i  = op_all[nodes]       # [B,1]
            op_j  = op_all[nbr]         # [B,K,1]

            sem_w = w_ij

            loss_app = appearance_contrastive_pair_loss(
                emb_nodes=emb[nodes],   # [B,D]
                emb_nbr=ej,             # [B,K,D]
                rgb_nodes=rgb_i,
                rgb_nbr=rgb_j,
                op_nodes=op_i,
                op_nbr=op_j,
                sem_w=sem_w,
                rgb_w=app_rgb_w,
                op_w=app_op_w,
                tau_pos=app_tau_pos,
                tau_neg=app_tau_neg,
                app_margin=app_margin,
                emb_margin=emb_margin,
                neg_scale=app_neg_scale,
            )
        else:
            loss_app = emb.new_tensor(0.0)

        ## ==================================================
        # 4) NEW: CENTROID PULL (pixel-distance-to-centroid logic)
        # ==================================================
        loss_cent = emb.new_tensor(0.0)
        if centroid_weight > 0.0 and topk is not None:
            inst_dict = mask_instances_dict.get(frame_name, {})
            if inst_dict:
                inst_keys = list(inst_dict.keys())
                ridx = torch.randint(0, len(inst_keys), (1,), device=device).item()
                inst_id = inst_keys[ridx]
                inst_data = inst_dict[inst_id]

                pix_list = inst_data.get("pixel_indices", [])
                centroid_pixel = inst_data.get("centroid_pixel", None)  # <--- NEW
                if centroid_pixel is None:
                    # (optional fallback) if old dict still has centroid xy:
                    # centroid_xy = inst_data.get("centroid", None)
                    # ... but you said you now store centroid_pixel, so just skip.
                    centroid_pixel = None

                if centroid_pixel is not None and pix_list:
                    inst_pix = torch.as_tensor(pix_list, device=device, dtype=torch.long)

                    cam_hw = topK_full.get("cam_hw", None)
                    if cam_hw is None or cam_hw[cam_idx] is None:
                        raise RuntimeError("Missing topK_full['cam_hw'][cam_idx]=(H,W).")

                    H, W = cam_hw[cam_idx]

                    # topk may be CPU initially; ensure it’s on device
                    topk0 = topk[:, 0].to(device).long()  # [P]

                    # centroid gaussian from map (preferred)
                    g_c = None
                    if centroid_gauss_map is not None:
                        g_c = centroid_gauss_map.get((frame_name, inst_id), None)

                    # optional fallback: if map missing, try centroid pixel top1
                    if g_c is None:
                        cp = int(centroid_pixel)
                        if 0 <= cp < topk0.shape[0]:
                            g_c = int(topk0[cp].item())

                    loss_cent = centroid_instance_pull_loss(
                        emb_norm=emb,
                        topk0=topk0,
                        inst_pix=inst_pix,
                        centroid_g=g_c,
                        centroid_pixel=int(centroid_pixel),
                        W=int(W),
                        sigma_px=float(centroid_sigma_px),
                        max_pix=int(centroid_max_pix),
                        min_valid=int(centroid_min_valid),
                    )

        # =========
        # TOTAL
        # =========
        loss = (
            pixel_weight * loss_pixel
            + smooth_weight * loss_smooth
            + app_weight * loss_app
            + centroid_weight * loss_cent
        )

        loss.backward()
        optimizer.step()

        if debug:
            with torch.no_grad():
                ema["loss"]   = 0.4 * loss.item() + 0.6 * ema["loss"]
                ema["pix"]    = 0.4 * loss_pixel.item() + 0.6 * ema["pix"]
                ema["smooth"] = 0.4 * loss_smooth.item() + 0.6 * ema["smooth"]
                ema["app"]    = 0.4 * loss_app.item() + 0.6 * ema["app"]
                ema["cent"]   = 0.4 * loss_cent.item() + 0.6 * ema["cent"]

                if it % 10 == 0:
                    progress_bar.set_postfix({
                        "loss": f"{ema['loss']:.6f}",
                        "pix": f"{ema['pix']:.6f}",
                        "smooth": f"{ema['smooth']:.6f}",
                        "app": f"{ema['app']:.6f}",
                        "cent": f"{ema['cent']:.6f}",
                    })

    if debug and hasattr(progress_bar, "close"):
        progress_bar.close()

    return gaussians.instance_embed.detach(), nbr_idx_cpu


def assign_gaussian_instance_from_pixels(
    gaussians,
    mask_instances_dict,
    topK_full,
    device,
    use_weights=False,       # placeholder if you later store per-pixel weights
    chunk_pixels=200000,
    debug=True,
):
    """
    Assign each Gaussian an instance id based on which (frame, instance) pixels
    it contributes to the most (using per-pixel top-K gaussians).

    IMPORTANT:
      - Correct global identity is (frame_name, inst_key), NOT inst_key alone.
      - This is DIAGNOSTIC (no training): purely "what pixels does this gaussian help explain?"
      - Returns compact integer instance IDs (0..M-1). Background/unseen gaussians -> -1.

    Returns:
      gaussian_instance_ids: LongTensor [N] on device, values in {-1, 0..M-1}
      inst_global_to_int: dict mapping (frame_name, inst_key) -> compact int
    """
    N = gaussians.get_xyz.shape[0]

    pixel_topk_gaussians = topK_full.get("pixel_topk_gaussians", None)
    if pixel_topk_gaussians is None:
        raise KeyError("topK_full missing key 'pixel_topk_gaussians'")

    # --------------------------------------------------
    # 0) Build stable global mapping: (frame, inst_key) -> compact int
    # --------------------------------------------------
    inst_global_to_int = {}
    next_id = 0
    frame_names = list(mask_instances_dict.keys())

    for fname in frame_names:
        for inst_key in mask_instances_dict[fname].keys():
            gkey = (fname, inst_key)   # frame-aware unique key
            if gkey not in inst_global_to_int:
                inst_global_to_int[gkey] = next_id
                next_id += 1

    M = next_id
    if debug:
        per_frame = {f: len(mask_instances_dict[f]) for f in frame_names}
        print(f"[DEBUG] Global instances (frame,inst): {M}")
        print(
            f"[DEBUG] Frames: {len(frame_names)} | per-frame instances: "
            f"min={min(per_frame.values())} max={max(per_frame.values())} "
            f"mean={sum(per_frame.values())/len(per_frame):.2f}"
        )
        print(f"[DEBUG] len(pixel_topk_gaussians): {len(pixel_topk_gaussians)}")

    # votes[g][inst_int] += count
    votes = defaultdict(lambda: defaultdict(float))

    # --------------------------------------------------
    # 1) Per-camera voting (LOCAL pixels only)
    #    Robust to topK length mismatch:
    #      - if more frames than topK entries, skip extra frames
    #      - if more topK entries than frames, ignore extras
    # --------------------------------------------------
    n_frames = len(frame_names)
    n_topk = len(pixel_topk_gaussians)
    n_iter = min(n_frames, n_topk)

    if debug and n_frames != n_topk:
        print(
            f"[WARN] Frame/topK length mismatch: frames={n_frames}, topK={n_topk}. "
            f"Processing first {n_iter} aligned entries; skipping {max(0, n_frames - n_iter)} frames without topK."
        )

    for cam_idx, fname in enumerate(tqdm(frame_names[:n_iter], desc="Assigning Gaussians (pixel-topK)")):
        topk = pixel_topk_gaussians[cam_idx]
        if topk is None:
            continue

        topk = topk.to(device)  # [P,K]
        P, K = topk.shape

        # Build local pixel -> compact instance id (size P only)
        pixel_inst = torch.full((P,), -1, dtype=torch.long, device=device)

        for inst_key, data in mask_instances_dict[fname].items():
            pix = torch.as_tensor(data["pixel_indices"], dtype=torch.long, device=device)
            if pix.numel() == 0:
                continue

            pix = pix[(pix >= 0) & (pix < P)]
            if pix.numel() == 0:
                continue

            gkey = (fname, inst_key)  # frame-aware key
            pixel_inst[pix] = inst_global_to_int[gkey]

        # Process pixels in chunks to keep memory stable
        for start in range(0, P, chunk_pixels):
            end = min(start + chunk_pixels, P)

            g_chunk = topk[start:end]          # [chunk,K]
            inst_chunk = pixel_inst[start:end] # [chunk]

            # keep only masked pixels
            m = inst_chunk >= 0
            if not m.any():
                continue

            g_chunk = g_chunk[m]                 # [Mpix,K]
            inst_chunk = inst_chunk[m]           # [Mpix]

            g_flat = g_chunk.reshape(-1)         # [Mpix*K]
            inst_flat = inst_chunk.repeat_interleave(K)

            v = g_flat >= 0
            if not v.any():
                continue

            g_flat = g_flat[v]
            inst_flat = inst_flat[v]

            # accumulate counts on CPU to avoid giant GPU vote matrices
            g_flat_cpu = g_flat.detach().to("cpu", non_blocking=False)
            inst_flat_cpu = inst_flat.detach().to("cpu", non_blocking=False)

            # Pack pairs: key = g * M + inst
            keys = g_flat_cpu.to(torch.int64) * M + inst_flat_cpu.to(torch.int64)
            uniq_keys, counts = torch.unique(keys, return_counts=True)

            gs = (uniq_keys // M).tolist()
            is_ = (uniq_keys % M).tolist()
            cs = counts.tolist()

            for g, inst, c in zip(gs, is_, cs):
                votes[g][inst] += float(c)

        # free per-camera stuff
        del topk, pixel_inst
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------
    # 2) Final assignment: argmax vote per gaussian
    # --------------------------------------------------
    gaussian_instance_ids = torch.full((N,), -1, dtype=torch.long, device=device)

    assigned = 0
    for g, d in votes.items():
        if not d:
            continue
        best_inst = max(d.items(), key=lambda x: x[1])[0]
        gaussian_instance_ids[g] = int(best_inst)
        assigned += 1

    # --------------------------------------------------
    # 3) Debug stats
    # --------------------------------------------------
    if debug:
        ids_cpu = gaussian_instance_ids.detach().to("cpu")
        bg = int((ids_cpu < 0).sum())
        asg = int((ids_cpu >= 0).sum())
        print("\n[DEBUG] Gaussian → pixel-dominant *GLOBAL* instance assignment stats")
        print(f"[DEBUG] Total Gaussians: {N}")
        print(f"[DEBUG] Background Gaussians (-1): {bg} ({100.0*bg/N:.2f}%)")
        print(f"[DEBUG] Assigned Gaussians: {asg} ({100.0*asg/N:.2f}%)")

        cnt = Counter(ids_cpu.tolist())
        cnt.pop(-1, None)
        print(f"[DEBUG] Instances with ≥1 Gaussian: {len(cnt)} (out of global {M})")

        if len(cnt) > 0:
            top10 = cnt.most_common(10)
            print("\n[DEBUG] Top instances by Gaussian count:")
            for inst_int, c in top10:
                print(f"  InstInt {inst_int:5d}: {c:7d} Gaussians")

    return gaussian_instance_ids, inst_global_to_int


def optimize_hdbscan_bo(
    D_csr,                       
    xyz: torch.Tensor,
    semantic_mask: torch.Tensor,
    emb: torch.Tensor,
    n_nodes: int,
    n_calls: int = 40,
    patience: int = 8,
    min_delta: float = 1e-5,
    mcs_min=None,
    mcs_max=None,
    ms_min=None,
    ms_max=None,
    lambda_emb: float = 1.0,
    noise_penalty: float = 2.0,
    k_hi_penalty: float = 0.15,
    k_lo_penalty: float = 0.30,
    k_target: int = None,
    k_knn: int = None,
    debug: bool = True,
):
    """
    BO over HDBSCAN parameters:
      - min_cluster_size
      - min_samples

    Clustering uses precomputed sparse distances D_csr.
    Score uses semantic_weighted_quality_score:
      (intra_xyz/inter_xyz) + lambda_emb*(intra_emb/inter_emb)
      + noise penalty + cluster-count penalties
    Lower is better.
    """

    if D_csr is None or getattr(D_csr, "nnz", 0) == 0:
        raise RuntimeError("Empty distance graph (CSR nnz=0); cannot optimize HDBSCAN.")

    M = int(n_nodes)
    if M < 10:
        raise RuntimeError(f"Too few nodes for BO HDBSCAN: n_nodes={M}")

    # -----------------------------
    # AUTO BOUNDS (robust defaults)
    # -----------------------------
    M = int(n_nodes)

    if mcs_min is None or mcs_max is None:
        # fallback if not provided
        mcs_min = 5
        mcs_max = int(np.clip(0.02 * M, 20, 500))
        mcs_max = min(mcs_max, M)

    # min_samples bounds: don’t choke it at min_deg; keep it meaningful but safe
    ms_max = int(min( ms_max, mcs_max)) 

    if debug:
        deg = np.diff(D_csr.indptr).astype(np.int32)
        print(f"[BO][HDB] CSR degree stats: min={int(deg.min())}, p50={int(np.median(deg))}, max={int(deg.max())}")
        print(f"[BO][HDB] using bounds:")
        print(f"[BO][HDB]   min_cluster_size: {mcs_min} → {mcs_max}")
        print(f"[BO][HDB]   min_samples:      {ms_min} → {ms_max}")
        # degree per node = number of nonzeros in each row
        deg = np.diff(D_csr.indptr).astype(np.int32)
        min_deg = int(deg.min()) if deg.size > 0 else 0

    if min_deg < 1:
        raise RuntimeError("[BO][HDB] Distance graph has isolated nodes (row nnz = 0). Increase K or rebuild kNN on FG.")

    space = [
        Integer(mcs_min, mcs_max, name="min_cluster_size"),
        Integer(ms_min,  ms_max,  name="min_samples"),
    ]
    opt = Optimizer(space, random_state=42)

    best_score = float("inf")
    best_params = None
    best_labels = None
    no_improve = 0

    for i in range(n_calls):
        [mcs, ms] = opt.ask()

        # enforce standard constraints: ms <= mcs and ms <= min_deg
        mcs = int(mcs)
        ms  = int(ms)

        ms = int(min(ms, k_knn - 1))  # standard constraint
        ms = max(1, ms)

        labels = cluster_utils.run_hdbscan_precomputed(
            D_csr=D_csr,
            min_cluster_size=int(mcs),
            min_samples=int(ms),
            eps=0.0,
            debug=False,
        )

        # Score function
        score, info = cluster_utils.semantic_weighted_quality_score(
            labels_np=labels,
            xyz=xyz,
            emb=emb,
            semantic_mask=semantic_mask,
            lambda_emb=1.0,
            microcluster_penalty=0.6,
            blob_penalty=0.3,
            size_floor=8,
            debug=False,
        )
        opt.tell([int(mcs), int(ms)], float(score))

        improved = score < (best_score - min_delta)
        if improved:
            best_score = score
            best_params = (int(mcs), int(ms))
            best_labels = labels
            no_improve = 0
        else:
            no_improve += 1

        if debug:
            # info contains the real cluster count used in score (after filtering tiny/no-weight)
            Sxyz = float(info.get("S_xyz", np.nan))
            Semb = float(info.get("S_emb", np.nan))
            K_true = int(info.get("K_true", 0))
            K_score = int(info.get("K", 0))

            print(
                f"[BO][HDB][{i:02d}], mcs={mcs:3d}, ms={ms:3d}, "
                f"K_true={K_true:3d}, K_score={K_score:3d}, "
                f"Sxyz={Sxyz:.4f}, Semb={Semb:.4f}, score={score:.6f} "
                f"{'✓' if improved else ''}"
            )

        if no_improve >= patience:
            if debug:
                print(f"[BO][HDB] Early stop: no improvement for {patience} iterations")
            break

    if best_params is None:
        raise RuntimeError("BO failed to find a valid HDBSCAN configuration.")

    best_mcs, best_ms = best_params
    if debug:
        print(f"[BO][HDB] Best: min_cluster_size={best_mcs}, min_samples={best_ms}, score={best_score:.6f}")

    return best_mcs, best_ms, best_labels


def optimize_seeded_dbscan_iou_bo(
    D_csr,
    seed_fg,
    xyz_fg,
    emb_fg,
    sem_fg,
    ms_min=8,
    ms_max=80,
    t_min=0.05,
    t_max=0.80,
    min_cluster_size=12,          # FINAL threshold (applied once at the end)
    min_cluster_size_bo=8,        # small threshold used INSIDE BO scoring
    n_calls=30,
    patience=8,
    min_delta=1e-5,
    noise_penalty=0.25,
    require_full=True,
    conflict_policy="largest",
    min_intersection=2,
    debug=True,
):
    """
    BO over:
      - min_samples (microcluster size)
      - IoU threshold for merging microclusters

    IMPORTANT:
      - Inside BO we label with min_cluster_size_bo (small), so we don't collapse to all-noise.
      - After BO, we rerun once with the best (ms,t) and apply min_cluster_size (final).
    """
    if not isinstance(D_csr, csr_matrix):
        D_csr = D_csr.tocsr()

    M = int(D_csr.shape[0])
    seed_fg = np.asarray(seed_fg, dtype=np.int32)

    if seed_fg.size == 0:
        if debug:
            print("[BO][Seeded-DBSCAN] No seeds; returning all noise.")
        return int(ms_min), float(t_min), -np.ones((M,), dtype=np.int32)

    # sanitize ranges
    ms_min_in = int(ms_min)
    ms_max_in = int(ms_max)
    if ms_min_in > ms_max_in:
        ms_min_in, ms_max_in = ms_max_in, ms_min_in  # swap quietly

    ms_min = max(2, int(ms_min_in))
    ms_max = max(ms_min, int(ms_max_in))

    t_min = float(np.clip(t_min, 0.0, 1.0))
    t_max = float(np.clip(t_max, t_min, 1.0))

    min_cluster_size = int(max(1, min_cluster_size))
    min_cluster_size_bo = int(max(1, min_cluster_size_bo))

    # -----------------------------
    # Clamp ms bounds to SEED feasibility on CSR
    # -----------------------------
    deg = np.diff(D_csr.indptr).astype(np.int32)  # nnz per row
    if deg.size == 0:
        if debug:
            print("[BO][Seeded-DBSCAN] Empty CSR degrees; returning all noise.")
        return int(ms_min), float(t_min), -np.ones((M,), dtype=np.int32)

    deg_seed = deg[seed_fg] if seed_fg.size > 0 else deg
    seed_deg_max = int(deg_seed.max()) if deg_seed.size > 0 else 0

    # ms includes the seed itself; need (ms-1) neighbors
    ms_cap = max(2, seed_deg_max + 1)

    ms_max = min(ms_max, ms_cap)
    if ms_min > ms_cap:
        # hard infeasible: you asked for ms larger than any seed can support
        ms_min = ms_cap

    # treat equality as "fixed ms" without scary warning
    fixed_ms = (ms_min >= ms_max)
    if fixed_ms:
        ms_fixed = int(ms_max)
        if debug:
            print(
                f"[BO][Seeded-DBSCAN] Fixed ms (no range): requested [{ms_min_in},{ms_max_in}] "
                f"-> using ms={ms_fixed} (cap={ms_cap})"
            )
            print(
                f"[BO][Seeded-DBSCAN] IoU threshold range: using [{t_min},{t_max}] "
            )


        # precompute neighbors once up to ms_fixed-1
        neigh_dict = cluster_utils.precompute_seed_sorted_neighbors(
            D_csr, seed_fg, max_take=int(ms_fixed - 1)
        )

        # BO only over t
        space = [Real(t_min, t_max, name="iou_t")]
        opt = Optimizer(space, random_state=42)

        best = {"score": float("inf"), "t": None}
        no_improve = 0

        for it in range(int(n_calls)):
            [t] = opt.ask()
            t = float(t)

            micro, kept, dropped = cluster_utils.build_seed_microclusters_from_precomputed(
                seed_fg=seed_fg,
                neigh_dict=neigh_dict,
                min_samples=ms_fixed,
                require_full=require_full,
            )

            if debug and len(micro) > 0:
                sizes = np.array([m.size for m in micro], dtype=np.int32)
                union = np.unique(np.concatenate(micro)).size
                coverage = union / float(M)

                # overlap: how many microclusters include each node
                hit = np.zeros((M,), dtype=np.int32)
                for m in micro:
                    hit[m] += 1
                overlap_nodes = int((hit > 1).sum())
                max_hit = int(hit.max())

                print(
                    f"[DBG][micro] seeds={seed_fg.size} kept={kept} drop={dropped} "
                    f"ms={ms} micro_n={len(micro)} "
                    f"sz[min/med/p95/max]={sizes.min()}/{int(np.median(sizes))}/{int(np.percentile(sizes,95))}/{sizes.max()} "
                    f"union={union} cov={100*coverage:.1f}% "
                    f"overlap_nodes={overlap_nodes} max_hit={max_hit}"
                )

            if len(micro) == 0:
                labels_bo = -np.ones((M,), dtype=np.int32)
            else:
                merged = cluster_utils.merge_microclusters_by_iou_unionfind(
                    microclusters=micro,
                    M=M,
                    iou_thresh=t,
                    min_intersection=min_intersection,
                )
                if debug and len(merged) > 0:
                    msizes = np.array([s.size for s in merged], dtype=np.int32)
                    munion = np.unique(np.concatenate(merged)).size
                    mcov = munion / float(M)
                    print(
                        f"[DBG][merge] merged_sets={len(merged)} "
                        f"sz[min/med/p95/max]={msizes.min()}/{int(np.median(msizes))}/{int(np.percentile(msizes,95))}/{msizes.max()} "
                        f"union={munion} cov={100*mcov:.1f}% iou_t={t:.3f}"
                    )
                else:
                    if debug:
                        print(f"[DBG][merge] merged_sets=0 (iou_t={t:.3f})")

                # BO labels use SMALL threshold
                labels_bo = cluster_utils.labels_from_merged_sets(
                    merged_sets=merged,
                    M=M,
                    min_cluster_size=min_cluster_size_bo,
                    conflict_policy=str(conflict_policy),
                )

                if debug:
                    lab = labels_bo
                    n_lab = int((lab >= 0).sum())
                    frac = n_lab / float(M)
                    K = int(np.unique(lab[lab >= 0]).size) if n_lab > 0 else 0
                    print(f"[DBG][label] K={K} labeled={n_lab}/{M} ({100*frac:.1f}%) mcs_used={min_cluster_size_bo}")


            score, info = cluster_utils.semantic_weighted_quality_score(
                labels_np=labels_bo,
                xyz=xyz_fg,
                emb=emb_fg,
                semantic_mask=sem_fg,
                lambda_emb=1.0,
                microcluster_penalty=0.6,
                blob_penalty=0.3,
                size_floor=8,
                debug=False,
            )

            noise_frac = float(np.mean(labels_bo < 0))
            score2 = float(score + float(noise_penalty) * noise_frac)

            opt.tell([t], score2)

            improved = score2 < (best["score"] - float(min_delta))
            if improved:
                best["score"] = score2
                best["t"] = t
                no_improve = 0
            else:
                no_improve += 1

            if debug:
                K_score = int(info.get("K", 0))
                Sxyz = float(info.get("S_xyz", np.nan))
                Semb = float(info.get("S_emb", np.nan))
                print(
                    f"[BO][Seeded-DBSCAN][{it:02d}] ms={ms_fixed:3d} t={t:.4f} "
                    f"micro={len(micro):3d} kept={kept:3d} drop={dropped:3d} "
                    f"K={K_score:3d} noise={100*noise_frac:5.1f}% "
                    f"Sxyz={Sxyz:.4f} Semb={Semb:.4f} score2={score2:.6f} "
                    f"{'✓' if improved else ''}"
                )

            if no_improve >= int(patience):
                if debug:
                    print(f"[BO][Seeded-DBSCAN] Early stop: no improvement for {patience} iterations")
                break

        best_t = float(best["t"]) if best["t"] is not None else float(t_min)

        # ---- FINAL rerun with FINAL min_cluster_size ----
        micro, _, _ = cluster_utils.build_seed_microclusters_from_precomputed(
            seed_fg=seed_fg,
            neigh_dict=neigh_dict,
            min_samples=ms_fixed,
            require_full=require_full,
        )
        if len(micro) == 0:
            final_labels = -np.ones((M,), dtype=np.int32)
        else:
            merged = cluster_utils.merge_microclusters_by_iou_unionfind(
                microclusters=micro,
                M=M,
                iou_thresh=best_t,
                min_intersection=min_intersection,
            )
            final_labels = cluster_utils.labels_from_merged_sets(
                merged_sets=merged,
                M=M,
                min_cluster_size=min_cluster_size,  # FINAL FILTER HERE
                conflict_policy=str(conflict_policy),
            )

        if debug:
            print(f"[BO][Seeded-DBSCAN] Best (fixed ms): ms={ms_fixed} t={best_t:.4f} best_score2={best['score']:.6f}")
            n_lab = int((final_labels >= 0).sum())
            K = int(np.unique(final_labels[final_labels >= 0]).size) if n_lab > 0 else 0
            print(f"[FINAL] K={K} labeled={n_lab}/{M} mcs_final={min_cluster_size}")
            if K == 0:
                print("[FINAL][WARN] All clusters filtered out by min_cluster_size. "
                    "Your merged set sizes are likely << min_cluster_size.")


        return int(ms_fixed), float(best_t), final_labels

    # -----------------------------
    # Normal BO path (ms range valid)
    # -----------------------------
    if debug:
        print(f"[BO][Seeded-DBSCAN] ms bounds requested [{ms_min_in},{ms_max_in}] -> used [{ms_min},{ms_max}] (cap={ms_cap})")

    neigh_dict = cluster_utils.precompute_seed_sorted_neighbors(
        D_csr, seed_fg, max_take=int(ms_max - 1)
    )

    space = [
        Integer(int(ms_min), int(ms_max), name="min_samples"),
        Real(float(t_min), float(t_max), name="iou_t"),
    ]
    opt = Optimizer(space, random_state=42)

    best = {"score": float("inf"), "ms": None, "t": None}
    no_improve = 0

    for it in range(int(n_calls)):
        ms, t = opt.ask()
        ms = int(ms)
        t = float(t)

        micro, kept, dropped = cluster_utils.build_seed_microclusters_from_precomputed(
            seed_fg=seed_fg,
            neigh_dict=neigh_dict,
            min_samples=ms,
            require_full=require_full,
        )

        if len(micro) == 0:
            labels_bo = -np.ones((M,), dtype=np.int32)
        else:
            merged = cluster_utils.merge_microclusters_by_iou_unionfind(
                microclusters=micro,
                M=M,
                iou_thresh=t,
                min_intersection=min_intersection,
            )
            # BO labels use SMALL threshold
            labels_bo = cluster_utils.labels_from_merged_sets(
                merged_sets=merged,
                M=M,
                min_cluster_size=min_cluster_size_bo,
                conflict_policy=str(conflict_policy),
            )

        score, info = cluster_utils.semantic_weighted_quality_score(
            labels_np=labels_bo,
            xyz=xyz_fg,
            emb=emb_fg,
            semantic_mask=sem_fg,
            lambda_emb=1.0,
            microcluster_penalty=0.6,
            blob_penalty=0.3,
            size_floor=8,
            debug=False,
        )

        noise_frac = float(np.mean(labels_bo < 0))
        score2 = float(score + float(noise_penalty) * noise_frac)

        opt.tell([ms, t], score2)

        improved = score2 < (best["score"] - float(min_delta))
        if improved:
            best.update({"score": score2, "ms": ms, "t": t})
            no_improve = 0
        else:
            no_improve += 1

        if debug:
            K_score = int(info.get("K", 0))
            Sxyz = float(info.get("S_xyz", np.nan))
            Semb = float(info.get("S_emb", np.nan))
            print(
                f"[BO][Seeded-DBSCAN][{it:02d}] ms={ms:3d} t={t:.4f} "
                f"micro={len(micro):3d} kept={kept:3d} drop={dropped:3d} "
                f"K={K_score:3d} noise={100*noise_frac:5.1f}% "
                f"Sxyz={Sxyz:.4f} Semb={Semb:.4f} score2={score2:.6f} "
                f"{'✓' if improved else ''}"
            )

        if no_improve >= int(patience):
            if debug:
                print(f"[BO][Seeded-DBSCAN] Early stop: no improvement for {patience} iterations")
            break

    if best["ms"] is None or best["t"] is None:
        if debug:
            print("[BO][Seeded-DBSCAN] Failed; returning all noise.")
        return int(ms_min), float(t_min), -np.ones((M,), dtype=np.int32)

    best_ms = int(best["ms"])
    best_t = float(best["t"])

    # ---- FINAL rerun with FINAL min_cluster_size ----
    micro, _, _ = cluster_utils.build_seed_microclusters_from_precomputed(
        seed_fg=seed_fg,
        neigh_dict=neigh_dict,
        min_samples=best_ms,
        require_full=require_full,
    )
    if len(micro) == 0:
        final_labels = -np.ones((M,), dtype=np.int32)
    else:
        merged = cluster_utils.merge_microclusters_by_iou_unionfind(
            microclusters=micro,
            M=M,
            iou_thresh=best_t,
            min_intersection=min_intersection,
        )
        final_labels = cluster_utils.labels_from_merged_sets(
            merged_sets=merged,
            M=M,
            min_cluster_size=min_cluster_size,  # FINAL FILTER HERE
            conflict_policy=str(conflict_policy),
        )

    if debug:
        print(f"[BO][Seeded-DBSCAN] Best: ms={best_ms} t={best_t:.4f} best_score2={best['score']:.6f}")

    return int(best_ms), float(best_t), final_labels
    

# -----------------------------
# Main
# -----------------------------
def main():
    
    parser = ArgumentParser(description="Gaussian Cluster Filtering Pipeline")

    # Required paths
    parser.add_argument("--colmap_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--mask_inst_dir", type=str, required=True)

    # Optional outputs
    parser.add_argument("--ply_iteration", type=int, default=30000)

    # Training settings
    parser.add_argument("--contribs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--smooth_dist_thresh", type=float, default=0.05)
    parser.add_argument("--smooth_max_points", type=int, default=2000)
    parser.add_argument("--smooth_chunk", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--fg_thresh", type=float, default=0.7)
    parser.add_argument("--k_centr", type=int, default=10)

    # Clustering algorithm choice
    parser.add_argument("--cluster_alg", type=str, default="hdbscan",
                        choices=["hdbscan", "dbscan"])

    args = parser.parse_args()

    # ----------------------------------------------------
    # Build paths based on iteration
    # ----------------------------------------------------
    ply_path = os.path.join(args.model_dir,
        f"point_cloud/iteration_{args.ply_iteration}/point_cloud.ply")

    out_ply_path = os.path.join(args.model_dir,
        f"point_cloud/iteration_{args.ply_iteration}/scene_clusters.ply")

    filtered_ply_path = os.path.join(args.model_dir,
        f"point_cloud/iteration_{args.ply_iteration}/filtered_scene_clusters.ply")


    # ----------------------------------------------------
    # Load scene
    # ----------------------------------------------------
    print("[Step] Loading scene...")
    scene, dataset = initialize_scene(
        args.colmap_dir,
        args.model_dir,
        args.mask_inst_dir
    )
    full_model = scene.gaussians
    N=full_model.get_xyz.shape[0]
    print(f"[INFO] Scene initialized. {N} Gaussians.")

    # ----------------------------------------------------
    # Compute responsibilities
    # ----------------------------------------------------
    print("[Step] Computing top-K contributors + responsibilities (merged)...")
    topK_full, r_point, r_gauss, r_vals = cluster_utils.compute_topK_contributors_and_responsibilities(scene, K=args.contribs,storage_device=args.device)

    pixel_best_gaussian = topK_full["pixel_best_gaussian"]

    # ----------------------------------------------------
    # Load mask instances
    # ----------------------------------------------------
    print("[Step] Loading mask instances...")
    mask_instances_dict = load_or_create_mask_instances(args.mask_inst_dir)
    mask_instances_dict = filter_mask_instances_by_scene_cameras(mask_instances_dict, scene, verbose=False)
    total_instances = 0
    total_pixels = 0

    # print("[DEBUG] Mask instances summary:")

    # for frame_name, inst_dict in mask_instances_dict.items():
    #     num_inst = len(inst_dict)
    #     num_pix = sum(len(v["pixel_indices"]) for v in inst_dict.values())

    #     print(f"  {frame_name}: {num_inst} instances, {num_pix} pixels")

    #     total_instances += num_inst
    #     total_pixels += num_pix

    # print(f"[DEBUG] Total frames: {len(mask_instances_dict)}")
    # print(f"[DEBUG] TOTAL mask instances: {total_instances}")
    # print(f"[DEBUG] TOTAL mask pixels: {total_pixels}")

    # ----------------------------------------------------
    # Compute accurate centroids
    # ----------------------------------------------------

    centroid_gauss_map, cent_stats = cluster_utils.compute_instance_centroid_gaussians(
        scene=scene,
        mask_instances_dict=mask_instances_dict,
        K_centroid=args.k_centr,                 # high K helps stability
        storage_device="cpu",
        method="max_semantic",
        dist_gate=0.05,                # optional: reject far contributors (scene units)
        min_keep=3,
        debug=True,
    )

    # ----------------------------------------------------
    # Build Gaussian ↔ Mask mappings
    # ----------------------------------------------------
    print("[Step] Building Gaussian ↔ Mask mappings...")

    gaussian_to_mask_ids, mask_id_to_pixel_idx, pixel_idx_to_gaussians, gaussian_to_pixels, cam_offset_dict, global_id_map = build_gaussian_mask_mappings(
        mask_instances_dict,
        pixel_best_gaussian,
        scene,
        device=full_model.get_xyz.device,
    )

    # mask_instance_ids, shared_masks = assign_gaussian_mask_ids(
    #     mask_instances_dict,
    #     pixel_best_gaussian,
    #     cam_offset_dict,
    #     global_id_map,
    #     num_gaussians=full_model.get_xyz.shape[0],
    #     device=full_model.get_xyz.device,
    # )

    # num_mapped_gaussians = (mask_instance_ids >= 0).sum().item()
    # unique_masks = set(mask_instance_ids.cpu().tolist()) - {-1}
   

    # # ----------------------------------------
    # # Basic graph statistics
    # # ----------------------------------------
    # num_edges = sum(len(v) for v in shared_masks.values())              # gaussian–mask edges
    # multi_mask_gaussians = sum(len(v) > 1 for v in shared_masks.values())
    # single_mask_gaussians = sum(len(v) == 1 for v in shared_masks.values())
    # unmapped_gaussians = full_model.get_xyz.shape[0] - num_mapped_gaussians

    # mask_usage = {}
    # for g, masks in shared_masks.items():
    #     for m in masks:
    #         mask_usage[m] = mask_usage.get(m, 0) + 1
    # num_masks_used = len(mask_usage)

    # print("\n[DEBUG] ---------- Mask ↔ Gaussian Graph ----------")
    # print(f"[DEBUG] Gaussians mapped to ≥1 mask: {num_mapped_gaussians}")
    # print(f"[DEBUG] Unmapped gaussians: {unmapped_gaussians}")
    # print(f"[DEBUG] Unique mask instances used: {num_masks_used}")
    # print(f"[DEBUG] Total gaussian–mask edges: {num_edges}")
    # print(f"[DEBUG] Gaussians touching exactly one mask: {single_mask_gaussians}")
    # print(f"[DEBUG] Gaussians touching >1 mask: {multi_mask_gaussians}")
    # print(f"[DEBUG] Avg masks per gaussian: {num_edges / max(len(shared_masks),1):.2f}")

    # # ----------------------------------------
    # # True multi-view coupling (frame-aware)
    # # ----------------------------------------
    # global_mask_to_frame = {gid: frame for (frame, inst), gid in global_id_map.items()}

    # true_multi_view_gaussians = 0
    # same_frame_multi_mask = 0
    # cross_frame_multi_mask = 0

    # for g, masks in shared_masks.items():
    #     if len(masks) > 1:
    #         frames = {global_mask_to_frame[m] for m in masks}
    #         if len(frames) > 1:
    #             true_multi_view_gaussians += 1
    #             cross_frame_multi_mask += 1
    #         else:
    #             same_frame_multi_mask += 1

    # print("\n[DEBUG] ---------- True Multi-View Statistics ----------")
    # print(f"[DEBUG] Gaussians seen in >1 frame (true multi-view anchors): {true_multi_view_gaussians}")
    # print(f"[DEBUG] Ambiguous gaussians within the same frame only: {same_frame_multi_mask}")
    # print(f"[DEBUG] Ambiguous gaussians across multiple frames: {cross_frame_multi_mask}")

    # # ----------------------------------------
    # # Distribution: how many masks per gaussian
    # # ----------------------------------------
    # mask_hist = Counter(len(v) for v in shared_masks.values())

    # print("\n[DEBUG] ---------- Mask-Per-Gaussian Distribution ----------")
    # for k in sorted(mask_hist):
    #     print(f"[DEBUG] Gaussians touched by {k} mask(s): {mask_hist[k]}")

    # # ----------------------------------------
    # # Distribution: how many gaussians per mask
    # # ----------------------------------------
    # gauss_per_mask = Counter(mask_usage.values())

    # print("\n[DEBUG] ---------- Gaussian-Per-Mask Distribution ----------")
    # for k in sorted(gauss_per_mask):
    #     print(f"[DEBUG] Masks touching {k} gaussian(s): {gauss_per_mask[k]}")

    # print("[Step] Assigning Gaussians to pixel-dominant instance IDs (diagnostic)...")
    # gauss_diag_ids, inst_map = assign_gaussian_instance_from_pixels(
    #     gaussians=full_model,
    #     mask_instances_dict=mask_instances_dict,
    #     topK_full=topK_full,
    #     device=full_model.get_xyz.device,
    #     use_weights=False,
    # )

    # full_model.instance_ids = gauss_diag_ids

    # diag_ply = os.path.join(
    #     args.model_dir,
    #     f"point_cloud/iteration_{args.ply_iteration}/pixel_dominant_instances.ply"
    # )

    # full_model.save_clustered_ply(
    #     diag_ply,
    #     cluster_ids=full_model.instance_ids
    # )

    # print(f"[OK] Diagnostic PLY saved: {diag_ply}")    

    # print("[DEBUG] Gaussian → pixel-dominant instance assignment stats")

    # gauss_ids = gauss_diag_ids.detach().cpu()

    # N = gauss_ids.numel()
    
    # num_bg = (gauss_ids == -1).sum().item()
    # num_fg = N - num_bg

    # print(f"[DEBUG] Total Gaussians: {N}")
    # print(f"[DEBUG] Background Gaussians (-1): {num_bg} ({100*num_bg/N:.2f}%)")
    # print(f"[DEBUG] Assigned Gaussians: {num_fg} ({100*num_fg/N:.2f}%)")

    # # ----------------------------------------
    # # Distribution per instance
    # # ----------------------------------------
    # inst_counts = Counter(gauss_ids.tolist())
    # inst_counts.pop(-1, None)

    # num_instances_used = len(inst_counts)
    # print(f"[DEBUG] Instances with ≥1 Gaussian: {num_instances_used}")

    # # Sort by size
    # top_instances = inst_counts.most_common(15)

    # print("\n[DEBUG] Top instances by Gaussian count:")
    # for inst, cnt in top_instances:
    #     print(f"  Instance {inst:4d}: {cnt:6d} Gaussians")

    # # ----------------------------------------
    # # Histogram-style summary
    # # ----------------------------------------
    # bins = [1, 5, 10, 50, 100, 500, 1000, 5000]
    # hist = {b: 0 for b in bins}
    # hist[">5000"] = 0

    # for cnt in inst_counts.values():
    #     placed = False
    #     for b in bins:
    #         if cnt <= b:
    #             hist[b] += 1
    #             placed = True
    #             break
    #     if not placed:
    #         hist[">5000"] += 1

    # print("\n[DEBUG] Gaussian-per-instance histogram:")
    # for k, v in hist.items():
    #     print(f"  ≤{k}: {v} instances")

    # ----------------------------------------------------
    # Train instance embeddings (NO hard assignments here)
    # ----------------------------------------------------
    print("\n[Step] Training instance embeddings (semantic-guided)...")


    cam_names = list(scene.gs_cameras.keys())
    # print(cam_names[:5])

    # cache appearance once (rgb + opacity) 
    rgb_all, op_all = full_model.get_rgb_opacity(clamp_rgb=True)
    rgb_all = rgb_all.to(full_model.get_xyz.device)
    op_all  = op_all.to(full_model.get_xyz.device)

    embeddings_final, nbr_idx_cpu = train_instance_embeddings_graph(
        gaussians=full_model,
        topK_full=topK_full,
        mask_instances_dict=mask_instances_dict,
        cam_names=cam_names,
        rgb_all=rgb_all,
        op_all=op_all,
        centroid_gauss_map=centroid_gauss_map,  
        iterations=args.iterations,
        lr=args.lr,
        embed_dim=args.embed_dim or 64,
        pixel_weight=1.0,
        smooth_weight=1.0,
        app_weight=1.0,
        centroid_weight=10.0,
        temperature=0.07,
        neg_per_pos=16,
        bg_neg_weight=8.0,
        debug=True,
    )

    # Store embeddings 
    full_model.instance_embed = embeddings_final

    print(
        "[INFO] Embedding training finished. "
        f"Embedding shape: {embeddings_final.shape}"
    )
    print(
        "[Embedding]",
        "norm min/max:",
        embeddings_final.norm(dim=1).min().item(),
        embeddings_final.norm(dim=1).max().item(),
    )
    
    emb = full_model.instance_embed.detach()
    # plot_cosine_histogram(emb)
    # plot_cosine_vs_index_distance(emb)
    # plot_cosine_by_semantic(emb, full_model.semantic_mask)

    # ----------------------------------------------------
    # BO-based clustering on embeddings
    # ----------------------------------------------------
    print("\n[Step] BO clustering over embedding space...")

    # Foreground filtering
    # sem = torch.sigmoid(full_model.semantic_mask).detach().view(-1)
    sem = full_model.semantic_mask.detach().view(-1)


    fg_thresh = args.fg_thresh
    fg_idx = (sem > fg_thresh).nonzero(as_tuple=True)[0]
    M = int(fg_idx.numel())
    print(f"[CC] fg_thresh={fg_thresh:.2f} -> {fg_idx.numel()} nodes")

    # Subselect embeddings/xyz to fg only (graph is on fg-nodes)
    emb_fg = full_model.instance_embed[fg_idx]
    xyz_fg = full_model.get_xyz[fg_idx]
    sem_fg = sem[fg_idx]
    rgb_fg = rgb_all[fg_idx]
    op_fg = op_all[fg_idx]

    print("[Step] Building embedded kNN distance graph (once)...")

    # restrict training knn to FG subset (reusing the KDTree neighbors you already computed)
    k_knn=100
    nbr_fg_cpu = cluster_utils.build_knn_graph_scipy(xyz_fg, k=k_knn)  # on FG directly
    # nbr_fg_cpu = cluster_utils.restrict_knn_to_subset(nbr_idx_cpu=nbr_idx_cpu, subset_idx=fg_idx)

    # build CSR distance matrix once 
    D_fg = cluster_utils.build_embedded_knn_distance_graph(
        xyz=xyz_fg, 
        emb=emb_fg, 
        sem=sem_fg, 
        nbr_sub_cpu=nbr_fg_cpu,
        rgb=rgb_fg, 
        op=op_fg,
        gamma_rgb=0.05,
        gamma_op=0.005,   # rgb >> op
        rgb_pow=1, 
        op_pow=1,
        debug=True
    )
        
    print("\n[Step] BO clustering over embedding space...")
    
    # -------------------------
    # SWITCH: HDBSCAN vs DBSCAN
    # -------------------------
    if args.cluster_alg == "hdbscan":
        print("[Clustering] Using HDBSCAN on EMBEDDED distance graph")

        mcs_min, mcs_max = cluster_utils.compute_hdbscan_bounds_from_mask(
            gaussian_to_mask_ids=gaussian_to_mask_ids,
            fg_idx=fg_idx,
            M=M,
            factor_hi=1.0,
            floor_min=5,
            robust=True,
            q_lo=0.10,
            q_hi=0.90,
            debug=True,
        )

        # good values Lemon and FruitNeRF
        ms_min=65
        ms_max=155  
        t_min=0.0                
        t_max=0.2
        
        # #good values Fuji
        # ms_min=65
        # ms_max=155  
        # t_min=0.0                
        # t_max=0.3

        # # #good values Brandeburg
        # ms_min=100
        # ms_max=135  
        # t_min=0.4               
        # t_max=0.8

        # # good values Multifruit
        # ms_min=100
        # ms_max=110
        # t_min=0.03              
        # t_max=0.04


        print("[Step] BO optimizing HDBSCAN params (min_cluster_size, min_samples)...")
        best_mcs, best_ms, labels_fg = optimize_hdbscan_bo(
            D_csr=D_fg,
            xyz=xyz_fg,
            emb=emb_fg,
            semantic_mask=sem_fg,
            n_nodes=M,
            n_calls=100,
            patience=8,
            debug=True,
            mcs_min=mcs_min,
            mcs_max=mcs_max,
            ms_min=15,
            ms_max=35,
            k_knn=k_knn,
        )
        print(f"[HDBSCAN] best min_cluster_size={best_mcs} best min_samples={best_ms}")

    elif args.cluster_alg == "dbscan":

        print("[Clustering] Seeded DBSCAN-like: microclusters around centroid seeds + IoU merge (BO over ms, IoU)")

        mcs_min, mcs_max = cluster_utils.compute_hdbscan_bounds_from_mask(
            gaussian_to_mask_ids=gaussian_to_mask_ids,
            fg_idx=fg_idx,
            M=M,
            factor_hi=1.0,
            floor_min=5,
            robust=True,
            q_lo=0.10,
            q_hi=0.90,
            debug=True,
        )


        # map centroid gaussians to FG-local seeds
        seed_fg = cluster_utils.map_centroid_gaussians_to_fg_seeds(
            centroid_gauss_map=centroid_gauss_map,
            fg_idx=fg_idx,
            N_total=N,
        )
        
        # ---- graph feasibility cap (based on CSR degrees for SEEDS) ----
        deg = np.diff(D_fg.indptr).astype(np.int32)
        deg_seed = deg[seed_fg] if seed_fg.size > 0 else deg
        ms_cap_seed = deg_seed + 1

        p10 = int(np.percentile(ms_cap_seed, 10))
        p50 = int(np.percentile(ms_cap_seed, 50))
        p80 = int(np.percentile(ms_cap_seed, 80))

        ms_min = max(15, p10)
        ms_max = min(p80, int(np.percentile(ms_cap_seed, 95)), 200)  # runtime cap

        # keep a *range* for BO if possible
        if ms_max <= ms_min:
            ms_max = min(ms_min + 20, int(np.percentile(ms_cap_seed, 95)), 200)

        # # good values Lemon and FruitNeRF
        # ms_min=65
        # ms_max=155  
        # t_min=0.0                
        # t_max=0.2
        
        #good values Fuji
        # ms_min=65
        # ms_max=155  
        # t_min=0.0                
        # t_max=0.3

        # # #good values Brandeburg
        ms_min=100
        ms_max=135  
        t_min=0.4               
        t_max=0.6

        # # good values Multifruit
        # ms_min=100
        # ms_max=110
        # t_min=0.03              
        # t_max=0.04


        best_ms, best_t, labels_fg = optimize_seeded_dbscan_iou_bo(
            D_csr=D_fg,
            seed_fg=seed_fg,
            xyz_fg=xyz_fg,
            emb_fg=emb_fg,
            sem_fg=sem_fg,
            ms_min=ms_min,
            ms_max=ms_max,
            t_min=t_min,                 # was 0.05
            t_max=t_max,                 # was 0.80
            min_cluster_size=mcs_min,  # final
            min_cluster_size_bo=mcs_min,    # bo
            n_calls=100,
            patience=8,
            noise_penalty=0.25,
            require_full=False,       
            conflict_policy="largest",
            min_intersection=2,       
            debug=True,
        )

        print(f"[Seeded-DBSCAN] best min_samples={best_ms}, best IoU t={best_t:.4f}")

    else:
        raise ValueError(f"Unknown cluster_alg: {args.cluster_alg}")

    # Lift labels back to full gaussian set
    ids_final = torch.full(
        (full_model.get_xyz.shape[0],),
        -1,
        device=full_model.get_xyz.device,
        dtype=torch.long
    )
    ids_final[fg_idx] = torch.as_tensor(labels_fg, device=ids_final.device, dtype=torch.long)

    # ----------------------------------------------------
    # FINAL MASK CONSISTENCY PASS (hard re-assign inside each mask instance)
    # ----------------------------------------------------
    # print("[Step] Enforcing per-mask instance cluster consistency (hard assignment)...")

    # ids_final_fixed, stats = cluster_utils.enforce_mask_instance_consistency(
    #     mask_instances_dict=mask_instances_dict,
    #     cam_names=cam_names,
    #     pixel_topk_gaussians=topK_full["pixel_topk_gaussians"],
    #     gaussian_instance_ids=ids_final,
    #     ignore_label=-1,
    #     prefer_non_noise=True,
    #     device=full_model.get_xyz.device,
    #     debug=True,
    # )
    # ids_final = ids_final_fixed
    
    full_model.instance_ids = ids_final

    filtered_ply_path = os.path.join(
        args.model_dir,
        f"point_cloud/iteration_{args.ply_iteration}/filtered_scene_clusters_cc.ply"
    )

    print(f"[Step] Saving clustered PLY → {filtered_ply_path}")
    full_model.save_clustered_ply(filtered_ply_path, cluster_ids=ids_final)

    print(f"[Step] Visualizing clusters")
    visualize_clusters_from_ply(filtered_ply_path)
    
    print(f"[Step] Visualizing clusters + mask instances centroids")
    visualize_fg_clusters_with_centroid_gaussians(
        ply_path=filtered_ply_path,
        centroid_gauss_map=centroid_gauss_map,
        marker_radius=0.01,
        marker_color=(1.0, 1.0, 1.0),
        color_markers_by_cluster=True,   # set True if you want marker colors to match clusters
        max_markers=None,                 # or e.g. 200
    )

if __name__ == "__main__":
    main()
