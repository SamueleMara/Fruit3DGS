#!/usr/bin/env python3
import os
import time
import torch
import torch.nn.functional as F

from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.system_utils import mkdir_p
from argparse import ArgumentParser, Namespace
from utils.read_write_model import read_model
from utils.general_utils import safe_state, get_expon_lr_func
from utils.loss_utils import total_cluster_loss
from utils import cluster_utils
from utils.visualize_clusters import visualize_clusters_from_ply, visualize_colmap_clusters
from gaussian_renderer import render

from tqdm import tqdm

from skopt.space import Real
from skopt.utils import use_named_args
from skopt import Optimizer

# -----------------------------
# Scene Initialization
# -----------------------------
def initialize_scene(colmap_dir, model_dir, mask_inst_dir, load_iteration=-1, num_its_BO=20, resolution=1, data_device="cpu"):
    """
    Initialize a Scene and optionally load a trained segmented Gaussian model and COLMAP-seeded Gaussians.

    Inputs:
        colmap_dir (str): Path to COLMAP reconstruction directory.
        model_dir (str): Path to Gaussian splatting model directory.
        mask_inst_dir (str): Path to mask instances directory.
        load_iteration (int): Iteration index of the trained model to load (-1 = latest).
        num_its_BO (int): Number of BO iterations (if bo_optimize True).
        resolution (int or float): resolution scale or integer used in the original script.
        data_device (str): 'cpu' or 'cuda' (or torch device string)

    Outputs:
        scene (Scene): Scene object containing all Gaussians and cameras.
        dataset (Namespace): Model parameter namespace.
        colmap_seed_gaussians (GaussianModel): COLMAP-seeded Gaussian model.
        trained_gs_seg (GaussianModel): Segmented Gaussian model (loaded if available).
        scene_info (object): dataset scene info
    """
    
    parser = ArgumentParser(add_help=False)
    model_args = ModelParams(parser)

    # Map provided values into model_args similarly to the original script
    model_args._source_path = os.path.abspath(colmap_dir)
    model_args._model_path = os.path.abspath(model_dir)
    model_args._images = "images"
    model_args._depths = ""
    model_args._resolution = resolution
    model_args._white_background = False
    model_args.train_test_exp = False
    model_args.data_device = data_device
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

    # create gaussian model and scene
    gaussians = GaussianModel(sh_degree=dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False, resolution_scales=[float(resolution)], mask_dir=None)

    print("[INFO] Loading COLMAP-seeded Gaussian model...")
    trained_gs_seg, colmap_seed_gaussians, scene_info = scene.load_with_colmap_seed(
        args=dataset,
        load_iteration=load_iteration,
        mask_dir=mask_inst_dir,
        num_its_BO=num_its_BO,
        bo_optimize=True,
        patience=10
    )

    print(f"[OK] Scene initialized. Segmented Gaussians: {trained_gs_seg.get_xyz.shape[0]}")
    
    return scene, dataset, colmap_seed_gaussians, trained_gs_seg, scene_info


# -----------------------------
# Train Dynamic Instances
# -----------------------------
def train_clusters_dynamic_instance(scene, trained_gs_seg, colmap_seed, r_point_idx, r_gauss_idx, r_vals,
                                    iterations=500, lr=1e-3, temperature=0.1, max_dist=0.05, debug=False):
    """
    Initialize and train Gaussian clusters dynamically using responsibilities and COLMAP seeds.
    """
    gaussians = trained_gs_seg

    # Ensure gaussians tensors & logits are on the correct device
    # prefer to use already stored device; fallback to cpu
    try:
        device = gaussians.get_xyz.device
    except Exception:
        device = torch.device("cpu")

    # Make sure the gaussians' internal tensors are on the right device if possible
    # If GaussianModel implements .to(), use it.
    if hasattr(gaussians, "to"):
        try:
            gaussians.to(device)
        except Exception:
            pass

    # Ensure COLMAP seed has instance_ids and is on the same device
    if hasattr(colmap_seed, "cluster_ids") and colmap_seed.cluster_ids is not None:
        try:
            colmap_seed.instance_ids = colmap_seed.cluster_ids.to(device)
        except Exception:
            colmap_seed.instance_ids = colmap_seed.cluster_ids
    else:
        colmap_seed.instance_ids = torch.arange(colmap_seed.get_xyz.shape[0], device=device)

    # Initialize instance logits from COLMAP clusters
    cluster_utils.initialize_gaussian_instances(
        gaussians,
        colmap_points=colmap_seed.get_xyz.detach(),
        colmap_cluster_ids=colmap_seed.instance_ids.detach(),
        temperature=temperature,
        max_dist=max_dist
    )

    # freeze xyz positions
    gaussians.get_xyz.requires_grad_(False)
    optimizer = torch.optim.Adam([gaussians.instance_logits], lr=lr)

    for it in range(iterations):
        optimizer.zero_grad()
        q_i = gaussians.instance_logits
        p_j = torch.softmax(q_i, dim=1)

        total_loss, loss_vals, grad_q = total_cluster_loss(
            gaussians,
            r_point_idx,
            r_gauss_idx,
            r_vals,
            p_j,
            q_i,
            pair_j=None, pair_k=None,
            A=None, Kmat=None,
            gaussians_mask=None,
            contrib_indices=None,
            contrib_opacities=None,
            gt_mask=None,
            alpha_mask=None,
            use_label_ce=True,
            use_pair_kl=False,
            use_prop=False,
            use_smooth=False,
            use_marg=False,
            use_instance_render=False,
            debug=debug
        )

        grad_q = torch.nan_to_num(grad_q)
        q_i.backward(grad_q)
        optimizer.step()

    # Final hard assignment
    with torch.no_grad():
        gaussians.instance_ids = torch.argmax(gaussians.instance_logits, dim=1)

    return gaussians.get_xyz.detach(), gaussians.instance_logits.detach(), gaussians.instance_ids


# -----------------------------
# Spatially-Aware Score for Dense Clusters
# -----------------------------
def spatial_coherence_score(
    logits,
    ids,
    xyz,
    block_size=2048,   # safe default (adjust if needed)
):
    """
    Exact coherence-weighted spatial compactness score
    using batched pairwise distances (OOM-safe).
    """
    device = xyz.device
    eps = 1e-8

    coherence = torch.softmax(logits, dim=1)[
        torch.arange(len(ids), device=device), ids
    ]

    unique_ids = ids.unique()
    cluster_scores = []

    for cid in unique_ids:
        mask = ids == cid
        n = int(mask.sum())
        if n < 2:
            continue

        X = xyz[mask]                 # [n, 3]
        w = coherence[mask]
        w = w / (w.sum() + eps)       # normalize

        # precompute norms
        X_norm = (X ** 2).sum(dim=1)  # [n]

        weighted_sum = 0.0

        # ---- blockwise pairwise accumulation ----
        for i0 in range(0, n, block_size):
            i1 = min(i0 + block_size, n)
            Xi = X[i0:i1]
            wi = w[i0:i1]
            ni = i1 - i0

            Xi_norm = X_norm[i0:i1]

            for j0 in range(0, n, block_size):
                j1 = min(j0 + block_size, n)
                Xj = X[j0:j1]
                wj = w[j0:j1]

                # ||x||^2 + ||y||^2 - 2 x·y
                d2 = (
                    Xi_norm[:, None]
                    + X_norm[j0:j1][None, :]
                    - 2.0 * (Xi @ Xj.T)
                )

                # numerical safety
                d2 = torch.clamp(d2, min=0.0)
                d = torch.sqrt(d2 + eps)

                # weighted accumulation
                weighted_sum += (d * (wi[:, None] * wj[None, :])).sum()

        # normalize exactly like original
        weighted_mean_dist = weighted_sum / (n * n)

        cluster_scores.append(1.0 / (weighted_mean_dist + 1e-6))

    if len(cluster_scores) == 0:
        return 0.0

    return torch.stack(cluster_scores).mean().item()

# -----------------------------
# BO Threshold Optimization Only
# -----------------------------
def optimize_threshold_only(trained_gs_seg, n_calls=5,patience = 10):
    """
    Bayesian Optimization over coherence *quantile* instead of raw threshold.
    This guarantees that some Gaussians always survive.
    """

    # ---------------------------------------------------------
    # Precompute coherence distribution ONCE
    # ---------------------------------------------------------
    with torch.no_grad():
        logits = trained_gs_seg.instance_logits.detach()
        ids = trained_gs_seg.instance_ids.detach()
        coherence_all = cluster_utils.compute_instance_coherence(logits, ids)

    c_min = coherence_all.min().item()
    c_max = coherence_all.max().item()

    print(
        f"[BO] Coherence stats: min={c_min:.6f}, max={c_max:.6f}, "
        f"mean={coherence_all.mean().item():.6f}"
    )

    # ---------------------------------------------------------
    # BO search space: quantile, not threshold
    # ---------------------------------------------------------
    # q = 0.9  → keep top 10% most coherent Gaussians
    space = [Real(0.60, 0.98, name="coherence_quantile")]
    opt = Optimizer(space, random_state=42)

    best_score = -float("inf")
    best_q = None
    best_threshold = None
    no_improve_count = 0  # counter for patience

    # ---------------------------------------------------------
    # BO loop
    # ---------------------------------------------------------
    for i in tqdm(range(n_calls), desc="BO Threshold (quantile-based)"):
        [q] = opt.ask()

        # Convert quantile -> actual threshold
        threshold = torch.quantile(coherence_all, q).item()

        # Apply filtering
        filtered_model, mask = cluster_utils.filter_coherent_gaussians(
            trained_gs_seg, threshold=threshold
        )

        kept = mask.sum().item()
        total = mask.numel()

        # Safety check (should almost never trigger)
        if kept == 0:
            print(f"[BO {i:03d}] q={q:.3f} -> th={threshold:.6f} | kept=0 (SKIP)")
            opt.tell([q], 1e6)
            continue

        # Compute score
        filtered_logits = filtered_model.instance_logits.detach()
        filtered_ids = filtered_model.instance_ids.detach()
        filtered_xyz = filtered_model.get_xyz.detach()

        score = spatial_coherence_score(
            filtered_logits, filtered_ids, filtered_xyz
        )

        opt.tell([q], -score)

        print(
            f"[BO {i:03d}] q={q:.3f} -> th={threshold:.6f} | "
            f"kept={kept}/{total} ({100*kept/total:.2f}%) | "
            f"score={score:.6f}"
        )

        # Check for improvement
        if score > best_score:
            best_score = score
            best_q = q
            best_threshold = threshold
            no_improve_count = 0  # reset patience
        else:
            no_improve_count += 1

        # Early stopping
        if no_improve_count >= patience:
            print(f"[BO] No improvement for {patience} iterations → stopping early at iteration {i}")
            break

    # ---------------------------------------------------------
    # Fallback (should not happen, but safe)
    # ---------------------------------------------------------
    if best_q is None:
        best_q = 0.9
        best_threshold = torch.quantile(coherence_all, best_q).item()
        print(
            "[BO Coherence Threshold] No valid parameter set found — "
            f"fallback q={best_q:.3f}, threshold={best_threshold:.6f}"
        )
    else:
        print(
            f"[BO Threshold Only] Best q={best_q:.3f} "
            f"(threshold={best_threshold:.6f})"
        )

    # ---------------------------------------------------------
    # Apply best threshold to the actual model
    # ---------------------------------------------------------
    trained_gs_seg, _ = cluster_utils.filter_coherent_gaussians(
        trained_gs_seg, threshold=best_threshold
    )

    return best_threshold, trained_gs_seg





# -----------------------------
# Main
# -----------------------------
def main():

    # -----------------------------
    # Safe CUDA initialization
    # -----------------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
        print(f"[INFO] GPU allocated: {torch.cuda.memory_allocated() / 1e9:.4f} GB")
    else:
        print("[INFO] CUDA not available, running on CPU")

    parser = ArgumentParser(description="Gaussian Cluster Filtering Pipeline (merged)")

    # Keep original script parameter names and add data_device
    parser.add_argument("--colmap_dir", type=str, required=True,
                        help="Path to COLMAP reconstruction directory")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to Gaussian splatting model directory")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="Path to semantic masks directory")
    parser.add_argument("--mask_inst_dir", type=str, required=True,
                        help="Path to instance mask directory")

    parser.add_argument("--ply_iteration", type=int, default=30000,
                        help="Iteration number where point_cloud.ply is stored")

    parser.add_argument("--bo_steps", type=int, default=100,
                        help="Number of Bayesian Optimization steps")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate for dynamic instance training")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Iterations for dynamic instance refinement")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Softmax temperature for cluster initialization")
    parser.add_argument("--max_dist", type=float, default=0.05,
                        help="Max dist for COLMAP-based initialization")
    parser.add_argument("--data_device", type=str, default="cpu",
                        help="Device to use for data/model (e.g. 'cpu' or 'cuda')")

    parser.add_argument("--resolution", type=float, default=1,
                        help="Resolution scale for scene loading")

    args = parser.parse_args()

    # build paths for ply outputs
    ply_path = os.path.join(args.model_dir,
        f"point_cloud/iteration_{args.ply_iteration}/point_cloud.ply")

    out_ply_path = os.path.join(args.model_dir,
        f"point_cloud/iteration_{args.ply_iteration}/scene_clusters.ply")

    filtered_ply_path = os.path.join(args.model_dir,
        f"point_cloud/iteration_{args.ply_iteration}/filtered_scene_clusters.ply")

    # Start timing
    start_total = time.time()
    print(f"[DEBUG] Starting Filter at {time.strftime('%H:%M:%S')}")

    # Initialize scene; pass data_device so Scene.__init__ sets self.device = args.data_device
    scene, dataset, colmap_seed, trained_gs_seg, scene_info = initialize_scene(
        colmap_dir=args.colmap_dir,
        model_dir=args.model_dir,
        mask_inst_dir=args.mask_inst_dir,
        load_iteration=-1,
        num_its_BO=args.bo_steps,
        resolution=args.resolution,
        data_device=args.data_device
    )

    full_model = scene.gaussians  # the original trained full-resolution GS model
    print("[INFO] Scene initialized.")

    # Compute Top-K per full-model Gaussian
    print("\n[Step] Computing top-K contributors on full model ...")
    topK_full = cluster_utils.compute_topK_contributors(scene, K=1)

    # Build mapping full → segmented
    print("\n[Step] Computing full->seg mapping...")
    full_to_seg, kept_full = cluster_utils.compute_full_to_seg_map(trained_gs_seg, full_model)

    # Convert full-model topK → segmented topK
    print("\n[Step] Mapping top-K to segmented model...")
    topK_seg = cluster_utils.map_full_topK_to_segmented(topK_full, full_to_seg, kept_full)

    # Convert top-K to responsibilities
    print("\n[Step] Computing responsibilities...")
    r_point_idx, r_gauss_idx, r_vals = cluster_utils.topK_to_responsibilities(topK_seg)

    # Visualize colmap clusters (if functionality present)
    print("\n[Step] Visualizing clusters from COLMAP...")
    visualize_colmap_clusters(scene_info, scene)

    # -----------------------------
    # Move all responsibility tensors to the device of the Gaussians
    # -----------------------------
    device = trained_gs_seg.get_xyz.device
    r_point_idx = r_point_idx.to(device)
    r_gauss_idx = r_gauss_idx.to(device)
    r_vals      = r_vals.to(device)

    # Train clustering using dynamic instance refinement
    print("\n[Step] Training instance clusters...")
    xyz_final, logits_final, ids_final = train_clusters_dynamic_instance(
        scene,
        trained_gs_seg,
        colmap_seed,
        r_point_idx,
        r_gauss_idx,
        r_vals,
        iterations=args.iterations,
        lr=args.lr,
        temperature=args.temperature,
        max_dist=args.max_dist,
        debug=True
    )

    # Optimize coherence threshold only
    print("\n[Step] Running Bayesian Optimization to find best threshold...")
    best_threshold, trained_gs_seg = optimize_threshold_only(
        trained_gs_seg,
        n_calls=args.bo_steps,
        patience = 10
    )

    # Save filtered PLY
    print(f"\n[Step] Saving filtered PLY to: {filtered_ply_path}")
    trained_gs_seg.save_clustered_ply(filtered_ply_path, cluster_ids=trained_gs_seg.instance_ids)

    print("\n[Step] Visualizing FILTERED PLY...")
    visualize_clusters_from_ply(filtered_ply_path)

    runtime = time.time() - start_total
    print(f"[TIME] Finished at {time.strftime('%H:%M:%S')}, total runtime: {runtime:.2f}s")


if __name__ == "__main__":
    main()
