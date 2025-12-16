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
        bo_optimize=True
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
def spatial_coherence_score(logits, ids, xyz):
    """
    Compute a coherence-weighted spatial compactness score for clustered Gaussians.
    """
    device = xyz.device
    coherence = torch.softmax(logits, dim=1)[torch.arange(len(ids), device=device), ids]

    unique_ids = ids.unique()
    cluster_scores = []

    for cid in unique_ids:
        mask = ids == cid
        if mask.sum() < 2:
            continue  # skip singleton clusters

        cluster_xyz = xyz[mask]
        cluster_coh = coherence[mask]

        # pairwise distances
        diff = cluster_xyz[:, None, :] - cluster_xyz[None, :, :]
        dists = torch.sqrt((diff ** 2).sum(-1) + 1e-8)

        # weighted intra-cluster distance
        w = cluster_coh / cluster_coh.sum()
        weighted_mean_dist = (dists * (w[:, None] @ w[None, :])).sum()

        # normalize by cluster size
        weighted_mean_dist /= mask.sum() ** 2
        cluster_scores.append(1.0 / (weighted_mean_dist + 1e-6))  # higher = more compact

    if len(cluster_scores) == 0:
        return 0.0
    return torch.stack(cluster_scores).mean().item()


# -----------------------------
# BO Dense Instance Cluster Optimization
# -----------------------------
def optimize_dense_instances(scene, trained_gs_seg, colmap_seed, r_point_idx, r_gauss_idx, r_vals,
                             n_calls=20, iterations=100, lr=5e-4):
    """
    Bayesian Optimization over temperature, max_dist, and coherence threshold
    for dense Gaussian clusters, keeping cluster assignments fixed and
    favoring spatially compact clusters.
    """

    # BO search space
    space = [
        Real(0.01, 1.0, name='temperature'),
        Real(0.001, 0.1, name='max_dist'),
        Real(0.05, 0.9, name='coherence_threshold')
    ]
    opt = Optimizer(space, random_state=42)
    best_score = -float('inf')
    best_params = None

    xyz_orig = trained_gs_seg.get_xyz.detach().cpu()
    ids_fixed = trained_gs_seg.instance_ids.detach().cpu()


    for i in tqdm(range(n_calls), desc="BO Dense Instance Clusters"):
        temperature, max_dist, coherence_threshold = opt.ask()

        # refine logits without modifying assignments
        xyz_final, logits_final, _ = train_clusters_dynamic_instance(
            scene, trained_gs_seg, colmap_seed,
            r_point_idx, r_gauss_idx, r_vals,
            iterations=iterations, lr=lr,
            temperature=temperature, max_dist=max_dist,
            debug=False
        )

        # spatially aware, coherence-weighted score
        score = spatial_coherence_score(logits_final, ids_fixed, xyz_orig)

        # maximize the score
        opt.tell([temperature, max_dist, coherence_threshold], -score)

        if score > best_score:
            best_score = score
            best_params = (temperature, max_dist, coherence_threshold)
            best_logits = logits_final.clone().detach()

    # Apply best parameters
    best_temperature, best_max_dist, best_threshold = best_params
    if best_params is None:
        best_temperature = temperature
        best_max_dist = max_dist
        best_threshold = 0.0
        print(
            "[BO Dense] No valid parameter set found — "
            f"using fallback: temperature={best_temperature:.3f}, "
            f"max_dist={best_max_dist:.3f}, threshold={best_threshold:.2f}"
        )
    else:
        best_temperature, best_max_dist, best_threshold = best_params
        print(
            f"[BO Dense] Best params: temperature={best_temperature:.3f}, "
            f"max_dist={best_max_dist:.3f}, threshold={best_threshold:.2f}")

    with torch.no_grad():
        trained_gs_seg.instance_logits.copy_(best_logits)
        trained_gs_seg.instance_ids.copy_(ids_fixed)  # keep IDs fixed

    # Filter coherent Gaussians using optimized threshold
    trained_gs_seg, mask_filt = cluster_utils.filter_coherent_gaussians(
        trained_gs_seg, threshold=best_threshold
    )

    return best_params, trained_gs_seg


# -----------------------------
# BO Threshold Optimization Only
# -----------------------------
def optimize_threshold_only(trained_gs_seg, n_calls=50):
    """
    Bayesian Optimization to find the best coherence threshold for
    already trained/fixed instance assignments.
    """

    space = [Real(0.05, 0.9, name='coherence_threshold')]
    opt = Optimizer(space, random_state=42)
    best_score = -float('inf')
    best_threshold = None

    ids_fixed = trained_gs_seg.instance_ids.detach()
    xyz = trained_gs_seg.get_xyz.detach()

    for i in tqdm(range(n_calls), desc="BO Threshold Only"):
        [coherence_threshold] = opt.ask()

        # Filter the model with the current threshold
        filtered_model, mask = cluster_utils.filter_coherent_gaussians(
            trained_gs_seg, threshold=coherence_threshold
        )

        # Skip thresholds that remove all Gaussians
        if filtered_model.get_xyz.shape[0] == 0:
            opt.tell([coherence_threshold], 1e6)  # very bad score
            continue

        # Compute spatial coherence score on filtered model
        filtered_logits = filtered_model.instance_logits.detach()
        filtered_ids = filtered_model.instance_ids.detach()
        filtered_xyz = filtered_model.get_xyz.detach()

        score = spatial_coherence_score(filtered_logits, filtered_ids, filtered_xyz)

        # maximize the score
        opt.tell([coherence_threshold], -score)

        if score > best_score:
            best_score = score
            best_threshold = coherence_threshold

    if best_threshold is None:
        threshold=0.557
        print(
            "[BO Coherence Threshold] No valid parameter set found — "
            f"using fallback:best_threshold={threshold:.3f}"

        )
    else:
        print(f"[BO Threshold Only] Best threshold: {best_threshold:.3f}")
        threshold=best_threshold
    
    # Apply best threshold to the actual model
    trained_gs_seg, _ = cluster_utils.filter_coherent_gaussians(
        trained_gs_seg, threshold
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
    topK_full = cluster_utils.compute_topK_contributors(scene, K=8)

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
        n_calls=args.bo_steps
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
