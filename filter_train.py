import os
import torch
from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.system_utils import mkdir_p
from argparse import ArgumentParser, Namespace
from utils.read_write_model import read_model
from utils.general_utils import safe_state, get_expon_lr_func
from utils.loss_utils import total_cluster_loss
from utils import cluster_utils
from utils.visualize_clusters import visualize_clusters_from_ply
from gaussian_renderer import render

from tqdm import tqdm


# -----------------------------
# Scene initialization
# -----------------------------
def initialize_scene(colmap_dir, model_dir, mask_inst_dir, load_iteration=30000):
    """
    Initialize the main Scene and also retrieve the COLMAP-seeded Gaussian model.
    
    Returns:
        scene: Scene instance (trained model loaded if available)
        dataset: configuration Namespace
        colmap_seed_gaussians: GaussianModel initialized from COLMAP point cloud
    """
    # -----------------------------
    # Parse arguments
    # -----------------------------
    parser = ArgumentParser()
    model_args = ModelParams(parser)

    model_args._source_path = os.path.abspath(colmap_dir)
    model_args._model_path = os.path.abspath(model_dir)
    model_args._images = "images"
    model_args._depths = ""
    model_args._resolution = 1.0
    model_args._white_background = False
    model_args.train_test_exp = False
    model_args.data_device = "cuda"
    model_args.eval = False

    # Wrap in Namespace for Scene compatibility
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

    # -----------------------------
    # Initialize trained scene
    # -----------------------------
    gaussians = GaussianModel(sh_degree=dataset.sh_degree)
    scene = Scene(dataset, gaussians,
                  load_iteration=load_iteration,
                  shuffle=False,
                  resolution_scales=[1.0])

    # -----------------------------
    # Load COLMAP-seeded Gaussians for clustering initialization
    # -----------------------------
    print("[INFO] Loading COLMAP-seeded Gaussian model...")
    trained_gs_seg, colmap_seed_gaussians, scene_info = scene.load_with_colmap_seed(
        args=dataset,
        load_iteration=load_iteration,
        mask_dir=mask_inst_dir
    )

    print("[OK] Scene initialized.")
    if trained_gs_seg is not None:
        print(f" - Loaded trained Segmented Gaussians: {trained_gs_seg.get_xyz.shape[0]} points")
    print(f" - Loaded COLMAP-seed Gaussians: {colmap_seed_gaussians.get_xyz.shape[0]} points")

    return scene, dataset, colmap_seed_gaussians, trained_gs_seg

# -----------------------------
# Compute top K contributors tensor 
# -----------------------------
def compute_topK_contributors(scene, K=8, bg_color=(0, 0, 0)):
    """
    Compute the top-K contributing pixels for each Gaussian across all training cameras.

    Args:
        scene (Scene): Scene object containing Gaussians and cameras.
        K (int): Number of top contributors to store per Gaussian.
        bg_color (tuple): Background color used for dummy rendering (RGB).

    Returns:
        dict: {
            "indices": LongTensor [N, num_cameras, K], pixel indices (-1 if none),
            "opacities": FloatTensor [N, num_cameras, K], 1.0 if contribution exists, else 0
        }
    """
    gaussians = scene.gaussians
    N = gaussians.get_xyz.shape[0]
    cameras = scene.getTrainCameras()
    num_cams = len(cameras)
    device = gaussians.get_xyz.device

    topK_contrib_indices = -torch.ones((N, num_cams, K), dtype=torch.long, device=device)
    topK_contrib_opacities = torch.zeros((N, num_cams, K), dtype=torch.float32, device=device)

    class DummyPipe:
        debug = False
        antialiasing = False
        compute_cov3D_python = False
        convert_SHs_python = False

    dummy_pipe = DummyPipe()
    bg_color_tensor = torch.tensor(bg_color, dtype=torch.float32, device=device)

    for cam_idx, cam in enumerate(tqdm(cameras, desc="Computing top-K contributors")):
        render_out = render(cam, gaussians, dummy_pipe, bg_color_tensor, contrib=True, K=K)

        if render_out is None or "contrib_indices" not in render_out:
            print(f"[WARN] No contributor data for {cam.image_name}, skipping.")
            continue

        contrib_indices = render_out["contrib_indices"]
        if contrib_indices is None:
            continue

        H, W, _ = contrib_indices.shape
        contrib_indices = contrib_indices.view(-1, K)

        valid_mask = contrib_indices >= 0
        g_ids_valid = contrib_indices[valid_mask]
        pixel_ids_valid = valid_mask.nonzero(as_tuple=True)[0]

        g_sorted, idx = torch.sort(g_ids_valid)
        pix_sorted = pixel_ids_valid[idx]

        unique_g = torch.unique_consecutive(g_sorted)
        start_mask = torch.ones_like(g_sorted, dtype=torch.bool)
        start_mask[1:] = g_sorted[1:] != g_sorted[:-1]
        start_idx = torch.nonzero(start_mask, as_tuple=True)[0]

        for g, s in zip(unique_g.tolist(), start_idx.tolist()):
            end = min(s + K, g_sorted.numel())
            slots = min(K, end - s)
            topK_contrib_indices[g, cam_idx, :slots] = pix_sorted[s:s + slots]
            topK_contrib_opacities[g, cam_idx, :slots] = 1.0

    return {
        "indices": topK_contrib_indices,
        "opacities": topK_contrib_opacities
    }

# ------------------------------------------
# Cluster-gaussians-modifying training loop
# ------------------------------------------
def train_clusters_dynamic(
    scene, colmap_seed_gaussians, trained_gs_seg, topk_contrib, mask_inst_dir,
    iterations=500, lr=1e-3,
    alpha=10.0, semantic_scale=1.0, radius=0.05,
    merge_distance=0.05, merge_semantic_diff=0.01,
    max_dispersion=0.05, max_semantic_std=0.1
):
    """
    Train Gaussian clusters dynamically using COLMAP seeds for initialization,
    **without moving the Gaussian positions**. Only the cluster assignments are refined
    using semantic + spatial losses, merging, and splitting.

    Args:
        scene (Scene): Scene object containing Gaussians and cameras
        colmap_seed_gaussians (GaussianModel): Seed points from COLMAP reconstruction
        trained_gs_seg (GaussianModel): Segmented Gaussian model
        topk_contrib (dict): Precomputed top-K pixel contributions per Gaussian
        mask_inst_dir (str): Instance mask directory
        iterations (int): Number of optimization iterations
        lr (float): Learning rate
        alpha (float): Weight for semantic term in hybrid loss
        semantic_scale (float): Weight for pseudo-ground-truth semantic loss
        radius (float): Neighbor radius for hybrid loss
        merge_distance (float): Distance threshold for merging clusters
        merge_semantic_diff (float): Semantic threshold for merging
        max_dispersion (float): Dispersion threshold for splitting clusters
        max_semantic_std (float): Semantic variance threshold for splitting clusters

    Returns:
        tuple: (xyz_final, sem_final, cluster_ids)
    """

    # Gaussian positions are fixed
    gaussians = trained_gs_seg
    device = gaussians.get_xyz.device

    # Only semantic mask is trainable
    gaussians.get_xyz.requires_grad_(False)
    gaussians.semantic_mask.requires_grad_(True)

    # Initialize clusters from COLMAP seeds (assignment only)
    gaussians_xyz = gaussians.get_xyz.detach()
    colmap_points = colmap_seed_gaussians.get_xyz.detach()
    colmap_cluster_ids = colmap_seed_gaussians.cluster_ids

    cluster_ids, cluster_centroids = cluster_utils.initialize_clusters_from_colmap(
        gaussians_xyz, colmap_points, colmap_cluster_ids
    )

    optimizer = torch.optim.Adam([gaussians.semantic_mask], lr=lr)
    print(f"[INFO] Initialized {len(torch.unique(cluster_ids))} clusters from COLMAP seeds")

    N = gaussians.get_xyz.shape[0]

    # ----------------------
    # Dynamic training loop
    # ----------------------
    for it in tqdm(range(iterations), desc="Dynamic Cluster Training"):
        optimizer.zero_grad()

        # Compute semantic predictions for active Gaussians
        semantic_pred = torch.sigmoid(gaussians.semantic_mask.squeeze())

        # Compute hybrid + semantic loss (cluster assignments fixed for this iteration)
        total_loss, hybrid_loss, semantic_loss = total_cluster_loss(
            gaussians,
            topK_contrib_indices=topk_contrib["indices"],
            mask_images=None,  # or your mask images if available
            alpha=alpha,
            neighbor_radius=radius,
            λ_sem=semantic_scale,
            debug=False
        )

        # Backpropagate
        total_loss.backward()
        optimizer.step()

        # -----------------------------
        # Cluster refinement: merge & split
        # -----------------------------
        with torch.no_grad():

            # -----------------------------------------
            # Compute cluster stats
            # -----------------------------------------
            num_clusters = int(cluster_ids.max().item() + 1)

            cluster_centroids, cluster_semantics, cluster_sizes = cluster_utils.compute_cluster_stats(
                gaussians_xyz,
                semantic_pred,
                cluster_ids,
                num_clusters
            )

            # -----------------------------------------
            # Build adjacency A between Gaussians × MaskInstances
            # -----------------------------------------
            # Use the active/trainable Gaussian subset

            seg_indices = torch.arange(gaussians.get_xyz.shape[0], device=gaussians_xyz.device)
            
            # Filter topK_contrib_indices
            topK_filtered = scene.topK_contrib_indices[seg_indices]  # [N_filtered, num_cameras, K]

            # Now call adjacency with filtered Gaussians
            A = scene.build_gaussian_instance_adjacency(
                topK_contrib_indices=topK_filtered,
                mask_images=scene.mask_instances,
                num_gaussians=gaussians.get_xyz.shape[0],
                num_mask_instances=scene.num_mask_instances,
                device=gaussians.get_xyz.device
            )

            # -----------------------------------------
            # Compute mask-center reliability weights
            # -----------------------------------------
            mask_center_weights = cluster_utils.compute_mask_center_weights(
                A,                                                     # sparse adjacency
                scene.instance_centroids,                              # [M,3] mask center pts
                gaussians_xyz                                          # [N,3]
            )                                                          # returns [N] weights ∈ [0,1]

            # -----------------------------------------
            # Merge (spatial + semantic + adjacency + center weights)
            # Uses the unified merge_clusters() function
            # -----------------------------------------
            cluster_ids, _ = cluster_utils.merge_clusters(
                cluster_centroids=cluster_centroids,
                cluster_semantics=cluster_semantics,
                cluster_ids=cluster_ids,
                merge_distance=merge_distance,
                merge_semantic_diff=merge_semantic_diff,
                A=A,                                # adjacency constraint
                mask_center_weights=mask_center_weights,
                adjacency_min_overlap=3,
                center_weight_pow=2.0
            )

            # -----------------------------------------
            # Multi-camera consistency
            # -----------------------------------------
            cluster_ids = cluster_utils.enforce_multicam_consistency_instance(
                cluster_ids,
                xyz=gaussians_xyz,
                cameras=scene.getTrainCameras(),
                mask_inst_dir=mask_inst_dir,
                pixel_tolerance=2.0,
                min_view_support=3
            )

            # -----------------------------------------
            # Splitting step (unchanged)
            # -----------------------------------------
            cluster_ids, _ = cluster_utils.split_cluster(
                gaussians_xyz,
                semantic_pred,
                cluster_ids,
                max_dispersion=max_dispersion,
                max_semantic_std=max_semantic_std
            )

            
            print(f"[INFO] {len(torch.unique(cluster_ids))} clusters")

    print("[OK] Dynamic cluster training completed.")
    return gaussians_xyz.detach(), gaussians.semantic_mask.detach(), cluster_ids


# # ---------------------------------
# # Cluster Assignment training loop
# # ---------------------------------
# def assign_clusters_dynamic(scene, colmap_seed_gaussians, trained_gs_seg, mask_inst_dir,
#                             merge_distance=0.05, merge_semantic_diff=0.01,
#                             max_dispersion=0.05, max_semantic_std=0.1):
#     """
#     Assign Gaussians to clusters using COLMAP seeds without moving their positions.
#     Optional semantic-aware merging/splitting is applied to cluster_ids only.

#     Args:
#         scene (Scene): Scene object containing Gaussians and cameras
#         colmap_seed_gaussians (GaussianModel): Seed points from COLMAP reconstruction
#         trained_gs_seg (GaussianModel): Segmented Gaussian model
#         mask_inst_dir (str): Path to mask instances for multi-camera consistency
#         merge_distance (float): Distance threshold for merging clusters
#         merge_semantic_diff (float): Semantic difference threshold for merging
#         max_dispersion (float): Dispersion threshold for splitting clusters
#         max_semantic_std (float): Semantic std threshold for splitting clusters

#     Returns:
#         cluster_ids (Tensor[N]): cluster assignment per Gaussian
#     """
#     gaussians = trained_gs_seg
#     device = gaussians.get_xyz.device

#     # Detach positions to prevent any accidental updates
#     xyz = gaussians.get_xyz.detach()
#     semantic_mask = torch.sigmoid(gaussians.semantic_mask.detach().squeeze())

#     # Initial assignment: nearest COLMAP seed per Gaussian
#     seed_xyz = colmap_seed_gaussians.get_xyz.detach()
#     dists = torch.cdist(xyz, seed_xyz)  # [N, K]
#     cluster_ids = torch.argmin(dists, dim=1)  # [N]

#     # Compute initial cluster stats
#     cluster_centroids, cluster_semantics, cluster_sizes = cluster_utils.compute_cluster_stats(
#         xyz, semantic_mask, cluster_ids, seed_xyz.shape[0]
#     )

#     # Step 1: spatial + semantic merging
#     cluster_ids, _ = cluster_utils.merge_clusters(
#         cluster_centroids,
#         cluster_semantics,
#         cluster_ids,
#         merge_distance=merge_distance,
#         merge_semantic_diff=merge_semantic_diff
#     )

#     # Step 2: multi-camera consistency
#     cluster_ids = cluster_utils.enforce_multicam_consistency_instance(
#         cluster_ids,
#         xyz=xyz,
#         cameras=scene.getTrainCameras(),
#         mask_inst_dir=mask_inst_dir,
#         pixel_tolerance=2.0,
#         min_view_support=3
#     )

#     # Step 3: optional re-split over-merged clusters
#     cluster_ids, _ = cluster_utils.split_cluster(
#         xyz,
#         semantic_mask,
#         cluster_ids,
#         max_dispersion=max_dispersion,
#         max_semantic_std=max_semantic_std
#     )

#     n_clusters = int(cluster_ids.unique().numel())
#     print(f"[OK] Cluster assignment completed. Total clusters: {n_clusters}")
#     return cluster_ids



# -----------------------------
# Main execution
# -----------------------------
def main():

    colmap_dir = "/home/samuelemara/colmap/samuele/lemons_only_sam2_masked"
    model_dir = "/home/samuelemara/gaussian-splatting-seg/output/28a55428-1"
    mask_dir = "/home/samuelemara/Grounded-SAM-2-autodistill/samuele/Lemon_only/masks"
    mask_inst_dir = "/home/samuelemara/Grounded-SAM-2-autodistill/samuele/Lemon_only/mask_instances"
    ply_path = os.path.join(model_dir, "point_cloud/iteration_30000/point_cloud.ply")
    out_ply_path = os.path.join(model_dir, "point_cloud/iteration_30000/scene_clusters.ply")

    # Load the tained model and the COLMAP points as seeds: [num_seeds, 3] tensor
    scene, dataset, colmap_seed,trained_gs_seg = initialize_scene(colmap_dir, model_dir, mask_inst_dir, load_iteration=-1)
    
    topK_contrib = compute_topK_contributors(scene, K=8)

    # Assign to the Scene so training code can access it
    scene.topK_contrib_indices = topK_contrib["indices"]
    scene.topK_contrib_opacities = topK_contrib["opacities"]

    xyz_final, sem_final, cluster_ids = train_clusters_dynamic(
        scene,
        colmap_seed,
        trained_gs_seg,
        topK_contrib,
        mask_inst_dir,
        iterations=500,
        lr=5e-4,
        alpha=10.0,
        semantic_scale=1.0,
        radius=0.05
    )
    # Update the Gaussian model tensors with final results
    with torch.no_grad():
        trained_gs_seg._xyz.copy_(xyz_final)
        trained_gs_seg.semantic_mask.copy_(sem_final)

    # cluster_ids = assign_clusters_dynamic(scene, colmap_seed, trained_gs_seg, mask_inst_dir)

    # Save clustered PLY from the segmented/trained Gaussians
    trained_gs_seg.save_clustered_ply(out_ply_path, cluster_ids=cluster_ids)

    # Visualize the clustered pointcloud
    visualize_clusters_from_ply(out_ply_path)

if __name__ == "__main__":
    main()
