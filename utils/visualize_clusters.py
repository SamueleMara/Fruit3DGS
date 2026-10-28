import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from plyfile import PlyData


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "Open3D is required only for interactive cluster visualization. "
            "Install open3d or run the pipeline with visualization disabled."
        ) from exc
    return o3d


def visualize_clusters_from_ply(ply_path):
    """
    Load a PLY saved by GaussianModel.save_clustered_ply() and visualize clusters.
    Each cluster gets a unique color.
    """
    o3d = _require_open3d()

    # -------------------------
    # Read PLY
    # -------------------------
    ply = PlyData.read(ply_path)
    v = ply['vertex'].data

    # Extract xyz
    xyz = np.vstack([v['x'], v['y'], v['z']]).T

    # Check cluster field
    if 'cluster_id' not in v.dtype.names:
        raise ValueError("PLY has no 'cluster_id' field. Make sure you saved clustered PLY.")

    cluster_ids = v['cluster_id'].astype(np.int32)

    # -------------------------
    # Create Open3D point cloud
    # -------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # -------------------------
    # Assign colors per cluster
    # -------------------------
    unique_clusters = np.unique(cluster_ids)
    rng = np.random.default_rng(42)   # deterministic colors

    # Map cluster → color
    cluster_color_map = {cid: rng.random(3) for cid in unique_clusters}

    # Expand to per-point color array
    colors = np.array([cluster_color_map[cid] for cid in cluster_ids], dtype=np.float32)

    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"[INFO] Visualizing {xyz.shape[0]} points across {len(unique_clusters)} clusters...")

    # -------------------------
    # Show
    # -------------------------
    o3d.visualization.draw_geometries([pcd])

    # -------------------------
    # Filter out all negative cluster points (background)
    # -------------------------
    mask = cluster_ids >= 0
    xyz = xyz[mask]
    cluster_ids = cluster_ids[mask]

    # -------------------------
    # Create Open3D point cloud
    # -------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # -------------------------
    # Assign colors per cluster
    # -------------------------
    unique_clusters = np.unique(cluster_ids)
    rng = np.random.default_rng(42)   # deterministic colors

    # Map cluster → color
    cluster_color_map = {cid: rng.random(3) for cid in unique_clusters}

    # Expand to per-point color array
    colors = np.array([cluster_color_map[cid] for cid in cluster_ids], dtype=np.float32)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"[INFO] Visualizing {xyz.shape[0]} points across {len(unique_clusters)} clusters (excluding background).")

    # -------------------------
    # Show
    # -------------------------
    o3d.visualization.draw_geometries([pcd])


def visualize_fg_clusters_with_centroid_gaussians(
    ply_path,
    centroid_gauss_map,              # dict (frame_name, inst_id) -> gaussian_id
    marker_radius=0.01,
    marker_color=(1.0, 1.0, 1.0),
    color_markers_by_cluster=False,  # if True, marker color = cluster color of that gaussian
    max_markers=None,               # optional cap on number of markers total
    seed=42,
):
    """
    Visualize ONLY foreground clustered points (cluster_id >= 0) from a clustered PLY,
    and overlay markers at the Gaussian positions given by centroid_gauss_map.

    Assumptions:
      - PLY contains per-vertex fields: x,y,z, cluster_id
      - Gaussian IDs in centroid_gauss_map refer to the *full model gaussian indexing*,
        which matches the vertex ordering used by save_clustered_ply().
    """
    o3d = _require_open3d()

    print(f"[Step] Visualizing FG clusters + centroid gaussians → {ply_path}")

    # -------------------------
    # Read PLY
    # -------------------------
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data

    if "cluster_id" not in v.dtype.names:
        raise ValueError("PLY has no 'cluster_id' field. Did you save clustered PLY?")

    xyz_all = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)  # [N,3] full indexing
    cluster_all = v["cluster_id"].astype(np.int32)                      # [N]

    # -------------------------
    # FG only
    # -------------------------
    fg_mask = cluster_all >= 0
    xyz_fg = xyz_all[fg_mask]
    cluster_fg = cluster_all[fg_mask]

    if xyz_fg.shape[0] == 0:
        print("[WARN] No FG points (cluster_id >= 0). Nothing to visualize.")
        return

    unique_clusters = np.unique(cluster_fg)
    rng = np.random.default_rng(seed)

    # Map cluster -> color
    cluster_color_map = {cid: rng.random(3) for cid in unique_clusters}
    colors_fg = np.array([cluster_color_map[cid] for cid in cluster_fg], dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_fg.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors_fg.astype(np.float64))

    # -------------------------
    # Build centroid markers
    # -------------------------
    marker_meshes = []
    used_gauss = set()

    # gather gaussian ids (dedup)
    gauss_ids = []
    for (frame_name, inst_id), g_id in centroid_gauss_map.items():
        if g_id is None:
            continue
        g_id = int(g_id)
        if g_id < 0 or g_id >= xyz_all.shape[0]:
            continue
        if g_id in used_gauss:
            continue
        used_gauss.add(g_id)
        gauss_ids.append(g_id)

    if max_markers is not None and len(gauss_ids) > int(max_markers):
        gauss_ids = rng.choice(gauss_ids, size=int(max_markers), replace=False).tolist()

    for g_id in gauss_ids:
        g_xyz = xyz_all[g_id].astype(np.float64)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(marker_radius))
        sphere.compute_vertex_normals()

        if color_markers_by_cluster and cluster_all[g_id] >= 0:
            c = cluster_color_map.get(int(cluster_all[g_id]), np.array(marker_color, dtype=np.float32))
            sphere.paint_uniform_color([float(c[0]), float(c[1]), float(c[2])])
        else:
            sphere.paint_uniform_color([float(marker_color[0]), float(marker_color[1]), float(marker_color[2])])

        sphere.translate(g_xyz)
        marker_meshes.append(sphere)

    print(
        f"[INFO] Visualizing FG only: {xyz_fg.shape[0]} points, "
        f"{len(unique_clusters)} clusters, {len(marker_meshes)} centroid-gaussian markers."
    )

    o3d.visualization.draw_geometries([pcd, *marker_meshes])


@torch.no_grad()
def plot_cosine_vs_index_distance(embeddings, max_pairs=200_000):
    emb = torch.nn.functional.normalize(embeddings, dim=1)
    N = emb.shape[0]

    # random pairs with bounded index distance
    i = torch.randint(0, N, (max_pairs,))
    j = torch.randint(0, N, (max_pairs,))

    idx_dist = (i - j).abs().cpu()
    cos_sim = (emb[i] * emb[j]).sum(dim=1).cpu()

    plt.figure(figsize=(7,5))
    plt.scatter(idx_dist, cos_sim, s=1, alpha=0.05)
    plt.xlabel("Gaussian index distance |i - j|")
    plt.ylabel("Cosine similarity")
    plt.title("Cosine similarity vs Gaussian index distance")
    plt.grid(True)
    plt.show()

@torch.no_grad()
def plot_cosine_histogram(embeddings, max_pairs=300_000):
    emb = torch.nn.functional.normalize(embeddings, dim=1)
    N = emb.shape[0]

    i = torch.randint(0, N, (max_pairs,))
    j = torch.randint(0, N, (max_pairs,))

    cos_sim = (emb[i] * emb[j]).sum(dim=1).cpu()

    plt.figure(figsize=(6,4))
    plt.hist(cos_sim.numpy(), bins=100, density=True)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title("Distribution of cosine similarities")
    plt.grid(True)
    plt.show()

    print(
        "[CosSim]",
        f"mean={cos_sim.mean():.4f}",
        f"std={cos_sim.std():.4f}",
        f">0.8={(cos_sim>0.8).float().mean()*100:.2f}%",
        f">0.5={(cos_sim>0.5).float().mean()*100:.2f}%"
    )

@torch.no_grad()
def plot_cosine_by_semantic(embeddings, semantic_mask, max_pairs=200_000):
    emb = torch.nn.functional.normalize(embeddings, dim=1)
    s = torch.sigmoid(semantic_mask).view(-1)

    # sample only foreground-heavy pairs
    idx = (s > 0.5).nonzero(as_tuple=True)[0]
    if idx.numel() < 2:
        print("Not enough high-semantic gaussians")
        return

    idx = idx[torch.randperm(idx.numel())[:max_pairs]]
    i = idx
    j = idx[torch.randperm(idx.numel())]

    cos_sim = (emb[i] * emb[j]).sum(dim=1).cpu()

    plt.figure(figsize=(6,4))
    plt.hist(cos_sim.numpy(), bins=100, density=True)
    plt.xlabel("Cosine similarity (semantic > 0.5)")
    plt.ylabel("Density")
    plt.title("Cosine similarity among semantic gaussians")
    plt.grid(True)
    plt.show()
