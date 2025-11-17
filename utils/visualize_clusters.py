import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData

def visualize_clusters_from_ply(ply_path):
    """
    Load a PLY saved by GaussianModel.save_clustered_ply() and visualize clusters.
    Each cluster gets a unique color.
    """
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

