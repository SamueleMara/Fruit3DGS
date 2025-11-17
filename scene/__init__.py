#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.read_write_model import read_model
from scene.colmap_masker import ColmapMaskFilter

from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement

from pathlib import Path
from tqdm import tqdm
import glob
from collections import defaultdict
import torch
from PIL import Image


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], mask_dir=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False, mask_dir=mask_dir
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True, mask_dir=mask_dir
            )

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, 
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    # --------------------------------------------------------------------------
    # NEW FUNCTION: load both trained model and COLMAP-seed Gaussian model
    # --------------------------------------------------------------------------
    def load_with_colmap_seed(self, args: ModelParams, load_iteration=None, mask_dir=None):
        """
        Loads both:
        - The trained Gaussian model (from .ply)
        - A new Gaussian model created from COLMAP point cloud as cluster seed
        with instance-aware clustering using mask instances.

        Returns:
            trained_gaussians: GaussianModel or None if not found
            seed_gaussians: GaussianModel created from COLMAP point cloud
            scene_info: metadata (train/test cameras, normalization, etc.)
        """
        model_path = self.model_path
        source_path = args.source_path

        # 1. Determine which iteration to load
        if load_iteration is not None and load_iteration == -1:
            load_iteration = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
            print(f"[Scene] Using trained model from iteration {load_iteration}")

        # 2. Load COLMAP or Blender dataset
        if os.path.exists(os.path.join(source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(source_path, "transforms_train.json")):
            print("[Scene] Found Blender transforms JSON — loading Blender dataset...")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                source_path, args.white_background, args.depths, args.eval)
        else:
            raise RuntimeError("Could not recognize scene type: no sparse/ or transforms_train.json found.")

        # 3. Load COLMAP model if needed and build 3D→2D mapping
        if not hasattr(self, "point_to_pixels"):
            if not hasattr(self, "points3D") or not hasattr(self, "images"):
                self.colmap_model_dir = os.path.join(source_path, "sparse", "0")
                self.load_colmap()
            self.build_point_pixel_mapping()

        # 4. Build instance-aware clusters from masks
        if mask_dir is None:
            raise RuntimeError("[Scene] mask_dir must be provided for instance-aware clustering")
        point_clusters = self.build_instance_seed_clusters(mask_dir)

        # 5. Create GaussianModel from COLMAP point cloud
        print("[Scene] Creating GaussianModel from COLMAP point cloud (cluster seed)...")
        seed_gaussians = GaussianModel(sh_degree=args.sh_degree)
        seed_gaussians.create_from_pcd(
            scene_info.point_cloud,
            scene_info.train_cameras,
            scene_info.nerf_normalization["radius"]
        )

        # 6. Map COLMAP points to corresponding Gaussians
        # Use the COLMAP point IDs from loaded COLMAP model
        colmap_pids = list(self.points3D.keys())
        colmap_pid_to_idx = {pid: idx for idx, pid in enumerate(colmap_pids)}

        gaussian_clusters = torch.full((seed_gaussians._xyz.shape[0],), -1, dtype=torch.long)
        for pid, cid in point_clusters.items():
            if pid in colmap_pid_to_idx:
                idx = colmap_pid_to_idx[pid]
                gaussian_clusters[idx] = cid

        # Attach cluster IDs to the Gaussian model
        seed_gaussians.cluster_ids = gaussian_clusters
        self.seed_gaussian_clusters = gaussian_clusters

        num_assigned = (gaussian_clusters >= 0).sum().item()
        print(f"[Scene] Assigned {num_assigned} clusters to COLMAP Gaussians.")
        num_clusters = len(torch.unique(gaussian_clusters[gaussian_clusters >= 0]))
        print(f"[Scene] Total {num_clusters} unique clusters assigned.")

        # 7. Load trained Gaussian model if available
        trained_gaussians = None
        if load_iteration is not None:
            trained_model_seg_path = os.path.join(
                model_path, "point_cloud", f"iteration_{load_iteration}", "scene_semantics_filtered.ply"
            )
            if os.path.exists(trained_model_seg_path):
                print(f"[Scene] Loading trained Gaussian model from {trained_model_seg_path}")
                trained_gaussians = GaussianModel(sh_degree=args.sh_degree)
                trained_gaussians.load_ply(trained_model_seg_path, args.train_test_exp)

        return trained_gaussians, seed_gaussians, scene_info

    # --------------------------------------------------------------------------
    # NEW FUNCTION: save the clustered ply with an additional cluster_id field
    # --------------------------------------------------------------------------
    def save_clustered_ply(self, path, gaussians, cluster_ids=None):
        """
            Save Gaussian points as PLY, optionally including cluster IDs for visualization.

            Args:
                path (str): output PLY path
                gaussians (GaussianModel): Gaussian model to save
                cluster_ids (Tensor[N], optional): cluster assignment per Gaussian
            """
        mkdir_p(os.path.dirname(path))
        xyz = gaussians._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = gaussians._opacity.detach().cpu().numpy()
        scale = gaussians._scaling.detach().cpu().numpy()
        rotation = gaussians._rotation.detach().cpu().numpy()

        semantics = None
        if gaussians.semantic_mask is not None:
            semantics = gaussians.semantic_mask.detach().cpu().numpy()[..., None]

        dtype_full = [(attribute, 'f4') for attribute in gaussians.construct_list_of_attributes()]
        if cluster_ids is not None:
            dtype_full.append(('cluster_id', 'i4'))

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attrs = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]
        attributes = np.concatenate([a if a.ndim == 2 else a.reshape(a.shape[0], -1) for a in attrs], axis=1)
        if semantics is not None:
            attributes = np.concatenate((attributes, semantics), axis=1)
        if cluster_ids is not None:
            cluster_np = cluster_ids.detach().cpu().numpy()[..., None]
            attributes = np.concatenate((attributes, cluster_np), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el], text=True).write(path)

        print(f"[OK] Clustered PLY saved at {path}")

    # ---------------------------------
    # NEW FUNCTION: load colmap model
    # ---------------------------------
    def load_colmap(self):
        """
        Load COLMAP model (cameras, images, and points3D) from self.colmap_model_dir.

        This wraps utils.read_write_model.read_model() and caches the loaded structures
        for later geometric reasoning, instance association, or cluster seeding.

        Expected directory structure:
            self.colmap_model_dir/
                cameras.txt
                images.txt
                points3D.txt

        Side effects:
            - Populates self.cameras, self.images, self.points3D
            - Resets camera graph and masks cache
        """
        print(f"[INFO] Loading COLMAP model from: {self.colmap_model_dir}")

        try:
            self.cameras, self.images, self.points3D = read_model(self.colmap_model_dir, ext=".txt")
        except Exception as e:
            print(f"[ERROR] Failed to load COLMAP model: {e}")
            raise

        print(f"[OK] Loaded {len(self.cameras)} cameras, {len(self.images)} images, {len(self.points3D)} 3D points")

        # Reset caches
        self.camera_graph = None
        self.camera_centers = {}
        if hasattr(self, "_masks_cache"):
            self._masks_cache.clear()

        return self.cameras, self.images, self.points3D

    # ---------------------------------
    # NEW FUNCTION: build 3D→2D point-pixel mapping
    # ---------------------------------
    def build_point_pixel_mapping(self, save_path=None):
        """
        Vectorized version: For each 3D COLMAP point, compute all 2D pixel coordinates in images where it is observed.

        Returns:
            dict[int, list[dict]]: Mapping point_id -> list of observations:
                {"image_id": int, "image_name": str, "xy": [float, float]}
        """

        if not hasattr(self, "points3D") or not hasattr(self, "images"):
            raise RuntimeError("[Scene] COLMAP model not loaded. Call `load_colmap()` first.")

        point_ids = []
        image_ids = []
        point2d_idxs = []

        # Collect all point observations
        for pid, pt in self.points3D.items():
            n_obs = len(pt.image_ids)
            if n_obs == 0:
                continue
            point_ids.append(np.full(n_obs, pid, dtype=int))
            image_ids.append(pt.image_ids)
            point2d_idxs.append(pt.point2D_idxs)

        if not point_ids:
            print("[Scene] No point observations found.")
            return {}

        point_ids = np.concatenate(point_ids)
        image_ids = np.concatenate(image_ids)
        point2d_idxs = np.concatenate(point2d_idxs)

        # Gather corresponding image names and pixel coordinates
        all_image_names = np.array([self.images[i].name for i in image_ids])
        all_xys = np.array([self.images[i].xys[idx] for i, idx in zip(image_ids, point2d_idxs)])

        # Build mapping dictionary
        point_to_pixels = {}
        for pid, img_name, xy, img_id in zip(point_ids, all_image_names, all_xys, image_ids):
            if pid not in point_to_pixels:
                point_to_pixels[pid] = []
            point_to_pixels[pid].append({
                "image_id": int(img_id),
                "image_name": img_name,
                "xy": xy.tolist()
            })

        print(f"[Scene] Built vectorized 3D→2D mapping for {len(point_to_pixels)} points")

        # Optionally save to JSON
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(point_to_pixels, f, indent=2)
            print(f"[Scene] Saved 3D→2D mapping to {save_path}")

        self.point_to_pixels = point_to_pixels
        return point_to_pixels
        
    # ------------------------------------------------------------
    # NEW FUNCTION: build instance-aware COLMAP seed clusters
    # ------------------------------------------------------------
    def build_instance_seed_clusters(self, mask_dir, device="cuda", save_path=None):
        """
        Build initial Gaussian cluster seeds based on mask instances.

        Args:
            mask_dir (str): Folder containing *_instance_*.png masks.
            device (str): Torch device for masks.
            save_path (str): Optional path to save JSON cluster mapping.

        Returns:
            dict: point_id -> cluster_id
        """
        

        if not hasattr(self, "point_to_pixels"):
            raise RuntimeError("[Scene] Run build_point_pixel_mapping() first.")

        #Load all mask instances (vectorized)
        mask_cache = {}  # frame_name -> tensor [N_instances, H, W]
        for mask_file in tqdm(sorted(glob.glob(os.path.join(mask_dir, "*_instance_*.png"))),
                            desc="[Scene] Loading instance masks"):
            basename = Path(mask_file).stem.rsplit("_instance_", 1)[0]
            if basename not in mask_cache:
                mask_cache[basename] = []

            mask = Image.open(mask_file).convert("L")
            mask_tensor = torch.from_numpy(np.array(mask, dtype=np.float32)/255.0).to(device)
            mask_cache[basename].append(mask_tensor)

        # Convert lists to stacked tensors
        for basename in mask_cache:
            mask_cache[basename] = torch.stack(mask_cache[basename], dim=0)  # [N_instances, H, W]

                # mask_cache: dict[frame_name] -> tensor[N_instances, H, W]
        self.mask_instances = [mask_cache[k] for k in sorted(mask_cache.keys())]  # list of tensors
        self.num_mask_instances = max([m.shape[0] for m in mask_cache.values()])  # max number of instances across frames

        #Assign points to mask instances
        point_to_instance = {}
        instance_to_points = defaultdict(list)

        for pid, observations in tqdm(self.point_to_pixels.items(), desc="[Scene] Assigning points to mask instances"):
            votes = {}
            for obs in observations:
                frame_name = Path(obs["image_name"]).stem
                xy = obs["xy"]
                if frame_name not in mask_cache:
                    continue
                masks = mask_cache[frame_name]  # [N_instances, H, W]
                x, y = int(round(xy[0])), int(round(xy[1]))
                # Clamp pixel coordinates
                x = max(0, min(x, masks.shape[2]-1))
                y = max(0, min(y, masks.shape[1]-1))
                # Which instances cover this pixel?
                instance_hits = (masks[:, y, x] > 0.5).nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
                for iid in instance_hits:
                    votes[iid] = votes.get(iid, 0) + 1

            if votes:
                # Assign to instance with most votes
                assigned_instance = max(votes, key=votes.get)
                point_to_instance[pid] = assigned_instance
                instance_to_points[assigned_instance].append(pid)
            else:
                # Assign singleton cluster for points not seen in any mask
                point_to_instance[pid] = -1
                instance_to_points[-1].append(pid)

        #Merge clusters transitively (if A↔1, B↔1, C↔B → A,B,C same cluster)
        # Here a simple union-find can vectorize the merging
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x, y):
            parent[find(x)] = find(y)

        # Union points that share the same instance
        for pts in instance_to_points.values():
            if len(pts) > 1:
                base = pts[0]
                for p in pts[1:]:
                    union(base, p)

        # Assign cluster IDs
        clusters = {}
        cluster_id_map = {}
        cid = 0
        for pid in point_to_instance.keys():
            root = find(pid)
            if root not in cluster_id_map:
                cluster_id_map[root] = cid
                cid += 1
            clusters[pid] = cluster_id_map[root]

        print(f"[Scene] Built {len(set(clusters.values()))} initial instance-aware seed clusters")

        # Optionally save to JSON
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(clusters, f, indent=2)
            print(f"[Scene] Saved instance seed clusters to {save_path}")

        self.point_to_instance_cluster = clusters

        cluster_counts = np.bincount(list(clusters.values()))
        print(f"[Scene] Cluster sizes: {cluster_counts}")

        return clusters


    # ------------------------------------------------
    # NEW FUNCTION: build gaussian instance adjacency
    # ------------------------------------------------
    def build_gaussian_instance_adjacency(self, 
                                      topK_contrib_indices,
                                      mask_images,
                                      num_gaussians,
                                      num_mask_instances,
                                      device="cuda"):
        """
        Fully vectorized construction of the sparse adjacency matrix A[gaussian_id, instance_id] = 1.

        Args:
            topK_contrib_indices: [N_gaussians, num_cameras, K] indices of pixels each Gaussian contributes to
            mask_images: list of per-camera masks, each [N_instances, H, W]
            num_gaussians: total number of Gaussians
            num_mask_instances: total number of mask instances
            device: torch device
        """

        N, num_cameras, K = topK_contrib_indices.shape

        gaussian_ids_all = []
        instance_ids_all = []

        for cam_idx in range(num_cameras):
            topK_cam = topK_contrib_indices[:, cam_idx, :]  # [N, K]

            # Get mask for this camera
            mask_cam = mask_images[cam_idx]  # [N_instances, H, W]
            mask_flat = mask_cam.argmax(dim=0).reshape(-1)  # [H*W]

            # Flatten topK indices
            topK_flat = topK_cam.reshape(-1)                # [N*K]
            gaussian_ids = torch.arange(N, device=device).unsqueeze(1).expand(-1, K).reshape(-1)  # [N*K]

            # Gather instance IDs at topK pixel locations
            instance_ids = mask_flat[topK_flat]

            # Keep only valid pixels
            valid = instance_ids > 0
            gaussian_ids_all.append(gaussian_ids[valid])
            instance_ids_all.append(instance_ids[valid].to(device))

        # Concatenate all cameras
        gaussian_ids_final = torch.cat(gaussian_ids_all)
        instance_ids_final = torch.cat(instance_ids_all)

        # Clamp instance IDs to valid range
        instance_ids_final = instance_ids_final.clamp(0, num_mask_instances - 1)

        # Build sparse adjacency matrix
        indices = torch.stack([gaussian_ids_final, instance_ids_final], dim=0)  # [2, NNZ]
        values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)
        A = torch.sparse_coo_tensor(
            indices,
            values,
            size=(num_gaussians, num_mask_instances)
        ).coalesce()

        return A



