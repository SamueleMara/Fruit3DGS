# Fruit_GS Repository Analysis for Ablation and Failure Studies

This document describes the current repository as it exists on disk, with special focus on:

- `train_sem.py`
- `filter_standalone.py`
- `render_images_standalone.py`
- `train.py`

The goal is to make the current manual workflow explicit enough to automate repeated ablation studies and failure studies, including studies where a clustering parameter such as a DBSCAN/seeded-DBSCAN `min_samples` or IoU threshold is deliberately changed.

## Executive Summary

This repository is a modified 3D Gaussian Splatting codebase. It keeps the standard 3DGS training/rendering structure, but adds:

- A learnable per-Gaussian semantic scalar called `semantic_mask`.
- A modified rasterizer path that can return top-K Gaussian contributors per pixel.
- Mask-supervised training in `train_sem.py`.
- Final semantic/mask-based point cloud filtering via `filter.py`.
- Standalone instance embedding training and clustering in `filter_standalone.py`.
- Paper-image export in `render_images_standalone.py`.

Your historical workflow is:

1. Train a semantic Gaussian model with `train_sem.py`.
2. Use the trained model directory and the same COLMAP dataset as inputs to `filter_standalone.py`.
3. Produce clustered PLY files, especially `filtered_scene_clusters_cc.ply` and `filtered_scene_clusters_cc_no_bkg.ply`.
4. Render paper figures with `render_images_standalone.py`.

The most important automation target is `filter_standalone.py`: many ablation knobs are hard-coded inside the script today, so a robust experiment runner should either add CLI/config parameters or patch those values before each run.

## Repo Layout and Main Components

Important top-level files:

- `train.py`: vanilla 3DGS training entry point, fixed to avoid semantic/filtering behavior.
- `train_sem.py`: semantic/mask-supervised 3DGS training.
- `filter.py`: final mask/semantic PLY filtering used by `train_sem.py`.
- `filter_standalone.py`: loads a trained semantic model, trains instance embeddings, clusters foreground Gaussians, and writes clustered PLYs.
- `render_images_standalone.py`: exports RGB, semantic, instance, Open3D, and composite images for paper figures.
- `render.py`, `metrics.py`, `full_eval.py`: standard-ish 3DGS rendering/evaluation utilities.
- `arguments/__init__.py`: shared CLI parameter groups for loading, pipeline, and optimization parameters.
- `scene/__init__.py`: `Scene` creation, camera loading, loading trained or filtered PLYs.
- `scene/gaussian_model.py`: Gaussian storage, semantic field, PLY load/save, clustered PLY save.
- `gaussian_renderer/__init__.py`: Python wrapper around the modified rasterizer.
- `utils/masks_utils.py`: mask instance JSON creation and frame-name matching.
- `utils/cluster_utils.py`: top-K contributor processing, mapping masks to Gaussians, graph construction, HDBSCAN, seeded DBSCAN-like clustering helpers.
- `utils/loss_utils.py`: RGB/depth/SSIM losses plus semantic mask and embedding losses.

Important submodules:

- `submodules/diff-gaussian-rasterization`: modified rasterizer. It supports `contrib=True` and `K=...` to return top-K per-pixel Gaussian contributor indices and opacities.
- `submodules/simple-knn`: standard 3DGS nearest-neighbor CUDA helper.
- `submodules/fused-ssim`: optional fused SSIM.
- `submodules/pytorch3d`: present but not central to the four scripts analyzed here.

## Data and Output Contracts

### COLMAP Dataset Contract

Most scripts expect a COLMAP-style dataset:

```text
<colmap_dir>/
  images/
  sparse/
    0/
      cameras.bin or cameras.txt
      images.bin or images.txt
      points3D.bin or points3D.txt
```

`Scene` decides whether the input is COLMAP or Blender by checking for `sparse/` or `transforms_train.json`.

### Semantic Mask Directory Contract

`train_sem.py` takes `--mask_dir`. This is passed to `Scene`, which passes it to camera loading. Each camera may receive:

- `viewpoint_cam.mask`
- `viewpoint_cam.alpha_mask`

Those fields are used during semantic-mask training. The same `--mask_dir` is also passed to `filter.py` at the final iteration to write a filtered PLY.

Mask file matching in `filter.py` uses the image basename, lowercased, without extension. For a camera named `rgb_000.png`, it looks for a mask with basename `rgb_000` in `mask_dir`.

### Instance Mask Directory Contract

`filter_standalone.py` takes `--mask_inst_dir`. This directory should contain instance masks named like:

```text
<frame>_instance_<id>.png
```

Accepted extensions include PNG, JPG, JPEG, BMP, TIF, and TIFF, with upper-case variants.

If `mask_instances.json` is missing, `utils/masks_utils.py` creates it. Each instance entry stores:

```json
{
  "centroid": [cx, cy],
  "centroid_pixel": 12345,
  "bbox": [xmin, ymin, xmax, ymax],
  "area": 1234,
  "pixel_indices": [0, 1, 2]
}
```

The keys are normalized to match `scene.gs_cameras` keys.

### Model Directory Contract

The standard trained model directory looks like:

```text
<model_dir>/
  cfg_args
  cameras.json
  exposure.json
  point_cloud/
    iteration_30000/
      point_cloud.ply
      scene_mask_filtered_renderer.ply
      filtered_scene_clusters_cc.ply
      filtered_scene_clusters_cc_no_bkg.ply
```

Not all files exist after every stage.

## Core Modification: Semantic Gaussians

`scene/gaussian_model.py` adds a per-Gaussian field:

```python
self.semantic_mask = None
```

When Gaussians are created from COLMAP points, this field is initialized to `0.5` for every Gaussian. During training setup, it is added to the optimizer with the opacity learning rate:

```python
l.append({"params": [self.semantic_mask], "lr": training_args.opacity_lr, "name": "semantic_mask"})
```

When PLYs are saved, `semantic_mask` is written as a `semantics` vertex property. When PLYs are loaded, the `semantics` property is loaded back as `self.semantic_mask`. If a PLY does not contain `semantics`, the loader initializes the semantic field to `0.5` and prints a warning.

This means downstream filtering and rendering depend on the trained PLY containing a meaningful `semantics` field.

## Core Modification: Top-K Contributor Rasterization

`gaussian_renderer.render()` accepts:

```python
contrib=False
K=8
```

When `contrib=True`, the modified rasterizer returns:

- `contrib_indices`: shape `[H, W, K]`
- `contrib_opacities`: shape `[H, W, K]`

These arrays are central to:

- `train_sem.py` semantic mask supervision.
- `filter.py` semantic/mask final filtering.
- `filter_standalone.py` top-K contributor cache, Gaussian-mask mapping, centroid Gaussian selection, and embedding supervision.

The output dict always includes:

- `render`
- `viewspace_points`
- `visibility_filter`
- `radii`
- `depth`

and includes the contributor tensors only if `contrib=True`.

## `train_sem.py`

### Purpose

`train_sem.py` is the main semantic training script. It starts from a COLMAP/Blender scene, trains standard 3DGS appearance parameters, additionally learns a per-Gaussian `semantic_mask`, and optionally filters the final point cloud using `filter.py`.

### CLI

It uses the shared 3DGS argument groups from `arguments/__init__.py`, plus:

```text
--ip
--port
--debug_from
--detect_anomaly
--test_iterations
--save_iterations
--quiet
--disable_viewer
--checkpoint_iterations
--start_checkpoint
--mask_dir
--topk_contrib
--sem_threshold
```

Important defaults:

- `--iterations`: inherited default `30000`
- `--topk_contrib`: `8`
- `--sem_threshold`: `0.7`
- `--test_iterations`: `7000 30000`
- `--save_iterations`: `7000 30000`, plus `args.iterations` appended automatically

Example:

```bash
python train_sem.py \
  -s /path/to/colmap_scene \
  -m /path/to/output/model \
  --mask_dir /path/to/semantic_masks \
  --topk_contrib 10 \
  --sem_threshold 0.7 \
  --percent_dense 0.001 \
  --disable_viewer
```

### Training Flow

1. Validate `--mask_dir`.
   - If it exists, final filtering is enabled.
   - If absent or invalid, training still runs, but final filtering is skipped.

2. Create output directory and write `cfg_args`.

3. Create model and scene:

```python
gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
scene = Scene(dataset, gaussians, mask_dir=mask_dir)
gaussians.training_setup(opt)
```

Passing `mask_dir` into `Scene` is what attaches mask data to cameras.

4. Optionally restore from checkpoint.

5. Set background.

6. Create schedules:
   - Depth loss schedule from `depth_l1_weight_init` to `depth_l1_weight_final`.
   - Mask loss schedule from `mask_weight_init` to `mask_weight_final`.

7. For every iteration:
   - Handle viewer connection unless disabled.
   - Update learning rate.
   - Increase SH degree every 1000 iterations.
   - Randomly sample a training camera.
   - Render.
   - Compute RGB loss: L1 plus DSSIM/fused SSIM.
   - If the sampled camera has a mask, compute semantic binary mask render loss.
   - Optionally compute depth regularization.
   - Backpropagate.
   - Densify/prune until `densify_until_iter`.
   - Step optimizer.
   - Save/checkpoint/report as requested.

### Semantic Mask Loss

The semantic part activates when:

```python
viewpoint_cam.mask is not None
```

Then `render()` is called with:

```python
contrib=True
K=topk_contrib
```

The loss is:

```python
binary_mask_render_loss(
    gaussians.semantic_mask,
    contrib_indices,
    contrib_opacities,
    gt_mask,
    alpha_mask=...
)
```

This loss composites the per-Gaussian semantic scalar through the top-K per-pixel contributors and compares the rendered semantic mask against the ground-truth binary mask.

The total loss is:

```text
appearance_loss + current_mask_weight * mask_loss + optional_depth_loss
```

`current_mask_weight` follows an exponential schedule.

### Final Filtering

At the last iteration, if `mask_dir` was valid, `train_sem.py` calls:

```python
filter_and_save(
    scene,
    mask_dir=mask_dir,
    iteration=iteration,
    K=topk_contrib,
    K_keep=topk_contrib,
    semantic_threshold=semantic_threshold,
    semantic_use_sigmoid=False,
    storage_device="cpu",
    missing_mask_policy="all_white",
)
```

This writes:

```text
<model_dir>/point_cloud/iteration_<iteration>/scene_mask_filtered_renderer.ply
```

Important detail: `semantic_use_sigmoid=False`, so `--sem_threshold` is applied to raw `semantic_mask` values, not sigmoid values.

### Outputs

Expected outputs:

```text
<model_dir>/
  cfg_args
  cameras.json
  exposure.json
  point_cloud/
    iteration_7000/point_cloud.ply
    iteration_30000/point_cloud.ply
    iteration_30000/scene_mask_filtered_renderer.ply
    iteration_30000/filter_visualizations/
```

### Automation-Relevant Parameters

Already exposed:

- `--iterations`
- `--mask_dir`
- `--topk_contrib`
- `--sem_threshold`
- `--mask_weight_init`
- `--mask_weight_final`
- `--lambda_dssim`
- `--depth_l1_weight_init`
- `--depth_l1_weight_final`
- `--percent_dense`
- `--densify_from_iter`
- `--densify_until_iter`
- `--densification_interval`
- `--densify_grad_threshold`
- `--optimizer_type`

Useful ablations:

- Mask loss weight schedule: `mask_weight_init`, `mask_weight_final`.
- Top-K used for semantic mask loss and final filtering: `topk_contrib`.
- Semantic threshold for final filtering: `sem_threshold`.
- `semantic_use_sigmoid` in final filtering is hard-coded to `False`. This should be exposed if comparing raw semantics vs sigmoid semantics.
- `missing_mask_policy` is hard-coded to `all_white`. This should be exposed for failure studies with missing masks.

## `filter.py`

### Purpose

`filter.py` is not the standalone clusterer. It is the final mask and semantic pruning helper called by `train_sem.py`.

### Filtering Logic

For every training camera:

1. Render with `contrib=True`.
2. Load the binary mask for that camera, or use the missing-mask policy.
3. For each pixel, rank contributors:
   - First prefer semantic-passing Gaussians.
   - Then rank by opacity.
4. Mark top `K_keep` contributors on white mask pixels.
5. At the end keep:

```text
semantic_keep AND contrib_keep
```

The saved file is:

```text
scene_mask_filtered_renderer.ply
```

If nothing is selected, the original scene is saved instead.

### Automation-Relevant Parameters

Function arguments:

- `K`
- `K_keep`
- `semantic_threshold`
- `semantic_use_sigmoid`
- `storage_device`
- `missing_mask_policy`

In the current `train_sem.py`, only `K`, `K_keep`, and `semantic_threshold` are effectively exposed through CLI. The other values are hard-coded at the call site.

## `filter_standalone.py`

### Purpose

`filter_standalone.py` is the main post-training instance clustering pipeline. It loads a trained semantic Gaussian model, computes top-K contributor maps, trains per-Gaussian instance embeddings using mask-instance supervision, clusters foreground Gaussians, and saves clustered PLYs.

This is the key script for ablations and failure studies.

### CLI

Current CLI:

```text
--colmap_dir       required
--model_dir        required
--mask_dir         required but currently not used in the main body
--mask_inst_dir    required
--ply_iteration    default 30000
--contribs         default 1
--iterations       default 1000
--lr               default 5e-4
--temperature      default 0.1
--smooth_dist_thresh default 0.05
--smooth_max_points  default 2000
--smooth_chunk       default 512
--embed_dim        default None, meaning 64
--device           default gpu
--fg_thresh        default 0.7
--k_centr          default 10
--cluster_alg      hdbscan or dbscan, default hdbscan
```

Critical issue: the default `--device gpu` is not a valid `torch.device` string in standard PyTorch. Use `--device cpu` or `--device cuda`. Since this is passed into `torch.device(storage_device)` in `cluster_utils.compute_topK_contributors_and_responsibilities`, the default can crash.

Recommended invocation:

```bash
python filter_standalone.py \
  --colmap_dir /path/to/colmap_scene \
  --model_dir /path/to/output/model \
  --mask_dir /path/to/semantic_masks \
  --mask_inst_dir /path/to/instance_masks \
  --ply_iteration 30000 \
  --contribs 8 \
  --iterations 1000 \
  --lr 5e-4 \
  --fg_thresh 0.7 \
  --k_centr 10 \
  --cluster_alg hdbscan \
  --device cpu
```

### Scene Loading

`initialize_scene()` builds a manual `Namespace` and calls:

```python
scene = Scene(
    dataset,
    gaussians,
    load_iteration=load_iteration,
    load_filtered=load_filtered,
    shuffle=False,
    resolution_scales=[1.0],
)
```

The default function signature is:

```python
load_iteration=-1
load_filtered=True
```

Therefore, by default it loads the latest iteration and uses the filtered trained model:

```text
scene_mask_filtered_renderer.ply
```

This is important. If you intended to cluster the unfiltered trained `point_cloud.ply`, set `load_filtered=False` or expose a CLI flag.

Current `main()` calls:

```python
initialize_scene(args.colmap_dir, args.model_dir, args.mask_inst_dir)
```

It does not pass `args.ply_iteration`, so it loads the latest iteration, not necessarily `--ply_iteration`. However, output paths are built from `--ply_iteration`. For automation, this should be made explicit and consistent.

### High-Level Pipeline

1. Build paths for:
   - `point_cloud.ply`
   - `scene_clusters.ply`
   - `filtered_scene_clusters.ply`

   Some of these variables are currently unused.

2. Load scene and `scene.gaussians`.

3. Compute top-K contributors and responsibilities:

```python
topK_full, r_point, r_gauss, r_vals = \
    cluster_utils.compute_topK_contributors_and_responsibilities(
        scene,
        K=args.contribs,
        storage_device=args.device,
    )
```

Because `skip_responsibilities=True` inside `cluster_utils` by default, `r_point`, `r_gauss`, and `r_vals` are empty tensors unless that helper is called differently. The useful outputs are in `topK_full`.

4. Load or create `mask_instances.json`.

5. Filter mask instances to scene camera keys.

6. Compute centroid Gaussians for each mask instance:

```python
cluster_utils.compute_instance_centroid_gaussians(
    scene=scene,
    mask_instances_dict=mask_instances_dict,
    K_centroid=args.k_centr,
    storage_device="cpu",
    method="max_semantic",
    dist_gate=0.05,
    min_keep=3,
    debug=True,
)
```

7. Build Gaussian-mask mappings using `pixel_best_gaussian`.

8. Train per-Gaussian instance embeddings.

9. Select foreground Gaussians:

```python
fg_idx = (sem > args.fg_thresh).nonzero(...)
```

Important detail: the code currently uses raw `full_model.semantic_mask`, not `sigmoid(full_model.semantic_mask)`.

10. Build a foreground kNN graph:

```python
k_knn = 100
nbr_fg_cpu = cluster_utils.build_knn_graph_scipy(xyz_fg, k=k_knn)
```

11. Build a sparse embedded distance graph using:

- xyz distance
- embedding cosine distance
- semantic difference
- RGB distance
- opacity distance

Hard-coded graph parameters:

```python
gamma_rgb=0.05
gamma_op=0.005
rgb_pow=1
op_pow=1
```

12. Cluster with either HDBSCAN or seeded DBSCAN-like microclusters.

13. Lift foreground labels back to all Gaussians.

14. Save:

```text
filtered_scene_clusters_cc.ply
filtered_scene_clusters_cc_no_bkg.ply
```

15. Run visualization helpers.

### Instance Embedding Training

`train_instance_embeddings_graph()` creates:

```python
gaussians.instance_embed = torch.nn.Parameter(0.1 * torch.randn(N, embed_dim))
```

It trains embeddings using four loss components:

1. Same-frame pixel contrastive loss:
   - Uses top-1 Gaussian per pixel from `topK_full["pixel_topk_gaussians"]`.
   - Uses `mask_instances_dict` to assign local instance labels per frame.

2. Semantic-weighted spatial smoothness:
   - Builds a scipy kNN graph on Gaussian xyz.
   - Penalizes embedding differences between neighboring Gaussians.
   - Weights by semantic foreground confidence.

3. Appearance contrastive loss:
   - Uses RGB from SH DC features and opacity.
   - Encourages similar-looking neighbors to be close and different-looking neighbors to separate.

4. Centroid pull loss:
   - Uses `centroid_pixel` from each instance.
   - Uses `centroid_gauss_map`.
   - Pulls instance pixels toward the selected centroid Gaussian embedding.

Current hard-coded call in `main()`:

```python
pixel_weight=1.0
smooth_weight=1.0
app_weight=1.0
centroid_weight=10.0
temperature=0.07
neg_per_pos=16
bg_neg_weight=8.0
```

The parser has `--temperature`, but it is not actually used in this call. The function receives `temperature=0.07` hard-coded.

### HDBSCAN Path

When `--cluster_alg hdbscan`:

1. The code derives `min_cluster_size` bounds from mask usage:

```python
factor_hi=1.0
floor_min=5
robust=True
q_lo=0.10
q_hi=0.90
```

2. It defines some commented dataset-specific values, but the actual optimizer call uses:

```python
n_calls=100
patience=8
mcs_min=mcs_min
mcs_max=mcs_max
ms_min=15
ms_max=35
k_knn=100
```

3. `optimize_hdbscan_bo()` runs Bayesian optimization over:

- `min_cluster_size`
- `min_samples`

4. Each candidate is evaluated by `cluster_utils.semantic_weighted_quality_score()`.

5. The best labels are returned.

Automation knobs that should be exposed:

- `hdb_n_calls`
- `hdb_patience`
- `hdb_mcs_min`
- `hdb_mcs_max`
- `hdb_ms_min`
- `hdb_ms_max`
- `hdb_factor_hi`
- `hdb_floor_min`
- `hdb_q_lo`
- `hdb_q_hi`
- `hdb_eps`
- `hdb_allow_single_cluster`
- scoring weights such as `microcluster_penalty`, `blob_penalty`, `size_floor`, `lambda_emb`

### Seeded DBSCAN-Like Path

When `--cluster_alg dbscan`, this is not sklearn DBSCAN. It is a custom seeded microcluster method:

1. Use centroid Gaussians from mask instances as seeds.
2. For each seed, take `min_samples - 1` nearest neighbors in the sparse distance graph.
3. Treat each seed neighborhood as a microcluster.
4. Merge microclusters whose IoU is above threshold.
5. Convert merged sets into labels.
6. Bayesian optimization searches over:
   - `min_samples`
   - `iou_t`

The helper is `optimize_seeded_dbscan_iou_bo()`.

Important exposed function parameters:

```python
ms_min
ms_max
t_min
t_max
min_cluster_size
min_cluster_size_bo
n_calls
patience
noise_penalty
require_full
conflict_policy
min_intersection
```

Current DBSCAN branch computes data-driven bounds, then overrides them with hard-coded Brandeburg-like values:

```python
ms_min = 100
ms_max = 135
t_min = 0.4
t_max = 0.6
```

Then it calls:

```python
n_calls=100
patience=8
noise_penalty=0.25
require_full=False
conflict_policy="largest"
min_intersection=2
min_cluster_size=mcs_min
min_cluster_size_bo=mcs_min
```

These are exactly the parameters to expose for ablation/failure studies.

### Current Risks and Bugs in `filter_standalone.py`

1. `--device` default is `gpu`, which is invalid for `torch.device()`. Use `cpu` or `cuda`.

2. `--mask_dir` is required but not meaningfully used in `main()`. Only `--mask_inst_dir` is used for instance masks.

3. `--ply_iteration` controls output paths, but `initialize_scene()` loads `load_iteration=-1` by default, meaning latest iteration. If latest differs from `--ply_iteration`, inputs and outputs can mismatch.

4. `initialize_scene()` defaults to `load_filtered=True`, so it loads `scene_mask_filtered_renderer.ply`, not `point_cloud.ply`. That is probably what you used after `train_sem.py`, but it should be explicit.

5. Some CLI args are parsed but unused or overridden:
   - `--temperature` is overridden by `temperature=0.07`.
   - `--smooth_dist_thresh`, `--smooth_max_points`, and `--smooth_chunk` are not used by the current main path.

6. HDBSCAN branch defines `ms_min`, `ms_max`, `t_min`, `t_max` comments, but actual HDBSCAN optimizer call uses hard-coded `ms_min=15`, `ms_max=35`. The `t_*` values are irrelevant to HDBSCAN.

7. DBSCAN branch computes percentile-based `ms_min` and `ms_max`, then overrides them with hard-coded values.

8. In `optimize_seeded_dbscan_iou_bo()`, the fixed-`ms` debug print uses `ms` instead of `ms_fixed` in one f-string. If that branch is reached with debug enabled and microclusters exist, it can raise `NameError`.

9. HDBSCAN optimizer has a likely scope issue: `min_deg` is assigned only inside `if debug:` but checked afterward. Because current calls use `debug=True`, it works. If `debug=False`, it can fail.

10. Many outputs are visualization side effects. For batch automation, visualizers may open windows or require display depending on implementation. Consider making visualization optional.

## `render_images_standalone.py`

### Purpose

This script renders the images used for the paper. It loads cameras from the COLMAP scene and loads PLYs directly from the model iteration directory.

It exports:

- RGB renderings on black background.
- Semantic renderings using a CloudCompare-like blue-green-yellow-red color ramp.
- Instance PLY renderings with real learned colors on white background.
- Instance PLY renderings with random cluster colors on white background.
- Open3D point-cloud renderings of instance clusters on white background.
- Composite images: dim RGB plus Open3D cluster points plus centroid markers.
- Optional OBB overlays.

### CLI

Important arguments:

```text
--colmap_dir
--model_dir
--ply_iteration
--images_folder
--sh_degree
--load_iteration
--load_filtered
--base_ply_name
--inst_ply_name
--semantic_use_sigmoid
--semantic_scale
--semantic_gamma
--semantic_intensity_power
--overlay_alpha
--o3d_disable_post
--o3d_force_pure_white
--o3d_no_force_pure_white
--o3d_white_thr
--o3d_black_bg_tol
--o3d_bgdiff_tol
--o3d_alpha_bg_thr
--seed
--point_size
--out_subdir
--o3d_save_rgba
--composite_base_alpha
--composite_o3d_alpha
--centroid_radius_px
--draw_obbs
--obbs_json
--obb_line_width
```

Typical command from `Istruzioni codice immagini.txt`:

```bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python render_images_standalone.py \
  --colmap_dir /workspace/samuele/Colmap/Lemons \
  --model_dir /workspace/output/Lemons \
  --ply_iteration 30000 \
  --base_ply_name point_cloud.ply \
  --inst_ply_name filtered_scene_clusters_cc_no_bkg.ply \
  --semantic_use_sigmoid \
  --overlay_alpha 0.6 \
  --point_size 8.0 \
  --semantic_gamma 5 \
  --o3d_disable_post \
  --o3d_force_pure_white
```

### Input PLYs

The script resolves:

```text
<model_dir>/point_cloud/iteration_<ply_iteration>/<base_ply_name>
<model_dir>/point_cloud/iteration_<ply_iteration>/<inst_ply_name>
```

Default:

```text
base_ply_name = point_cloud.ply
inst_ply_name = filtered_scene_clusters_cc_no_bkg.ply
```

`base_ply` must contain `semantics`, or the script tries:

```text
scene_semantics_filtered.ply
```

`inst_ply` must contain `cluster_id`.

### Rendering Flow

For each camera:

1. Render `base_g` with black background and save RGB.
2. Override Gaussian SH colors with semantic-ramp colors and save semantic image.
3. Render `inst_g` with original colors on white background and save.
4. Override `inst_g` colors with cluster colormap and save.
5. If Open3D works:
   - Render the clustered point cloud on white and black backgrounds.
   - Use depth and white/black recovery to remove background halo.
   - Save Open3D output.
   - Optionally save RGBA.
6. Project cluster centroids into the camera and save composite.
7. Optionally draw OBBs.

### Output Folders

By default under:

```text
<model_dir>/output/paper_images_simple/
```

Subfolders:

```text
rgb_full_blackbg/
semantic_cloudcompare_blackbg/
instances_real_rgb_whitebg/
instances_cluster_colors_whitebg/
open3d_instances_whitebg/
open3d_instances_rgba/
composite_rgbdim_o3d_centroids/
```

`open3d_instances_rgba/` is created only with `--o3d_save_rgba`.

### Automation-Relevant Parameters

For paper reproduction:

- `--ply_iteration`
- `--base_ply_name`
- `--inst_ply_name`
- `--semantic_use_sigmoid`
- `--semantic_gamma`
- `--semantic_scale`
- `--semantic_intensity_power`
- `--point_size`
- `--composite_base_alpha`
- `--composite_o3d_alpha`
- `--centroid_radius_px`
- `--out_subdir`
- `--seed`

For failure modes:

- Missing Open3D should not stop 3DGS image outputs, but Open3D and composites will be incomplete.
- Headless Open3D can fail depending on environment.
- `--o3d_disable_post` switches to legacy processing. This changes background/halo behavior.

## `train.py`

### Intended Purpose

In vanilla 3DGS, `train.py` should train only RGB/depth 3DGS without semantic masks or final filtering.

### Current State in This Repo

`train.py` has been cleaned back to the vanilla training path for ablation baselines:

```python
gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
```

The semantic/filtering-specific command-line options and final filtering call were removed from `train.py`; those behaviors remain in `train_sem.py` and `filter.py`.

Runtime note: in the current shell used for this analysis, `python train.py --help` cannot import `diff_gaussian_rasterization` because the active Python is 3.13 while the checked-in compiled extension binaries are for Python 3.10 and 3.7. Use the project training environment, or rebuild/install the CUDA extensions for the active Python version.

### Differences from `train_sem.py`

Compared with `train_sem.py`, the current `train.py`:

- Does not import `binary_mask_render_loss`.
- Does not compute semantic mask loss during training.
- Does not request top-K contributor outputs during training.
- Has no `--sem_threshold`.
- Has no `--mask_dir` or `--topk_contrib`.
- Does not call `filter_and_save`.
- Does not pass `mask_dir` into `Scene`, so cameras do not receive masks for semantic supervision.

### Baseline Recommendation

For ablation studies, define baselines clearly:

1. Vanilla 3DGS baseline:
   - Use `train.py`.
   - No semantic mask supervision.
   - No final filtering.

2. Semantic 3DGS baseline:
   - Use `train_sem.py` with fixed `--mask_dir`, `--topk_contrib`, `--sem_threshold`, and seed.

3. Post-processing baseline:
   - Use a fixed trained semantic model.
   - Run `filter_standalone.py` with fixed embedding and clustering parameters.

## End-to-End Manual Workflow

### Stage 1: Train Semantic Model

```bash
python train_sem.py \
  -s "$COLMAP_DIR" \
  -m "$MODEL_DIR" \
  --mask_dir "$SEM_MASK_DIR" \
  --topk_contrib 10 \
  --sem_threshold 0.7 \
  --percent_dense 0.001 \
  --disable_viewer
```

Primary outputs:

```text
$MODEL_DIR/point_cloud/iteration_30000/point_cloud.ply
$MODEL_DIR/point_cloud/iteration_30000/scene_mask_filtered_renderer.ply
```

### Stage 2: Standalone Filtering and Clustering

```bash
python filter_standalone.py \
  --colmap_dir "$COLMAP_DIR" \
  --model_dir "$MODEL_DIR" \
  --mask_dir "$SEM_MASK_DIR" \
  --mask_inst_dir "$INST_MASK_DIR" \
  --ply_iteration 30000 \
  --contribs 8 \
  --iterations 1000 \
  --lr 5e-4 \
  --fg_thresh 0.7 \
  --k_centr 10 \
  --cluster_alg hdbscan \
  --device cpu
```

Primary outputs:

```text
$MODEL_DIR/point_cloud/iteration_30000/filtered_scene_clusters_cc.ply
$MODEL_DIR/point_cloud/iteration_30000/filtered_scene_clusters_cc_no_bkg.ply
```

### Stage 3: Render Paper Images

```bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python render_images_standalone.py \
  --colmap_dir "$COLMAP_DIR" \
  --model_dir "$MODEL_DIR" \
  --ply_iteration 30000 \
  --base_ply_name point_cloud.ply \
  --inst_ply_name filtered_scene_clusters_cc_no_bkg.ply \
  --semantic_use_sigmoid \
  --overlay_alpha 0.6 \
  --point_size 8.0 \
  --semantic_gamma 5 \
  --o3d_disable_post \
  --o3d_force_pure_white
```

Primary outputs:

```text
$MODEL_DIR/output/paper_images_simple/
```

## Recommended Automation Design

### Experiment Unit

Each run should have one immutable config object:

```yaml
name: lemon_hdbscan_ms_15_35
dataset:
  colmap_dir: /path/to/colmap
  sem_mask_dir: /path/to/semantic_masks
  inst_mask_dir: /path/to/instance_masks
training:
  script: train_sem.py
  model_dir: /path/to/output/lemon_hdbscan_ms_15_35
  iterations: 30000
  topk_contrib: 10
  sem_threshold: 0.7
  percent_dense: 0.001
standalone_filter:
  ply_iteration: 30000
  load_filtered: true
  contribs: 8
  embed_iterations: 1000
  embed_dim: 64
  lr: 0.0005
  fg_thresh: 0.7
  k_centr: 10
  cluster_alg: hdbscan
  hdb_ms_min: 15
  hdb_ms_max: 35
render:
  enabled: true
  semantic_use_sigmoid: true
  semantic_gamma: 5
  point_size: 8.0
  overlay_alpha: 0.6
```

### Directory Layout

Use one model directory per experiment:

```text
experiments/
  lemon_hdbscan_ms_15_35/
    config.yaml
    train.log
    filter.log
    render.log
    point_cloud/
    output/
```

Do not reuse the same `model_dir` for different ablations unless the stage is intentionally shared.

### Separate Expensive and Cheap Stages

Training is expensive. Clustering is cheaper. Structure experiments as:

1. Train once per training setting.
2. Reuse the same trained model for many `filter_standalone.py` clustering ablations.
3. Render only selected successful clustering outputs.

For example:

```text
training seed A
  clustering ablation hdbscan ms 15-35
  clustering ablation hdbscan ms 35-75
  clustering ablation dbscan ms 80-120 iou 0.2-0.5
  clustering ablation dbscan ms 100-135 iou 0.4-0.6
```

### Minimum Changes Needed for Good Automation

Before writing a runner, expose currently hard-coded values in `filter_standalone.py`.

Recommended new CLI arguments:

```text
--load_iteration
--load_filtered / --load_unfiltered
--semantic_use_sigmoid_for_fg
--k_knn
--graph_gamma_rgb
--graph_gamma_op
--graph_rgb_pow
--graph_op_pow
--pixel_weight
--smooth_weight
--app_weight
--centroid_weight
--neg_per_pos
--bg_neg_weight
--embedding_temperature
--hdb_n_calls
--hdb_patience
--hdb_ms_min
--hdb_ms_max
--hdb_mcs_min
--hdb_mcs_max
--db_ms_min
--db_ms_max
--db_iou_min
--db_iou_max
--db_n_calls
--db_patience
--db_noise_penalty
--db_require_full
--db_conflict_policy
--db_min_intersection
--skip_visualization
--output_suffix
```

The `--output_suffix` is important so many ablations can run in the same model iteration directory without overwriting `filtered_scene_clusters_cc.ply`.

Example output names:

```text
filtered_scene_clusters_cc_hdb_ms15_35.ply
filtered_scene_clusters_cc_hdb_ms15_35_no_bkg.ply
filtered_scene_clusters_cc_db_ms100_135_iou04_06.ply
filtered_scene_clusters_cc_db_ms100_135_iou04_06_no_bkg.ply
```

### Failure Studies to Add

Good failure-study axes:

- Too low/high semantic foreground threshold:
  - `fg_thresh`: 0.3, 0.5, 0.7, 0.9
- Raw semantics vs sigmoid semantics:
  - `sem > threshold` versus `sigmoid(sem) > threshold`
- Too few top-K contributors:
  - `contribs`: 1, 2, 4, 8, 16
- Bad centroid K:
  - `k_centr`: 1, 5, 10, 32, 64
- DBSCAN-like min_samples:
  - `ms_min`, `ms_max` ranges
  - fixed values by setting min=max
- DBSCAN-like IoU merge threshold:
  - low thresholds merge too much
  - high thresholds split too much
- Final minimum cluster size:
  - `min_cluster_size` and `min_cluster_size_bo`
- Missing masks:
  - Remove some instance masks.
  - Compare with and without `missing_mask_policy` changes in `filter.py`.
- No centroid pull:
  - `centroid_weight=0`
- No appearance loss:
  - `app_weight=0`
- No smoothness:
  - `smooth_weight=0`
- No pixel contrastive:
  - `pixel_weight=0`
- kNN graph density:
  - `k_knn`: 30, 50, 100, 150
- Graph appearance weights:
  - `gamma_rgb`, `gamma_op`

### Metrics to Log Per Run

At minimum, parse and store:

- Number of input Gaussians.
- Number of foreground Gaussians after `fg_thresh`.
- Number of mask instances.
- Number of centroid Gaussians found.
- Number of clusters.
- Number of background/noise Gaussians (`cluster_id < 0`).
- Number of no-background Gaussians.
- Cluster size distribution:
  - min
  - median
  - p90/p95
  - max
- BO best parameters.
- BO best score.
- Runtime per stage.
- Whether Open3D rendering succeeded.

These metrics can be extracted from logs and PLY fields.

## Suggested First Code Refactor

The smallest useful refactor is:

1. Add CLI arguments to `filter_standalone.py` for the hard-coded HDBSCAN and DBSCAN parameters.
2. Add `--output_suffix`.
3. Add `--skip_visualization`.
4. Change `--device` default from `gpu` to `cpu`.
5. Pass `--ply_iteration` into `initialize_scene(load_iteration=...)`.
6. Expose `--load_filtered` behavior.

After that, a separate runner script can call:

```bash
python train_sem.py ...
python filter_standalone.py ... --output_suffix ...
python render_images_standalone.py ... --inst_ply_name ...
```

without patching source code between runs.

## Current High-Priority Issues Before Automation

1. The active Python environment must be compatible with the compiled 3DGS CUDA extensions.
2. `filter_standalone.py --device gpu` default can crash.
3. `filter_standalone.py --ply_iteration` and actual loaded iteration can disagree.
4. `filter_standalone.py` overwrites `filtered_scene_clusters_cc.ply` on every run.
5. DBSCAN and HDBSCAN ablation parameters are mostly hard-coded.
6. Some parser arguments are unused, which can mislead experiment configs.
7. Visualizations are always run, which is inconvenient for batch jobs.

## Practical Baseline Matrix

Start with a small matrix before launching large sweeps:

| Experiment | Training | Filter Input | Cluster Alg | Main Ablation |
| --- | --- | --- | --- | --- |
| vanilla_rgb | `train.py` | none | none | pure 3DGS baseline |
| sem_default | `train_sem.py` | `scene_mask_filtered_renderer.ply` | hdbscan | default semantic pipeline |
| sem_raw_pointcloud | `train_sem.py` | `point_cloud.ply` | hdbscan | effect of final renderer filtering |
| hdb_ms_low | shared semantic model | filtered | hdbscan | `ms_min=5`, `ms_max=15` |
| hdb_ms_default | shared semantic model | filtered | hdbscan | `ms_min=15`, `ms_max=35` |
| db_iou_low | shared semantic model | filtered | dbscan | low IoU merge threshold |
| db_iou_high | shared semantic model | filtered | dbscan | high IoU merge threshold |
| no_centroid_pull | shared semantic model | filtered | hdbscan | `centroid_weight=0` |
| contrib1 | shared semantic model | filtered | hdbscan | `contribs=1` |
| contrib8 | shared semantic model | filtered | hdbscan | `contribs=8` |

## Added Ablation Automation

The repository now has an `ablations/` folder with runnable automation:

- `ablations/direct_dbscan_ablation.py`: removes the embedding-clustering stage entirely and runs plain DBSCAN on Gaussian XYZ positions. It reads a trained model PLY, writes a labeled PLY with `cluster_id`, writes a `_no_bkg.ply`, and records metrics in JSON.
- `ablations/run_ablation_studies.py`: manifest-driven launcher for direct DBSCAN wide sweeps, direct DBSCAN fixed-parameter runs, vanilla-vs-full timing/resource benchmarks, and optional contrastive-gaussian-clustering runs.
- `ablations/fruit_ablation_manifest.example.json`: starting config for the local datasets under `samuele/datasets`.
- `ablations/README.md`: command examples and output layout.

Main commands:

```bash
python ablations/run_ablation_studies.py \
  --config ablations/fruit_ablation_manifest.example.json \
  --section direct_dbscan_wide

python ablations/run_ablation_studies.py \
  --config ablations/fruit_ablation_manifest.example.json \
  --section direct_dbscan_fixed

python ablations/run_ablation_studies.py \
  --config ablations/fruit_ablation_manifest.example.json \
  --section benchmark

python ablations/run_ablation_studies.py \
  --config ablations/fruit_ablation_manifest.example.json \
  --section contrastive
```

The direct DBSCAN script is intentionally independent from the CUDA renderer. It only needs NumPy and scikit-learn, and it includes its own minimal binary PLY vertex-table reader/writer. This makes it useful for quick clustering sweeps even when the active Python environment cannot import the compiled 3DGS rasterizer.

The benchmark and contrastive sections still need the correct 3DGS/CGC Python-CUDA environment, because they call `train.py`, `train_sem.py`, `filter_standalone.py`, `render.py`, and the contrastive checkout.

## Notes for Reproducibility

- `safe_state(args.quiet)` in training sets the repo's RNG behavior, but `filter_standalone.py` has its own calls to `torch.manual_seed(seed)` inside embedding training. There is no CLI seed for the whole standalone pipeline yet.
- `skopt.Optimizer` uses `random_state=42` in both clustering optimizers, so BO is deterministic for fixed inputs.
- Random cluster colors in `render_images_standalone.py` are controlled by `--seed`.
- If multiple runs share the same model directory, later PLY outputs will overwrite earlier ones unless filenames are changed.
