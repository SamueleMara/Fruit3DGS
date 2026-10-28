# Fruit3DGS ablation automation plan — corrected contrastive-learning version

This document supersedes the previous version where `output_filter_standalone.py` was incorrectly treated as the contrastive clustering stage.

The correct contrastive-learning / instance-clustering stage is:

```text
filter_standalone.py
```

The other standalone file is:

```text
output_filter_standalone.py
```

and it should be treated only as a foreground Gaussian filtering utility, not as the instance embedding or DBSCAN/HDBSCAN clustering implementation.

---

## 1. Correct high-level pipeline

The pipeline used for the paper is conceptually:

```text
COLMAP dataset
   |
   v
train_sem.py
   |
   |  trains 3DGS radiance field + per-Gaussian semantic foreground score
   |  optionally filters final foreground Gaussians using binary masks/top-K contributors
   v
trained semantic Gaussian model
   |
   v
filter_standalone.py
   |
   |  computes pixel-to-Gaussian top-K contributor maps
   |  loads per-frame mask instances
   |  trains per-Gaussian contrastive instance embeddings
   |  clusters foreground Gaussians using HDBSCAN or seeded DBSCAN-like clustering
   v
filtered_scene_clusters_cc.ply
   |
   v
render_images_standalone.py
   |
   |  renders RGB, semantic heatmaps, instance-color views,
   |  Open3D point overlays, centroid overlays, optional OBBs
   v
paper images / qualitative outputs
```

Vanilla 3DGS comparison should use the normal `train.py` without the semantic mask loss, without per-Gaussian semantic mask learning, and without `filter_standalone.py`.

---

## 2. `train_sem.py`: what it does

`train_sem.py` is the modified 3DGS training stage. It keeps the normal 3DGS optimization loop but adds semantic foreground supervision from binary masks.

### Inputs

Typical inputs:

```bash
python train_sem.py \
  -s <COLMAP_DATASET> \
  -m <OUTPUT_MODEL_DIR> \
  --mask_dir <BINARY_MASK_DIR> \
  --topk_contrib 8 \
  --sem_threshold 0.7 \
  --disable_viewer
```

Expected dataset:

```text
COLMAP_DATASET/
  images/
  sparse/0/
    cameras.bin/txt
    images.bin/txt
    points3D.bin/txt
```

Expected masks:

```text
BINARY_MASK_DIR/
  <image_name>.png
  <image_name>.jpg
  ...
```

The masks are foreground/background masks, not instance masks.

### Core behavior

For every training iteration:

1. Select one random training camera.
2. Render the current Gaussians.
3. Compute the normal 3DGS photometric loss:
   - L1 RGB loss.
   - DSSIM / SSIM loss.
4. If the selected camera has a mask:
   - ask the renderer for top-K Gaussian contributors per pixel;
   - compute a binary mask render loss using `gaussians.semantic_mask`;
   - add the semantic mask loss with an exponential weight schedule.
5. Optionally add depth regularization if the camera has a reliable inverse depth map.
6. Run normal 3DGS densification/pruning.
7. Save Gaussians at the requested iterations.
8. At the final iteration, if `mask_dir` is valid, it can call `filter_and_save(...)` to write a foreground-filtered model.

### Important output

The important learned field is:

```python
gaussians.semantic_mask
```

This is a per-Gaussian scalar foreground score. Later scripts threshold it with `fg_thresh`.

### Critical note

`train_sem.py` does **not** perform fruit instance clustering. It only learns the foreground semantic score and trains the Gaussian scene.

---

## 3. `output_filter_standalone.py`: what it actually does

This script should be considered a foreground filtering helper.

It:

1. loads a trained semantic scene;
2. renders top-K contributors for each training camera;
3. loads binary foreground masks;
4. keeps Gaussians that:
   - have semantic score above a threshold, and
   - contribute to foreground pixels in at least one mask;
5. writes:

```text
point_cloud/iteration_<ITER>/scene_mask_filtered_renderer.ply
```

It does **not** train contrastive embeddings.
It does **not** run DBSCAN/HDBSCAN.
It does **not** create fruit instances.

Use this script only if you want a separate foreground-only PLY before clustering, or as a baseline foreground filter.

---

## 4. `filter_standalone.py`: actual contrastive instance-learning stage

This is the important file for the ablations.

It performs four main blocks:

```text
A. Scene loading
B. Top-K contributor / responsibility computation
C. Contrastive per-Gaussian embedding training
D. Foreground clustering using HDBSCAN or seeded DBSCAN-like clustering
```

---

## 4.1 Scene loading

Function:

```python
initialize_scene(colmap_dir, model_dir, mask_inst_dir, load_iteration=-1, load_filtered=True)
```

It builds a `Scene` object with:

```python
source_path = colmap_dir
model_path  = model_dir
images      = "images"
resolution  = 1.0
data_device = "cpu"
eval        = False
```

and then calls:

```python
Scene(dataset, gaussians, load_iteration=load_iteration, load_filtered=load_filtered, ...)
```

### Critical issue

In `main()`, `--ply_iteration` is used to build output paths, but it is **not passed** to `initialize_scene(...)`.

Current behavior:

```python
scene, dataset = initialize_scene(
    args.colmap_dir,
    args.model_dir,
    args.mask_inst_dir
)
```

Since `initialize_scene()` defaults to `load_iteration=-1`, the script loads the latest available model, while saving into:

```text
point_cloud/iteration_<args.ply_iteration>/
```

This can silently mismatch the loaded model and the output directory.

### Required patch

Add:

```python
parser.add_argument("--load_iteration", type=int, default=None)
parser.add_argument("--load_filtered", action="store_true", default=True)
parser.add_argument("--no_load_filtered", dest="load_filtered", action="store_false")
```

and load with:

```python
load_iteration = args.ply_iteration if args.load_iteration is None else args.load_iteration

scene, dataset = initialize_scene(
    args.colmap_dir,
    args.model_dir,
    args.mask_inst_dir,
    load_iteration=load_iteration,
    load_filtered=args.load_filtered,
)
```

---

## 4.2 Contributor computation

Main block:

```python
topK_full, r_point, r_gauss, r_vals = \
    cluster_utils.compute_topK_contributors_and_responsibilities(
        scene,
        K=args.contribs,
        storage_device=args.device
    )
```

Important returned fields used later:

```python
topK_full["pixel_topk_gaussians"]
topK_full["pixel_best_gaussian"]
topK_full["cam_hw"]
```

Meaning:

- `pixel_topk_gaussians`: for every camera/pixel, the top-K contributing Gaussian IDs.
- `pixel_best_gaussian`: strongest Gaussian per pixel.
- `cam_hw`: image size metadata needed by centroid-pull loss.

This is the bridge between 2D masks and 3D Gaussians.

---

## 4.3 Mask instance loading

The script expects instance masks, not just binary foreground masks.

It calls:

```python
mask_instances_dict = load_or_create_mask_instances(args.mask_inst_dir)
mask_instances_dict = filter_mask_instances_by_scene_cameras(mask_instances_dict, scene)
```

Expected semantic structure:

```python
mask_instances_dict[frame_name][instance_id] = {
    "pixel_indices": [...],
    "centroid_pixel": ...
}
```

The important parts are:

- per-frame local instance IDs;
- pixel indices belonging to each 2D instance mask;
- centroid pixel for centroid-guided embedding supervision.

The global identity used internally is frame-aware:

```python
(frame_name, instance_id)
```

This is important because `instance_id == 0` in two different frames does **not** necessarily mean the same fruit.

---

## 4.4 Centroid Gaussian computation

Before embedding training, the script computes a representative Gaussian for each 2D instance centroid:

```python
centroid_gauss_map, cent_stats = cluster_utils.compute_instance_centroid_gaussians(
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

This produces a mapping like:

```python
centroid_gauss_map[(frame_name, inst_id)] = gaussian_id
```

This is later used by the centroid-pull contrastive term and by the seeded DBSCAN-like clustering.

---

## 4.5 Gaussian-mask mappings

The script builds a graph between Gaussians and 2D mask instances:

```python
gaussian_to_mask_ids, mask_id_to_pixel_idx, pixel_idx_to_gaussians, \
gaussian_to_pixels, cam_offset_dict, global_id_map = build_gaussian_mask_mappings(...)
```

The only active downstream use in the current script is:

```python
cluster_utils.compute_hdbscan_bounds_from_mask(...)
```

which estimates reasonable `min_cluster_size` bounds from the Gaussian-to-mask statistics.

Large diagnostic blocks are present but commented out. These are useful for failure studies and should be exposed as a flag:

```bash
--dump_mapping_diagnostics
```

---

## 4.6 Contrastive embedding training

Function:

```python
train_instance_embeddings_graph(...)
```

This creates:

```python
gaussians.instance_embed = torch.nn.Parameter(...)
```

with shape:

```text
[N_gaussians, embed_dim]
```

Default `embed_dim` is 64.

### Loss 1 — same-frame pixel contrastive loss

At each iteration, the function samples one camera:

```python
cam_idx = torch.randint(...)
frame_name = cam_names[cam_idx]
topk = pixel_topk[cam_idx]
pix_inst = pixel_inst_list[cam_idx]
g_at_pix = topk[:, 0]
```

It uses the strongest Gaussian at each pixel and the 2D instance ID of that pixel.

Then it calls:

```python
frame_pixel_contrastive_loss(
    emb_norm=emb,
    g_at_pix=g_at_pix,
    pix_inst=pix_inst,
    pixels_per_iter=pixels_per_iter,
    neg_per_pos=neg_per_pos,
    temperature=temperature,
    bg_neg_weight=bg_neg_weight,
)
```

Purpose:

- pull together Gaussians explaining pixels from the same 2D mask instance;
- push apart Gaussians explaining different 2D instances;
- treat background as strong negatives.

This is the main 2D-to-3D instance supervision.

### Loss 2 — semantic-weighted spatial smoothness

The function builds a kNN graph in 3D:

```python
nbr_idx_cpu = cluster_utils.build_knn_graph_scipy(xyz, k=k_neighbors)
```

At every iteration it samples nodes and neighbors:

```python
nodes = torch.randint(...)
nbr = nbr_idx_cpu[nodes.cpu()].to(device)
```

and computes:

```python
w_ij = semantic_score_i * semantic_score_j
loss_smooth = weighted_embedding_difference
```

Purpose:

- nearby foreground Gaussians should have locally smooth embeddings;
- background or low-semantic-score nodes contribute weakly.

### Loss 3 — appearance-based contrastive pair loss

The script caches Gaussian RGB and opacity:

```python
rgb_all, op_all = full_model.get_rgb_opacity(clamp_rgb=True)
```

For kNN neighbors it calls:

```python
appearance_contrastive_pair_loss(
    emb_nodes=emb[nodes],
    emb_nbr=emb[nbr],
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
```

Purpose:

- use color/opacity similarity as a weak cue;
- encourage visually similar nearby foreground Gaussians to have compatible embeddings;
- penalize confusing visually dissimilar neighbors.

### Loss 4 — centroid pull loss

For one random 2D instance in the selected frame, the function finds:

```python
inst_pix
centroid_pixel
centroid_g
```

and calls:

```python
centroid_instance_pull_loss(...)
```

Purpose:

- pull the Gaussians belonging to one 2D instance toward the embedding of a centroid Gaussian;
- stabilize the instance around a physically meaningful seed;
- later provide seeds for seeded DBSCAN-like clustering.

### Total loss

Current total:

```python
loss = (
    pixel_weight * loss_pixel
    + smooth_weight * loss_smooth
    + app_weight * loss_app
    + centroid_weight * loss_cent
)
```

In `main()`, the current hardcoded call uses:

```python
pixel_weight    = 1.0
smooth_weight   = 1.0
app_weight      = 1.0
centroid_weight = 10.0
temperature     = 0.07
neg_per_pos     = 16
bg_neg_weight   = 8.0
```

These must become command-line parameters for ablation/failure studies.

---

## 4.7 Foreground selection before clustering

After embedding training, the script thresholds the semantic mask:

```python
sem = full_model.semantic_mask.detach().view(-1)
fg_idx = (sem > args.fg_thresh).nonzero(as_tuple=True)[0]
```

Important detail:

```python
sem = full_model.semantic_mask.detach().view(-1)
```

The sigmoid version is commented out. This means `fg_thresh` is applied directly to stored semantic scores.

For reproducible ablations, add:

```bash
--semantic_use_sigmoid
```

and use:

```python
sem_raw = full_model.semantic_mask.detach().view(-1)
sem = torch.sigmoid(sem_raw) if args.semantic_use_sigmoid else sem_raw
```

---

## 4.8 Embedded distance graph

The script builds a foreground-only kNN graph:

```python
k_knn = 100
nbr_fg_cpu = cluster_utils.build_knn_graph_scipy(xyz_fg, k=k_knn)
```

Then it builds a sparse precomputed distance graph:

```python
D_fg = cluster_utils.build_embedded_knn_distance_graph(
    xyz=xyz_fg,
    emb=emb_fg,
    sem=sem_fg,
    nbr_sub_cpu=nbr_fg_cpu,
    rgb=rgb_fg,
    op=op_fg,
    gamma_rgb=0.05,
    gamma_op=0.005,
    rgb_pow=1,
    op_pow=1,
    debug=True
)
```

This graph is what HDBSCAN and seeded DBSCAN-like clustering operate on.

Critical hardcoded values:

```python
k_knn = 100
gamma_rgb = 0.05
gamma_op  = 0.005
rgb_pow   = 1
op_pow    = 1
```

These must be exposed to CLI.

---

## 4.9 HDBSCAN clustering branch

Enabled with:

```bash
--cluster_alg hdbscan
```

The script:

1. Computes `min_cluster_size` bounds from Gaussian-mask mappings.
2. Runs Bayesian optimization over:
   - `min_cluster_size`
   - `min_samples`
3. Clusters with a precomputed sparse distance matrix.
4. Scores using:

```python
cluster_utils.semantic_weighted_quality_score(...)
```

Current call:

```python
best_mcs, best_ms, labels_fg = optimize_hdbscan_bo(
    D_csr=D_fg,
    xyz=xyz_fg,
    emb=emb_fg,
    semantic_mask=sem_fg,
    n_nodes=M,
    n_calls=100,
    patience=8,
    mcs_min=mcs_min,
    mcs_max=mcs_max,
    ms_min=15,
    ms_max=35,
    k_knn=k_knn,
)
```

Critical hardcoded values:

```text
n_calls = 100
patience = 8
ms_min = 15
ms_max = 35
```

Several dataset-specific parameter notes are left as comments, but they are not actually used.

---

## 4.10 Seeded DBSCAN-like branch

Enabled with:

```bash
--cluster_alg dbscan
```

This is **not vanilla sklearn DBSCAN**.

It is a seeded microcluster method:

1. Use centroid Gaussians as seeds.
2. For each seed, take nearest neighbors in the embedded graph.
3. Build seed-centered microclusters with `min_samples`.
4. Merge microclusters by IoU threshold.
5. Convert merged sets to labels.
6. Score with the same semantic/embedding quality score.
7. Run Bayesian optimization over:
   - `min_samples`
   - IoU threshold `t`

The function is:

```python
optimize_seeded_dbscan_iou_bo(...)
```

Important parameters:

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

### Critical current problem

The code computes adaptive `ms_min` and `ms_max` from seed graph degree statistics, then immediately overwrites them with hardcoded Brandenburg values:

```python
ms_min = 100
ms_max = 135
t_min  = 0.4
t_max  = 0.6
```

This is incompatible with large automatic sweeps across all datasets.

### Required patch

Add CLI parameters:

```bash
--dbscan_ms_min
--dbscan_ms_max
--dbscan_iou_min
--dbscan_iou_max
--dbscan_n_calls
--dbscan_patience
--dbscan_noise_penalty
--dbscan_min_intersection
--dbscan_require_full
--dbscan_conflict_policy
--dbscan_min_cluster_size
--dbscan_min_cluster_size_bo
```

Then only use dataset-specific overrides from a config file, never hardcoded values in the script.

---

## 4.11 Output

The final labels are lifted back to all Gaussians:

```python
ids_final = torch.full((N,), -1)
ids_final[fg_idx] = labels_fg
```

Then saved as:

```text
point_cloud/iteration_<ITER>/filtered_scene_clusters_cc.ply
```

Visualizations:

```python
visualize_clusters_from_ply(filtered_ply_path)
visualize_fg_clusters_with_centroid_gaussians(...)
```

### Critical issue

The output filename is always:

```text
filtered_scene_clusters_cc.ply
```

This is unsafe for ablation sweeps because every parameter run overwrites the previous result.

### Required patch

Create run-specific output directories:

```text
ablation_runs/
  <dataset_name>/
    <method_name>/
      <run_id>/
        clustered.ply
        metrics.json
        stdout.log
        stderr.log
        command.txt
        config.yaml
        timings.json
        gpu_stats.csv
```

Example run ID:

```text
direct_dbscan_xyz_eps0.015_min20_fg0.70
full_hdbscan_mcsauto_ms15-35_embed64_iter1000
seeded_dbscan_ms100-135_iou0.40-0.60_embed64_iter1000
```

---

# 5. Required ablation implementation

## 5.1 Ablation A — remove contrastive embeddings and directly run DBSCAN

This is the first ablation you described.

### What must be removed

Remove:

- `train_instance_embeddings_graph(...)`
- `instance_embed`
- embedded kNN graph
- HDBSCAN/seeded DBSCAN over embeddings

### What remains

Use the semantic foreground Gaussians from `train_sem.py`:

```python
sem = full_model.semantic_mask.detach().view(-1)
fg_idx = sem > fg_thresh
xyz_fg = full_model.get_xyz[fg_idx]
rgb_fg, op_fg = full_model.get_rgb_opacity(clamp_rgb=True)
```

Then run vanilla DBSCAN over a direct feature space.

### Baseline variants

Run at least these two variants:

#### A1 — geometry-only DBSCAN

Feature:

```python
X = xyz_fg
```

Parameters:

```text
eps = radius
min_samples = min_samples
```

#### A2 — geometry + color DBSCAN

Feature:

```python
X = concat([
    xyz_fg / xyz_scale,
    rgb_weight * rgb_fg
])
```

This tests whether color can partially replace learned embeddings.

Recommended CLI:

```bash
python cluster_direct_dbscan.py \
  --colmap_dir <COLMAP_DATASET> \
  --model_dir <TRAIN_SEM_MODEL> \
  --ply_iteration 30000 \
  --fg_thresh 0.7 \
  --eps 0.015 \
  --min_samples 20 \
  --feature_space xyz \
  --out_dir <RUN_DIR>
```

For RGB:

```bash
python cluster_direct_dbscan.py \
  --feature_space xyz_rgb \
  --eps 0.020 \
  --min_samples 20 \
  --rgb_weight 0.2
```

### Wide sweep

For each dataset:

```yaml
eps_values: [0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05]
min_samples_values: [3, 5, 8, 10, 15, 20, 30, 50, 80]
fg_thresh_values: [0.5, 0.6, 0.7, 0.8]
```

Do not evaluate only one global radius: each dataset has different scale.

### Fixed-parameter test

After the wide sweep, select one fixed parameter set and run it across all datasets.

Example fixed candidates to test:

```yaml
fixed_direct_dbscan:
  feature_space: xyz
  eps: 0.015
  min_samples: 20
  fg_thresh: 0.7
```

and:

```yaml
fixed_direct_dbscan_rgb:
  feature_space: xyz_rgb
  eps: 0.020
  min_samples: 20
  fg_thresh: 0.7
  rgb_weight: 0.2
```

The final paper comparison should report:

```text
Best per dataset
Fixed parameter across all datasets
```

This shows both upper-bound behavior and robustness/generalization.

---

## 5.2 Ablation B — full contrastive pipeline with parameterized clustering

This is your real method.

Run:

```bash
python filter_standalone.py \
  --colmap_dir <COLMAP_DATASET> \
  --model_dir <TRAIN_SEM_MODEL> \
  --mask_dir <BINARY_MASK_DIR> \
  --mask_inst_dir <INSTANCE_MASK_DIR> \
  --ply_iteration 30000 \
  --iterations 1000 \
  --embed_dim 64 \
  --contribs 1 \
  --fg_thresh 0.7 \
  --cluster_alg hdbscan
```

and:

```bash
python filter_standalone.py \
  --cluster_alg dbscan \
  --dbscan_ms_min ... \
  --dbscan_ms_max ... \
  --dbscan_iou_min ... \
  --dbscan_iou_max ...
```

### Required for fair sweeps

Add:

```bash
--save_embeddings <path>
--load_embeddings <path>
--skip_embedding_training
```

This avoids retraining embeddings for every DBSCAN/HDBSCAN parameter combination.

Correct structure:

```text
1. train embeddings once per dataset/seed
2. save embeddings
3. sweep clustering parameters many times
```

Otherwise timing and ablation results are confounded.

---

## 5.3 Ablation C — loss ablations inside contrastive training

Expose all loss weights:

```bash
--pixel_weight
--smooth_weight
--app_weight
--centroid_weight
```

Suggested runs:

```text
full:              pixel=1, smooth=1, app=1, centroid=10
no_appearance:     pixel=1, smooth=1, app=0, centroid=10
no_centroid:       pixel=1, smooth=1, app=1, centroid=0
no_smooth:         pixel=1, smooth=0, app=1, centroid=10
pixel_only:        pixel=1, smooth=0, app=0, centroid=0
centroid_only_bad: pixel=0, smooth=0, app=0, centroid=10
```

This will tell you whether the result is mostly driven by masks, appearance, centroid seeds, or actual multi-view contrastive lifting.

---

## 5.4 Ablation D — seeded DBSCAN vs HDBSCAN

Use the same trained embeddings and compare:

```text
HDBSCAN on embedded graph
Seeded DBSCAN-like microcluster/IoU merge
Direct vanilla DBSCAN on xyz
Direct vanilla DBSCAN on xyz+rgb
```

This is the clearest clustering-method ablation.

---

# 6. Timing and resource comparison

You need separate measurements for:

```text
1. train_sem.py
2. filter_standalone.py embedding training
3. filter_standalone.py clustering only
4. render_images_standalone.py
5. vanilla train.py
6. vanilla rendering
```

Do not report only end-to-end time.

## 6.1 Metrics to record

For every command:

```json
{
  "dataset": "...",
  "method": "...",
  "stage": "...",
  "wall_time_sec": 0,
  "max_gpu_mem_mb": 0,
  "avg_gpu_util_percent": 0,
  "max_gpu_util_percent": 0,
  "avg_cpu_percent": 0,
  "max_cpu_percent": 0,
  "peak_ram_mb": 0,
  "num_gaussians": 0,
  "num_fg_gaussians": 0,
  "num_clusters": 0,
  "noise_fraction": 0,
  "command": "...",
  "git_commit": "...",
  "docker_image": "..."
}
```

## 6.2 GPU logging

Use a wrapper script that launches:

```bash
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu \
  --format=csv \
  -lms 500
```

during each run.

Also call:

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

inside Python stages where possible.

## 6.3 CPU/RAM logging

Use `psutil` in the wrapper to sample:

```python
cpu_percent
memory_info().rss
```

for the process tree.

---

# 7. Evaluation outputs

For each clustering result, save:

```text
clustered.ply
cluster_summary.json
clusters_rendered/
centroids.json
obbs.json
metrics.json
```

Minimum useful metrics:

```json
{
  "num_clusters": 0,
  "num_noise_gaussians": 0,
  "noise_fraction": 0,
  "cluster_size_min": 0,
  "cluster_size_median": 0,
  "cluster_size_max": 0,
  "semantic_fg_count": 0,
  "assigned_fg_count": 0,
  "expected_fruit_count": null,
  "count_error": null
}
```

If you have ground-truth fruit counts per dataset, add:

```text
absolute count error
relative count error
precision/recall if GT centers are available
centroid localization error if GT centers are available
```

---

# 8. Docker integration

Everything should run inside the existing Docker environment.

Add a single entrypoint-style runner:

```bash
python experiments/run_fruit3dgs_ablation.py \
  --config experiments/fruit3dgs_ablation.yaml \
  --docker_mode inside
```

The outer host command can be:

```bash
docker run --gpus all --rm -it \
  -v /home/rhea/shared/samueleWS/Fruit_GS:/workspace/Fruit_GS \
  -v /home/rhea/shared/datasets:/workspace/datasets \
  -v /home/rhea/shared/results:/workspace/results \
  fruit3dgs:latest \
  bash -lc "cd /workspace/Fruit_GS && python experiments/run_fruit3dgs_ablation.py --config experiments/fruit3dgs_ablation.yaml"
```

---

# 9. Recommended config file

Create:

```text
experiments/fruit3dgs_ablation.yaml
```

Example:

```yaml
global:
  iteration: 30000
  image_folder: images
  fg_thresh: 0.7
  seeds: [0]
  docker_image: fruit3dgs:latest
  output_root: /workspace/results/fruit3dgs_ablation

datasets:
  - name: lemon
    colmap_dir: /workspace/datasets/lemon/colmap
    binary_mask_dir: /workspace/datasets/lemon/masks_binary
    instance_mask_dir: /workspace/datasets/lemon/masks_instances
    train_sem_model_dir: /workspace/results/models/lemon_sem
    vanilla_model_dir: /workspace/results/models/lemon_vanilla
    expected_count: null

  - name: fruitnerf
    colmap_dir: /workspace/datasets/fruitnerf/colmap
    binary_mask_dir: /workspace/datasets/fruitnerf/masks_binary
    instance_mask_dir: /workspace/datasets/fruitnerf/masks_instances
    train_sem_model_dir: /workspace/results/models/fruitnerf_sem
    vanilla_model_dir: /workspace/results/models/fruitnerf_vanilla
    expected_count: null

experiments:
  direct_dbscan_wide:
    enabled: true
    feature_spaces: [xyz, xyz_rgb]
    eps_values: [0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05]
    min_samples_values: [3, 5, 8, 10, 15, 20, 30, 50, 80]
    fg_thresh_values: [0.5, 0.6, 0.7, 0.8]
    rgb_weights: [0.1, 0.2, 0.5]

  direct_dbscan_fixed:
    enabled: true
    feature_space: xyz
    eps: 0.015
    min_samples: 20
    fg_thresh: 0.7

  full_hdbscan:
    enabled: true
    embedding_iterations: 1000
    embed_dim: 64
    contribs: 1
    hdbscan_ms_min: 15
    hdbscan_ms_max: 35
    hdbscan_n_calls: 100

  seeded_dbscan:
    enabled: true
    embedding_iterations: 1000
    embed_dim: 64
    contribs: 1
    ms_min_values: [20, 40, 65, 100]
    ms_max_values: [60, 100, 155]
    iou_min_values: [0.0, 0.2, 0.4]
    iou_max_values: [0.2, 0.6, 0.8]

  timing:
    enabled: true
    compare:
      - vanilla_3dgs_train
      - fruit3dgs_train_sem
      - fruit3dgs_filter_standalone
      - vanilla_render
      - fruit3dgs_render_images
```

---

# 10. Required code files to implement

## 10.1 `experiments/run_fruit3dgs_ablation.py`

Responsibilities:

- read YAML config;
- iterate over datasets;
- run training if outputs do not exist;
- run direct DBSCAN wide sweeps;
- select best per-dataset parameters;
- run fixed-parameter DBSCAN;
- run full contrastive pipeline;
- run timing comparison;
- write a single `summary.csv` and `summary.json`.

## 10.2 `experiments/cluster_direct_dbscan.py`

Responsibilities:

- load trained semantic Gaussian model;
- load correct iteration;
- threshold foreground semantic mask;
- build features:
  - `xyz`
  - `xyz_rgb`
  - optionally `xyz_rgb_opacity`
- run sklearn DBSCAN;
- lift labels to all Gaussians;
- save clustered PLY;
- save metrics JSON.

## 10.3 Patch `filter_standalone.py`

Minimum patch list:

```text
[ ] pass --ply_iteration into scene loading
[ ] expose --load_iteration
[ ] expose --load_filtered / --no_load_filtered
[ ] expose all embedding loss weights
[ ] expose k_knn
[ ] expose gamma_rgb/gamma_op/rgb_pow/op_pow
[ ] expose HDBSCAN BO ranges and n_calls
[ ] expose seeded DBSCAN ms/t ranges and n_calls
[ ] remove hardcoded Brandenburg params
[ ] add --save_embeddings
[ ] add --load_embeddings
[ ] add --skip_embedding_training
[ ] add --out_dir / --run_name
[ ] stop overwriting filtered_scene_clusters_cc.ply
[ ] save metrics.json
[ ] save timings.json
[ ] save command.txt
```

## 10.4 `experiments/profile_run.py`

Responsibilities:

- run a subprocess;
- sample GPU with `nvidia-smi`;
- sample CPU/RAM with `psutil`;
- save logs and timing JSON.

## 10.5 `experiments/render_eval_outputs.py`

Responsibilities:

- call `render_images_standalone.py` on selected results;
- render consistent views for qualitative comparison;
- optionally render the same camera IDs for all methods.

---

# 11. Concrete command examples

## 11.1 Train semantic Fruit3DGS

```bash
python train_sem.py \
  -s /workspace/datasets/lemon/colmap \
  -m /workspace/results/models/lemon_sem \
  --mask_dir /workspace/datasets/lemon/masks_binary \
  --topk_contrib 8 \
  --sem_threshold 0.7 \
  --iterations 30000 \
  --disable_viewer
```

## 11.2 Run full contrastive HDBSCAN

```bash
python filter_standalone.py \
  --colmap_dir /workspace/datasets/lemon/colmap \
  --model_dir /workspace/results/models/lemon_sem \
  --mask_dir /workspace/datasets/lemon/masks_binary \
  --mask_inst_dir /workspace/datasets/lemon/masks_instances \
  --ply_iteration 30000 \
  --load_iteration 30000 \
  --iterations 1000 \
  --embed_dim 64 \
  --contribs 1 \
  --fg_thresh 0.7 \
  --cluster_alg hdbscan \
  --hdbscan_ms_min 15 \
  --hdbscan_ms_max 35 \
  --hdbscan_n_calls 100 \
  --out_dir /workspace/results/fruit3dgs_ablation/lemon/full_hdbscan
```

## 11.3 Train embeddings once, then sweep clustering

```bash
python filter_standalone.py \
  --colmap_dir /workspace/datasets/lemon/colmap \
  --model_dir /workspace/results/models/lemon_sem \
  --mask_dir /workspace/datasets/lemon/masks_binary \
  --mask_inst_dir /workspace/datasets/lemon/masks_instances \
  --ply_iteration 30000 \
  --load_iteration 30000 \
  --iterations 1000 \
  --embed_dim 64 \
  --save_embeddings /workspace/results/embeddings/lemon_embed64_iter1000.pt \
  --cluster_alg hdbscan \
  --out_dir /workspace/results/fruit3dgs_ablation/lemon/full_hdbscan_train_embed
```

Then:

```bash
python filter_standalone.py \
  --colmap_dir /workspace/datasets/lemon/colmap \
  --model_dir /workspace/results/models/lemon_sem \
  --mask_dir /workspace/datasets/lemon/masks_binary \
  --mask_inst_dir /workspace/datasets/lemon/masks_instances \
  --ply_iteration 30000 \
  --load_iteration 30000 \
  --skip_embedding_training \
  --load_embeddings /workspace/results/embeddings/lemon_embed64_iter1000.pt \
  --cluster_alg dbscan \
  --dbscan_ms_min 65 \
  --dbscan_ms_max 155 \
  --dbscan_iou_min 0.0 \
  --dbscan_iou_max 0.2 \
  --out_dir /workspace/results/fruit3dgs_ablation/lemon/seeded_dbscan_ms65-155_iou0-02
```

## 11.4 Direct vanilla DBSCAN ablation

```bash
python experiments/cluster_direct_dbscan.py \
  --colmap_dir /workspace/datasets/lemon/colmap \
  --model_dir /workspace/results/models/lemon_sem \
  --ply_iteration 30000 \
  --load_iteration 30000 \
  --fg_thresh 0.7 \
  --feature_space xyz \
  --eps 0.015 \
  --min_samples 20 \
  --out_dir /workspace/results/fruit3dgs_ablation/lemon/direct_dbscan_xyz_eps0015_min20
```

## 11.5 Vanilla 3DGS timing baseline

```bash
python train.py \
  -s /workspace/datasets/lemon/colmap \
  -m /workspace/results/models/lemon_vanilla \
  --iterations 30000 \
  --disable_viewer
```

---

# 12. Most important implementation priorities

Implement in this order:

1. Patch `filter_standalone.py` so it loads the correct iteration and does not overwrite outputs.
2. Add `--save_embeddings`, `--load_embeddings`, and `--skip_embedding_training`.
3. Implement `cluster_direct_dbscan.py`.
4. Implement the YAML-driven sweep runner.
5. Add profiling wrapper.
6. Add summary CSV/JSON generation.
7. Add rendering of selected representative outputs.

---

# 13. Key conceptual corrections

The correct interpretation is:

```text
train_sem.py
  = semantic foreground-aware 3DGS training

output_filter_standalone.py
  = optional foreground Gaussian filtering

filter_standalone.py
  = contrastive per-Gaussian instance embedding + HDBSCAN/seeded DBSCAN clustering

render_images_standalone.py
  = paper-quality rendering/export script

vanilla train.py
  = baseline 3DGS training without semantics/instances
```

Therefore, the main ablation should not modify `output_filter_standalone.py`.
It should either:

```text
A. bypass filter_standalone.py entirely and run direct DBSCAN on foreground Gaussians
```

or:

```text
B. patch filter_standalone.py to expose/disable individual pieces:
   - no appearance loss
   - no centroid loss
   - no smoothness loss
   - HDBSCAN vs seeded DBSCAN
   - loaded embeddings vs retrained embeddings
```

