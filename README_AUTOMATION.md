# Fruit3DGS ablation automation code

This folder contains a proposed experiment framework to automate Fruit3DGS ablations inside the existing Docker/repo.

It is designed to be copied into the root of your Fruit3DGS/Fruit_GS repository:

```bash
cp -r experiments /workspace/Fruit_GS/
cd /workspace/Fruit_GS
```

Main goals:

1. Run the no-embedding direct-DBSCAN ablation.
2. Train contrastive embeddings once per dataset.
3. Sweep fixed HDBSCAN parameters over all datasets.
4. Compute count-level metrics from GT fruit counts.
5. Log all runs and parameter-effect figures to Weights & Biases.
6. Compare runtime and GPU/CPU/RAM usage for Fruit3DGS vs vanilla 3DGS.

---

## Important metric note

If the only ground truth is the number of fruits in each dataset, then precision/recall/F1 cannot be true detection metrics because there is no object-level matching.

The code therefore computes count-level proxy metrics:

```text
pred_count = number of non-noise clusters
gt_count   = known number of fruits

tp_proxy = min(pred_count, gt_count)

precision_count = tp_proxy / pred_count
recall_count    = tp_proxy / gt_count
f1_count        = 2 * precision_count * recall_count / (precision_count + recall_count)
```

This correctly penalizes over-counting through precision and under-counting through recall, but it does not verify whether the predicted clusters are spatially aligned with real fruits.

If later you add GT fruit centers, extend `counting_metrics.py` with Hungarian matching.

---

## Files

```text
experiments/
  fruit3dgs_ablation.yaml
  run_fruit3dgs_ablation.py
  profile_run.py
  counting_metrics.py
  cluster_direct_dbscan.py
  train_contrastive_embeddings.py
  cluster_hdbscan_fixed.py
  plot_hdbscan_parameter_effects.py
```

---

## Recommended workflow

### 1. Edit the config

```bash
nano experiments/fruit3dgs_ablation.yaml
```

Set:

- dataset names;
- COLMAP dataset roots;
- binary mask directories;
- instance mask directories;
- semantic model output directories;
- vanilla 3DGS model directories;
- GT fruit counts.

### 2. Run everything

```bash
python experiments/run_fruit3dgs_ablation.py \
  --config experiments/fruit3dgs_ablation.yaml
```

### 3. Produce W&B-ready HDBSCAN figures from the summary

```bash
python experiments/plot_hdbscan_parameter_effects.py \
  --summary_csv /workspace/results/fruit3dgs_ablation/summary.csv \
  --out_dir /workspace/results/fruit3dgs_ablation/hdbscan_param_figures \
  --wandb_project Fruit3DGS-Ablations \
  --wandb_run_name hdbscan_parameter_effects
```

---

## Why HDBSCAN is split from embedding training

The original `filter_standalone.py` trains embeddings and then immediately performs clustering.

For parameter sweeps, this is inefficient and confounded:

```text
bad:
  train embeddings again for every min_cluster_size/min_samples pair

good:
  train embeddings once
  save embeddings
  sweep clustering parameters only
```

This framework does the good version:

```text
train_contrastive_embeddings.py
  -> embeddings.pt

cluster_hdbscan_fixed.py
  -> run many times with different HDBSCAN parameters
```

---

## W&B logging

If W&B is installed and you pass a project name in the config, each run logs:

- dataset name;
- method;
- HDBSCAN parameters;
- predicted count;
- GT count;
- count precision;
- count recall;
- count F1;
- count absolute error;
- runtime;
- GPU/CPU/RAM summaries.

The plotting script also logs aggregate heatmaps showing how:

```text
min_cluster_size x min_samples
```

affects:

```text
precision_count
recall_count
f1_count
count_abs_error
```

over all datasets.

---

## Expected output structure

```text
/workspace/results/fruit3dgs_ablation/
  summary.csv
  summary.json
  lemon/
    direct_dbscan_xyz_eps0.015_min20_fg0.70/
      clustered.ply
      metrics.json
      resources.json
      gpu_stats.csv
      stdout.log
      stderr.log
      command.txt

    embeddings/
      embeddings_iter1000_dim64.pt

    hdbscan_mcs20_ms15/
      clustered.ply
      metrics.json
      resources.json
      gpu_stats.csv
      stdout.log
      stderr.log
      command.txt

  hdbscan_param_figures/
    aggregate_f1_count_heatmap.png
    aggregate_precision_count_heatmap.png
    aggregate_recall_count_heatmap.png
    aggregate_count_abs_error_heatmap.png
```

---

## Minimum patches still recommended in original code

Even though this framework avoids rewriting `filter_standalone.py`, I still recommend patching the original script later:

1. Pass `--ply_iteration` into `initialize_scene(..., load_iteration=...)`.
2. Remove hardcoded dataset-specific DBSCAN ranges.
3. Add unique output directories.
4. Add `--save_embeddings`, `--load_embeddings`, and `--skip_embedding_training`.
5. Expose all loss weights and graph parameters as CLI arguments.

The scripts here already implement these ideas externally.
---

## True W&B HDBSCAN sweep

For the W&B sweep panel shown in the screenshot, use the dedicated sweep config:

```bash
wandb login
wandb sweep experiments/wandb_hdbscan_sweep.yaml
wandb agent <entity>/<project>/<sweep_id>
```

This uses `experiments/wandb_hdbscan_sweep_agent.py`.

One W&B run corresponds to one pair:

```text
hdbscan_min_cluster_size
hdbscan_min_samples
```

evaluated on all configured datasets. The run logs:

```text
macro_precision_count
macro_recall_count
macro_f1_count
macro_count_abs_error
```

plus per-dataset metrics.

Afterwards, create local heatmaps:

```bash
python experiments/plot_hdbscan_parameter_effects.py \
  --summary_csv /workspace/results/fruit3dgs_ablation/wandb_hdbscan_sweep/wandb_hdbscan_sweep_summary.csv \
  --out_dir /workspace/results/fruit3dgs_ablation/wandb_hdbscan_sweep/figures \
  --wandb_project Fruit3DGS-HDBSCAN-Sweep \
  --wandb_run_name hdbscan_parameter_heatmaps
```


---

## Automated seeded DBSCAN-like ablation

This package now includes:

```text
experiments/cluster_seeded_dbscan_auto.py
```

This is the clean automated wrapper for your `filter_standalone.py --cluster_alg dbscan` logic.

It reuses saved contrastive embeddings and then runs the seeded microcluster/IoU DBSCAN pipeline with automatic dataset-aware presets:

```text
2022-09-06_scaled -> brandeburg  -> ms=[100,135], t=[0.4,0.6]
tree_01/tree_02_merged/tree_03_merged -> fruitnerf -> ms=[65,155], t=[0.0,0.2]
Fuji -> fuji -> ms=[65,155], t=[0.0,0.3]
lemons_2026-02-19 -> lemon -> ms=[65,155], t=[0.0,0.2]
multifruit_2026-02-19 -> multifruit -> ms=[100,110], t=[0.03,0.04]
```

Enable it in:

```yaml
seeded_dbscan_auto:
  enabled: true
  preset: auto
```

Then run:

```bash
python experiments/run_fruit3dgs_ablation.py \
  --config experiments/fruit3dgs_ablation.yaml
```

Per dataset output:

```text
<output_root>/<dataset>/seeded_dbscan_auto_auto/
  clustered.ply
  metrics.json
  resources.json
  gpu_stats.csv
  cpu_stats.csv
  debug_seeded_dbscan.json
```

You can also run it manually:

```bash
python experiments/cluster_seeded_dbscan_auto.py \
  --colmap_dir /workspace/datasets/Fuji \
  --model_dir /workspace/results/models/fuji_sem \
  --mask_inst_dir /workspace/datasets/Fuji/masks_instances \
  --embeddings_path /workspace/results/fruit3dgs_ablation/Fuji/embeddings/embeddings_iter1000_dim64.pt \
  --out_dir /workspace/results/fruit3dgs_ablation/Fuji/seeded_dbscan_auto \
  --ply_iteration 30000 \
  --load_iteration 30000 \
  --dbscan_preset auto \
  --gt_count 42 \
  --dataset_name Fuji
```
