# W&B HDBSCAN sweep for Fruit3DGS

This adds a true W&B Sweep around the HDBSCAN ablation.

The important design choice is:

```text
one W&B run = one HDBSCAN parameter pair
              evaluated on all datasets
```

So the W&B parallel coordinates plot and parameter importance plot show the effect of:

```text
hdbscan_min_cluster_size
hdbscan_min_samples
```

on aggregate metrics over all datasets:

```text
macro_precision_count
macro_recall_count
macro_f1_count
macro_count_abs_error
```

Each run also logs per-dataset metrics:

```text
per_dataset/<dataset_name>/precision_count
per_dataset/<dataset_name>/recall_count
per_dataset/<dataset_name>/f1_count
per_dataset/<dataset_name>/count_abs_error
per_dataset/<dataset_name>/pred_count
per_dataset/<dataset_name>/gt_count
```

---

## Metric warning

With only GT fruit counts, precision/recall/F1 are count-level proxies:

```text
tp_proxy = min(pred_count, gt_count)
precision_count = tp_proxy / pred_count
recall_count = tp_proxy / gt_count
f1_count = harmonic mean
```

This is good for counting ablations, but it is not object-level detection precision/recall. True detection metrics need GT fruit centers, masks, or boxes.

---

## How to run the W&B sweep

From inside the Docker/container and repo root:

```bash
cd /workspace/Fruit_GS

wandb login
wandb sweep experiments/wandb_hdbscan_sweep.yaml
```

W&B prints a command like:

```bash
wandb agent <entity>/<project>/<sweep_id>
```

Run it:

```bash
wandb agent <entity>/<project>/<sweep_id>
```

To run multiple workers on the same machine or across machines, start more agents with the same sweep id. Make sure they share the same output directory and dataset mounts, or use a different `output_root` per worker.

---

## What the W&B UI will show

W&B automatically creates sweep visualizations such as parallel coordinates, parameter importance, and scatter plots after sweep runs exist.

For your case, use the metric dropdown to inspect:

```text
macro_f1_count
macro_precision_count
macro_recall_count
macro_count_abs_error
```

and also the per-dataset metrics.

---

## Local/paper figures

After the sweep, the agent writes a CSV:

```text
<output_root>/wandb_hdbscan_sweep/wandb_hdbscan_sweep_summary.csv
```

Create heatmaps:

```bash
python experiments/plot_hdbscan_parameter_effects.py \
  --summary_csv /workspace/results/fruit3dgs_ablation/wandb_hdbscan_sweep/wandb_hdbscan_sweep_summary.csv \
  --out_dir /workspace/results/fruit3dgs_ablation/wandb_hdbscan_sweep/figures \
  --wandb_project Fruit3DGS-HDBSCAN-Sweep \
  --wandb_run_name hdbscan_parameter_heatmaps
```

These figures show:

```text
min_cluster_size × min_samples -> precision_count
min_cluster_size × min_samples -> recall_count
min_cluster_size × min_samples -> f1_count
min_cluster_size × min_samples -> count_abs_error
```

aggregated over all datasets, plus one set per dataset.
