# Fruit_GS Ablation Runner

This folder contains a small automation layer for the ablation/failure studies:

- `direct_dbscan_ablation.py` runs DBSCAN directly on trained Gaussian XYZ positions. It skips the learned instance-embedding clustering path in `filter_standalone.py`.
- `run_ablation_studies.py` reads a JSON/YAML manifest and launches sweeps, fixed-parameter runs, timing benchmarks, and optional contrastive-gaussian-clustering runs.
- `fruit_ablation_manifest.example.json` is a starting manifest for the datasets currently present under `samuele/datasets`.

Run these commands from the repository root. The direct DBSCAN ablation only needs NumPy and scikit-learn. The benchmark and contrastive sections must be run from the Python/CUDA environment that can run the 3DGS training code.

## Path And Stage Wiring

Set the COLMAP dataset location in the manifest:

```json
{
  "repo_root": ".",
  "datasets": [
    {
      "name": "lemons",
      "colmap_dir": "/workspace/samuele/Colmap/lemons_2026-02-19",
      "model_dir": "/workspace/output/lemons_2026-02-19",
      "sem_mask_dir": "/workspace/samuele/Colmap/lemons_2026-02-19/semantics",
      "mask_inst_dir": "/workspace/samuele/Colmap/lemons_2026-02-19/mask_instances"
    }
  ]
}
```

Paths may be absolute, as above, or relative to `repo_root`. In the local example manifest, paths like `samuele/datasets/lemons_2026-02-19` are relative to this repository.

The benchmark section chains the full pipeline automatically:

```text
train_sem.py -s <colmap_dir> -m <results_root>/benchmark/<dataset>/full_pipeline_model --mask_dir <sem_mask_dir>
filter_standalone.py --colmap_dir <colmap_dir> --model_dir <same full_pipeline_model> --mask_dir <sem_mask_dir> --mask_inst_dir <mask_inst_dir>
```

Vanilla 3DGS is trained separately:

```text
train.py -s <colmap_dir> -m <results_root>/benchmark/<dataset>/vanilla_model
```

The direct DBSCAN ablation chooses its input model with `direct_dbscan.model_source`:

- `manifest_model_dir`: use `datasets[*].model_dir`, usually an already-trained `train_sem.py` model.
- `benchmark_full_pipeline_model`: use the `train_sem.py` output created by the benchmark section.
- `benchmark_vanilla_model`: use the vanilla `train.py` output created by the benchmark section.

If you want one command to train `train_sem.py` and then run direct DBSCAN on that new model, set:

```json
{
  "run_sections": ["benchmark", "direct_dbscan_wide"],
  "direct_dbscan": {
    "model_source": "benchmark_full_pipeline_model"
  }
}
```

## Direct DBSCAN Ablation

This is the "no embedding clustering" ablation. It uses `ablations/direct_dbscan_ablation.py`, which calls `sklearn.cluster.DBSCAN` directly on Gaussian XYZ coordinates from the trained model PLY.

It is not the same as `filter_standalone.py --cluster_alg dbscan`: that original path still computes mask responsibilities, trains instance embeddings, and then clusters in the embedding/graph pipeline.

Wide sweep on all enabled datasets:

```bash
python ablations/run_ablation_studies.py \
  --config ablations/fruit_ablation_manifest.example.json \
  --section direct_dbscan_wide
```

Fixed parameter set on all enabled datasets:

```bash
python ablations/run_ablation_studies.py \
  --config ablations/fruit_ablation_manifest.example.json \
  --section direct_dbscan_fixed
```

Single direct DBSCAN run:

```bash
python ablations/direct_dbscan_ablation.py \
  --model_dir output/lemons_2026-02-19 \
  --ply_iteration 30000 \
  --input_ply_name scene_mask_filtered_renderer.ply \
  --eps 0.01 \
  --min_samples 20 \
  --xyz_normalize bbox_diag
```

`--eps` is the DBSCAN radius. With `--xyz_normalize bbox_diag`, the radius is relative to the robust scene bounding-box diagonal, which makes a fixed parameter set more comparable across datasets. Use `--xyz_normalize none` if you want radius in the original Gaussian coordinate units.

Outputs are written under:

```text
ablation_runs/direct_dbscan/<wide|fixed>/<dataset>/<run_name>/
```

Each run stores:

- `stdout.log`: command output.
- `command.json`: exact command and cwd.
- `resource_samples.json`: sampled CPU/GPU/memory usage when available.
- `summary.json`: return code, elapsed time, and resource summary.
- `metrics.json`: DBSCAN metrics such as number of clusters, clustered points, noise fraction, and output PLY paths.
- `ply/*.ply`: full labeled PLY and `_no_bkg.ply` with only non-noise clusters.

The runner also writes:

```text
ablation_runs/direct_dbscan/wide_summary.csv
ablation_runs/direct_dbscan/fixed_summary.csv
```

## Timing Benchmark

This compares vanilla 3DGS against the full pipeline:

- vanilla: `train.py`, optionally followed by `render.py`
- full: `train_sem.py`, `filter_standalone.py`, optionally followed by `render.py`
- optional paper render: `render_images_standalone.py`

```bash
python ablations/run_ablation_studies.py \
  --config ablations/fruit_ablation_manifest.example.json \
  --section benchmark
```

The summary CSV is:

```text
ablation_runs/benchmark/summary.csv
```

`psutil` enables process CPU/RAM sampling. `nvidia-smi` enables GPU utilization and memory sampling. If either is missing, the run still proceeds and records whichever metrics are available.

## Contrastive Gaussian Clustering

The manifest can also run the local `contrastive-gaussian-clustering` checkout on the same dataset list:

```bash
python ablations/run_ablation_studies.py \
  --config ablations/fruit_ablation_manifest.example.json \
  --section contrastive
```

The summary CSV is:

```text
ablation_runs/contrastive/summary.csv
```

By default the manifest passes `--objmasks masks`, because the local datasets use a `masks` folder. Change `contrastive_objmasks` per dataset if a dataset should use another object-mask folder name.

## Manifest Fields To Edit First

- `datasets[*].model_dir`: trained model used by direct DBSCAN.
- `datasets[*].colmap_dir`: COLMAP dataset root used by training benchmarks.
- `datasets[*].sem_mask_dir`: semantic masks for `train_sem.py` and `filter_standalone.py`.
- `datasets[*].mask_inst_dir`: instance masks for `filter_standalone.py`.
- `direct_dbscan.model_source`: whether direct DBSCAN uses `datasets[*].model_dir` or a model produced by the benchmark section.
- `direct_dbscan.wide.eps_values`: DBSCAN radius sweep.
- `direct_dbscan.wide.min_samples_values`: DBSCAN min-samples sweep.
- `direct_dbscan.fixed`: fixed DBSCAN parameters for the all-dataset comparison.
- `benchmark.*_args`: training/filter/render arguments for timing comparisons.

The `scaled_both_sides_merged` entry currently points to `output/Fuji_new`, because that appears to be the existing trained model used for that dataset in the local outputs. Check that mapping before launching long direct-DBSCAN sweeps.
