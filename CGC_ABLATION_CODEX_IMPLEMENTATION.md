# Codex implementation plan — Contrastive Gaussian Clustering / Gaussian Grouping ablation for Fruit3DGS

## Goal

Add a final ablation method based on **Contrastive Gaussian Clustering / Gaussian Grouping (CGC)** to the existing Fruit3DGS comparison pipeline.

The baseline must:

1. Train CGC on every dataset using the **same COLMAP/source paths** already used by our method.
2. Load the final CGC Gaussian point cloud.
3. Extract the learned CGC instance embedding channels from the PLY, i.e. `f_seg_0 ... f_seg_15`.
4. Build the same **Euclidean + instance-embedding distance matrix** used by our clustering method.
5. Since CGC does not train/use our semantic field, set the semantic score/probability of every Gaussian to `1.0`.
6. Run the same Bayesian optimization clustering procedure used by our method.
7. Add CGC results to the same metrics tables and visualizations as the previous ablations.

This is an ablation baseline, not a replacement for our pipeline.

---

## Important repository paths

Assume the container layout used during the experiments:

```bash
/workspace/Fruit_GS                              # Fruit3DGS repo, if mounted as subfolder
/workspace                                      # Fruit3DGS repo, if mounted as root
/workspace/contrastive-gaussian-clustering      # CGC repo
/workspace/output_cgc                           # output root for CGC trainings
```

Code must not hardcode only one Fruit3DGS root. Detect it robustly:

```python
candidate_roots = [Path('/workspace/Fruit_GS'), Path('/workspace')]
fruitgs_root = first root containing train_sem.py / experiments / scene / utils
```

The CGC repo root should be configurable, default:

```bash
/workspace/contrastive-gaussian-clustering
```

The CGC output root should be configurable, default:

```bash
/workspace/output_cgc
```

---

## Do not break the existing rasterizer

The normal Fruit3DGS installation uses:

```python
import diff_gaussian_rasterization
```

The CGC baseline must use the renamed Gaussian Grouping rasterizer:

```python
import diff_gaussian_rasterization_cgc
```

Only patch this file inside CGC:

```bash
/workspace/contrastive-gaussian-clustering/gaussian_renderer/__init__.py
```

Required import in CGC renderer:

```python
from diff_gaussian_rasterization_cgc import GaussianRasterizationSettings, GaussianRasterizer
```

Do **not** change Fruit3DGS renderer imports.

---

## CGC dataset bridge status

CGC's original loader expects one mask label image per RGB:

```text
seg_masks/rgb_000.png
```

Fruit3DGS datasets have per-instance binary masks:

```text
mask_instances/rgb_000_instance_0.png
mask_instances/rgb_000_instance_1.png
mask_instances/rgb_000_instance_2.png
...
```

The CGC loader must support this by composing per-instance masks into one integer label image on the fly:

```text
0 = background
1 = instance_0
2 = instance_1
3 = instance_2
...
```

The updated CGC loader should live at:

```bash
/workspace/contrastive-gaussian-clustering/scene/dataset_readers.py
```

It must keep compatibility with normal CGC direct masks if available.

---

## Training CGC for one dataset

Example command for lemons:

```bash
cd /workspace/contrastive-gaussian-clustering

rm -rf /workspace/output_cgc/lemons_2026-02-19

PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python train.py \
  -s /workspace/samuele/deep-image-matching/lemons_2026-02-19 \
  -m /workspace/output_cgc/lemons_2026-02-19 \
  --objmasks mask_instances \
  --images images \
  --iterations 30000 \
  --save_iterations 30000 \
  --test_iterations 30000 \
  --data_device cpu
```

The comparison code must generate equivalent commands for **all datasets**.

---

## Dataset configuration

Use the same dataset/source paths as our method.

Add CGC-specific fields to the existing ablation YAML/config, or create a dedicated section like this:

```yaml
cgc:
  repo_root: /workspace/contrastive-gaussian-clustering
  output_root: /workspace/output_cgc
  iterations: 30000
  save_iterations: [30000]
  test_iterations: [30000]
  images_dir: images
  objmasks_dir: mask_instances
  data_device: cpu
  py_env:
    PYTHONPATH: /workspace/contrastive-gaussian-clustering
    PYTORCH_CUDA_ALLOC_CONF: max_split_size_mb:128
```

Each dataset entry should already contain the source path. If the existing config only points to a trained Fruit3DGS model, recover the source path from the model's `cfg_args` file, exactly as the existing train-if-missing gateway does.

Expected dataset mapping pattern:

```yaml
datasets:
  lemons_2026-02-19:
    source_path: /workspace/samuele/deep-image-matching/lemons_2026-02-19
    fruitgs_model_path: /workspace/output/lemons_2026-02-19
    cgc_model_path: /workspace/output_cgc/lemons_2026-02-19

  multifruit_2026-02-19:
    source_path: /workspace/samuele/deep-image-matching/multifruit_2026-02-19
    fruitgs_model_path: /workspace/output/multifruit_2026-02-19
    cgc_model_path: /workspace/output_cgc/multifruit_2026-02-19

  Tree_01_new:
    source_path: /workspace/samuele/Colmap/tree_01/tree_01_filtered_scaled
    fruitgs_model_path: /workspace/output/Tree_01_new
    cgc_model_path: /workspace/output_cgc/Tree_01_new

  Tree_02_new:
    source_path: /workspace/samuele/Colmap/tree_02_merged
    fruitgs_model_path: /workspace/output/Tree_02_new
    cgc_model_path: /workspace/output_cgc/Tree_02_new

  Tree_03_new:
    source_path: /workspace/samuele/Colmap/tree_03_merged
    fruitgs_model_path: /workspace/output/Tree_03_new
    cgc_model_path: /workspace/output_cgc/Tree_03_new

  Fuji_yolo_seg:
    source_path: /workspace/samuele/Colmap/scaled_both_sides_merged
    fruitgs_model_path: /workspace/output/Fuji_yolo_seg
    cgc_model_path: /workspace/output_cgc/Fuji_yolo_seg

  2022-09-06_scaled:
    source_path: /workspace/samuele/deep-image-matching/2022-09-06_scaled
    fruitgs_model_path: /workspace/output/2022-09-06_scaled
    cgc_model_path: /workspace/output_cgc/2022-09-06_scaled
```

Do not assume this list is complete. Use the existing ablation dataset config as the source of truth and derive the CGC output path by replacing the output root.

---

## New comparison method name

Add a new method key:

```text
cgc_bo
```

Human-readable display name:

```text
CGC + BO
```

Meaning:

```text
CGC learned instance embeddings + our Euclidean/embedding distance matrix + our Bayesian optimization clustering.
```

This is different from a raw CGC/DBSCAN result. The point is to evaluate whether CGC's learned instance features can replace our contrastive features when everything else downstream is kept identical.

---

## Implementation files to add or modify

Exact filenames may differ in the current repo. Codex must search the existing `experiments/` and `utils/` code and integrate with the existing style.

### Add: `experiments/cgc_baseline.py`

Responsibilities:

1. Build the CGC train command for one dataset.
2. Check if CGC output already exists.
3. Train CGC if missing, unless `--skip-train` is set.
4. Locate the final PLY.
5. Load the PLY and extract:
   - xyz positions: `x`, `y`, `z`
   - opacity if needed: `opacity`
   - instance features: `f_seg_0 ... f_seg_15`
   - semantic scores: synthetic vector of ones
6. Package the data in the same structure used by the existing clustering/evaluation code.

Pseudo-code:

```python
from pathlib import Path
import os
import subprocess
import numpy as np
from plyfile import PlyData


def cgc_output_dir(output_root: Path, dataset_name: str) -> Path:
    return output_root / dataset_name


def cgc_final_ply(model_path: Path, iteration: int = 30000) -> Path:
    return model_path / 'point_cloud' / f'iteration_{iteration}' / 'point_cloud.ply'


def train_cgc_if_missing(dataset_name, source_path, cgc_repo, output_root, cfg, force=False):
    model_path = cgc_output_dir(output_root, dataset_name)
    ply_path = cgc_final_ply(model_path, cfg.iterations)

    if ply_path.exists() and not force:
        return ply_path

    cmd = [
        'python', 'train.py',
        '-s', str(source_path),
        '-m', str(model_path),
        '--objmasks', cfg.objmasks_dir,
        '--images', cfg.images_dir,
        '--iterations', str(cfg.iterations),
        '--save_iterations', str(cfg.iterations),
        '--test_iterations', str(cfg.iterations),
        '--data_device', cfg.data_device,
    ]

    env = os.environ.copy()
    env['PYTHONPATH'] = str(cgc_repo)
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    subprocess.run(cmd, cwd=str(cgc_repo), env=env, check=True)

    if not ply_path.exists():
        raise FileNotFoundError(f'CGC training finished but final PLY was not found: {ply_path}')

    return ply_path


def load_cgc_gaussians(ply_path: Path):
    ply = PlyData.read(str(ply_path))
    v = ply['vertex'].data

    xyz = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)

    seg_names = sorted(
        [name for name in v.dtype.names if name.startswith('f_seg_')],
        key=lambda s: int(s.split('_')[-1]),
    )
    if len(seg_names) == 0:
        raise ValueError(f'No f_seg_* fields found in CGC PLY: {ply_path}')

    embeddings = np.stack([v[name] for name in seg_names], axis=1).astype(np.float32)

    semantic = np.ones((xyz.shape[0],), dtype=np.float32)

    opacity = None
    if 'opacity' in v.dtype.names:
        opacity = np.asarray(v['opacity']).astype(np.float32)

    return {
        'xyz': xyz,
        'instance_embeddings': embeddings,
        'semantic': semantic,
        'opacity': opacity,
        'ply_path': str(ply_path),
        'feature_names': seg_names,
    }
```

### Modify: existing ablation runner

Likely file:

```text
experiments/run_fruit3dgs_ablation.py
```

Add method dispatch:

```python
if method == 'cgc_bo':
    ply_path = train_cgc_if_missing(...)
    data = load_cgc_gaussians(ply_path)
    result = run_existing_bo_clustering_on_loaded_gaussians(data, same_bo_cfg_as_ours)
```

Important: do not create a separate BO implementation. Reuse the same BO objective, search space, random seed logic, and evaluation function used by our method.

If the current BO code is not reusable, refactor it into a shared function:

```python
def run_bo_clustering_from_arrays(
    xyz,
    embeddings,
    semantic_probs,
    gt_instances,
    bo_config,
    distance_config,
    dataset_name,
    method_name,
):
    ...
```

Then call the same function from both:

```text
ours
cgc_bo
```

---

## Distance matrix for CGC

The CGC distance matrix must use the **same formula and normalization** as our method.

Codex must locate the existing distance construction code. Search for terms like:

```bash
grep -R "distance_matrix\|embedding\|semantic\|Bayesian\|BO\|dbscan\|eps" -n experiments utils scripts
```

Then reuse or factor out the existing code.

Expected conceptual inputs:

```python
xyz:              [N, 3]
embeddings:       [N, D]       # CGC f_seg channels
semantic_probs:   [N]          # all ones for CGC
```

Semantic handling for CGC:

```python
semantic_probs = np.ones((N,), dtype=np.float32)
```

Do not filter out Gaussians by semantic confidence for CGC, because all are foreground by definition in this ablation. If existing code requires a semantic threshold, ensure it keeps all points:

```python
semantic_threshold = 0.0
foreground_mask = semantic_probs >= semantic_threshold
```

Distance matrix must be exactly whatever our method uses, e.g. if our method does:

```python
D = alpha_xyz * D_xyz + alpha_emb * D_emb + alpha_sem * D_sem
```

then CGC should use the same code with:

```python
D_sem = 0
```

because all semantic values are one.

Do not invent new weights for CGC. The weights, scaling, normalization, and BO ranges must be identical to ours.

---

## Bayesian optimization

Run BO with the same settings as ours:

- same search space
- same number of calls/trials
- same random seeds
- same metric objective
- same constraints
- same DBSCAN/HDBSCAN backend if used by our method
- same post-processing

The CGC method must appear in output rows as:

```text
method = cgc_bo
method_display = CGC + BO
```

Store per-run information:

```json
{
  "method": "cgc_bo",
  "dataset": "lemons_2026-02-19",
  "source_path": "...",
  "cgc_model_path": "...",
  "ply_path": ".../point_cloud.ply",
  "feature_dim": 16,
  "semantic_assumption": "all_ones",
  "bo_best_params": {...},
  "metrics": {...}
}
```

---

## Metrics

Use exactly the existing metrics from the comparison code. At minimum keep:

```text
precision
recall
F1
predicted_count
ground_truth_count
absolute_count_error
relative_count_error
```

If the existing code already computes matching/assignment metrics between predicted clusters and GT fruit instances, reuse them unchanged.

Do not evaluate CGC with a special metric.

---

## Visualization requirements from sketches

The hand sketches define the final reporting style. Implement this for **all methods**, including previous ablations and the new CGC baseline.

### Methods to show

Use a consistent order and legend:

```text
Ours
DBSCAN-auto
DBSCAN-best
CGC + BO
```

If some older ablation names differ in the code, map them to these display names.

### Plot 1 — per-dataset Precision / Recall / F1 grouped bars

For each dataset, show three metrics:

```text
Precision
Recall
F1
```

Each metric group contains bars for all methods:

```text
Ours, DBSCAN-auto, DBSCAN-best, CGC + BO
```

Requirements:

- Same y-axis for all three metrics: `[0, 1]`.
- Show dataset names on x-axis.
- Rotate dataset labels if needed.
- Put legend outside or above the plot.
- Export PNG and PDF.
- Also save the raw CSV used for plotting.

Recommended filenames:

```text
comparison_precision_recall_f1_by_dataset.png
comparison_precision_recall_f1_by_dataset.pdf
comparison_precision_recall_f1_by_dataset.csv
```

### Plot 2 — count prediction comparison

For each dataset, show:

```text
GT count
predicted count from each method
```

Recommended grouped bar structure:

```text
Dataset 1: GT, Ours, DBSCAN-auto, DBSCAN-best, CGC + BO
Dataset N: GT, Ours, DBSCAN-auto, DBSCAN-best, CGC + BO
```

Requirements:

- Use the same y-axis for all datasets.
- Add value labels above bars if readable.
- Export PNG/PDF/CSV.

Recommended filenames:

```text
comparison_counts_by_dataset.png
comparison_counts_by_dataset.pdf
comparison_counts_by_dataset.csv
```

### Plot 3 — count error / absolute error

For each dataset and method, show:

```text
abs(predicted_count - GT_count)
```

Recommended filenames:

```text
comparison_count_error_by_dataset.png
comparison_count_error_by_dataset.pdf
comparison_count_error_by_dataset.csv
```

### Plot 4 — compact instance-quality summary

The first sketch emphasizes "only pred and F1 score" for the instance comparison. Create a compact summary table/plot with:

```text
dataset
method
predicted_count
ground_truth_count
F1
```

Recommended outputs:

```text
comparison_instance_summary.csv
comparison_instance_summary.md
comparison_instance_summary.html
```

### Plot 5 — optional per-dataset qualitative HTML

For each dataset, produce an HTML page showing point clouds/clusters if the existing code already supports this.

For every method, visualize clusters with the same camera/viewpoint conventions if possible:

```text
Ours
DBSCAN-auto
DBSCAN-best
CGC + BO
```

Recommended folder:

```text
outputs/ablation_visualizations/<dataset>/
```

Recommended files:

```text
ours_clusters.html
dbscan_auto_clusters.html
dbscan_best_clusters.html
cgc_bo_clusters.html
```

---

## Output folder layout

Use a single comparison output root, e.g.:

```text
/workspace/output_ablation_comparison
```

Suggested structure:

```text
/workspace/output_ablation_comparison/
  config_resolved.yaml
  all_results.csv
  all_results.json
  all_results.md

  cgc_training_logs/
    lemons_2026-02-19.log
    Tree_01_new.log
    ...

  plots/
    comparison_precision_recall_f1_by_dataset.png
    comparison_precision_recall_f1_by_dataset.pdf
    comparison_precision_recall_f1_by_dataset.csv
    comparison_counts_by_dataset.png
    comparison_counts_by_dataset.pdf
    comparison_counts_by_dataset.csv
    comparison_count_error_by_dataset.png
    comparison_count_error_by_dataset.pdf
    comparison_count_error_by_dataset.csv
    comparison_instance_summary.csv
    comparison_instance_summary.md
    comparison_instance_summary.html

  per_dataset/
    lemons_2026-02-19/
      cgc_bo_result.json
      cgc_bo_best_params.json
      cgc_bo_clusters.ply
      cgc_bo_clusters.html
    ...
```

---

## CLI interface

Extend the existing ablation runner with these options:

```bash
--methods ours dbscan_auto dbscan_best cgc_bo
--include-cgc
--cgc-repo /workspace/contrastive-gaussian-clustering
--cgc-output-root /workspace/output_cgc
--cgc-iterations 30000
--cgc-force-train
--cgc-skip-train
--only-visualize
```

Expected full run:

```bash
cd /workspace/Fruit_GS   # or /workspace if repo root is mounted there

PYTHONPATH=. python -m experiments.run_fruit3dgs_ablation \
  --config experiments/configs/fruit3dgs_ablation_all_datasets.yaml \
  --methods ours dbscan_auto dbscan_best cgc_bo \
  --include-cgc \
  --cgc-repo /workspace/contrastive-gaussian-clustering \
  --cgc-output-root /workspace/output_cgc \
  --cgc-iterations 30000 \
  --output-root /workspace/output_ablation_comparison
```

For debugging only one dataset:

```bash
PYTHONPATH=. python -m experiments.run_fruit3dgs_ablation \
  --config experiments/configs/fruit3dgs_ablation_all_datasets.yaml \
  --datasets lemons_2026-02-19 \
  --methods cgc_bo \
  --include-cgc \
  --cgc-repo /workspace/contrastive-gaussian-clustering \
  --cgc-output-root /workspace/output_cgc \
  --cgc-iterations 100 \
  --output-root /workspace/output_ablation_debug
```

---

## Validation checks before running all datasets

Add a preflight command or automatic checks:

```python
assert cgc_repo.exists()
assert (cgc_repo / 'train.py').exists()
assert source_path.exists()
assert (source_path / 'sparse').exists()
assert (source_path / images_dir).exists()
assert (source_path / objmasks_dir).exists()
```

Check rasterizer:

```python
import inspect
from diff_gaussian_rasterization_cgc import GaussianRasterizer
sig = str(inspect.signature(GaussianRasterizer.forward))
assert 'sh_objs' in sig or 'obj' in sig or feature-compatible condition
```

Check the CGC renderer imports the renamed package:

```python
renderer_text = (cgc_repo / 'gaussian_renderer' / '__init__.py').read_text()
assert 'diff_gaussian_rasterization_cgc' in renderer_text
```

Check final PLY fields:

```python
required_xyz = ['x', 'y', 'z']
required_emb = ['f_seg_0']
```

Fail clearly if missing.

---

## Minimal smoke test acceptance criteria

On lemons with 100 iterations:

```bash
cd /workspace/contrastive-gaussian-clustering

rm -rf /workspace/output_cgc/debug_lemons_100

PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python train.py \
  -s /workspace/samuele/deep-image-matching/lemons_2026-02-19 \
  -m /workspace/output_cgc/debug_lemons_100 \
  --objmasks mask_instances \
  --images images \
  --iterations 100 \
  --save_iterations 100 \
  --test_iterations 100 \
  --data_device cpu
```

Must produce:

```text
/workspace/output_cgc/debug_lemons_100/point_cloud/iteration_100/point_cloud.ply
```

Then the Fruit3DGS ablation runner must be able to run:

```bash
PYTHONPATH=. python -m experiments.run_fruit3dgs_ablation \
  --datasets lemons_2026-02-19 \
  --methods cgc_bo \
  --cgc-output-root /workspace/output_cgc \
  --cgc-iterations 100 \
  --cgc-skip-train
```

and produce a metrics row for:

```text
method = cgc_bo
dataset = lemons_2026-02-19
```

---

## Full experiment acceptance criteria

The final implementation is complete when:

1. CGC can be trained for every dataset in the existing comparison config.
2. The final CGC PLY is found for every dataset.
3. The PLY loader extracts `xyz` and all `f_seg_*` channels.
4. Semantic field is set to all ones for CGC.
5. CGC uses the same Euclidean + instance-embedding distance matrix code as ours.
6. CGC uses the same BO search space/settings/objective as ours.
7. Results are written to the same global `all_results.csv/json/md` as the other methods.
8. Plots include `CGC + BO` next to `Ours`, `DBSCAN-auto`, and `DBSCAN-best`.
9. Visualizations from the sketches are generated:
   - Precision/Recall/F1 grouped bars.
   - GT vs predicted counts.
   - Count error.
   - Compact instance summary with predicted count and F1.
   - Optional per-dataset qualitative HTMLs.
10. Existing Fruit3DGS training/inference remains unaffected and still imports the normal `diff_gaussian_rasterization`.

---

## Notes for Codex

Do not duplicate entire clustering pipelines. Refactor shared pieces if necessary.

The correct abstraction is:

```text
source of Gaussian features changes
BO + clustering + evaluation stays the same
```

So:

```text
Ours    -> our trained semantic + contrastive fields
CGC+BO  -> CGC f_seg embeddings + semantic all ones
```

Everything after feature loading must be identical.
