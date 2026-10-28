# Fruit3DGS Ablation Automation Debug Context

This file is intended for Codex / an implementation agent working inside the Fruit3DGS repo at `/workspace`.

The goal is simple: **make the ablation automation run end-to-end without manual debugging**. The immediate smoke test is the quick Lemon run:

```bash
cd /workspace
touch experiments/__init__.py
PYTHONPATH=/workspace python -m experiments.run_fruit3dgs_ablation \
  --config experiments/fruit3dgs_ablation_quick_lemon_train_gate_cfgargs_patched.yaml
```

The runner should finish at least the quick Lemon ablation and produce:

```text
/workspace/results/fruit3dgs_ablation_quick_lemon/summary.csv
/workspace/results/fruit3dgs_ablation_quick_lemon/summary.json
/workspace/results/fruit3dgs_ablation_quick_lemon/lemons/**/metrics.json
```

At the moment it fails repeatedly on missing imports / brittle path assumptions. Fix the code and repo utilities so this does not keep happening.

---

## Current environment / repo assumptions

Repo path inside container:

```text
/workspace
```

Important: `/workspace` is a Docker bind mount from the host repo. Do **not** assume the repo is inside `/workspace/Fruit_GS` unless `cfg_args` says so. The correct dataset/source path for a trained model must be recovered from the model's `cfg_args` whenever possible.

The active Python environment is the container's Gaussian Splatting env, e.g.:

```text
/opt/conda/envs/gaussian_splatting/bin/python
```

3DGS CUDA extensions should be installed from local repo submodules:

```bash
python -m pip install --no-cache-dir --no-build-isolation -e submodules/simple-knn
python -m pip install --no-cache-dir --no-build-isolation -e submodules/diff-gaussian-rasterization
```

Do **not** rely on `git submodule update --init --recursive` because this repo currently has a broken `.gitmodules` entry:

```text
fatal: No url found for submodule path 'submodules/pytorch3d' in .gitmodules
```

If `submodules/simple-knn` and `submodules/diff-gaussian-rasterization` already exist, install them directly and skip recursive submodule update.

---

## Errors already encountered and what they mean

The ablation runner currently hides failures behind repeated:

```text
[WARN] Command failed with return code 1
```

The actual errors are in each profile `stderr.log`. The implementation must make these visible automatically.

Errors hit so far:

### 1. Missing Python deps

Observed failures:

```text
ModuleNotFoundError: No module named 'pandas'
ModuleNotFoundError: No module named 'plyfile'
ModuleNotFoundError: No module named 'cv2'
ModuleNotFoundError: No module named 'imageio'
```

The code should provide a clear dependency preflight instead of failing one missing import at a time.

Minimum Python packages needed for the ablation scripts and 3DGS repo imports:

```text
pandas
pyyaml
psutil
matplotlib
scikit-learn
scikit-optimize
hdbscan
wandb
plyfile
opencv-python-headless
imageio
imageio-ffmpeg
pillow
tqdm
scipy
ninja
```

Potentially needed for visualization/export stages:

```text
open3d
```

Do not blindly import `open3d` in core ablation paths unless it is actually needed.

### 2. Python package path issue

Direct execution failed with:

```text
ModuleNotFoundError: No module named 'experiments'
```

The correct way to run is:

```bash
PYTHONPATH=/workspace python -m experiments.run_fruit3dgs_ablation --config ...
```

Also ensure:

```text
experiments/__init__.py
```

exists in the code package.

### 3. Missing 3DGS CUDA extensions

Observed:

```text
ModuleNotFoundError: No module named 'simple_knn'
```

Then a naive install failed because pip build isolation could not see torch:

```text
ModuleNotFoundError: No module named 'torch'
```

Correct install command:

```bash
python -m pip install --no-cache-dir --no-build-isolation -e submodules/simple-knn
python -m pip install --no-cache-dir --no-build-isolation -e submodules/diff-gaussian-rasterization
```

Add a doctor/preflight that tests:

```python
from simple_knn._C import distCUDA2
import diff_gaussian_rasterization
```

### 4. Wrong trained model path

The initial quick config used a fake model path:

```text
/workspace/results/models/lemons_2026-02-19_sem
```

But the actual trained Lemon output exists at:

```text
/workspace/output/lemons_2026-02-19
```

The model checkpoint condition should check:

```text
<train_sem_model_dir>/point_cloud/iteration_30000/point_cloud.ply
```

For Lemon this exists:

```text
/workspace/output/lemons_2026-02-19/point_cloud/iteration_30000/point_cloud.ply
```

### 5. Wrong COLMAP/data path for existing trained models

The quick config originally used:

```text
/workspace/Fruit_GS/samuele/datasets/lemons_2026-02-19
```

But the trained model `/workspace/output/lemons_2026-02-19` was actually trained from:

```text
/workspace/samuele/deep-image-matching/lemons_2026-02-19
```

This was recovered from:

```text
/workspace/output/lemons_2026-02-19/cfg_args
```

The `cfg_args` file stores the original source directory as `source_path`. For example, a `Tree_03_new/cfg_args` contained:

```text
source_path='/workspace/samuele/Colmap/tree_03_merged'
```

Therefore: **when a trained model exists, the runner should read `model_dir/cfg_args` and override the dataset paths from `source_path`**.

Expected mapping after reading cfg_args:

```yaml
colmap_dir: <source_path>
binary_mask_dir: <source_path>/masks
instance_mask_dir: <source_path>/mask_instances
```

Also update `train_if_missing.source_dir` and `train_if_missing.mask_dir` to the same values where appropriate.

---

## Known good trained model folders

Host paths provided by the user:

```text
/home/rhea/shared/samueleWS/Fruit_GS/output/Tree_01_new
/home/rhea/shared/samueleWS/Fruit_GS/output/Tree_02_new
/home/rhea/shared/samueleWS/Fruit_GS/output/Fuji_yolo_seg
/home/rhea/shared/samueleWS/Fruit_GS/output/Tree_03_new
/home/rhea/shared/samueleWS/Fruit_GS/output/2022-09-06_scaled
/home/rhea/shared/samueleWS/Fruit_GS/output/multifruit_2026-02-19
/home/rhea/shared/samueleWS/Fruit_GS/output/lemons_2026-02-19
/home/rhea/shared/samueleWS/Fruit_GS/output/Fuji_yolo_sam
```

Inside the container these are:

```text
/workspace/output/Tree_01_new
/workspace/output/Tree_02_new
/workspace/output/Fuji_yolo_seg
/workspace/output/Tree_03_new
/workspace/output/2022-09-06_scaled
/workspace/output/multifruit_2026-02-19
/workspace/output/lemons_2026-02-19
/workspace/output/Fuji_yolo_sam
```

There may also be:

```text
/workspace/output/girasoli_new
```

but the user did not list it among confirmed good folders. If missing, the train-if-missing gateway must train it.

---

## User-provided training settings for train-if-missing

The user provided the original `train_sem.py`, `output_filter_standalone.py`, and `filter_standalone.py` commands used in the project. The ablation automation must preserve these settings when retraining missing semantic models.

Common training settings:

```bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python train_sem.py \
  -s <source_dir> \
  --mask_dir <source_dir>/masks \
  --topk_contrib 2 \
  --percent_dense 0.001 \
  --data_device cpu
```

Some datasets additionally require:

```text
-r 1
--sem_threshold <value>
```

Important: if training from the ablation runner, train into the expected output folder by passing the model/output path argument supported by `train_sem.py` (usually `-m` or `--model_path`). Verify the parser in `train_sem.py`. The output should be:

```text
<train_sem_model_dir>/point_cloud/iteration_30000/point_cloud.ply
```

If `train_sem.py` does not support `-m`, update the runner to locate the actual output folder after training, but prefer explicit `-m <train_sem_model_dir>` if available.

Dataset-specific training commands from user context:

```bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python train_sem.py -s /workspace/samuele/Colmap/Lemons --mask_dir /workspace/samuele/Colmap/Lemons/masks --topk_contrib 2 --percent_dense 0.001 --data_device cpu

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python train_sem.py -s /workspace/samuele/Colmap/tree_01/tree_01_filtered_scaled --mask_dir /workspace/samuele/Colmap/tree_01/tree_01_filtered_scaled/masks --topk_contrib 2 --percent_dense 0.001 --data_device cpu

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python train_sem.py -s /workspace/samuele/Colmap/tree_02_merged --mask_dir /workspace/samuele/Colmap/tree_02_merged/masks --topk_contrib 2 --percent_dense 0.001 --data_device cpu

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python train_sem.py -s /workspace/samuele/Colmap/tree_03_merged --mask_dir /workspace/samuele/Colmap/tree_03_merged/masks --topk_contrib 2 --percent_dense 0.001 --data_device cpu

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python train_sem.py -s /workspace/samuele/Colmap/scaled_both_sides_merged --mask_dir /workspace/samuele/Colmap/scaled_both_sides_merged/masks --topk_contrib 2 --percent_dense 0.001 --data_device cpu --sem_threshold 0.3

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python train_sem.py -s /workspace/samuele/Colmap/Fuji --mask_dir /workspace/samuele/Colmap/Fuji/masks --topk_contrib 2 --percent_dense 0.001 --data_device cpu --sem_threshold 0.3

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python train_sem.py -s /workspace/samuele/Colmap/Fuji_yolo --mask_dir /workspace/samuele/Colmap/Fuji_yolo/masks --topk_contrib 2 --percent_dense 0.001 --data_device cpu --sem_threshold 0.3

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python train_sem.py -s /workspace/samuele/Colmap/girasoli_real/sparse_filtered_scaled_920x1992 --mask_dir /workspace/samuele/Colmap/girasoli_real/sparse_filtered_scaled_920x1992/masks --topk_contrib 2 --percent_dense 0.001 --data_device cpu -r 1

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python train_sem.py -s /workspace/samuele/deep-image-matching/2022-09-06_scaled --mask_dir /workspace/samuele/deep-image-matching/2022-09-06_scaled/masks --topk_contrib 2 --percent_dense 0.001 --data_device cpu -r 1 --sem_threshold 0.3

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python train_sem.py -s /workspace/samuele/deep-image-matching/lemons_2026-02-19 --mask_dir /workspace/samuele/deep-image-matching/lemons_2026-02-19/masks --topk_contrib 2 --percent_dense 0.001 --data_device cpu -r 1 --sem_threshold 0.7

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python train_sem.py -s /workspace/samuele/deep-image-matching/multifruit_2026-02-19 --mask_dir /workspace/samuele/deep-image-matching/multifruit_2026-02-19/masks --topk_contrib 2 --percent_dense 0.001 --data_device cpu -r 1 --sem_threshold 0.7
```

---

## Required code changes

### A. Add a robust dependency / environment doctor

Create or update:

```text
experiments/doctor_ablation_env.py
```

It should:

1. Print active Python path and version.
2. Print torch version, CUDA availability, torch CUDA version.
3. Check imports:

```python
required = [
    "pandas", "yaml", "psutil", "matplotlib", "sklearn", "skopt",
    "hdbscan", "wandb", "plyfile", "cv2", "imageio", "PIL",
    "tqdm", "scipy",
]
```

4. Check 3DGS extensions:

```python
from simple_knn._C import distCUDA2
import diff_gaussian_rasterization
```

5. Check that repo files exist:

```text
train_sem.py
filter_standalone.py
output_filter_standalone.py
scene/
gaussian_renderer/
submodules/simple-knn
submodules/diff-gaussian-rasterization
```

6. For each dataset in a config, check:

```text
train_sem_model_dir/cfg_args
train_sem_model_dir/point_cloud/iteration_<N>/point_cloud.ply
source_path recovered from cfg_args
source_path/images or source_path/input images
source_path/sparse or source_path/manual_sparse
source_path/masks
source_path/mask_instances
```

7. Check PLY fields. At minimum report semantic-like fields:

```python
[n for n in names if "semantic" in n.lower() or "mask" in n.lower()]
```

If no semantic field exists, report that the model is probably vanilla and cannot be used for semantic foreground ablation.

8. Exit non-zero if required core dependencies are missing.

### B. Make runner run doctor/preflight before ablations

In `experiments/run_fruit3dgs_ablation.py`:

1. Add a `--skip_preflight` option, default false.
2. Before running any ablations, call internal preflight checks or import from `doctor_ablation_env.py`.
3. Fail early with a readable message if dependencies are missing.
4. Do not launch dozens of runs if `scene` import will immediately fail due to missing `imageio`, `cv2`, etc.

### C. Make failure logs visible

Currently the user only sees:

```text
[WARN] Command failed with return code 1
```

Modify `profile_run.py` or `run_fruit3dgs_ablation.py` so every failed command prints:

```text
- command
- return code
- path to stderr.log
- last 80-160 lines of stderr.log
```

This alone will avoid repeated manual `find ... stderr.log` debugging.

### D. Integrate cfg_args source_path recovery into the runner

Do not require a separate patch script.

In `run_fruit3dgs_ablation.py`, add helper:

```python
def read_cfg_args_source_path(model_dir: Path) -> str | None:
    cfg = model_dir / "cfg_args"
    if not cfg.exists():
        return None
    txt = cfg.read_text(errors="replace")
    # Handle Namespace(source_path='...') and double quotes.
    ...
```

Before running ablations for a dataset:

1. If `<train_sem_model_dir>/point_cloud/iteration_<load_iteration>/point_cloud.ply` exists:
   - read `<train_sem_model_dir>/cfg_args`
   - if source_path exists and the directory exists, override:

```python
ds["colmap_dir"] = source_path
ds["binary_mask_dir"] = source_path + "/masks"
ds["instance_mask_dir"] = source_path + "/mask_instances"
```

2. Print:

```text
[OK] semantic model exists: ...
[OK] source_path from cfg_args: ...
```

3. If `cfg_args` source_path does not exist in the current container, keep config path but print a warning.

### E. Improve train-if-missing gateway

Current gateway correctly detects existing model for Lemon:

```text
[OK] semantic model exists: /workspace/output/lemons_2026-02-19/point_cloud/iteration_30000/point_cloud.ply
```

But implement it robustly:

1. Function:

```python
def model_ply_exists(model_dir: Path, iteration: int) -> bool:
    return (model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply").exists()
```

2. If missing:
   - read `train_if_missing` section from dataset config.
   - build `train_sem.py` command with user settings.
   - include `-m <train_sem_model_dir>` if `train_sem.py` supports it.
   - set env `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`.
   - run through profiler.
   - after training, check model ply again.
   - if still missing, print stderr tail and skip this dataset.

3. Do not run downstream DBSCAN/embedding if training failed or semantic model still missing.

### F. Do not import full Scene for direct DBSCAN if possible

`cluster_direct_dbscan.py` currently imports `Scene`, which triggers many unnecessary deps:

```text
scene -> utils.camera_utils -> scene.cameras -> cv2/imageio
```

For the **direct DBSCAN baseline**, it should be possible to read the trained PLY directly instead of initializing the full Scene. Implement this if feasible.

Direct DBSCAN only needs:

```text
xyz positions
RGB/DC color or features if using xyz_rgb
opacity if using xyz_rgb_opacity
semantic/semantic_mask field to threshold foreground
```

Use `plyfile.PlyData` and avoid loading cameras/rasterizer for this baseline. That makes the no-contrastive baseline cheaper and less fragile.

Fallback: if some fields are not available or the PLY schema differs, use the current Scene-based loader, but direct PLY loading should be the default.

### G. Keep Scene-based loading for embedding training, but ensure deps

`train_contrastive_embeddings.py` needs `filter_standalone.initialize_scene`, top-K contributor computation, image/mask loading, and camera rendering. It can keep Scene imports, but preflight must guarantee these modules exist:

```text
cv2
imageio
PIL
torch
simple_knn
diff_gaussian_rasterization
```

### H. Make configs unambiguous

Add/update configs:

```text
experiments/fruit3dgs_ablation_quick_lemon_train_gate.yaml
experiments/fruit3dgs_ablation_train_gate.yaml
```

Each dataset should have:

```yaml
name: lemons
gt_count: 3
train_sem_model_dir: /workspace/output/lemons_2026-02-19
colmap_dir: auto_from_cfg_args_or_fallback
binary_mask_dir: auto_from_cfg_args_or_fallback/masks
instance_mask_dir: auto_from_cfg_args_or_fallback/mask_instances
train_if_missing:
  enabled: true
  source_dir: /workspace/samuele/deep-image-matching/lemons_2026-02-19
  mask_dir: /workspace/samuele/deep-image-matching/lemons_2026-02-19/masks
  topk_contrib: 2
  percent_dense: 0.001
  data_device: cpu
  resolution: 1
  sem_threshold: 0.7
```

But if `cfg_args` exists, source_path from `cfg_args` wins.

---

## Current quick Lemon status

Command run:

```bash
PYTHONPATH=/workspace python -m experiments.run_fruit3dgs_ablation \
  --config experiments/fruit3dgs_ablation_quick_lemon_train_gate_cfgargs_patched.yaml
```

It correctly detects:

```text
[OK] semantic model exists: /workspace/output/lemons_2026-02-19/point_cloud/iteration_30000/point_cloud.ply
```

After cfg_args patch, it also correctly runs with:

```text
--colmap_dir /workspace/samuele/deep-image-matching/lemons_2026-02-19
--model_dir /workspace/output/lemons_2026-02-19
--mask_inst_dir /workspace/samuele/deep-image-matching/lemons_2026-02-19/mask_instances
```

But it still fails before actual ablation due to missing `imageio`:

```text
ModuleNotFoundError: No module named 'imageio'
```

Immediate manual fix:

```bash
python -m pip install --no-cache-dir imageio imageio-ffmpeg
```

But the requested code fix is to add the preflight/requirements/doctor so all missing dependencies are caught at once.

---

## Exact commands to validate after fixes

### 1. Install dependencies once

```bash
python -m pip install --no-cache-dir \
  pandas pyyaml psutil matplotlib scikit-learn scikit-optimize \
  hdbscan wandb plyfile opencv-python-headless imageio imageio-ffmpeg \
  pillow tqdm scipy ninja

python -m pip install --no-cache-dir --no-build-isolation -e submodules/simple-knn
python -m pip install --no-cache-dir --no-build-isolation -e submodules/diff-gaussian-rasterization
```

### 2. Doctor

```bash
PYTHONPATH=/workspace python -m experiments.doctor_ablation_env \
  --config experiments/fruit3dgs_ablation_quick_lemon_train_gate.yaml
```

Expected: no missing core dependencies, Lemon model found, cfg_args source_path found, masks and mask_instances found.

### 3. Quick Lemon run

```bash
rm -rf /workspace/results/fruit3dgs_ablation_quick_lemon

PYTHONPATH=/workspace python -m experiments.run_fruit3dgs_ablation \
  --config experiments/fruit3dgs_ablation_quick_lemon_train_gate.yaml
```

Expected:

```text
summary.csv exists
summary.json exists
at least one direct_dbscan metrics.json exists
embeddings_iter50_dim64.pt exists
HDBSCAN/seeded DBSCAN stages run if enabled in config
```

Check:

```bash
cat /workspace/results/fruit3dgs_ablation_quick_lemon/summary.csv
find /workspace/results/fruit3dgs_ablation_quick_lemon -name metrics.json | head -20
find /workspace/results/fruit3dgs_ablation_quick_lemon -name 'embeddings_iter50_dim64.pt'
```

### 4. Full run

Patch/source paths should be automatic from cfg_args now, so run:

```bash
PYTHONPATH=/workspace python -m experiments.run_fruit3dgs_ablation \
  --config experiments/fruit3dgs_ablation_train_gate.yaml
```

---

## Docker persistence note

The user may commit the working container after installing deps:

```bash
docker commit fruit_gs_gpu0 fruit_gs:ablation_ready
```

But `/workspace` is a bind mount, so files under `/workspace` are host repo files, not image files. Python packages installed into the conda env are saved by `docker commit`; repo edits are already saved on host.

For a permanent solution, also update the Dockerfile or add:

```text
experiments/requirements_ablation.txt
```

with the pip dependencies listed above.

---

## Implementation priority

1. Install/codify dependencies (`requirements_ablation.txt`, Dockerfile note, doctor script).
2. Add `experiments/__init__.py` to package.
3. Add runner preflight and readable stderr tail on failure.
4. Integrate cfg_args source_path recovery directly into runner.
5. Make train-if-missing train into the expected `train_sem_model_dir` and skip downstream if training fails.
6. Make direct DBSCAN read PLY directly to avoid unnecessary Scene/camera imports.
7. Validate quick Lemon until it actually produces metrics and embeddings.
8. Only then run the full all-dataset ablation.

Do not stop after fixing one missing import. Run the quick Lemon command repeatedly, inspect logs, and continue fixing until the ablation completes.
