# Train-if-missing gateway

This patch adds a semantic-model gate to `experiments/run_fruit3dgs_ablation.py`.

Before any DBSCAN / embedding / HDBSCAN stage, the runner checks:

```text
<train_sem_model_dir>/point_cloud/iteration_<global.iteration>/point_cloud.ply
```

If it exists, the runner proceeds normally.
If it is missing and `training.train_sem_if_missing: true`, the runner launches `train_sem.py` with the dataset-specific `train_if_missing` block.

Example command generated for Lemons:

```bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python train_sem.py \
  -s /workspace/samuele/deep-image-matching/lemons_2026-02-19 \
  -m /workspace/output/lemons_2026-02-19 \
  --mask_dir /workspace/samuele/deep-image-matching/lemons_2026-02-19/masks \
  --topk_contrib 2 \
  --percent_dense 0.001 \
  --data_device cpu \
  --iterations 30000 \
  -r 1 \
  --sem_threshold 0.7 \
  --disable_viewer
```

Use:

```bash
PYTHONPATH=/workspace python -m experiments.run_fruit3dgs_ablation \
  --config experiments/fruit3dgs_ablation_quick_lemon_train_gate.yaml
```

Full config:

```bash
PYTHONPATH=/workspace python -m experiments.run_fruit3dgs_ablation \
  --config experiments/fruit3dgs_ablation_train_gate.yaml
```
