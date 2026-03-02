
FRUIT3DGS
Fruit Counting and Localization with 3D Gaussian Splatting and Contrastive Learning

Fruit3DGS: a general fruit counting and localization framework leveraging 
semantic-guided 3D Gaussian Splatting and contrastive learning

Authors:
Samuele Mara, Angelo Moroncelli, Marco Maccarini, Loris Roveda

============================================================
PAPER
============================================================

Fruit3DGS: Fruit Counting and Localization with 3D Gaussian 
Splatting and Contrastive Learning
Computers and Electronics in Agriculture (2026)

============================================================
GRAPHICAL ABSTRACT
============================================================

Place the graphical abstract image inside:
assets/Fruit3DGS_visual_abstract.png

============================================================
OVERVIEW
============================================================

Fruit3DGS is a fruit-agnostic 3D perception framework for:

- Fruit counting from unordered multi-view RGB images
- Accurate 3D fruit localization
- Robotic picking integration
- Generalization across fruit types and orchard conditions

Unlike fruit-specific pipelines, Fruit3DGS:

- Does NOT rely on geometric fruit templates
- Does NOT require fruit-specific retraining
- Works across apples, lemons, sunflowers, and multi-fruit scenes
- Produces metrically accurate 3D outputs

============================================================
BUILT ON 3D GAUSSIAN SPLATTING
============================================================

This repository extends:

3D Gaussian Splatting for Real-Time Radiance Field Rendering
Kerbl et al., ACM TOG 2023

Official repository:
https://github.com/graphdeco-inria/gaussian-splatting

============================================================
METHODOLOGY
============================================================

1) Robot-Friendly Dataset Acquisition
- Franka EMIKA Panda
- Intel RealSense D405 (end-effector mounted)
- Hand–eye calibration
- Ground-truth 3D target measurement

2) Data Preparation
- COLMAP SfM initialization
- Undistorted multi-view dataset
- Mask generation (Grounded-SAM)

3) Semantic-Aware 3DGS Training (train_sem.py)
- Per-Gaussian semantic coefficients
- BCE mask supervision
- Top-K Gaussian contributor export
- Foreground filtering

4) Instance Embedding Field Training
- Learnable embedding vector per Gaussian
- Contrastive InfoNCE loss
- Semantic-gated smoothness constraint

5) Fruit-Agnostic Instance Clustering (instance_embedding_clustering.py)
- kNN graph construction
- Embedding + geometry metric
- DBSCAN / HDBSCAN
- Bayesian Optimization of hyperparameters

6) 3D Oriented Bounding Box Estimation
- PCA-based OBB fitting
- 3D centroid extraction
- JSON export for robotic picking

============================================================
FRUIT COUNTING RESULTS (APPLE BENCHMARKS)
============================================================

Dataset      | FruitNeRF | FruitLangGS | Fruit3DGS(HDBSCAN) | Fruit3DGS(Ours) | GT
-------------|-----------|-------------|--------------------|-----------------|----
Tree01       | 173       | 173         | 193                | 187             | 179
Tree02       | 112       | 110         | 158                | 108             | 113
Tree03       | 264       | 287         | 304                | 273             | 291
Fuji-SfM     | 1459      | 1443        | 2795               | 1445            | 1455

============================================================
GENERALIZATION TO UNSEEN DATASETS
============================================================

Dataset       | FruitNeRF | Fruit3DGS(HDBSCAN) | Fruit3DGS(Ours) | GT
--------------|-----------|--------------------|-----------------|----
Lemons        | 124       | 3                  | 3               | 3
Sunflowers    | 109       | 8                  | 6               | 5
Multi-Fruit   | 18        | 8                  | 7               | 6
Brandenburg   | 0         | 1199               | 2227            | 3293

============================================================
LOCALIZATION RESULTS – LEMON SCENE
============================================================

Pipeline        | Mean(cm) | Min(cm) | Max(cm)
----------------|----------|---------|--------
FruitNeRF       | 172.1    | 164.3   | 186.1
Fruit3DGS(Ours) | 6.4      | 5.4     | 7.4

============================================================
LOCALIZATION RESULTS – MULTI-FRUIT SCENE
============================================================

Pipeline        | Mean(cm) | Min(cm) | Max(cm)
----------------|----------|---------|--------
FruitNeRF       | --       | --      | --
Fruit3DGS(Ours) | 3.2      | 2.2     | 4.9

============================================================
DOCKER SETUP
============================================================

xhost +local:docker

docker run -it   --gpus '"device=all"'   --ipc=host   --memory=110g   --memory-swap=110g   --memory-reservation=90g   -e DISPLAY=$DISPLAY   -e NVIDIA_DRIVER_CAPABILITIES=all   -v /tmp/.X11-unix:/tmp/.X11-unix   -v <path_to_workspace>:/workspace   -p 7007:7007   --name fruit_gs   fruit_gs:latest   /bin/bash

============================================================
TRAINING
============================================================

Semantic Training:

python train_sem.py   -s <COLMAP_dataset>   --mask_dir <mask_dir>   --topk_contrib 2   --percent_dense 0.001   --sem_threshold 0.3

Instance Clustering:

python instance_embedding_clustering.py   --colmap_dir <dataset>   --model_dir <model_output>   --mask_dir <semantic_masks>   --mask_inst_dir <instance_masks>   --cluster_alg dbscan

============================================================
DATASET
============================================================

The Fruit3DGS localization dataset includes:

- Multi-view RGB captures
- Camera intrinsics & extrinsics
- Ground-truth 3D target positions
- Robotic acquisition setup

Repository & dataset:
https://github.com/SamueleMara/Fruit_3DGS

============================================================
CITATION
============================================================

@article{mara2026fruit3dgs,
  title   = {Fruit3DGS: Fruit Counting and Localization with 3D Gaussian Splatting and Contrastive Learning},
  author  = {Mara, Samuele and Moroncelli, Angelo and Maccarini, Marco and Roveda, Loris},
  journal = {Computers and Electronics in Agriculture},
  year    = {2026}
}

============================================================
CONTACT
============================================================

Samuele Mara
IDSIA / SUPSI / USI
GitHub: https://github.com/SamueleMara
