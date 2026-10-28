# Fruit_GS Docker Setup and Installation Guide

This guide explains how to build the Docker image, run the container with GPU/X11 support, install all Python/CUDA dependencies, install the Gaussian Splatting submodules, and verify that the environment works.

Tested environment:

- GPU: NVIDIA GeForce RTX 5090
- CUDA: 12.8
- PyTorch: 2.12.0.dev...+cu128
- Python: 3.10

Important: do not blindly install the old `environment.yml` if it contains Python 3.7, PyTorch 1.12.1, and CUDA 11.6. That environment is not suitable for RTX 5090 / CUDA 12.8.

---

## 1. Expected repository structure

```text
/workspace
├── Dockerfile
├── train.py
├── render.py
├── gaussian_renderer/
├── scene/
├── utils/
├── submodules/
│   ├── diff-gaussian-rasterization/
│   ├── simple-knn/
│   ├── fused-ssim/
│   └── pytorch3d/
├── environment.yml
└── README.md
```

If `submodules/` is missing or incomplete, restore it with:

```bash
cd /workspace
unzip -o submodules.zip -d .
```

Or, if the repository has Git submodule metadata:

```bash
cd /workspace
git submodule update --init --recursive
```

---

## 2. Build the Docker image on the host

On the host machine:

```bash
cd /home/rhea/shared/samueleWS/Fruit_GS
docker build -t fruit_gs:latest .
```

Check that the image exists:

```bash
docker image ls | grep fruit_gs
```

---

## 3. Alternative: load Docker image from backup tar

If you have a backup image:

```bash
cd /home/rhea/shared/samueleWS/Fruit_GS
docker load -i fruit_gs.tar
```

Check the image:

```bash
docker image ls
```

If the loaded image appears as `<none>:<none>`, tag it manually:

```bash
docker tag IMAGE_ID fruit_gs:latest
```

Example:

```bash
docker tag 1e1e35d9ab43 fruit_gs:latest
```

---

## 4. Allow Docker to use X11

On the host:

```bash
xhost +local:docker
```

---

## 5. Run the Docker container

On the host:

```bash
cd /home/rhea/shared/samueleWS/Fruit_GS

docker run -it \
  --gpus device=0 \
  --ipc=host \
  --memory=110g \
  --memory-swap=110g \
  --memory-reservation=90g \
  -e DISPLAY=$DISPLAY \
  -e XDG_RUNTIME_DIR=/tmp/runtime-$UID \
  -e QT_XCB_GL_INTEGRATION=xcb_egl \
  -e QT_QPA_PLATFORM=xcb \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace \
  -p 7007:7007 \
  --name fruit_gs_gpu0 \
  fruit_gs:latest \
  /bin/bash
```

If the container name already exists:

```bash
docker rm fruit_gs_gpu0
```

Then run the Docker command again.

---

## 6. Check the environment inside Docker

Inside the container:

```bash
cd /workspace

python --version
pip --version

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

nvcc --version || true
```

Expected example:

```text
Python 3.10.x
torch: 2.12.0.dev...+cu128
torch cuda: 12.8
cuda available: True
gpu: NVIDIA GeForce RTX 5090
Cuda compilation tools, release 12.8
```

---

## 7. Install system dependencies inside Docker

Inside the container:

```bash
cd /workspace

apt update
apt install -y \
  git \
  build-essential \
  cmake \
  ninja-build \
  python3-dev \
  libgl1 \
  libglib2.0-0 \
  libglew-dev
```

---

## 8. Install Python dependencies

Inside the container:

```bash
pip install --no-cache-dir \
  packaging \
  ninja \
  plyfile \
  tqdm \
  opencv-python \
  joblib \
  psutil \
  numpy \
  scipy \
  scikit-learn \
  matplotlib \
  imageio \
  imageio-ffmpeg \
  lpips \
  tensorboard
```

Optional:

```bash
pip install --no-cache-dir open3d
```

---

## 9. Set CUDA environment variables

Inside the container:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="12.0"
```

Check GPU compute capability:

```bash
python - <<'PY'
import torch
print(torch.cuda.get_device_capability(0))
PY
```

For RTX 5090, this should be compatible with:

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
```

---

## 10. Patch diff-gaussian-rasterization for CUDA 12.8

With CUDA 12.8 and newer compilers, `diff-gaussian-rasterization` may fail with errors like:

```text
namespace "std" has no member "uintptr_t"
identifier "uint32_t" is undefined
identifier "uint64_t" is undefined
```

Fix this by adding `#include <cstdint>` to:

```text
submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
```

Run:

```bash
cd /workspace/submodules/diff-gaussian-rasterization

cp cuda_rasterizer/rasterizer_impl.h cuda_rasterizer/rasterizer_impl.h.bak

grep -q "#include <cstdint>" cuda_rasterizer/rasterizer_impl.h || \
  sed -i '/#pragma once/a #include <cstdint>' cuda_rasterizer/rasterizer_impl.h
```

---

## 11. Install Gaussian Splatting submodules

Inside the container:

```bash
cd /workspace

pip install --no-build-isolation --no-cache-dir -e submodules/diff-gaussian-rasterization
pip install --no-build-isolation --no-cache-dir -e submodules/simple-knn
pip install --no-build-isolation --no-cache-dir -e submodules/fused-ssim
```

Optional PyTorch3D:

```bash
pip install --no-build-isolation --no-cache-dir -e submodules/pytorch3d
```

If PyTorch3D fails, the core Gaussian Splatting pipeline may still work.

---

## 12. Full one-shot install command inside Docker

Paste this inside a fresh container:

```bash
cd /workspace

apt update
apt install -y \
  git \
  build-essential \
  cmake \
  ninja-build \
  python3-dev \
  libgl1 \
  libglib2.0-0 \
  libglew-dev

pip install --no-cache-dir \
  packaging \
  ninja \
  plyfile \
  tqdm \
  opencv-python \
  joblib \
  psutil \
  numpy \
  scipy \
  scikit-learn \
  matplotlib \
  imageio \
  imageio-ffmpeg \
  lpips \
  tensorboard

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="12.0"

cd /workspace/submodules/diff-gaussian-rasterization

cp cuda_rasterizer/rasterizer_impl.h cuda_rasterizer/rasterizer_impl.h.bak || true

grep -q "#include <cstdint>" cuda_rasterizer/rasterizer_impl.h || \
  sed -i '/#pragma once/a #include <cstdint>' cuda_rasterizer/rasterizer_impl.h

cd /workspace

pip install --no-build-isolation --no-cache-dir -e submodules/diff-gaussian-rasterization
pip install --no-build-isolation --no-cache-dir -e submodules/simple-knn
pip install --no-build-isolation --no-cache-dir -e submodules/fused-ssim

pip install --no-build-isolation --no-cache-dir -e submodules/pytorch3d || true
```

---

## 13. Verify installation

Inside the container:

```bash
cd /workspace

python - <<'PY'
import sys
print("Python:", sys.version)

import torch
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

mods = [
    "numpy",
    "torch",
    "cv2",
    "tqdm",
    "plyfile",
    "joblib",
    "psutil",
    "scipy",
    "sklearn",
    "matplotlib",
    "imageio",
    "lpips",
    "simple_knn",
    "fused_ssim",
    "diff_gaussian_rasterization",
    "gaussian_renderer",
    "scene",
]

for m in mods:
    try:
        __import__(m)
        print(f"{m}: OK")
    except Exception as e:
        print(f"{m}: FAILED -> {e}")

try:
    import pytorch3d
    print("pytorch3d: OK")
except Exception as e:
    print("pytorch3d: FAILED ->", e)
PY
```

Expected important outputs:

```text
simple_knn: OK
fused_ssim: OK
diff_gaussian_rasterization: OK
gaussian_renderer: OK
scene: OK
```

---

## 14. Save the working Docker image

After the environment works, save it.

On the host, first check containers:

```bash
docker ps -a
```

Commit the configured container:

```bash
docker commit fruit_gs_gpu0 fruit_gs:installed
```

Save the image:

```bash
docker save fruit_gs:installed -o fruit_gs_installed.tar
```

Reload later with:

```bash
docker load -i fruit_gs_installed.tar
```

---

## 15. Common errors

### Docker image not found

Error:

```text
Unable to find image 'fruit_gs:latest' locally
pull access denied for fruit_gs
```

Fix:

```bash
docker image ls
cd /home/rhea/shared/samueleWS/Fruit_GS
docker build -t fruit_gs:latest .
```

Or load from tar:

```bash
docker load -i fruit_gs.tar
docker image ls
docker tag IMAGE_ID fruit_gs:latest
```

### Container name already exists

Fix:

```bash
docker rm fruit_gs_gpu0
```

### Missing psutil

Error:

```text
ModuleNotFoundError: No module named 'psutil'
```

Fix:

```bash
pip install --no-cache-dir psutil
```

### diff-gaussian-rasterization compile error

Error:

```text
namespace "std" has no member "uintptr_t"
identifier "uint32_t" is undefined
identifier "uint64_t" is undefined
```

Fix:

```bash
cd /workspace/submodules/diff-gaussian-rasterization

grep -q "#include <cstdint>" cuda_rasterizer/rasterizer_impl.h || \
  sed -i '/#pragma once/a #include <cstdint>' cuda_rasterizer/rasterizer_impl.h

cd /workspace

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="12.0"

pip install --no-build-isolation --no-cache-dir -e submodules/diff-gaussian-rasterization
```

### X11 / GUI / OpenGL issue

On the host:

```bash
xhost +local:docker
```

Make sure Docker is run with:

```bash
-e DISPLAY=$DISPLAY
-v /tmp/.X11-unix:/tmp/.X11-unix
-e QT_XCB_GL_INTEGRATION=xcb_egl
-e QT_QPA_PLATFORM=xcb
```

Inside Docker:

```bash
apt update
apt install -y libgl1 libglib2.0-0 libglew-dev
```

---

## 16. Minimal checklist

Host:

```bash
cd /home/rhea/shared/samueleWS/Fruit_GS
xhost +local:docker

docker run -it \
  --gpus device=0 \
  --ipc=host \
  --memory=110g \
  --memory-swap=110g \
  --memory-reservation=90g \
  -e DISPLAY=$DISPLAY \
  -e XDG_RUNTIME_DIR=/tmp/runtime-$UID \
  -e QT_XCB_GL_INTEGRATION=xcb_egl \
  -e QT_QPA_PLATFORM=xcb \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace \
  -p 7007:7007 \
  --name fruit_gs_gpu0 \
  fruit_gs:latest \
  /bin/bash
```

Container:

```bash
cd /workspace

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="12.0"

python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

import diff_gaussian_rasterization
import simple_knn
import fused_ssim
import gaussian_renderer
print("Fruit_GS environment OK")
PY
```
