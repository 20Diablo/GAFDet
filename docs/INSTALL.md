# Installation and data preparation

## Environment

This is a legacy CUDA research codebase. Use Linux with an NVIDIA GPU, a
CUDA toolkit compatible with the installed PyTorch build, and a C++ compiler.
macOS is suitable for editing/uploading this repository, not for reproducing
its CUDA detection pipeline.

The supplied `CRN.yaml` is a historical environment export (Python 3.8.18).
Its machine-specific `prefix` has been removed, but it **is not a complete
training-environment lock**: PyTorch, CUDA, MMCV and NATTEN are not recorded
there. The upstream CRN README documents its original stack; do not assume
that stack is the one used for the GAFDet manuscript or an RTX 4090.

Before configuring a new machine, record the actual working server environment:

```bash
python --version
python -m pip freeze > environment-working-server.txt
nvcc --version
nvidia-smi
```

Review the export for private paths, private package URLs and credentials
before publishing it. Install the matching PyTorch/torchvision, legacy
MMCV-with-CUDA-ops, MMDetection and PyTorch Lightning packages from that
environment. The source uses the MMDetection 2.x / MMCV 1.x APIs; it is not
a migration to MMDetection 3.x / MMCV 2.x.

The NATTEN version must provide these legacy symbols:

```python
from natten.functional import natten2dav, natten2dqkrpb
from natten import NeighborhoodAttention2D
```

Do not install an arbitrary current NATTEN version without checking those
APIs and its PyTorch/CUDA compatibility. Other direct dependencies include
NumPy, nuScenes-devkit, pyquaternion, Pillow, OpenCV, matplotlib, tqdm and
torchviz. The preserved visualization scripts may require additional packages.
The optional ConvNeXt experiment also imports MMEngine and MMPretrain; those
are not required by the documented ResNet-50 entry point.

After the matching environment has been installed, build the bundled
MMDetection3D source and project CUDA operators, from the repository root:

```bash
python -m pip install -v -e ./mmdetection3d
python setup.py develop
```

The bundled MMDetection3D version reports `1.0.0rc4`. Retain this local source
until any project-specific changes have been audited; replacing it with an
arbitrary package version is not equivalent. The original project distribution
name in `setup.py` remains `CRN` for compatibility. Build dependencies before
running either command; these commands are not dependency installers for a
fresh Python environment.

## nuScenes

Acquire nuScenes from its official distribution under its applicable terms.
Dataset images, point clouds, annotations and generated pickle files are not
redistributed in this repository. From the repository root, link your dataset:

```bash
mkdir -p data
ln -s /absolute/path/to/nuScenes data/nuScenes
```

Run preprocessing from the repository root:

```bash
PYTHONPATH=. python scripts/gen_info.py
PYTHONPATH=. python scripts/gen_depth_gt.py
PYTHONPATH=. python scripts/gen_radar_bev.py
PYTHONPATH=. python scripts/gen_radar_pv.py
```

The depth-target generator requires LiDAR keyframes for supervision. This is
not a claim that LiDAR is an inference input. The radar generators require
the radar blobs/sweeps in addition to keyframes.

Expected local layout:

```text
data/nuScenes/
  samples/
  sweeps/
  maps/
  v1.0-trainval/
  nuscenes_infos_train.pkl
  nuscenes_infos_val.pkl
  depth_gt/
  radar_bev_filter/
  radar_pv_filter/
```

The weather utilities are retained as source, with their local dataset path
changed to `data/nuScenes`. Their split rules and generated files should be
checked against the experiment being reproduced. They are not substitutes
for a confirmed validation split manifest. No verified GAFDet VoD pipeline
has been established for this local snapshot.

## Training and evaluation entry point

From the repository root, in the matching CUDA environment:

```bash
# Example: train the preserved ResNet-50 configuration on one GPU.
PYTHONPATH=. python exps/det/CRN_r50_256x704_128x128_4key.py \
    --amp_backend native --gpus 1 -b 2

# Example: evaluate a checkpoint matching this source/configuration.
PYTHONPATH=. python exps/det/CRN_r50_256x704_128x128_4key.py \
    --amp_backend native --gpus 1 -b 1 -e \
    --ckpt_path /absolute/path/to/matching_checkpoint.ckpt
```

These are entry-point examples based on the existing CLI, not a completed
clean-environment reproduction test. The checkpoint must match the source
version, configuration and state-dict naming. Official CRN weights are not
automatically compatible with the added modules.

The local ResNet-50 experiment currently registers feature-visualization
hooks that copy features to CPU. They were preserved rather than silently
changing experiment behavior. Do not treat timings from that entry point as
the manuscript's measured inference speed without matching the profiling
configuration and checking those hooks.

The inherited `test/` files refer to old modules/fixtures and are not a
validated GAFDet regression suite. Dataset fixtures were intentionally
excluded from redistribution. A full unit-test pass is not claimed.
