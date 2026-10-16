# Installation and data preparation

## Environment

Use Linux with an NVIDIA GPU, a CUDA toolkit compatible with the installed
PyTorch build, and a C++ compiler. The supplied `CRN.yaml` lists Python
3.8.18 and supporting packages. Install compatible PyTorch, torchvision,
MMCV, MMDetection, PyTorch Lightning and NATTEN packages separately.
The code uses MMDetection 2.x and MMCV 1.x APIs.

The NATTEN version must provide these legacy symbols:

```python
from natten.functional import natten2dav, natten2dqkrpb
from natten import NeighborhoodAttention2D
```

Select a NATTEN version supporting these APIs and your PyTorch/CUDA build.
Other direct dependencies include
NumPy, nuScenes-devkit, pyquaternion, Pillow, OpenCV, matplotlib, tqdm and
torchviz. The visualization scripts may require additional packages.
The optional ConvNeXt experiment also imports MMEngine and MMPretrain; those
are not required by the documented ResNet-50 entry point.

After the matching environment has been installed, build the bundled
MMDetection3D source and project CUDA operators, from the repository root:

```bash
python -m pip install -v -e ./mmdetection3d
python setup.py develop
```

Build the bundled MMDetection3D source (`1.0.0rc4`) rather than replacing it
with a different package version. The distribution name in `setup.py` remains
`CRN` for compatibility. Install the dependencies before running the build
commands above.

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

The depth-target generator uses LiDAR keyframes for training supervision;
inference uses camera and radar inputs. The radar generators require
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

The weather utilities use `data/nuScenes`. Select the split rules and generated
files appropriate to the experiment configuration.

## Training and evaluation entry point

From the repository root, in the matching CUDA environment:

```bash
# Train the ResNet-50 configuration on one GPU.
PYTHONPATH=. python exps/det/CRN_r50_256x704_128x128_4key.py \
    --amp_backend native --gpus 1 -b 2

# Example: evaluate a checkpoint matching this source/configuration.
PYTHONPATH=. python exps/det/CRN_r50_256x704_128x128_4key.py \
    --amp_backend native --gpus 1 -b 1 -e \
    --ckpt_path /absolute/path/to/matching_checkpoint.ckpt
```

Use a checkpoint matching the source version, configuration and state-dict
naming. CRN checkpoints require compatible parameter shapes and module names
when loading the added components.

The ResNet-50 configuration includes feature-visualization hooks that copy
features to CPU. Disable these hooks when profiling inference latency.
