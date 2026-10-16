# GAFDet: Geometry-Guided Adaptive Fusion for Robust Camera-Radar 3D Object Detection under Adverse Weather

GAFDet is a geometry-guided adaptive camera-radar fusion framework for robust 3D object detection, particularly under adverse weather and low-light conditions. To address spatial misalignment caused by inaccurate monocular depth estimation and insufficiently adaptive multi-modal fusion, GAFDet introduces three key components: Depth-guided Feature Enhancement (DFE), Geometric Interactive View Transformation (GIVT), and Weather-aware Adaptive Feature Fusion (WAFF). These components leverage radar depth priors to enhance image representations, improve geometric consistency during image-to-BEV transformation, and adaptively fuse camera and radar features based on environmental cues and cross-modal consistency. Extensive experiments demonstrate that GAFDet achieves competitive detection performance and improved robustness under challenging conditions such as rain and nighttime.

## Overview

The principal components and source locations are:

| Paper component | Source locations |
| --- | --- |
| DFE | `layers/fuser/radar_depth_aware_fusion.py`, used by `layers/backbones/rvt_lss_fpn.py` |
| GIVT | `layers/backbones/rvt_lss_fpn.py`, `layers/modules/neighborhood_cross_attention.py` |
| WAFF | `layers/fuser/weather_multimodal_feature_aggregation.py` |
| Detection pipeline | `models/camera_radar_net_det.py` |
| ResNet-50 configuration | `exps/det/CRN_r50_256x704_128x128_4key.py` |

Configuration files retain the `CRN_*` naming convention for compatibility.

## Getting started

See [installation and data preparation](docs/INSTALL.md) for environment
requirements, build steps and preprocessing commands.

The expected dataset location is `data/nuScenes`. Obtain nuScenes separately
under its applicable terms. CUDA operators must be rebuilt for the target
machine; precompiled binaries are intentionally excluded.

Once the matching environment and prepared data are available, run from
the repository root:

```bash
# Training
PYTHONPATH=. python exps/det/CRN_r50_256x704_128x128_4key.py \
    --amp_backend native --gpus 1 -b 2

# Evaluation: specify a checkpoint matching the configuration
PYTHONPATH=. python exps/det/CRN_r50_256x704_128x128_4key.py \
    --amp_backend native --gpus 1 -b 1 -e \
    --ckpt_path /absolute/path/to/matching_checkpoint.ckpt
```

## Repository contents

```text
callbacks/       Training callbacks
datasets/        Dataset loader source (not dataset files)
evaluators/      Evaluation code and configurations
exps/            Model/training configurations and CLI
layers/          Feature, geometry and fusion modules
models/          Detection pipeline
ops/             CUDA/C++ operator source and Python bindings
scripts/         Data preparation and visualization utilities
test/            Test utilities
mmdetection3d/   Bundled legacy dependency source
docs/            Installation instructions and upstream documentation
```

Weights, dataset blobs, generated annotations/results, logs, caches, compiled
libraries and private IDE settings are excluded. The `datasets/` Python
package is included; the `data/` dataset directory is not.

## Acknowledgements

This project is built upon [CRN](https://github.com/youngskkim/CRN). We thank
the authors for making their code publicly available.
