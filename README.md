# GAFDet — Camera–Radar 3D Object Detection

A source-only snapshot of the locally modified CRN-based implementation
associated with the GAFDet manuscript. The original CRN code, third-party
notices and compatible experiment names are retained.

This snapshot is prepared for repository upload. It has **not** yet been
validated as the exact server version used for all paper experiments or
tested in a fresh CUDA environment. No model checkpoint or dataset is
included. See the [reproducibility checklist](docs/REPRODUCIBILITY.md) before
describing it as a complete paper release.

## Overview

The implementation contains radar-guided image enhancement, geometric
interaction during view transformation, and environment-conditioned BEV
fusion. The principal source locations are:

| Paper component | Source locations |
| --- | --- |
| DFE | `layers/fuser/radar_depth_aware_fusion.py`, used by `layers/backbones/rvt_lss_fpn.py` |
| GIVT | `layers/backbones/rvt_lss_fpn.py`, `layers/modules/neighborhood_cross_attention.py` |
| WAFF | `layers/fuser/weather_multimodal_feature_aggregation.py` |
| Detection pipeline | `models/camera_radar_net_det.py` |
| ResNet-50 configuration | `exps/det/CRN_r50_256x704_128x128_4key.py` |

Configuration files retain their historical `CRN_*` names. They import the
locally modified model, so filenames must not be interpreted as evidence
that they are unchanged official CRN baselines. The environment branch in
the current source produces an eight-dimensional latent representation;
its channels are not assigned explicit day/rain/night category labels.

## Getting started

Read [installation and data preparation](docs/INSTALL.md) for the Linux/CUDA
requirements, legacy dependencies, build steps and preprocessing commands.
The historical `CRN.yaml` alone does not specify the complete working
environment, particularly PyTorch/CUDA, MMCV and NATTEN.

The expected dataset location is `data/nuScenes`. Obtain nuScenes separately
under its applicable terms. CUDA operators must be rebuilt for the target
machine; precompiled binaries are intentionally excluded.

Once the matching environment and prepared data are available, run from
the repository root:

```bash
# Training entry-point example
PYTHONPATH=. python exps/det/CRN_r50_256x704_128x128_4key.py \
    --amp_backend native --gpus 1 -b 2

# Evaluation entry-point example: supply a checkpoint matching this source
PYTHONPATH=. python exps/det/CRN_r50_256x704_128x128_4key.py \
    --amp_backend native --gpus 1 -b 1 -e \
    --ckpt_path /absolute/path/to/matching_checkpoint.ckpt
```

These commands document the existing CLI; they are not a claim that a
clean-environment reproduction has already been completed. The preserved
R50 configuration registers visualization hooks that copy features to CPU;
see the installation notes before measuring inference efficiency.

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
test/            Inherited test sources; not a validated GAFDet test suite
mmdetection3d/   Bundled legacy dependency source
docs/            Installation, provenance and release checks
```

Weights, dataset blobs, generated annotations/results, logs, caches, compiled
libraries and private IDE settings are excluded. The `datasets/` Python
package is included; the `data/` dataset directory is not.

No verified GAFDet VoD experiment pipeline or complete set of paper
checkpoints is provided by this local snapshot. Confirm the missing release
items before promising reproduction of those results.

## Acknowledgements and licensing

This project is based on [CRN](https://github.com/youngskkim/CRN), with
components originating from BEVDepth, MMDetection3D and neighborhood-attention
research code. See [third-party notices](THIRD_PARTY_NOTICES.md).

The root [MIT license](LICENSE) is the preserved CRN license. Bundled
components retain their own licenses and copyright headers. Project-specific
authors/citation and permission to publish the modifications should be
confirmed by the GAFDet authors. The original CRN documentation and citation
are preserved in [the upstream README](docs/UPSTREAM_CRN_README.md); its
performance tables and checkpoints describe CRN, not GAFDet.
