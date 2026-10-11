# Reproducibility and release checklist

## What this snapshot contains

- A source-only copy of the local `my_CRN` directory.
- The modified model, experiment configurations and preprocessing utilities.
- CUDA/C++ source files and build scripts; compiled `.so` files are excluded.
- Bundled MMDetection3D source and its original license.
- Original CRN and neighborhood-attention copyright/license notices.

Model computation, module names and state-dict keys were not changed during
release preparation. The `CRN_*` configuration filenames are retained for
compatibility. A filename alone does not establish that it is the official
CRN baseline, because these configurations import the modified model.

## What must be confirmed before declaring a paper release

- [ ] Compare the local model/configuration files with the working server
      experiment version and record the chosen Git commit.
- [ ] Record the exact Python, PyTorch, torchvision, PyTorch Lightning, CUDA,
      MMCV, MMDetection, NATTEN and bundled MMDetection3D versions.
- [ ] Check preprocessing settings, validation split manifests, sample
      counts and data licensing against the reported experiments.
- [ ] Identify the paper checkpoint, verify that it loads with the released
      source, and document its configuration and checksum.
- [ ] Test data preparation and checkpoint evaluation in a fresh CUDA
      environment, then document the commands and observed metrics.
- [ ] Confirm the implementation/training interpretation of the environment
      branch against the version used for the manuscript.
- [ ] Provide the actual VoD configuration/pipeline if the release is intended
      to reproduce the paper's VoD results. The nuScenes pipeline is not a
      substitute for it.
- [ ] Match efficiency-measurement settings, including precision, frame
      caching, decoding/NMS and visualization hooks.
- [ ] Confirm the exact WAFF perturbation operation, seed(s), checkpoint and
      evaluation split before adding a script claiming to reproduce that
      experiment.
- [ ] Obtain the authors'/supervisor's approval to publish project-specific
      modifications and choose their licensing terms.

No training, inference or clean-environment reproduction was run as part of
this local packaging step. Published-paper performance is not asserted for
this unverified local snapshot.

## Excluded files

The publication copy excludes pretrained/trained weights, datasets and test
dataset fixtures, generated annotations/results, training/TensorBoard logs,
Python caches, compiled binaries, IDE settings, machine metadata and an
unrelated code-to-text merging utility (`main.py`). These remain in the
original local folder; exclusion from this copy does not delete them.

Two third-party pretrained weights were excluded: an EfficientNet-B7
backbone and a DeepLabV3+ segmentation checkpoint. Optional scripts relying
on them require separately acquired matching weights. A GAFDet paper
checkpoint has not been supplied by this source-only snapshot. Do not
mislabel either of these third-party weights as the GAFDet checkpoint.

Dataset helper paths were normalized to `data/nuScenes`; the historical
Conda export's absolute `prefix` and an inactive server-checkpoint comment
were removed. These are portability/documentation changes, not new results.
