# Third-party code and attribution

This repository is derived from existing research code. Preserve upstream
copyright headers and license files when redistributing it.

| Component | Source / attribution | Notice retained here |
| --- | --- | --- |
| CRN | [youngskkim/CRN](https://github.com/youngskkim/CRN), Youngseok Kim and collaborators | Root `LICENSE` (MIT); original README in `docs/UPSTREAM_CRN_README.md` |
| MMDetection3D | [OpenMMLab/MMDetection3D](https://github.com/open-mmlab/mmdetection3d) | `mmdetection3d/LICENSE` and source headers |
| BEVDepth-derived code | [Megvii-BaseDetection/BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth) | Existing Megvii source headers and original CRN acknowledgements |
| Neighborhood attention components | Ali Hassani / [NATTEN](https://github.com/SHI-Labs/NATTEN) | Full MIT notice in `layers/modules/neighborhood_cross_attention.py` and `channel_cross_attention.py` |

This inventory highlights directly identified sources; it is not a claim
that every bundled third-party file has been independently provenance-audited.
Keep other notices found in the source as well. Dependencies and pretrained
weights retain their own terms. The root MIT license is the preserved CRN
license, not a replacement for licenses applicable to bundled dependencies.

The upstream CRN paper is:

```bibtex
@inproceedings{kim2023crn,
  title={CRN: Camera Radar Net for Accurate, Robust, Efficient 3D Perception},
  author={Kim, Youngseok and Shin, Juyeb and Kim, Sanmin and Lee, In-Jae and Choi, Jun Won and Kum, Dongsuk},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={17615--17626},
  year={2023}
}
```

The original CRN results, authors and model-zoo links in the archived upstream
README refer to CRN, not to GAFDet. Project-specific authorship/citation and
licensing should be finalized by the GAFDet authors before a public release.
