# Third-party code and attribution

Third-party code retains its original copyright and license notices.

| Component | Source / attribution | Notice retained here |
| --- | --- | --- |
| CRN | [youngskkim/CRN](https://github.com/youngskkim/CRN), Youngseok Kim and collaborators | [MIT license](third_party/licenses/CRN.txt); [upstream README](docs/UPSTREAM_CRN_README.md) |
| MMDetection3D | [OpenMMLab/MMDetection3D](https://github.com/open-mmlab/mmdetection3d) | `mmdetection3d/LICENSE` and source headers |
| BEVDepth-derived code | [Megvii-BaseDetection/BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth) | Existing Megvii source headers and original CRN acknowledgements |
| Neighborhood attention components | Ali Hassani / [NATTEN](https://github.com/SHI-Labs/NATTEN) | Full MIT notice in `layers/modules/neighborhood_cross_attention.py` and `channel_cross_attention.py` |

Bundled components and dependencies remain subject to their respective
licenses and source-file notices.

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

The results and model-zoo links in the upstream README refer to CRN.
