# Lorentz Entailment Cone for Semantic Segmentation

**Official PyTorch Implementation**

This is a PyTorch/GPU implementation of the paper [Lorentz Entailment Cone for Semantic Segmentation]()

```
@inproceedings{zahid2026lorentz,
  title={Lorentz Entailment Cone for Semantic Segmentation},
  author={Hasan, Zahid and Ahmed, Masud and Roy, Nirmalya},
  booktitle={2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={},
  year={2026},
  organization={IEEE}
}
```


## Abstract
Semantic segmentation in hyperbolic space can capture hierarchical structure in low dimensions with uncertainty quantification. Existing approaches choose the Poincaré ball model for hyperbolic geometry, which suffers from numerical instabilities, optimization, and computational challenges. We propose a novel, tractable, architecture-agnostic semantic segmentation framework in the hyperbolic Lorentz model. We employ text embeddings with semantic and visual cues to guide hierarchical pixel-level representations in Lorentz space. This enables stable and efficient optimization without requiring a Riemannian optimizer, and easily integrates with existing Euclidean architectures. Beyond segmentation, our approach yields free uncertainty estimation, confidence map, boundary delineation, hierarchical and text-based retrieval,  and zero-shot performance, reaching generalized flatter minima. We further introduce a novel uncertainty and confidence indicator in Lorentz cone embeddings. Extensive experiments on ADE20K, COCO-Stuff-164k, Pascal-VOC, and Cityscapes with state-of-the-art models (DeepLabV3 and SegFormer) validate the effectiveness and generality of our approach. Our results demonstrate the potential of hyperbolic Lorentz embeddings for robust and uncertainty-aware semantic segmentation.

## Result
Trained on Cityscape dataset and tested on SemanticKITTI, ACDC, CADEdgeTune dataset
<p align="center">
  <img src="demo/qualitative.png" width="720">
</p>
