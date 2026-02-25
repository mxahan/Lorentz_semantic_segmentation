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
Qualitative segmentation results across Cityscapes, Pascal-VOC (DeepLabV3), ADE20K, and COCO-Stuff test samples
<p align="center">
  <img src="demo/qualitative.png" width="720">
</p>

Uncertainty maps across Cityscapes, Pascal-VOC, COCO-Stuff, and ADE20K datasets (columns). The first row shows angle-based uncertainty, while the second row shows length-based uncertainty.
<p align="center">
  <img src="demo/uncertainty_map.png" width="720">
</p>

Image mask captured by different keywords. We point to the corresponding histogram; the accurate class text has the highest confidence, and so on (more results in the supplementary section).
<p align="center">
  <table>
      <tr>
        <td align="center"><img src="demo/elephant.png" width="350"/><br>Elephant</td>
        <td align="center"><img src="demo/motorcycle.png" width="350"/><br>Motorcycle</td>
      </tr>
      <tr>
        <td align="center"><img src="demo/orange.png" width="350"/><br>Orange</td>
        <td align="center"><img src="demo/laptop.png" width="350"/><br>Laptop</td>
      </tr>
  </table>
</p>
