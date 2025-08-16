#**PointTFAm: Multi-Modal, Training-Free Adaptation for Point Cloud Understanding**

#**Introduction**

High-dimensional data are more sparsely distributed in space compared to low-dimensional data of the same size
(e.g., 3D point cloud vs 2D images) —a phenomenon known as the “Curse of Dimensionality” (COD). Consequently, more
samples are required to effectively fine-tune models for highdimensional tasks like 3D point cloud understanding, leading
to increased computational costs. Meanwhile, although 3D point clouds provide comprehensive spatial details, 2D images projected from specific viewpoints often capture sufficient information for understanding visual content. To address the COD challeng and leverage the complementary nature of 3D-2D data, we introduce a multi-modal, training-free approach named PointTFAm, an extended version of our original PointTFA. This new approach incorporates 2D view images projected from 3D point clouds in traning-free manner to augment cloud classification. Specifically, PointTFAm contains two training-free branches that process 3D point clouds and 2D view images independently. Each branch includes its own Representative Memory Cache (RMC), Cloud/Image Query Refactor (CQR or IQR), and Training-Free Adapter (TFA). The model combines the outputs from both branches through score fusion to make effective multi-modal predictions. PointTFAm improves upon single-modal PointTFA by accuracy gains of 1.01%, 1.32%, and 4.64% on the ModelNet40, ModelNet10, and ScanObjectNN benchmarks, respectively, setting new state-of-the-art performance for training-free point cloud understanding approaches.
<img width="843" height="583" alt="image" src="https://github.com/user-attachments/assets/a0037e33-5a68-40c8-8435-842ba6b6b9bd" />
