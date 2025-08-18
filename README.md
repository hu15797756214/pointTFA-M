**PointTFAm: Multi-Modal, Training-Free Adaptation for Point Cloud Understanding**
====================================================================================
**Introduction**
-------------------------------------------------------------------------------------
High-dimensional data are more sparsely distributed in space compared to low-dimensional data of the same size
(e.g., 3D point cloud vs 2D images) —a phenomenon known as the “Curse of Dimensionality” (COD). Consequently, more
samples are required to effectively fine-tune models for high dimensional tasks like 3D point cloud understanding, leading
to increased computational costs. Meanwhile, although 3D point clouds provide comprehensive spatial details, 2D images projected from specific viewpoints often capture sufficient information for understanding visual content. To address the COD challeng and leverage the complementary nature of 3D-2D data, we introduce a multi-modal, training-free approach named PointTFAm, an extended version of our original PointTFA. This new approach incorporates 2D view images projected from 3D point clouds in traning-free manner to augment cloud classification. Specifically, PointTFAm contains two training-free branches that process 3D point clouds and 2D view images independently. Each branch includes its own Representative Memory Cache (RMC), Cloud/Image Query Refactor (CQR or IQR), and Training-Free Adapter (TFA). The model combines the outputs from both branches through score fusion to make effective multi-modal predictions. PointTFAm improves upon single-modal PointTFA by accuracy gains of 1.01%, 1.32%, and 4.64% on the ModelNet40, ModelNet10, and ScanObjectNN benchmarks, respectively, setting new state-of-the-art performance for training-free point cloud understanding approaches.

**The overall model network is built on PointTFA, with basically the same environment and operation . We have realized the generation from point clouds to images through the projection network, thereby achieving multi-modal prediction.**

<img width="843" height="583" alt="image" src="https://github.com/user-attachments/assets/a0037e33-5a68-40c8-8435-842ba6b6b9bd" />

**Install environments**
---------------------------------------------
_The code is tested with CUDA==11.0 and pytorch==1.10.1_<br>
_conda create -n tfa python=3.7.15_<br>
_conda activate tfa_<br>
_conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge_<br>
_pip install -r requirements.txt_<br>


Download datasets and pre-trained models, put them in the right paths
--------------------------------------------------------------------------
For specific model data and pre-trained weights, please refer to **[PointTFA](https://github.com/user-attachments/assets/a0037e33-5a68-40c8-8435-842ba6b6b9bd)**.

./DATA |<br>
-- labels.json |<br>
-- templates.json |<br>
-- /modelnet40<br>
  -- modelnet40_normal_resampled |<br>

./pretrained_ckpt |<br>
-- ckpt_pointbert_ULIP-2.pt<br>

 **projection net**<br>
 --------------------------
The projection network adopts the **_PointCLIP_** model, because it features training-free capability and can well balance model complexity and performance.

Get Started
--------------------------------------------------------
**Configs**

The running configurations can be modified in **"configs/dataset.yaml"**. It includes parameter settings related to two categories: images and point clouds, and can also handle separate tuning for **1~16 shot** counts. You can edit **search_scale**, **search_step**, **init_beta**, **init_alpha**, **init_gamma**, and **init_alpha** for fine-grained tuning and hyperparameter search settings. Note that for the first run, the parameters **load_cache, load_pre_feat, load_RMC, load_img_feat, load_image_cache, and image_load_RMC** are set to **False** for data preprocessing. For subsequent runs, they can be set to True to speed up hyperparameter tuning.

*Running*
-------------------------------------------------
For modelnet40 dataset:<br>
CUDA_VISIBLE_DEVICES=0 python **main_modelnet40_image_mm.py** --config **configs/modelnet40.yaml**

For modelnet10 dataset:<br>
CUDA_VISIBLE_DEVICES=0 python **main_modelnet10_image_mm.py** --config  **configs/modelnet10.yaml**

For ScanObjectNN dataset:<br>
CUDA_VISIBLE_DEVICES=0 python **main_ScanObjectNN_image_mm.py** --config **configs/scanobjectnn.yaml**
