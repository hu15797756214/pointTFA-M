**PointTFAm: Multi-Modal, Training-Free Adaptation for Point Cloud Understanding**
====================================================================================
**Introduction**
-------------------------------------------------------------------------------------
High-dimensional data are more sparsely distributed in space compared to low-dimensional data of the same size
(e.g., 3D point cloud vs 2D images) —a phenomenon known as the “Curse of Dimensionality” (COD). Consequently, more
samples are required to effectively fine-tune models for high dimensional tasks like 3D point cloud understanding, leading
to increased computational costs. Meanwhile, although 3D point clouds provide comprehensive spatial details, 2D images projected from specific viewpoints often capture sufficient information for understanding visual content. To address the COD challeng and leverage the complementary nature of 3D-2D data, we introduce a multi-modal, training-free approach named PointTFAm, an extended version of our original PointTFA. This new approach incorporates 2D view images projected from 3D point clouds in traning-free manner to augment cloud classification. Specifically, PointTFAm contains two training-free branches that process 3D point clouds and 2D view images independently. Each branch includes its own Representative Memory Cache (RMC), Cloud/Image Query Refactor (CQR or IQR), and Training-Free Adapter (TFA). The model combines the outputs from both branches through score fusion to make effective multi-modal predictions. PointTFAm improves upon single-modal PointTFA by accuracy gains of 1.01%, 1.32%, and 4.64% on the ModelNet40, ModelNet10, and ScanObjectNN benchmarks, respectively, setting new state-of-the-art performance for training-free point cloud understanding approaches.
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
./DATA |<br>
-- labels.json |<br>
-- templates.json |<br>
-- /modelnet40<br>
  -- modelnet40_normal_resampled |<br>

./pretrained_ckpt |<br>
-- ckpt_pointbert_ULIP-2.pt<br>


Get Started
--------------------------------------------------------
*Configs*

The running configurations can be modified in "configs/dataset.yaml".
For simplicity, we provide the hyperparamters achieving the overall best performance on 1~16 shots for a dataset, which accord with the scores reported in the paper. If respectively tuned for different shot numbers, the 1-16-shot performance can be further improved. You can edit the **_search_scale_**, **_search_step_**,**_init_beta_** and **_init_alpha_** for **_fine-grained_** tuning.
Note that the default load_cache and load_pre_feat are False for the first running, which will store the cache model and test features in cache/dataset/. For later running, they can be set as True for faster hyperparamters tuning.

*Running*
-------------------------------------------------
For modelnet40 dataset:

CUDA_VISIBLE_DEVICES=0 python main.py --config configs/modelnet40.yaml
