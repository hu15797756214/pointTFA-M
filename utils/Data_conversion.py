import numpy as np
import torch
from tqdm import tqdm

def data_conversion(path):
    point_cloud = torch.load(path)

    # ModelNet40 类别列表
    modelnet40 = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup',
                  'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
                  'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
                  'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

    # 初始化字典，每个类别对应一个点云列表
    processed_data = {cls: [] for cls in modelnet40}

    # 获取 xyz 和 labels
    xyz = point_cloud["xyz"]  # Shape: (7679, 3, 1024)
    labels = point_cloud["labels"]  # Shape: (7679)

    # 遍历数据，将数据按类别存入字典
    for i in tqdm(range(len(labels)), desc="Processing Data", unit="sample"):
        class_index = labels[i].item()  # 获取类别索引
        class_name = modelnet40[class_index]  # 获取类别名称

        # 由于 xyz[i] 形状为 (3, 1024)，需要转置为 (1024, 3)
        processed_data[class_name].append(xyz[i].T)

    return processed_data

# 运行转换
# path = "/your/path/to/modelnet40_data.pt"  # 修改为你的路径
# processed_data = data_conversion(path)

# 保存为 npy 文件
# save_path = "/media/www/新加卷/gxydate/OpenShape/meta_data/modelnet40/shape-e_modelnet40_pc.npy"
# np.save(save_path, processed_data, allow_pickle=True)
# print(f"数据已保存到 {save_path}")