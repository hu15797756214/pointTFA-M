import os
import random
import argparse
import yaml
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.nn as nn

from datasets.modelnet40 import *

from utils.utils import get_dataset
import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
from utils import utils
from datasets.modelnet40 import customized_collate_fn
from utils_tfa import *

def get_arguments():
    # Data
    parser = argparse.ArgumentParser(description='PointTFA on modelnet10', add_help=False)
    parser.add_argument('--pretrain_dataset_name', default='modelnet10', type=str)
    parser.add_argument('--pretrain_dataset_prompt', default='modelnet10_64', type=str)
    parser.add_argument('--validate_dataset_name', default='modelnet10_test', type=str)
    parser.add_argument('--validate_dataset_prompt', default='modelnet10_64', type=str)
    parser.add_argument('--use_height', action='store_true')
    parser.add_argument('--npoints', default=1024, type=int, help='number of points used for pre-train and test.')
    parser.add_argument('--data_addr', default='./DATA/modelnet10/modelnet10_normal_resampled',
                        help='the data addr')

    # ULIP_model
    parser.add_argument('--model', default='ULIP_PointBERT', type=str)
    parser.add_argument('--ckpt_addr', default='./pretrained_ckpt/checkpoint_pointbert.pt',
                        help='the ckpt to ulip 3d encoder')
    parser.add_argument('--evaluate_3d', default='True', help='eval 3d only')

    # cfg
    parser.add_argument('--config', dest='config', help='settings of PointTFA in yaml format')
    args = parser.parse_args()

    return args


def run_PointTFA(cfg, cache_keys, cache_values, test_features, test_features_R, test_labels, ulip_weights):
    # Zero-shot ULIP
    ulip_logits = 100. * test_features @ ulip_weights
    class_accuracies, all_classes_average_accuracy = cls_acc(ulip_logits, test_labels)
    # print("\n**** Zero-shot ULIP's test accuracy: {:.2f}. ****\n".format(acc))

    # 打印每个类别的准确率
    print("Zero-shot ULIP每个类别的准确率:")
    for class_idx, acc in class_accuracies.items():
        print(f"类别 {class_idx}: {acc:.2f}%")

    # 打印所有类别的平均准确率
    print(f"\nZero-shot ULIP所有类别的平均准确率: {all_classes_average_accuracy:.2f}%")

    # PointTFA
    beta, alpha = cfg['init_beta'], cfg['init_alpha']

    affinity = test_features_R @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

    tfa_logits = ulip_logits + cache_logits * alpha

    class_accuracies, all_classes_average_accuracy = cls_acc(tfa_logits , test_labels)

    # 打印每个类别的准确率
    print("PointTFA每个类别的准确率:")
    for class_idx, acc in class_accuracies.items():
        print(f"类别 {class_idx}: {acc:.2f}%")

    # 打印所有类别的平均准确率
    print(f"\nPointTFA所有类别的平均准确率: {all_classes_average_accuracy:.2f}%")

    # acc = cls_acc(tfa_logits, test_labels)
    # print("**** PointTFA's test accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    _ = search_hp(cfg, cache_keys, cache_values, test_features, test_features_R, test_labels, ulip_weights)

def run_PointTFA_mm(cfg, cache_keys, cache_values, test_features, test_features_R, test_labels, ulip_weights, image_features):
    # Zero-shot ULIP
    ulip_logits = 100. * test_features @ ulip_weights
    acc = cls_acc(ulip_logits, test_labels)
    print("**** Zero-shot ULIP's test accuracy: {:.2f}. ****\n".format(acc))

    # PointTFA_mm
    beta, alpha, gamma = cfg['init_beta'], cfg['init_alpha'], cfg['init_gamma']
    print("Searching image view weights:")
    if image_features.shape[0]==9080:
        img_logits = search_image_view_weight(cfg, image_features, test_labels, ulip_weights)

    else:
        img_logits = search_image_R_view_weight(cfg, image_features, test_labels, ulip_weights)
    affinity = test_features_R @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

    mm_logits = ulip_logits + cache_logits * alpha + img_logits * gamma
    acc = cls_acc(mm_logits, test_labels)
    print("**** PointTFA_mm's test accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    _ = search_mm_hp(cfg, cache_keys, cache_values, test_features, test_features_R, test_labels, ulip_weights, img_logits)

def run_PointTFA_R_mm(cfg, cache_keys, cache_values,image_cache_keys, image_cache_values,test_features, test_features_R, test_labels, ulip_weights,image_features,image_features_R):
    # Zero-shot ULIP
    weights = torch.tensor([[0.75, 0.75, 0.75, 0.25, 0.5,1,1, 1, 0.25, 0.25]]).to("cuda")
    weights = weights.view(1, 10, 1)

    if test_features.shape[0]==9080:
        test_features = test_features.view(-1, 10, test_features.shape[-1])
        weights_tensor = test_features * weights
        test_features = weights_tensor.sum(dim=1) / weights.sum(dim=1)

    clip_logits = 100. * test_features @ ulip_weights
    class_accuracies, all_classes_average_accuracy = cls_acc(clip_logits, test_labels)
    # print("\n**** Zero-shot ULIP's test accuracy: {:.2f}. ****\n".format(acc))

    # 打印每个类别的准确率
    print("Zero-shot ULIP每个类别的准确率:")
    for class_idx, acc in class_accuracies.items():
        print(f"类别 {class_idx}: {acc:.2f}%")

    # 打印所有类别的平均准确率
    print(f"\nZero-shot ULIP所有类别的平均准确率: {all_classes_average_accuracy:.2f}%")
    # PointTFA_mm
    beta, alpha, gamma = cfg['init_beta'], cfg['init_alpha'], cfg['init_gamma']
    print("Searching image view weights:")

    if image_features.shape[0]==9080:
        img_logits = search_image_view_weight(cfg, image_features, test_labels, ulip_weights)

    else:
        img_logits = search_image_R_view_weight(cfg, image_features, test_labels, ulip_weights)

    if image_cache_keys.shape[1] ==39910:
        image_cache_keys =  image_cache_keys.permute(1,0)
        image_cache_keys = image_cache_keys.view(-1, 10, image_cache_keys.shape[-1])
        image_cache_keys = image_cache_keys *weights
        image_cache_keys = image_cache_keys.sum(dim=1) / weights.sum(dim=1)
        image_cache_keys = image_cache_keys.permute(1, 0)

        image_cache_values=  image_cache_values.view(-1,10,image_cache_values.shape[1])
        image_cache_values =image_cache_values [:,0,:]

    affinity = test_features_R @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values


    affinity_image = image_features_R @ image_cache_keys
    clip_logits_image =  ((-1) * (beta - beta * affinity_image)).exp() @ image_cache_values

    mm_logits = clip_logits + cache_logits * alpha + img_logits +clip_logits_image* gamma
    class_accuracies, all_classes_average_accuracy  = cls_acc(mm_logits, test_labels)
    # 打印每个类别的准确率
    print("PointTFA_mm每个类别的准确率:")
    for class_idx, acc in class_accuracies.items():
        print(f"类别 {class_idx}: {acc:.2f}%")

    # 打印所有类别的平均准确率
    print(f"\nPointTFA_mm所有类别的平均准确率: {all_classes_average_accuracy:.2f}%")
    # PointTFA_mm
    beta, alpha, gamma = cfg['init_beta'], cfg['init_alpha'], cfg['init_gamma']
    print("Searching image view weights:")

    # print("**** PointTFA_mm's test accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    _ = search_mm_R_hp(cfg, cache_keys, cache_values,image_cache_keys,image_cache_values, test_features, test_features_R, test_labels, ulip_weights,
                     img_logits,image_features_R)

def run_PointTFA_R_mm_no_point_cache(cfg, image_cache_keys, image_cache_values,test_features, test_features_R, test_labels, ulip_weights,image_features,image_features_R):
    # Zero-shot ULIP
    weights = torch.tensor([[0.75, 0.75, 0.75, 0.25, 0.5,1,1, 1, 0.25, 0.25]]).to("cuda")
    weights = weights.view(1, 10, 1)

    if test_features.shape[0]==9080:
        test_features = test_features.view(-1, 10, test_features.shape[-1])
        weights_tensor = test_features * weights
        test_features = weights_tensor.sum(dim=1) / weights.sum(dim=1)

    clip_logits = 100. * test_features @ ulip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("**** Zero-shot ULIP's test accuracy: {:.2f}. ****\n".format(acc))

    # PointTFA_mm
    beta, alpha, gamma = cfg['init_beta'], cfg['init_alpha'], cfg['init_gamma']
    print("Searching image view weights:")

    if image_features.shape[0]==9080:
        img_logits = search_image_view_weight(cfg, image_features, test_labels, ulip_weights)

    else:
        img_logits = search_image_R_view_weight(cfg, image_features, test_labels, ulip_weights)

    if image_cache_keys.shape[1] ==39910:
        image_cache_keys =  image_cache_keys.permute(1,0)
        image_cache_keys = image_cache_keys.view(-1, 10, image_cache_keys.shape[-1])
        image_cache_keys = image_cache_keys *weights
        image_cache_keys = image_cache_keys.sum(dim=1) / weights.sum(dim=1)
        image_cache_keys = image_cache_keys.permute(1, 0)

        image_cache_values=  image_cache_values.view(-1,10,image_cache_values.shape[1])
        image_cache_values =image_cache_values [:,0,:]



    affinity_image = image_features_R @ image_cache_keys
    clip_logits_image =  ((-1) * (beta - beta * affinity_image)).exp() @ image_cache_values

    mm_logits = clip_logits  + img_logits +clip_logits_image* gamma
    acc = cls_acc(mm_logits, test_labels)
    print("**** PointTFA_mm's test accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    _ = search_mm_R_hp_no_point_cache(cfg, image_cache_keys,image_cache_values, test_features, test_features_R, test_labels, ulip_weights,
                     img_logits,image_features_R)


def main():
    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")


    # ULIP
    ckpt = torch.load(args.ckpt_addr, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    try:
        model = getattr(models, old_args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.ckpt_addr))
    except:
        model = getattr(models, args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.ckpt_addr))
    model.eval()

    # Dataset
    random.seed(1)
    torch.manual_seed(1)

    tokenizer = SimpleTokenizer()

    print("Preparing modelnet10 dataset.")
    train_dataset = get_dataset(None, tokenizer, args, 'train')
    test_dataset = get_dataset(None, tokenizer, args, 'val')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=10, shuffle=False,
                                              pin_memory=True, sampler=None, drop_last=False)
    train_loader_cache = torch.utils.data.DataLoader(train_dataset, batch_size=64, num_workers=10, shuffle=False,
                                                     pin_memory=True, sampler=None, drop_last=False)

    # Textual features
    print("Getting textual features as ULIP's classifier.")
    ulip_weights = ulip_classifier(args, model, tokenizer)

    # Construct the cache model by training set
    print("\nConstructing cache model by visual features and labels.")
    cache_keys, cache_values = build_pc_cache_model(cfg, model, train_loader_cache)

    # Construct the image-cache model by point-cache
    print("Constructing image-cache model by point-cache cache_keys features and cache_values.")
    image_cache_keys, image_cache_values = build_image_cache_model(cfg, model, train_loader_cache)

    # Data efficiency with Kmeans
    print("\nRepresentative Cache Memory")
    cache_keys, cache_values = data_efficiency(cfg, cache_keys, cache_values, train_loader_cache)

    # image-Data-efficiency
    print("image-cache model with Data-efficiency.")
    image_cache_keys, image_cache_values = image_data_efficiency(cfg, image_cache_keys, image_cache_values,
                                                                 train_loader_cache)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_pc_features(cfg, "test", model, test_loader)

    # Query Cloud Refactor
    print("\nTransfer cache knowledge to test features")
    test_features_R = reinvent_query(test_features, cache_keys)

    # Getting 2D images and extracting its features
    print("Getting 2D images and extracting its features:")
    image_features = project_pc_and_gets_img_features(cfg, 'test', model, test_loader)

    # 2D - chace重构3Dtest
    print("Transfer cache knowledge to test features.")
    cache2D_3Dtest_features_R = reinvent_query(test_features, image_cache_keys)

    # 3D - chace重构2Dtest
    print("Transfer cache knowledge to test features.")
    cache3D_2Dtest_features_R = image_reinvent_query(image_features, cache_keys)

    # image Query Refactor
    image_test_features_R = image_reinvent_query(image_features, image_cache_keys)


    # ------------------------------------------ PointTFA ------------------------------------------
    # 2D-TFA
    # run_PointTFA(cfg, image_cache_keys, image_cache_values, image_features, image_test_features_R, test_labels,ulip_weights)

    # 3D-chace重构2Dtest  2D-TFA
    # run_PointTFA(cfg, image_cache_keys, image_cache_values, cache_keys, cache_values,image_features, cache3D_2Dtest_features_R, test_labels,ulip_weights)

    # 2Dand3D-chace重构2Dtest  2D-TFA
    # run_PointTFA_mm_2d_3d_mm(cfg, cache_keys,cache_values,image_cache_keys, image_cache_values, image_features, image_test_features_R, cache3D_2Dtest_features_R , test_labels, ulip_weights, image_features)

    # ----------------------------------------------------------------------------------------------------------------------------------

    # 3D-TFA
    run_PointTFA(cfg, cache_keys, cache_values, test_features, test_features_R, test_labels, ulip_weights)

    # 2D-chace重构3Dtest    3D-TFA
    # run_PointTFA(cfg, image_cache_keys, image_cache_values, image_features, cache2D_3Dtest_features_R, test_labels,ulip_weights)

    # -----------------------------------------------------------------------------------------------------------------------------------------------

    #  三模态完整版   没有重构图片特征 3D-cache-3dtest            2D,3D,TFA
    #run_PointTFA_mm(cfg, cache_keys, cache_values, test_features, test_features_R, test_labels, ulip_weights,image_features)

    #   三模态完整版      2D-cache-3dtest       2D,3D,TFA
    # run_PointTFA_mm(cfg, cache_keys, cache_values, test_features, cache2D_3Dtest_features_R , test_labels, ulip_weights, image_features)

    #  三模态完整版，重构图片特征    3D-cache-3dtest            2D,3D,TFA
    run_PointTFA_R_mm(cfg, cache_keys, cache_values,image_cache_keys, image_cache_values, test_features, test_features_R, test_labels, ulip_weights,image_features,image_test_features_R)

   #  三模态完整版，重构图片特征    3D-cache-3dtest            2D,3D,TFA
   #  run_PointTFA_R_mm_no_point_cache(cfg, image_cache_keys, image_cache_values, test_features, test_features_R, test_labels, ulip_weights,image_features,image_test_features_R)

    #  三模态完整版             2Dand3D-cache-3dtest
    # run_PointTFA_mm_2d_3d_mm(cfg, image_cache_keys, image_cache_values,cache_keys, cache_values, test_features, test_features_R,cache2D_3Dtest_features_R , test_labels, ulip_weights, image_features)
    #

if __name__ == '__main__':
    main()