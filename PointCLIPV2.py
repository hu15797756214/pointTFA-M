import os
import random
import argparse
import numpy as np
import torch
import yaml
from tqdm import tqdm
from collections import OrderedDict

from datasets.modelnet40 import *
import torch.nn.functional as F
from utils.utils import get_dataset
import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
from utils import utils
from datasets.modelnet40 import customized_collate_fn
from utils_tfa import *

import clip
import time
import statistics

def get_arguments():
    # Data
    parser = argparse.ArgumentParser(description='PointTFA on modelnet40', add_help=False)
    parser.add_argument('--pretrain_dataset_name', default='modelnet40', type=str)
    parser.add_argument('--pretrain_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--validate_dataset_name', default='modelnet40_test', type=str)
    parser.add_argument('--validate_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--use_height', action='store_true')
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')

    # ULIP_model
    parser.add_argument('--model', default='ULIP_PointBERT', type=str)
    parser.add_argument('--ckpt_addr', default='./pretrained_ckpt/ckpt_pointbert_ULIP-2.pt',
                        help='the ckpt to ulip 3d enconder')
    parser.add_argument('--evaluate_3d', default='True', help='eval 3d only')

    # cfg
    parser.add_argument('--config', dest='config', help='settings of PointTFA in yaml format')
    args = parser.parse_args()

    return args


def run_PointTFA(cfg, cache_keys, cache_values, test_features_R, test_labels, ulip_weights, logit_scale):
    
    print("\nRun PointTFA:")
    ulip_weights, cache_keys = ulip_weights.T, cache_keys.T
    # 3D Zero-shot
    ulip_logits = logit_scale * test_features_R @ ulip_weights * 1.0
    acc = cls_acc(ulip_logits, test_labels)
    print("\n**** Zero-shot ULIP's test accuracy: {:.2f}. ****\n".format(acc))

    # 3D TFA
    beta, alpha = cfg['init_beta'], cfg['init_alpha']

    affinity = test_features_R @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

    tfa_logits = ulip_logits + cache_logits * alpha
    acc = cls_acc(tfa_logits, test_labels)
    print("**** PointTFA's test accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    _ = search_hp(cfg, cache_keys, cache_values, test_features_R, test_labels, ulip_weights, logit_scale)


def sort(features, labels):

    sorted_indices = np.argsort(labels)

    sorted_features = features[sorted_indices]
    sorted_labels = labels[sorted_indices]

    return sorted_features, sorted_labels

def RMC(cfg, train_features, train_labels):

    #full-set
    cache_keys = train_features
    cache_values = F.one_hot(train_labels).half()
    cache_values = cache_values.cuda()

    # RMC
    if cfg['Kmeans_DATA'] == True:
        list_of_labels = [train_labels[i].item() for i in range(len(train_labels))]
        current_class = list_of_labels[0]
        start_idx = 0
        values_class = []
        keys_class = []

        for i in range(len(list_of_labels)):
            if list_of_labels[i] != current_class:
                values_class.append(cache_values[start_idx: i])
                keys_class.append(cache_keys[start_idx: i])
                current_class = list_of_labels[i]
                start_idx = i

        values_class.append(cache_values[start_idx:])
        keys_class.append(cache_keys[start_idx:])

        # Data augmentation for the Kmeans cache model
        cache_keys = []
        for augment_idx in range(cfg['augment_epoch']):
            new_keys = []
            new_values = []
            for key in keys_class:
                if cfg['n_clusters'] != 1:
                    cluster_idx_x, cluster_centers = kmeans(X=key, num_clusters=cfg['n_clusters'], distance='euclidean',
                                                            device=torch.device('cuda:0'))
                else:
                    cluster_centers = key.mean(dim=0).unsqueeze(0)
                new_keys.append(cluster_centers)

            cache_keys.append(torch.cat(new_keys, dim=0).unsqueeze(0))

            if augment_idx == 0:
                for value in values_class:
                    for i in range(cfg['n_clusters']):
                        value_i = value[i]
                        new_values.append(value_i)

                cache_values = torch.stack(new_values).cuda()
                cache_values = cache_values.half()

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.half().cuda()

    return cache_keys, cache_values

def main():
    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()
    logit_scale = clip_model.logit_scale

    # Train features
    #train_features = torch.load("./PointCLIPV2_ZS/vit_b16/modelnet40_train/features.pt")
    #train_labels = torch.load("./PointCLIPV2_ZS/vit_b16/modelnet40_train/labels.pt")

    # Test features
    #test_features = torch.load("./PointCLIPV2_ZS/vit_b16/modelnet40_test/features.pt")
    #test_labels = torch.load("./PointCLIPV2_ZS/vit_b16/modelnet40_test/labels.pt")

    #train_features, train_labels = train_features.cpu().numpy(), train_labels.cpu().numpy()
    #train_features, train_labels = sort(train_features, train_labels)
    #test_features, test_labels = test_features.cpu().numpy(), test_labels.cpu().numpy()
    #test_features, test_labels = sort(test_features, test_labels)

    #train_features, train_labels, test_features, test_labels = torch.tensor(train_features).cuda(), torch.tensor(train_labels).cuda(), torch.tensor(test_features).cuda(), torch.tensor(test_labels).cuda()

    #torch.save(train_features, "PointCLIPV2_sort" + "/" "train_features.pt")
    #torch.save(train_labels, "PointCLIPV2_sort" + "/" + "train_labels.pt")
    #torch.save(test_features, "PointCLIPV2_sort" + "/" + "test_features.pt")
    #torch.save(test_labels, "PointCLIPV2_sort" + "/" + "test_labels.pt")

    # Textual features
    print("Getting textual features as ULIP's classifier.")
    text_weights = torch.load("./PointCLIPV2_ZS/vit_b16/modelnet40_test/text_classifier.pt")

    # Train features
    train_features = torch.load("./PointCLIPV2_sort/train_features.pt")
    train_labels = torch.load("./PointCLIPV2_sort/train_labels.pt")

    # Test features
    test_features = torch.load("./PointCLIPV2_sort/test_features.pt")
    test_labels = torch.load("./PointCLIPV2_sort/test_labels.pt")

    train_features /= train_features.norm(dim=-1, keepdim=True)
    #test_features /= test_features.norm(dim=-1, keepdim=True)
    #text_weights /= text_weights.norm(dim=-1, keepdim=True)


    # Data-efficiency
    print("\ncache model with Data-efficiency.")
    cache_keys, cache_values = RMC(cfg, train_features, train_labels)

    # Cloud Query Refactor
    print("\nTransfer cache knowledge to test features.")
    test_features = reinvent_query(test_features, cache_keys)

    # ------------------------------------------ PointTFA ------------------------------------------
    run_PointTFA(cfg, cache_keys, cache_values, test_features, test_labels, text_weights, logit_scale)

if __name__ == '__main__':
    main()