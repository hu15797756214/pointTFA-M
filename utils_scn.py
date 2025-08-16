from tqdm import tqdm
#from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans
from mv_utils_zs import Realistic_Projection

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import os
import json

from utils import utils

def build_scn_cache_model(cfg, model, train_loader_cache):

    if cfg['load_cache'] == False:
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (_, _, pc, target) in enumerate(tqdm(train_loader_cache)):
                    pc = pc.cuda()
                    pc_features = utils.get_model(model).encode_pc(pc)
                    train_features.append(pc_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

            cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            cache_keys = cache_keys.permute(1, 0)

            cache_values = torch.cat(cache_values, dim=0)
            cache_values = cache_values.to(torch.int64)
            cache_values = F.one_hot(cache_values).float()

            torch.save(cache_keys, cfg['cache_dir'] + "/keys.pt")
            torch.save(cache_values, cfg['cache_dir'] + "/values.pt")
    else:
        cache_keys = torch.load(cfg['cache_dir'] + "/keys.pt")
        cache_values = torch.load(cfg['cache_dir'] + "/values.pt")

    return cache_keys, cache_values


def build_image_scn_cache_model(cfg, model, train_loader_cache):

    if cfg['load_image_cache'] == False:
        pc_views = Realistic_Projection()
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (_, _, pc, target) in enumerate(tqdm(train_loader_cache)):
                    pc = pc.cuda()
                    pc = pc[:, :, 0:3].cuda()
                    image = pc_views.get_img(pc)
                    image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=True)
                    image_feat = utils.get_model(model).encode_image(image)
                    image_feat /= image_feat.norm(dim=-1, keepdim=True)
                    train_features.append(image_feat)

                    if augment_idx == 0:
                        target = target.repeat_interleave(10).cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

            cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            cache_keys = cache_keys.permute(1, 0)

            cache_values = torch.cat(cache_values, dim=0)
            cache_values = cache_values.to(torch.int64)
            cache_values = F.one_hot(cache_values).float()

            torch.save(cache_keys, cfg['cache_dir'] + "/image_keys.pt")
            torch.save(cache_values, cfg['cache_dir'] + "/image_values.pt")
    else:
        cache_keys = torch.load(cfg['cache_dir'] + "/image_keys.pt")
        cache_values = torch.load(cfg['cache_dir'] + "/image_values.pt")

    return cache_keys, cache_values

def pre_load_scn_pc_features(cfg, split, model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (_, _, pc, target) in enumerate(tqdm(loader)):
                pc = pc.cuda()
                pc_features = utils.get_model(model).encode_pc(pc)
                pc_features /= pc_features.norm(dim=-1, keepdim=True)
                features.append(pc_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)
        labels = labels.to(torch.int64)
        labels = labels.cuda()

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")

    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")

    return features, labels


def scn_data_efficiency(cfg, cache_keys, cache_values, train_loader_cache):
    # Aggregating data for each category
    cache_keys = cache_keys.permute(1, 0)
    list_of_labels = train_loader_cache.dataset.labels
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

    # RandomExtract data on origin cache model
    if cfg['Extract_DATA'] == True:
        new_keys = []
        new_values = []
        for key in keys_class:
            key.cpu()
            key_size = key.shape[0]
            key_size_extract = int(key_size * (cfg['Extract_rate']) / 100)
            indices = np.random.choice(key_size, key_size_extract, replace=False)
            key = key[indices]
            new_keys.append(key)

        cache_keys = torch.cat(new_keys, dim=0)
        cache_keys = cache_keys.permute(1, 0).cuda()

        for value in values_class:
            value_size = value.shape[0]
            value_size_extract = int(value_size * (cfg['Extract_rate'] / 100))
            indices = np.random.choice(value_size, value_size_extract, replace=False)
            value = value[indices]
            new_values.append(value)

        cache_values = torch.cat(new_values, dim=0).cuda()

    if cfg['Fewshot_DATA'] == True:
        # Construction of few-shot cache model
        new_keys = []
        new_values = []
        for key in keys_class:
            key.cpu()
            key_size = key.shape[0]
            key_size_extract = cfg['shots']
            indices = np.random.choice(key_size, key_size_extract, replace=False)
            key = key[indices]
            new_keys.append(key)

        cache_keys = torch.cat(new_keys, dim=0)
        cache_keys = cache_keys.permute(1, 0).cuda()

        for value in values_class:
            value_size = value.shape[0]
            value_size_extract = cfg['shots']
            indices = np.random.choice(value_size, value_size_extract, replace=False)
            value = value[indices]
            new_values.append(value)

        cache_values = torch.cat(new_values, dim=0).cuda()

    if cfg['Kmeans_DATA'] == True:
        if cfg['load_RMC'] == False:
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
                            value_class = value[i]
                            new_values.append(value_class)

                    cache_values = torch.stack(new_values).cuda()

            cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            cache_keys = cache_keys.permute(1, 0).cuda()

            torch.save(cache_keys, cfg['cache_dir'] + "/keys_RMC.pt")
            torch.save(cache_values, cfg['cache_dir'] + "/values_RMC.pt")

        else:
            cache_keys = torch.load(cfg['cache_dir'] + "/keys_RMC.pt")
            cache_values = torch.load(cfg['cache_dir'] + "/values_RMC.pt")

    return cache_keys, cache_values

def scn_image_data_efficiency(cfg, cache_keys, cache_values, train_loader_cache):
    # Aggregating data for each category
    cache_keys = cache_keys.permute(1, 0)
    list_of_labels =  torch.tensor(train_loader_cache.dataset.labels)
    list_of_labels = list_of_labels.repeat_interleave(10).tolist()
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

    # RandomExtract data on origin cache model
    if cfg['image_Extract_DATA'] == True:
        new_keys = []
        new_values = []
        for key in keys_class:
            key.cpu()
            key_size = key.shape[0]
            key_size_extract = int(key_size * (cfg['image_Extract_rate']) / 100)
            indices = np.random.choice(key_size, key_size_extract, replace=False)
            key = key[indices]
            new_keys.append(key)

        cache_keys = torch.cat(new_keys, dim=0)
        cache_keys = cache_keys.permute(1, 0).cuda()

        for value in values_class:
            value_size = value.shape[0]
            value_size_extract = int(value_size * (cfg['image_Extract_rate'] / 100))
            indices = np.random.choice(value_size, value_size_extract, replace=False)
            value = value[indices]
            new_values.append(value)

        cache_values = torch.cat(new_values, dim=0).cuda()

    if cfg['image_Fewshot_DATA'] == True:
        # Construction of few-shot cache model
        new_keys = []
        new_values = []
        for key in keys_class:
            key.cpu()
            key_size = key.shape[0]
            key_size_extract = cfg['shots']
            indices = np.random.choice(key_size, key_size_extract, replace=False)
            key = key[indices]
            new_keys.append(key)

        cache_keys = torch.cat(new_keys, dim=0)
        cache_keys = cache_keys.permute(1, 0).cuda()

        for value in values_class:
            value_size = value.shape[0]
            value_size_extract = cfg['shots']
            indices = np.random.choice(value_size, value_size_extract, replace=False)
            value = value[indices]
            new_values.append(value)

        cache_values = torch.cat(new_values, dim=0).cuda()

    if cfg['image_Kmeans_DATA'] == True:
        if cfg['image_load_RMC'] == False:
            # Data augmentation for the Kmeans cache model
            cache_keys = []
           # weights = torch.tensor([0.75, 0.75, 0.75, 1.00, 0.75, 0.50, 1.0, 0.75, 0.25, 0.25]).to("cuda")
            weights = torch.tensor([0.75, 0.75, 0.75, 1, 0.75,0.5,1, 0.75, 0.25, 0.25]).to("cuda")
            weights = weights.view(1, 10, 1)
            for augment_idx in range(cfg['augment_epoch']):
                new_keys = []
                new_values = []
                for key in keys_class:
                    if cfg['image_n_clusters'] != 1:

                        key = key.view(-1, 10, key.shape[-1])
                        weights_tensor = key * weights
                        # key = weights_tensor.sum(dim=1)/ weights.sum(dim=1)
                        key = weights_tensor.sum(dim=1) / weights.shape[1]

                        cluster_idx_x, cluster_centers = kmeans(X=key, num_clusters=cfg['image_n_clusters'], distance='euclidean',
                                                                device=torch.device('cuda:0'))
                    else:
                        cluster_centers = key.mean(dim=0).unsqueeze(0)
                    new_keys.append(cluster_centers)

                cache_keys.append(torch.cat(new_keys, dim=0).unsqueeze(0))

                if augment_idx == 0:
                    for value in values_class:
                        for i in range(cfg['image_n_clusters']):
                            value_class = value[i]
                            new_values.append(value_class)

                    cache_values = torch.stack(new_values).cuda()

            cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            cache_keys = cache_keys.permute(1, 0).cuda()

            torch.save(cache_keys, cfg['cache_dir'] + "/keys_image_RMC.pt")
            torch.save(cache_values, cfg['cache_dir'] + "/values_image_RMC.pt")

        else:
            cache_keys = torch.load(cfg['cache_dir'] + "/keys_RMC.pt")
            cache_values = torch.load(cfg['cache_dir'] + "/values_RMC.pt")

    return cache_keys, cache_values

    # if cfg['image_Kmeans_DATA'] == True:
    #     if cfg['image_load_RMC'] == False:
    #         # Data augmentation for the Kmeans cache model
    #         cache_keys = []
    #
    #         for augment_idx in range(cfg['augment_epoch']):
    #
    #             new_values = []
    #             for key in keys_class:
    #                 new_keys = []
    #                 key = key.view(-1, 10, key.shape[-1])
    #
    #                 if cfg['n_clusters'] != 1:
    #                     for view in range(10):
    #                         view_data = key[:,view,:]
    #
    #                         cluster_idx_x, cluster_centers = kmeans(X=view_data, num_clusters=cfg['n_clusters'], distance='euclidean',
    #                                                             device=torch.device('cuda:0'))
    #                         new_keys.append(cluster_centers)
    #
    #                     cache_keys.append(torch.cat(new_keys, dim=0))
    #                 else:
    #                     cluster_centers = key.mean(dim=0).unsqueeze(0)
    #                     cache_keys.append(torch.cat(new_keys, dim=0))
    #
    #
    #             if augment_idx == 0:
    #                 for value in values_class:
    #                     for i in range(cfg['n_clusters']):
    #                         value_class = value[i]
    #                         new_values.append(value_class)
    #
    #                 cache_values = torch.stack(new_values).cuda()
    #             cache_values = cache_values.repeat(10,1)
    #
    #
    #         cache_keys = torch.cat(cache_keys, dim=0)
    #         cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    #         cache_keys = cache_keys.permute(1, 0).cuda()
    #
    #         torch.save(cache_keys, cfg['cache_dir'] + "/keys_image_RMC.pt")
    #         torch.save(cache_values, cfg['cache_dir'] + "/values_image_RMC.pt")
    #
    #     else:
    #         cache_keys = torch.load(cfg['cache_dir'] + "/keys_image_RMC.pt")
    #         cache_values = torch.load(cfg['cache_dir'] + "/values_image_RMC.pt")
    #
    # return cache_keys, cache_values




def project_scnpc_and_gets_img_features(cfg, split, model, loader):
    if cfg['load_img_feat'] == False:
        pc_views = Realistic_Projection()
        image_features = []
        with torch.no_grad():
            for i, (_, _, pc, target) in enumerate(tqdm(loader)):
                pc = pc.cuda()
                image = pc_views.get_img(pc)
                image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=True)
                image_feat = utils.get_model(model).encode_image(image)
                image_feat /= image_feat.norm(dim=-1, keepdim=True)
                image_features.append(image_feat)

        image_features = torch.cat(image_features)
        torch.save(image_features, cfg['cache_dir'] + "/" + split + "_img.pt")

    else:
        image_features = torch.load(cfg['cache_dir'] + "/" + split + "_img.pt")

    return image_features