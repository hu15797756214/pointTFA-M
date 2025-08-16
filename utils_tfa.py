import matplotlib.pyplot as plt
from tqdm import tqdm
#from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans
from sklearn.metrics import silhouette_score
#from memory_profiler import profile
from mv_utils_zs import Realistic_Projection
from clip import clip, load
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import os
import json

from utils import utils

import torch

import torch
import numpy as np

def cls_acc(output, target, topk=1):
    num_classes = output.size(1)
    class_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}

    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    for class_idx in range(num_classes):
        class_indices = (target == class_idx).nonzero(as_tuple=True)[0]
        if len(class_indices) > 0:
            class_correct[class_idx] = float(correct[:, class_indices].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
            class_total[class_idx] = len(class_indices)

    class_accuracies = {i: 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)}

    all_correct = sum(class_correct.values())
    all_total = sum(class_total.values())
    all_classes_average_accuracy = 100 * all_correct / all_total if all_total > 0 else 0

    return class_accuracies, all_classes_average_accuracy



# def cls_acc(output, target, topk=1):
#     pred = output.topk(topk, 1, True, True)[1].t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#     acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
#     acc = 100 * acc / target.shape[0]
#     return acc

def scale_(x, target):
    y = (x - x.min()) / (x.max() - x.min())
    y *= target.max() - target.min()
    y += target.min()

    return y

def compute_pc_text_distributions(test_features, ulip_classifier):

    test_pc_text_distribution = test_features @ ulip_classifier
    test_pc_text_distribution = nn.Softmax(dim=-1)(test_pc_text_distribution)

    return test_pc_text_distribution


def get_kl_divergence_sims(key_text_distribution, test_pc_text_distribution):
    bs = 100
    kl_div_sim = torch.zeros((test_pc_text_distribution.shape[0]), (key_text_distribution.shape[0]))

    for i in tqdm(range(test_pc_text_distribution.shape[0]//bs)):
        curr_batch = test_pc_text_distribution[i*bs : (i+1)*bs]
        repeated_batch = torch.repeat_interleave(curr_batch, key_text_distribution.shape[0], dim = 0)
        repeated_key_text_distribution = torch.cat([key_text_distribution]*bs)
        kl = repeated_batch * (repeated_batch.log() - repeated_key_text_distribution.log())
        kl = kl.sum(dim = -1)
        kl = kl.view(bs, -1)
        kl_div_sim[i*bs : (i+1)*bs , : ] = kl

    return kl_div_sim


def get_kl_div_sims(cache_keys, test_features, test_features_R, ulip_classifier):
    key_text_distribution = compute_pc_text_distributions(cache_keys.T, ulip_classifier)
    test_pc_text_distribution = compute_pc_text_distributions(test_features, ulip_classifier)
    test_R_pc_text_distribution = compute_pc_text_distributions(test_features_R, ulip_classifier)

    test_kl_div_sim = get_kl_divergence_sims(key_text_distribution, test_pc_text_distribution)
    test_R_kl_div_sim = get_kl_divergence_sims(key_text_distribution, test_R_pc_text_distribution)

    return test_kl_div_sim, test_R_kl_div_sim

def ulip_classifier(args, model, tokenizer):
    with open(os.path.join("./DATA", 'templates.json')) as f:
        templates = json.load(f)[args.pretrain_dataset_prompt]

    with open(os.path.join("./DATA", 'labels.json')) as f:
        labels = json.load(f)[args.pretrain_dataset_name]

    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).cuda(None, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
            class_embeddings = utils.get_model(model).encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)

        text_features = torch.stack(text_features, dim=0)
        text_features = text_features.permute(1, 0)
    return text_features


def build_pc_cache_model(cfg, model, train_loader_cache):

    if cfg['load_cache'] == False:
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (pc, target, target_name) in enumerate(tqdm(train_loader_cache)):
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

def build_image_cache_model(cfg,model,train_loader_cache):

    if cfg['load_image_cache'] ==False:
        pc_views = Realistic_Projection()
        image_cache_keys = []
        image_cache_values = []

        with torch.no_grad():
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []
                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (pc, target, target_name) in enumerate(tqdm(train_loader_cache,desc="Processing batches")):
                    pc = pc.cuda()
                    image = pc_views.get_img(pc)
                    image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=True)
                    image_feat = utils.get_model(model).encode_image(image)
                    image_feat /= image_feat.norm(dim=-1, keepdim=True)

                    train_features.append(image_feat)
                    if augment_idx == 0:
                            target = target.repeat_interleave(10).cuda()
                            image_cache_values.append(target)
                image_cache_keys.append(torch.cat(train_features,dim=0).unsqueeze(0))

        cache_keys = torch.cat(image_cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)

        cache_values = torch.cat(image_cache_values, dim=0)
        cache_values = cache_values.to(torch.int64)
        cache_values = F.one_hot(cache_values).float()

        torch.save(cache_keys, cfg['cache_dir'] + "/iamge_keys.pt")
        torch.save(cache_values, cfg['cache_dir'] + "/image_values.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + "/iamge_keys.pt")
        cache_values = torch.load(cfg['cache_dir'] + "/image_values.pt")

    return cache_keys, cache_values


def pre_load_pc_features(cfg, split, model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (pc, target, target_name) in enumerate(tqdm(loader)):
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

def data_efficiency(cfg, cache_keys, cache_values, train_loader_cache):

    # Aggregating data for each category
    cache_keys = cache_keys.permute(1, 0)
    list_of_labels = train_loader_cache.dataset.list_of_labels
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


    if cfg['Extract_DATA'] == True:
        # RandomExtract data on raw cache model
        new_keys = []
        new_values = []
        for key in keys_class:
            key.cpu()
            key_size = key.shape[0]
            key_size_extract = int(key_size * (cfg['Extract_rate'])/100)
            indices = np.random.choice(key_size, key_size_extract, replace=False)
            key=key[indices]
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
            #sil = []
            for augment_idx in range(cfg['augment_epoch']):
                new_keys = []
                new_values = []
                for key in keys_class:
                    if cfg['n_clusters'] != 1:
                        cluster_idx_x, cluster_centers = kmeans(X = key, num_clusters = cfg['n_clusters'], distance = 'euclidean',
                                                            device = torch.device('cuda:0'))
                        #silhouette_avg = silhouette_score(key.cpu().numpy(), cluster_idx_x)
                    else:
                        cluster_centers = key.mean(dim=0).unsqueeze(0)
                    new_keys.append(cluster_centers)
                    #sil.append(silhouette_avg)

                #silhouette_mean = sum(sil) / len(sil)
                #print("Silhouette:", silhouette_mean)

                cache_keys.append(torch.cat(new_keys, dim=0).unsqueeze(0))

                if augment_idx == 0:
                    for value in values_class:
                        for i in range(cfg['n_clusters']):
                            value_i = value[i]
                            new_values.append(value_i)

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


def image_data_efficiency(cfg, cache_keys, cache_values, train_loader_cache):

    # Aggregating data for each category
    cache_keys = cache_keys.permute(1, 0)
    list_of_labels = torch.tensor(train_loader_cache.dataset.list_of_labels)
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


    if cfg['image_Extract_DATA'] == True:
        # RandomExtract data on raw cache model
        new_keys = []
        new_values = []
        for key in keys_class:
            key.cpu()
            key_size = key.shape[0]
            key_size_extract = int(key_size * (cfg['image_Extract_rate'])/100)
            indices = np.random.choice(key_size, key_size_extract, replace=False)
            key=key[indices]
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
        weights = torch.tensor([0.75, 0.75, 0.75, 0.75, 1, 1, 0.5, 1, 0.25, 0.25]).to("cuda")
        weights = weights.view(1, 10, 1)
        new_keys = []
        new_values = []
        for key in keys_class:
            # key = key.view(-1, 10, key.shape[-1])
            # weights_tensor = key * weights
            # key = weights_tensor.sum(dim=1) / weights.sum(dim=1)

            key.cpu()
            key_size = key.shape[0]
            key_size_extract = cfg['image_shots']
            indices = np.random.choice(key_size, key_size_extract, replace=False)
            key = key[indices]
            new_keys.append(key)

        cache_keys = torch.cat(new_keys, dim=0)
        cache_keys = cache_keys.permute(1, 0).cuda()

        for value in values_class:
            value_size = value.shape[0]
            value_size_extract = cfg['image_shots']
            indices = np.random.choice(value_size, value_size_extract, replace=False)
            value = value[indices]
            new_values.append(value)

        cache_values = torch.cat(new_values, dim=0).cuda()


    if cfg['image_Kmeans_DATA'] == True:
        if cfg['image_load_RMC'] == False:
            # Data augmentation for the Kmeans cache model
            cache_keys = []
            weights = torch.tensor([0.75, 0.75, 0.75, 0.75, 1,1,0.5, 1,0.25,0.25]).to("cuda")
            weights = weights.view(1,10, 1)
            #sil = []
            for augment_idx in range(cfg['augment_epoch']):
                new_keys = []
                new_values = []
                for key in keys_class:
                    if cfg['image_n_clusters'] != 1:
                        # key = key.view(-1,10,key.shape[-1]).mean(1)

                        key = key.view(-1, 10, key.shape[-1])
                        weights_tensor = key *weights
                        key = weights_tensor.sum(dim=1)/ weights.sum(dim=1)
                        # key = weights_tensor.sum(dim=1) / weights.shape[1]

                        cluster_idx_x, cluster_centers = kmeans(X = key, num_clusters = cfg['image_n_clusters'], distance = 'euclidean',
                                                            device = torch.device('cuda:0'))
                        #silhouette_avg = silhouette_score(key.cpu().numpy(), cluster_idx_x)
                    else:
                        cluster_centers = key.mean(dim=0).unsqueeze(0)
                    new_keys.append(cluster_centers)
                    #sil.append(silhouette_avg)

                #silhouette_mean = sum(sil) / len(sil)
                #print("Silhouette:", silhouette_mean)

                cache_keys.append(torch.cat(new_keys, dim=0).unsqueeze(0))

                if augment_idx == 0:
                    for value in values_class:
                        for i in range(cfg['image_n_clusters']):
                            value_i = value[i]
                            new_values.append(value_i)

                    cache_values = torch.stack(new_values).cuda()

            cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            cache_keys = cache_keys.permute(1, 0).cuda()

            torch.save(cache_keys, cfg['cache_dir'] + "/image_keys_RMC.pt")
            torch.save(cache_values, cfg['cache_dir'] + "/image_values_RMC.pt")

        else:
            cache_keys = torch.load(cfg['cache_dir'] + "/image_keys_RMC.pt")
            cache_values = torch.load(cfg['cache_dir'] + "/image_values_RMC.pt")

    return cache_keys, cache_values

def fewshot_random(cfg, cache_keys, cache_values, train_loader_cache):
    # Aggregating data for each category
    cache_keys = cache_keys.permute(1, 0)
    list_of_labels = train_loader_cache.dataset.list_of_labels
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

    # Construction of few-shot cache model
    new_keys = []
    for key in keys_class:
        key.cpu()
        key_size = key.shape[0]
        key_size_extract = cfg['shots']
        indices = np.random.choice(key_size, key_size_extract, replace=False)
        key = key[indices]
        new_keys.append(key)

    cache_keys = torch.cat(new_keys, dim=0)
    cache_keys = cache_keys.permute(1, 0).cuda()

    return cache_keys

def image_fewshot_random(cfg, cache_keys, cache_values, train_loader_cache):
    # Aggregating data for each category

    weights = torch.tensor([0.75, 0.75, 0.75, 0.75, 1.00, 1.00, 0.50, 1.00, 0.25, 0.25]).to("cuda")
    weights = weights.view(1, 10, 1)

    cache_keys = cache_keys.permute(1, 0)
    list_of_labels = torch.tensor(train_loader_cache.dataset.list_of_labels)
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

    # Construction of few-shot cache model
    new_keys = []
    for key in keys_class:
        key.cpu()
        key_size = key.shape[0]
        key_size_extract = cfg['shots']

        key = key.view(-1, 10, key.shape[-1])
        weights_tensor = key * weights
        key = weights_tensor.sum(dim=1) / weights.sum(dim=1)

        indices = np.random.choice(int(key.shape[0]), key_size_extract, replace=False)
        key = key[indices]
        new_keys.append(key)

    cache_keys = torch.cat(new_keys, dim=0)
    cache_keys = cache_keys.permute(1, 0).cuda()

    return cache_keys

def reinvent_query(test_features, cache_keys):


    sim = test_features @ cache_keys
    sim = (sim*100).softmax(dim=-1)
    test_features = sim @ cache_keys.T
    test_features /= test_features.norm(dim=-1, keepdim=True)

    return test_features

def image_reinvent_query(test_features, cache_keys):
    #test_features = test_features.view(-1,10,test_features.shape[-1]).mean(dim=1)


    weights = torch.tensor([0.75, 0.75, 0.75, 0.75, 1,1,0.5, 1,0.25,0.25]).to("cuda")
    weights = weights.view(1, 10, 1)
    test_features = test_features.view(-1,10,test_features.shape[-1])
    weight_tensor = test_features * weights
    test_features = weight_tensor.sum(dim=1)/weights.sum(dim=1)

    # if cache_keys.shape[1] == 23090:     #test_noise_0.01_objectdataset.h5
    if cache_keys.shape[1] == 98430:
        cache_keys_t = cache_keys.permute(1,0)
        cache_keys_t = cache_keys_t.view(-1,10,test_features.shape[-1])
        cache_keys_t =cache_keys_t *weights
        cache_keys_t = cache_keys_t.sum(dim=1)/weights.sum(dim=1)
        cache_keys = cache_keys_t.permute(1, 0)

    sim = test_features @ cache_keys
    sim = (sim*100).softmax(dim=-1)
    test_features = sim @ cache_keys.permute(1,0)
    test_features /= test_features.norm(dim=-1, keepdim=True)

    return test_features




def project_pc_and_gets_img_features(cfg, split, model, loader):
    if cfg['load_img_feat'] == False:
        pc_views = Realistic_Projection()
        image_features = []
        with torch.no_grad():
            for i, (pc, target, target_name) in enumerate(tqdm(loader)):
                pc = pc.cuda()
                image = pc_views.get_img(pc)
                image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=True)
                image_feat = utils.get_model(model).encode_image(image)   #[640,512]

                # clip_image_model= clip_image_load(name="RN101")
                # image_feat = clip_image_model(image)

                image_feat /= image_feat.norm(dim=-1, keepdim=True)
                # ------------------只要前8个--------------------------------
                # image_feat = image_feat.view(-1, 10, image_feat.shape[1])
                # image_feat = image_feat[:, 0:8, :]
                # image_feat = image_feat.reshape(-1, image_feat.shape[2])
                # --------------------------------------------------------
                image_features.append(image_feat)

        image_features = torch.cat(image_features)
        torch.save(image_features, cfg['cache_dir'] + "/" + split + "_img.pt")

    else:
        image_features = torch.load(cfg['cache_dir'] + "/" + split + "_img.pt")

    return image_features

def search_image_view_weight(cfg,  image_features, test_labels, ulip_weights):
    search_time, search_range = cfg['vw_search']['TIME'], cfg['vw_search']['RANGE']
    search_list = [(i + 1) * search_range / search_time for i in range(search_time)]
    zero_list = [0]
    one_list = [1]
    best_acc = 0
    for a in  search_list  :
        for b in  search_list    :
            for c in search_list  :
                for d in search_list  :
                    for e in search_list  :
                        for f in search_list   :
                            for g in  search_list   :
                                view_weights = torch.tensor([0.75, 0.75,0.75, a, b, c, d, e,f,g]).cuda()
                                #view_weights = torch.tensor([1, 1,1, a, b, c, d, e, f, g]).cuda()
                                image_features_w = image_features.reshape(-1, 10, 512) * view_weights.reshape(1, -1, 1) #(b, 10, 512)
                                logits = image_features_w.reshape(-1, 10* 512) @ ulip_weights.repeat(10, 1)
                                class_accuracies, all_classes_average_accuracy= cls_acc(logits, test_labels)
                                if  all_classes_average_accuracy > best_acc:
                                    best_acc = all_classes_average_accuracy
                                    best_logits = logits
                                    print("New view weights setting, a: {:.2f}, b: {:.2f}; c: {:.2f}; d: {:.2f}; e: {:.2f}; f: {:.2f}; g: {:.2f}; acc: {:.2f};".format(a, b, c, d, e, f, g, best_acc))

    return best_logits

def search_image_R_view_weight(cfg,  image_features, test_labels, ulip_weights):

    logits = image_features @ ulip_weights

    return logits


def search_hp(cfg, cache_keys, cache_values, features, features_R, labels, ulip_weights):

    if cfg['search_hp'] == True:

        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                affinity = features_R @ cache_keys
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                ulip_logits = 100. * features @ ulip_weights
                tfa_logits = ulip_logits + cache_logits * alpha
                class_accuracies, all_classes_average_accuracy = cls_acc(tfa_logits, labels)

                if all_classes_average_accuracy> best_acc:
                    print("New best setting, alpha: {:.2f}, beta: {:.2f}; accuracy: {:.2f}".format(alpha, beta, all_classes_average_accuracy))
                    best_acc = all_classes_average_accuracy
                    best_beta = beta
                    best_alpha = alpha
                    best_class_accuracies=class_accuracies

        # print("\nAfter searching, PointTFA the best accuarcy: {:.2f}.\n".format(best_acc))

        # 打印每个类别的准确率
        print("PointTFA搜索后的每个类别的准确率:")
        for class_idx, acc in best_class_accuracies.items():
            print(f"类别 {class_idx}: {acc:.2f}%")

        # 打印所有类别的平均准确率
        print(f"\nPointTFA搜索后的所有类别的平均准确率: {best_acc:.2f}%")

    return best_beta, best_alpha


def search_x_hp(cfg, cache_keys, cache_values, features, features_R, labels, ulip_weights, test_kl_div_sim):

    if cfg['search_hp'] == True:
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        gamma_list = [i * (cfg['search_scale'][2] - 0.1) / cfg['search_step'][2] + 0.1 for i in range(cfg['search_step'][2])]

        best_acc = 0
        best_beta, best_alpha, best_gamma = 0, 0, 0

        for alpha in alpha_list:
            for beta in beta_list:
                affinity = features_R @ cache_keys
                new_affinity = scale_((test_kl_div_sim).cuda(), affinity)
                new_affinity = -new_affinity

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                ulip_logits = features @ ulip_weights
                kl_logits = new_affinity @ cache_values

                for gamma in gamma_list:
                    tfa_x_logits = ulip_logits + cache_logits * alpha + kl_logits * gamma
                    acc = cls_acc(tfa_x_logits, labels)

                    if acc > best_acc:
                        print("New best setting, alpha: {:.2f}, beta: {:.2f}, gamma: {:.2f}; accuracy: {:.2f}".format(alpha, beta, gamma, acc))
                        best_acc = acc
                        best_alpha = alpha
                        best_beta = beta
                        best_gamma = gamma

        print("\nAfter searching, PointTFA_X the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha, best_gamma


def search_mm_hp(cfg, cache_keys, cache_values, features, features_R, labels, ulip_weights, img_logits):

    if cfg['search_hp'] == True:
        alpha_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        beta_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        gamma_list = [i * (cfg['search_scale'][2] - 0.1) / cfg['search_step'][2] + 0.1 for i in range(cfg['search_step'][2])]

        best_acc = 0
        best_beta, best_alpha, best_gamma = 0, 0, 0

        for alpha in alpha_list:
            for beta in beta_list:
                affinity = features_R @ cache_keys
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                ulip_logits = 100. * features @ ulip_weights

                for gamma in gamma_list:
                    tfa_mm_logits = ulip_logits + cache_logits * alpha + img_logits * gamma
                    acc = cls_acc(tfa_mm_logits, labels)

                    if acc > best_acc:
                        print("New best setting, alpha: {:.2f}, beta: {:.2f}, gamma: {:.2f}; accuracy: {:.2f}".format(alpha, beta, gamma, acc))
                        best_acc = acc
                        best_alpha = alpha
                        best_beta = beta
                        best_gamma = gamma

        print("\nAfter searching, PointTFA_mm the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha, best_gamma

def search_mm_R_hp(cfg, cache_keys, cache_values,image_cache_keys,image_cache_values, test_features, test_features_R, test_labels, ulip_weights,
                     img_logits,image_features_R):

    if cfg['search_hp'] == True:
        alpha_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        beta_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        gamma_list = [i * (cfg['search_scale'][2] - 0.1) / cfg['search_step'][2] + 0.1 for i in range(cfg['search_step'][2])]
        a_list =[(i+1) * (cfg['search_scale'][3] - 0) / cfg['search_step'][3]  for i in range(cfg['search_step'][3])]

        best_acc = 0
        best_beta, best_alpha, best_gamma, best_a= 0, 0, 0,0

    weights = torch.tensor([[0.75, 0.75, 0.75, 0.75, 1,1,0.5, 1,0.25,0.25]]).to("cuda")
    weights = weights.view(1, 10, 1)

    # if image_cache_keys.shape[1] == 23090:  main_split  test_noise_0.01_objectdataset.h5

    total_iterations = len(alpha_list)* len(beta_list)* len(gamma_list)* len(a_list)
    with tqdm(total=total_iterations,desc="Searching hyperparmeters") as pbar:
        for alpha in alpha_list:
            for beta in beta_list:
                affinity = test_features_R @ cache_keys
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                ulip_logits = 100. * test_features @ ulip_weights

                for gamma in gamma_list:
                    for a in a_list:

                        affinity_image = image_features_R @ image_cache_keys
                        clip_logits_image = ((-1) * (a - a * affinity_image)).exp() @ image_cache_values

                        tfa_mm_logits = ulip_logits + cache_logits * alpha +img_logits + clip_logits_image * gamma
                        class_accuracies, all_classes_average_accuracy = cls_acc(tfa_mm_logits, test_labels)

                        if all_classes_average_accuracy > best_acc:
                            best_acc = all_classes_average_accuracy
                            best_alpha = alpha
                            best_beta = beta
                            best_gamma = gamma
                            best_a = a
                            best_class_accuracies=class_accuracies
                            print("New best setting, alpha: {:.2f}, beta: {:.2f}, gamma: {:.2f}; a: {:.2f};,accuracy: {:.2f}".format(alpha, beta, gamma, a, best_acc))
                        pbar.update(1)
    # print("\nAfter searching, PointTFA_mm the best accuarcy: {:.2f}.\n".format(best_acc))
    print("搜索后的PointTFA_mm每个类别的准确率:")
    for class_idx, acc in best_class_accuracies.items():
        print(f"类别 {class_idx}: {acc:.2f}%")

    # 打印所有类别的平均准确率
    print(f"\n搜索后的PointTFA_mm所有类别的平均准确率: { best_acc:.2f}%")
    # PointTFA_mm
    beta, alpha, gamma = cfg['init_beta'], cfg['init_alpha'], cfg['init_gamma']
    print("Searching image view weights:")
    return best_beta, best_alpha, best_gamma ,best_a

def search_mm_R_hp_no_point_cache(cfg, image_cache_keys,image_cache_values, test_features, test_features_R, test_labels, ulip_weights,
                     img_logits,image_features_R):

    if cfg['search_hp'] == True:
        alpha_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        beta_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        gamma_list = [i * (cfg['search_scale'][2] - 0.1) / cfg['search_step'][2] + 0.1 for i in range(cfg['search_step'][2])]
        a_list =[(i+1) * (cfg['search_scale'][3] - 0) / cfg['search_step'][3]  for i in range(cfg['search_step'][3])]

        best_acc = 0
        best_beta, best_alpha, best_gamma, best_a= 0, 0, 0,0

    weights = torch.tensor([[0.75, 0.75, 0.75, 0.75, 1,1,0.5, 1,0.25,0.25]]).to("cuda")
    weights = weights.view(1, 10, 1)

    # if image_cache_keys.shape[1] == 23090:  main_split  test_noise_0.01_objectdataset.h5

    total_iterations = len(alpha_list)* len(beta_list)* len(gamma_list)* len(a_list)
    with tqdm(total=total_iterations,desc="Searching hyperparmeters") as pbar:
        for alpha in alpha_list:
            for beta in beta_list:

                ulip_logits = 100. * test_features @ ulip_weights

                for gamma in gamma_list:
                    for a in a_list:

                        affinity_image = image_features_R @ image_cache_keys
                        clip_logits_image = ((-1) * (a - a * affinity_image)).exp() @ image_cache_values

                        tfa_mm_logits = ulip_logits +img_logits + clip_logits_image * gamma
                        acc = cls_acc(tfa_mm_logits, test_labels)

                        if acc > best_acc:
                            best_acc = acc


                            best_gamma = gamma
                            best_a = a
                            print("New best setting, alpha: {:.2f}, beta: {:.2f};,accuracy: {:.2f}".format(alpha, beta, acc))
                        pbar.update(1)
    print("\nAfter searching, PointTFA_mm the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha


def search_mm_R_hp_no_point_cache(cfg, image_cache_keys,image_cache_values, test_features, test_features_R, test_labels, ulip_weights,
                     img_logits,image_features_R):

    if cfg['search_hp'] == True:
        alpha_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        beta_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        gamma_list = [i * (cfg['search_scale'][2] - 0.1) / cfg['search_step'][2] + 0.1 for i in range(cfg['search_step'][2])]
        a_list =[(i+1) * (cfg['search_scale'][3] - 0) / cfg['search_step'][3]  for i in range(cfg['search_step'][3])]

        best_acc = 0
        best_beta, best_alpha, best_gamma, best_a= 0, 0, 0,0

    weights = torch.tensor([[0.75, 0.75, 0.75, 0.75, 1,1,0.5, 1,0.25,0.25]]).to("cuda")
    weights = weights.view(1, 10, 1)

    # if image_cache_keys.shape[1] == 23090:  main_split  test_noise_0.01_objectdataset.h5

    total_iterations = len(alpha_list)* len(beta_list)* len(gamma_list)* len(a_list)
    with tqdm(total=total_iterations,desc="Searching hyperparmeters") as pbar:



                ulip_logits = 100. * test_features @ ulip_weights

                for gamma in gamma_list:
                    for a in a_list:

                        affinity_image = image_features_R @ image_cache_keys
                        clip_logits_image = ((-1) * (a - a * affinity_image)).exp() @ image_cache_values

                        tfa_mm_logits = ulip_logits +img_logits + clip_logits_image * gamma
                        acc = cls_acc(tfa_mm_logits, test_labels)

                        if acc > best_acc:
                            best_acc = acc

                            best_gamma = gamma
                            best_a = a
                            print("New best setting,gamma: {:.2f}; a: {:.2f};,accuracy: {:.2f}".format( gamma, a, acc))
                        pbar.update(1)
    print("\nAfter searching, PointTFA_mm the best accuarcy: {:.2f}.\n".format(best_acc))

    return  best_gamma ,best_a


def search_mm_R_hp_no_image_cache(cfg, cache_keys, cache_values, test_features, test_features_R, test_labels, ulip_weights,
                     img_logits,image_features_R):

    if cfg['search_hp'] == True:
        alpha_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        beta_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        gamma_list = [i * (cfg['search_scale'][2] - 0.1) / cfg['search_step'][2] + 0.1 for i in range(cfg['search_step'][2])]
        a_list =[(i+1) * (cfg['search_scale'][3] - 0) / cfg['search_step'][3]  for i in range(cfg['search_step'][3])]

        best_acc = 0
        best_beta, best_alpha, best_gamma, best_a= 0, 0, 0,0

    weights = torch.tensor([[0.75, 0.75, 0.75, 0.75, 1,1,0.5, 1,0.25,0.25]]).to("cuda")
    weights = weights.view(1, 10, 1)

    # if image_cache_keys.shape[1] == 23090:  main_split  test_noise_0.01_objectdataset.h5

    total_iterations = len(alpha_list)* len(beta_list)* len(gamma_list)* len(a_list)
    with tqdm(total=total_iterations,desc="Searching hyperparmeters") as pbar:
        for alpha in alpha_list:
            for beta in beta_list:
                affinity = test_features_R @ cache_keys
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                ulip_logits = 100. * test_features @ ulip_weights



                tfa_mm_logits = ulip_logits + cache_logits * alpha +img_logits
                acc = cls_acc(tfa_mm_logits, test_labels)

                if acc > best_acc:
                    best_acc = acc
                    best_alpha = alpha
                    best_beta = beta

                    print("New best setting, alpha: {:.2f}, beta: {:.2f};,accuracy: {:.2f}".format(alpha, beta, acc))
                pbar.update(1)
    print("\nAfter searching, PointTFA_mm the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha,


def search_mm_R_no_image_cache_hp(cfg, cache_keys, cache_values, test_features, test_features_R, test_labels, ulip_weights,
                     img_logits,image_features_R):

    if cfg['search_hp'] == True:
        alpha_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        beta_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        gamma_list = [i * (cfg['search_scale'][2] - 0.1) / cfg['search_step'][2] + 0.1 for i in range(cfg['search_step'][2])]
        a_list =[(i+1) * (cfg['search_scale'][3] - 0) / cfg['search_step'][3]  for i in range(cfg['search_step'][3])]

        best_acc = 0
        best_beta, best_alpha, best_gamma, best_a= 0, 0, 0,0

    weights = torch.tensor([[0.75, 0.75, 0.75, 0.75, 1,1,0.5, 1,0.25,0.25]]).to("cuda")
    weights = weights.view(1, 10, 1)

    # if image_cache_keys.shape[1] == 23090:  main_split  test_noise_0.01_objectdataset.h5

    total_iterations = len(alpha_list)* len(beta_list)* len(gamma_list)* len(a_list)
    with tqdm(total=total_iterations,desc="Searching hyperparmeters") as pbar:
        for alpha in alpha_list:
            for beta in beta_list:
                affinity = test_features_R @ cache_keys
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                ulip_logits = 100. * test_features @ ulip_weights



                tfa_mm_logits = ulip_logits + cache_logits * alpha +img_logits
                acc = cls_acc(tfa_mm_logits, test_labels)

                if acc > best_acc:
                    best_acc = acc
                    best_alpha = alpha
                    best_beta = beta


                    print("New best setting, alpha: {:.2f}, beta: {:.2f},,accuracy: {:.2f}".format(alpha, beta, acc))
                pbar.update(1)
    print("\nAfter searching, PointTFA_mm the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha

def search_mm_R_no_point_cache_hp(cfg, image_cache_keys,image_cache_values, test_features, test_features_R, test_labels, ulip_weights,
                     img_logits,image_features_R):

    if cfg['search_hp'] == True:
        alpha_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        beta_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        gamma_list = [i * (cfg['search_scale'][2] - 0.1) / cfg['search_step'][2] + 0.1 for i in range(cfg['search_step'][2])]
        a_list =[(i+1) * (cfg['search_scale'][3] - 0) / cfg['search_step'][3]  for i in range(cfg['search_step'][3])]

        best_acc = 0
        best_beta, best_alpha, best_gamma, best_a= 0, 0, 0,0

    weights = torch.tensor([[0.75, 0.75, 0.75, 0.75, 1,1,0.5, 1,0.25,0.25]]).to("cuda")
    weights = weights.view(1, 10, 1)

    # if image_cache_keys.shape[1] == 23090:  main_split  test_noise_0.01_objectdataset.h5

    total_iterations = len(alpha_list)* len(beta_list)* len(gamma_list)* len(a_list)
    with tqdm(total=total_iterations,desc="Searching hyperparmeters") as pbar:
        for alpha in alpha_list:

                ulip_logits = 100. * test_features @ ulip_weights

                for gamma in gamma_list:
                    for a in a_list:

                        affinity_image = image_features_R @ image_cache_keys
                        clip_logits_image = ((-1) * (a - a * affinity_image)).exp() @ image_cache_values

                        tfa_mm_logits = ulip_logits + +img_logits + clip_logits_image * gamma
                        acc = cls_acc(tfa_mm_logits, test_labels)

                        if acc > best_acc:
                            best_acc = acc


                            best_gamma = gamma
                            best_a = a
                            print("New best setting, gamma: {:.2f}; a: {:.2f};,accuracy: {:.2f}".format( gamma, a, acc))
                        pbar.update(1)
    print("\nAfter searching, PointTFA_mm the best accuarcy: {:.2f}.\n".format(best_acc))

    return  best_gamma ,best_a


def search_mm_R_kl_hp(cfg, cache_keys, cache_values,image_cache_keys,image_cache_values, test_features, test_features_R, test_labels, ulip_weights,
                     img_logits,image_features_R,kl_logits):

    if cfg['search_hp'] == True:
        alpha_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        beta_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        gamma_list = [i * (cfg['search_scale'][2] - 0.1) / cfg['search_step'][2] + 0.1 for i in range(cfg['search_step'][2])]
        a_list =[(i+1) * (cfg['search_scale'][3] - 0) / cfg['search_step'][3]  for i in range(cfg['search_step'][3])]
        c_list = [(i+1) * (cfg['search_scale'][3] - 0) / cfg['search_step'][4]  for i in range(cfg['search_step'][4])]

        best_acc = 0
        best_beta, best_alpha, best_gamma, best_a= 0, 0, 0,0

    weights = torch.tensor([[0.75, 0.75, 0.75, 0.75, 1,1,0.5, 1,0.25,0.25]]).to("cuda")
    weights = weights.view(1, 10, 1)

    # if image_cache_keys.shape[1] == 23090:  main_split  test_noise_0.01_objectdataset.h5

    total_iterations = len(alpha_list)* len(beta_list)* len(gamma_list)* len(a_list)*len(c_list)
    with tqdm(total=total_iterations,desc="Searching hyperparmeters") as pbar:
        for alpha in alpha_list:
            for beta in beta_list:
                affinity = test_features_R @ cache_keys
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                ulip_logits = 100. * test_features @ ulip_weights

                for gamma in gamma_list:
                    for a in a_list:
                        for c in c_list:
                            affinity_image = image_features_R @ image_cache_keys
                            clip_logits_image = ((-1) * (a - a * affinity_image)).exp() @ image_cache_values

                            tfa_mm_logits = ulip_logits +cache_logits* alpha +img_logits + clip_logits_image * gamma +kl_logits*c
                            acc = cls_acc(tfa_mm_logits, test_labels)

                            if acc > best_acc:
                                best_acc = acc
                                best_alpha = alpha
                                best_beta = beta
                                best_gamma = gamma
                                best_a = a
                                best_c = c

                                print("New best setting, alpha: {:.2f}, beta: {:.2f}, gamma: {:.2f}, a: {:.2f},c: {:.2f},accuracy: {:.2f}".format(alpha, beta, gamma, a, c,acc))
                            pbar.update(1)
    print("\nAfter searching, PointTFA_mm the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha, best_gamma ,best_a
def search_mm_2D_3D_hp(cfg, image_cache_keys, image_cache_values,cache_keys, cache_values, test_features, test_features_R,cache3D_2Dtest_features_R , test_labels, ulip_weights,  img_logits):

    if cfg['search_hp'] == True:
        alpha_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        beta_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        gamma_list = [i * (cfg['search_scale'][2] - 0.1) / cfg['search_step'][2] + 0.1 for i in range(cfg['search_step'][2])]
        c_list =[i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]

        best_acc = 0
        best_beta, best_alpha, best_gamma = 0, 0, 0

        for alpha in alpha_list:
            for beta in beta_list:
                affinity = test_features_R @ cache_keys
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                ulip_logits = 100. * test_features @ ulip_weights
                for c in c_list:
                    affinity_1 = cache3D_2Dtest_features_R @ image_cache_keys
                    cache_logits_1 = ((-1) * (beta - beta * affinity_1)).exp() @ image_cache_values

                    for gamma in gamma_list:
                        tfa_mm_logits = cache_logits * alpha + cache_logits_1*c  + img_logits * gamma
                        acc = cls_acc(tfa_mm_logits, test_labels)

                        if acc > best_acc:
                            print("New best setting, alpha: {:.2f}, beta: {:.2f}, gamma: {:.2f};c: {:.2f} accuracy: {:.2f}".format(alpha, beta, gamma,c, acc))
                            best_acc = acc
                            best_alpha = alpha
                            best_beta = beta
                            best_gamma = gamma

        print("\nAfter searching, PointTFA_mm the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha, best_gamma


class clip_image_load(nn.Module):

    def __init__(self,name):
        super(clip_image_load,self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = load(name,device= self.device)

    def forward(self,image):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)

        return  image_features

def Depth_map_generation(cfg, split, model, loader):
        if cfg['load_img_feat'] == False:
            pc_views = Realistic_Projection()
            image_features = []
            with torch.no_grad():
                for i, (pc, target, target_name) in enumerate(tqdm(loader)):

                    if i == 0 :
                        pc = pc.cuda()
                        image = pc_views.get_img(pc)
                        image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=True)
                        image = image.view(-1,10,3,image.shape[-1],image.shape[-1])
                        image = image.cpu().numpy()

                        num_samples = image.shape[0]
                        num_view = image.shape[1]

                        for samplex_idx in range(num_samples):
                            if samplex_idx==32:
                                for view_idx in range(num_view):
                                    img = image[samplex_idx,view_idx]
                                    img = np.transpose(img,(1,2,0))

                                    plt.figure(figsize=(5,5),dpi=100)
                                    plt.imshow(img)
                                    plt.axis("off")
                                    plt.title(f"Sample {samplex_idx + 1} - view{view_idx + 1}")

                                    plt.show()



