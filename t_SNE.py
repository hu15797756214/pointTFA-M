import argparse
from collections import OrderedDict

import torch

from datasets.modelnet40 import *

from utils.utils import get_dataset
import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
from utils_tfa import *

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

    # modelnet40 dataset
    random.seed(1)
    torch.manual_seed(1)

    tokenizer = SimpleTokenizer()

    print("Preparing modelnet40 dataset.")
    train_dataset = get_dataset(None, tokenizer, args, 'train')
    test_dataset = get_dataset(None, tokenizer, args, 'val')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=10, shuffle=False,
                                              pin_memory=True, sampler=None, drop_last=False)
    train_loader_cache = torch.utils.data.DataLoader(train_dataset, batch_size=64, num_workers=10, shuffle=False,
                                                     pin_memory=True, sampler=None, drop_last=False)

    # Construct the cache model by training set
    print("\nConstructing cache model by training set visual features and labels.")
    cache_keys, cache_values = build_pc_cache_model(cfg, model, train_loader_cache)

    # Data-efficiency
    print("\ncache model with Data-efficiency.")
    cache_keys_RMC, cache_values_RMC = data_efficiency(cfg, cache_keys, cache_values, train_loader_cache)
    cache_keys_radom = fewshot_random(cfg, cache_keys, cache_values, train_loader_cache)
    cache_keys_RMC, cache_keys_radom = cache_keys_RMC.permute(1,0), cache_keys_radom.permute(1,0)
    label = [i for i in range(40) for _ in range(16)]
    label = torch.tensor(label)

    # t-SNE
    cache_keys_RMC, cache_keys_radom, cache_values_RMC = cache_keys_RMC.to('cpu'), cache_keys_radom.to('cpu'), label
    cache_keys_RMC, cache_keys_radom, cache_values_RMC = cache_keys_RMC.numpy(), cache_keys_radom.numpy(), cache_values_RMC.numpy()
    tsne = TSNE(n_components=2, perplexity=4, random_state=3, learning_rate='auto')
    embedded_test_features = tsne.fit_transform(cache_keys_radom)
    embedded_test_features_R = tsne.fit_transform(cache_keys_RMC)

    plt.figure(figsize=(12,6), dpi=1080)

    plt.subplot(1,2,1)
    for i in range(40):
        indices = cache_values_RMC == i
        plt.scatter(embedded_test_features[indices,0], embedded_test_features[indices,1], s = 6, label=f'Class {i}')
    plt.title('Random few-shot features', fontsize = 20)
    plt.axis('off')
    #plt.legend()

    plt.subplot(1,2,2)
    for i in range(40):
        indices = cache_values_RMC ==i
        plt.scatter(embedded_test_features_R[indices,0], embedded_test_features_R[indices,1], s = 6, label=f'Class {i}')
    plt.title('RMC few-shot features', fontsize = 20)
    plt.axis('off')

    plt.savefig('figure', dpi=1080)

if __name__ == '__main__':
    main()
