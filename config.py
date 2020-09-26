# --------------------------------------------------------
# MOCT Classfication Config File
# Ruibing 2020.07.29
# --------------------------------------------------------

import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()


config.arch = 'resnet18'
config.workers = 4
config.epochs = 90
config.start_epoch = 0
config.batch_size = 256
config.lr = 0.1
config.lr_epoch = None
config.lr_inter = 30
config.lr_factor = 0.1
config.warmup = False
config.warmup_lr = 0.1
config.warmup_epoch = 20
config.momentum = 0.9
config.weight_decay = 1e-4
config.print_freq = 10
config.resume = ''
config.evaluate = False
config.pretrained = False
config.world_size = -1
config.rank = -1
config.dist_url = 'tcp://224.66.41.62:23456'
config.dist_backend = 'nccl'
config.seed = None
config.gpu = None
config.multiprocessing_distributed = False
config.distributed = False

config.image_sets = ''
config.num_cls = 1
config.vis = False





def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    if k == 'network':
                        if 'PIXEL_MEANS' in v:
                            v['PIXEL_MEANS'] = np.array(v['PIXEL_MEANS'])
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    if v == 'None':
                        config[k] = None
                    elif k == 'lr_epoch':
                        step_list = [int(x) for x in v.split(',')]
                        config[k] = step_list
                    else:
                        config[k] = v
            else:
                raise ValueError("key must exist in config.py")
