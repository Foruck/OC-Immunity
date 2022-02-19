import torch
import os
import re
import time
import yaml
import copy
import numpy as np
import argparse
import pickle, json
import logging, tqdm
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from prefetch_generator import BackgroundGenerator
from dataset import hico_train_set, element_train_set
from model import unified_classifier, human_classifier, object_classifier, spatial_classifier
from utils import Timer, AverageMeter, loss_weight, restruct_pose


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

models = {
    'human_classifier':   human_classifier,
    'object_classifier':  object_classifier, 
    'spatial_classifier': spatial_classifier, 
    'human_unsup':   human_classifier,
    'object_unsup':  object_classifier, 
    'spatial_unsup': spatial_classifier, 
    'unified_classifier': unified_classifier,
}

optims = {}
optims['RMSprop'] = optim.RMSprop
optims['SGD']     = optim.SGD

gpus = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
device = torch.device('cuda:{}'.format(gpus[0]))
verb_mapping = pickle.load(open('verb_mapping.pkl', 'rb'), encoding='latin1')

def parse_arg():
    parser = argparse.ArgumentParser(description='Generate detection file')
    parser.add_argument('--exp', dest='exp',
            help='Define exp name',
            default='_'.join(time.asctime(time.localtime(time.time())).split()), type=str)
    parser.add_argument('--config_path', dest='config_path',
            help='Select config file',
            default='configs/default.yml', type=str)
    args = parser.parse_args()
    return args

def get_config(args):
    loader = yaml.FullLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    
    config = edict(yaml.load(open(args.config_path, 'r'), Loader=loader))
    return config
    
def get_logger(cur_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)

    handler = logging.FileHandler(os.path.join(cur_path, "log.txt"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    writer = SummaryWriter(os.path.join(cur_path, 'tb'))
    
    return logger, writer

args = parse_arg()

cur_path = os.path.join(os.getcwd(), 'exp', args.exp)
assert not os.path.exists(cur_path), 'Duplicate exp name'
os.mkdir(cur_path)

config = get_config(args)
yaml.dump(dict(config), open(os.path.join(cur_path, 'config.yml'), 'w'))

logger, writer = get_logger(cur_path)
logger.info("Start print log")

net = models[config.MODE](config.MODEL, loss_weight)
logger.info(net)
if len(gpus) > 1:
    net = torch.nn.DataParallel(net.to(device), device_ids=gpus, output_device=gpus[0])
else:
    net = net.to(device)

cur_epoch, step = 0, 0
if config.TRAIN.OPTIMIZER.GROUP == 'FUSION':
    optimizer = optims[config.TRAIN.OPTIMIZER.TYPE](net.fusion.parameters(), lr=config.TRAIN.OPTIMIZER.lr, momentum=config.TRAIN.OPTIMIZER.momentum, weight_decay=config.TRAIN.OPTIMIZER.weight_decay)
else:
    optimizer = optims[config.TRAIN.OPTIMIZER.TYPE](net.parameters(), lr=config.TRAIN.OPTIMIZER.lr, momentum=config.TRAIN.OPTIMIZER.momentum, weight_decay=config.TRAIN.OPTIMIZER.weight_decay)

db_train   = pickle.load(open(os.path.join(config.DATA_DIR[0], 'db_trainval.pkl'), 'rb'))
db_test    = pickle.load(open(os.path.join(config.DATA_DIR[0], 'db_test.pkl'), 'rb'))
train_set, unsup_set = None, None

if 'human' in config.MODE:
    entry      = pickle.load(open(os.path.join(config.DATA_DIR[0], 'human_train.pkl'), 'rb'))
    train_set  = element_train_set(config, config.DATA_DIR[0], 'trainval', db_train, entry)
elif 'object' in config.MODE:
    entry      = pickle.load(open(os.path.join(config.DATA_DIR[0], 'object_train.pkl'), 'rb'))
    train_set  = element_train_set(config, config.DATA_DIR[0], 'trainval', db_train, entry)
elif 'spatial' in config.MODE:
    cand_pos   = pickle.load(open(os.path.join(config.DATA_DIR[0], 'cand_positives_trainval.pkl'), 'rb'))
    cand_neg   = pickle.load(open(os.path.join(config.DATA_DIR[0], 'cand_negatives_trainval.pkl'), 'rb'))
    pose_train = restruct_pose(json.load(open(os.path.join(config.DATA_DIR[0], 'train_pose_result.json'))))
    train_set  = hico_train_set(config, config.DATA_DIR[0], 'trainval', db_train, cand_pos, cand_neg, pose_train)
elif 'unified' in config.MODE:
    cand_val   = pickle.load(open(os.path.join(config.DATA_DIR[0], 'cand_val_sel.pkl'), 'rb'))
    pose_val   = restruct_pose(json.load(open(os.path.join(config.DATA_DIR[0], 'train_pose_result.json'))))
    train_set  = hico_test_set(config, config.DATA_DIR[0], 'trainval', db_train, cand_val, pose_val)

train_loader = DataLoaderX(train_set, batch_size=config.TRAIN.DATASET.BATCH_SIZE[0], shuffle=True, num_workers=config.TRAIN.DATASET.NUM_WORKERS, collate_fn=train_set.collate_fn, pin_memory=False, drop_last=False)

if 'unsup' in config.MODE:
    db_unsup    = pickle.load(open(os.path.join(config.DATA_DIR[1], 'db_openimage.pkl'), 'rb'))
    if 'human' in config.MODE:
        entry_unsup = pickle.load(open(os.path.join(config.DATA_DIR[1], 'human_open.pkl'), 'rb'))
        unsup_set   = element_train_set(config, config.DATA_DIR[1], '', db_unsup, entry_unsup)
    elif 'object' in config.MODE:
        entry_unsup = pickle.load(open(os.path.join(config.DATA_DIR[1], 'object_open.pkl'), 'rb'))
        unsup_set   = element_train_set(config, config.DATA_DIR[1], '', db_unsup, entry_unsup)
    elif 'spatial' in config.MODE:
        cand_unsup  = pickle.load(open(os.path.join(config.DATA_DIR[1], 'cand_openimage.pkl'), 'rb'))
        pose_unsup  = restruct_pose(json.load(open(os.path.join(config.DATA_DIR[1], 'pose_openimage.json'))))
        unsup_set   = hico_test_set(config, config.DATA_DIR[1], '', db_unsup, cand_unsup, pose_unsup)
    unsup_loader    = DataLoaderX(unsup_set, batch_size=config.TRAIN.DATASET.BATCH_SIZE[1], shuffle=True, num_workers=config.TRAIN.DATASET.NUM_WORKERS, collate_fn=unsup_set.collate_fn, pin_memory=False, drop_last=False)

logger.info("Data set loaded")

def train(net, loader, optimizer, epoch):
    net.train()
    global step

    meters = {
        'loss': AverageMeter()
    }
    for i, batch in tqdm.tqdm(enumerate(loader)):
        for key in config.F_keys:
            batch[key] = batch[key].cuda(non_blocking=True)
            n = batch[key].shape[0]
        for key in config.L_keys:
            batch[key] = batch[key].cuda(non_blocking=True)
        output = net(batch)
        loss   = net.add_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        meters['loss'].update(loss.detach().cpu().data, n)
        if i % 400 == 0:
            writer.add_scalar('loss', loss.detach().cpu().data, step)
        step += 1
    
    return net, meters
    
def unsup(net, train_loader, unsup_loader, optimizer, epoch):
    net.train()
    global step

    meters = {
        'loss': AverageMeter()
    }
    for i, (train_batch, unsup_batch) in tqdm.tqdm(enumerate(zip(train_loader, unsup_loader))):
        for key in config.F_keys:
            train_batch[key] = train_batch[key].cuda(non_blocking=True)
            unsup_batch[key] = unsup_batch[key].cuda(non_blocking=True)
            n = unsup_batch[key].shape[0]
        for key in config.L_keys:
            train_batch[key] = train_batch[key].cuda(non_blocking=True)
            unsup_batch[key] = unsup_batch[key].cuda(non_blocking=True)
        train_output = net(train_batch)
        loss   = net.add_loss()
        unsup_output = net(unsup_batch)
        loss   += net.add_loss(train_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        meters['loss'].update(loss.detach().cpu().data, n)
        if i % 400 == 0:
            writer.add_scalar('loss', loss.detach().cpu().data, step)
        step += 1
    
    return net, meters

for i in range(config.TRAIN.MAX_EPOCH):
    if 'unsup' in config.MODE:
        net, train_meters = unsup(net, train_loader, unsup_loader, optimizer, i)
    else:
        net, train_meters = train(net, train_loader, optimizer, i)
    train_str = "%03d epoch training" % i

    for (key, value) in train_meters.items():
        train_str += ", %s=%.4f" % (key, value.avg)
    logger.info(train_str)

    try:
        state = {
            'state': net.state_dict(),
            'optim_state': optimizer.state_dict(),
        }
    except:
        state = {
            'state': net.state_dict(),
            'optim_state': optimizer.state_dict(),
        }
    torch.save(state, os.path.join(cur_path, 'epoch_%d.pth' % i))