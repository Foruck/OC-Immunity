import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from collections import OrderedDict
import torchvision.models as models

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)

acti = {
    'ReLU': nn.ReLU,
    'GELU': nn.GELU,
    'Swish': Swish,
}

class human_classifier(nn.Module):
    def __init__(self, config, loss_weight):
    
        super(human_classifier, self).__init__()
        
        if config.ENCODER.RESNET:
            self.encoder         = models.resnet18()
            self.encoder.fc      = nn.Identity()
        else:
            self.encoder = OrderedDict()
            for i in range(1, len(config.ENCODER.LAYER_SIZE)):
                self.encoder['conv%d' % i] = nn.Conv2d(config.ENCODER.LAYER_SIZE[i - 1], config.ENCODER.LAYER_SIZE[i], config.ENCODER.KERNEL_SIZE[i], 1, config.ENCODER.PADDING[i])
                if config.ENCODER.BN:
                    self.encoder['bn%d' % i] = nn.BatchNorm2d(config.ENCODER.LAYER_SIZE[i])
                self.encoder['act%d' % i]    = acti[config.ENCODER.ACT]()
                self.encoder['pool%d' % i]   = nn.MaxPool2d(2)
            self.encoder['avgpool'] = nn.AdaptiveAvgPool2d((1, 1))
            self.encoder['flatten'] = nn.Flatten()
            self.encoder = nn.Sequential(self.encoder)

        self.key  = config.KEY
        
        self.mean = OrderedDict()
        for i in range(1, len(config.LAYER_SIZE)):
            self.mean['fc%d' % i] = nn.Linear(config.LAYER_SIZE[i - 1], config.LAYER_SIZE[i])
            if config.BN:
                self.mean['bn%d' % i] = nn.BatchNorm1d(config.LAYER_SIZE[i])
            self.mean['act%d' % i] = acti[config.ACT]()
        self.mean['fc_out'] = nn.Linear(config.LAYER_SIZE[-1], config.NUM_CLASSES)
        self.mean = nn.Sequential(self.mean)
        
        self.logvar = OrderedDict()
        for i in range(1, len(config.LAYER_SIZE)):
            self.logvar['fc%d' % i] = nn.Linear(config.LAYER_SIZE[i - 1], config.LAYER_SIZE[i])
            if config.BN:
                self.logvar['bn%d' % i] = nn.BatchNorm1d(config.LAYER_SIZE[i])
            self.logvar['act%d' % i] = acti[config.ACT]()
        self.logvar['fc_out']  = nn.Linear(config.LAYER_SIZE[-1], config.NUM_CLASSES)
        self.logvar['act_out'] = nn.Tanh()
        self.logvar = nn.Sequential(self.logvar)
        
        self.cls_loss  = nn.BCEWithLogitsLoss(pos_weight=loss_weight[self.key[1]])
        self.unsup_fac = config.UNSUP_FAC
        if config.CHECKPOINT:
            self.load_state_dict(torch.load(config.CHECKPOINT)['state'])
            
        self.predictions = {}

    def forward(self, batch):
        x      = batch[self.key[0]]
        z      = self.encoder(x) # fc7_P
        # z      = torch.mean(z, dim=[2, 3])
        mean   = self.mean(z)
        logvar = self.logvar(z)
        
        if self.training:
            self.predictions[self.key[1]] = batch[self.key[1]]
            self.predictions['mean']      = mean
            self.predictions['logvar']    = logvar
        output = {
            'z': z,
            'mean': mean,
            'logvar': logvar,
            self.key[1]: batch[self.key[1]],
        }
        
        return output
    
    def add_loss(self, ref=None):
        mean   = self.predictions['mean']
        logvar = self.predictions['logvar']
        label  = self.predictions[self.key[1]]
        if ref is not None:
            prob      = torch.sigmoid(mean * torch.exp(-logvar))
            prob_rev  = 1. - prob
            prob_ref  = torch.sigmoid(ref['mean'] * torch.exp(-ref['logvar']))
            label_ref = ref[self.key[1]]
            
            e_thresh  = torch.mean(ref['logvar'], dim=0, keepdim=True)
            p_pos1    = torch.mean(prob_ref * label_ref, dim=0, keepdim=True)
            p_pos2    = torch.max(prob_ref * (1 - label_ref), dim=0, keepdim=True)[0]
            p_pos     = torch.max(p_pos1, p_pos2)
            p_neg1    = torch.mean(prob_ref * (1 - label_ref), dim=0, keepdim=True)
            p_neg2    = torch.min(prob_ref * label_ref, dim=0, keepdim=True)[0]
            p_neg     = torch.min(p_neg1, p_neg2)
            p_mid     = (p_pos + p_neg) / 2
            
            mask_fp  = torch.logical_and(prob > p_pos, logvar > e_thresh)
            tmp_fp   = prob * (1 - p_mid) + prob_rev * p_mid
            if torch.sum(mask_fp) > 0:
                loss_fp  = torch.mean(tmp_fp[mask_fp]) - torch.mean(logvar[mask_fp])
            else:
                loss_fp  = 0.
            
            mask_tp  = torch.logical_and(prob > p_pos, logvar < e_thresh)
            if torch.sum(mask_tp) > 0:
                loss_tp  = torch.mean(prob_rev[mask_tp])
            else:
                loss_tp  = 0.
            
            mask_fn  = torch.logical_and(prob < p_neg, logvar > e_thresh)
            tmp_fn   = prob * (1 - p_mid) + prob_rev * p_mid
            if torch.sum(mask_fn) > 0:
                loss_fn  = torch.mean(tmp_fn[mask_fn]) - torch.mean(logvar[mask_fn])
            else:
                loss_fn  = 0.
            
            mask_tn  = torch.logical_and(prob < p_neg, logvar < e_thresh)
            if torch.sum(mask_tn) > 0:
                loss_tn  = torch.mean(prob[mask_tn])
            else:
                loss_tn  = 0.
            
            loss     = (loss_fp + loss_tp + loss_fn + loss_tn) * self.unsup_fac
        else:
            loss_cls = self.cls_loss(torch.exp(-logvar) * mean, label)
            loss_var = torch.mean(logvar)
            loss     = loss_cls + loss_var
        return loss

class object_classifier(nn.Module):
    def __init__(self, config, loss_weight):
        super(object_classifier, self).__init__()
        self.key  = config.KEY
        
        self.synth = OrderedDict()
        for i in range(1, len(config.LAYER_SIZE)):
            self.synth['fc%d' % i] = nn.Linear(config.SYN.LAYER_SIZE[i - 1], config.SYN.LAYER_SIZE[i])
            if config.BN:
                self.synth['bn%d' % i] = nn.BatchNorm1d(config.SYN.LAYER_SIZE[i])
            self.synth['act%d' % i] = acti[config.ACT]()
        self.synth['fc_out'] = nn.Linear(config.SYN.LAYER_SIZE[-1], config.LAYER_SIZE[0])
        self.synth = nn.Sequential(self.synth)
        
        self.mean = OrderedDict()
        for i in range(1, len(config.LAYER_SIZE)):
            self.mean['fc%d' % i] = nn.Linear(config.LAYER_SIZE[i - 1], config.LAYER_SIZE[i])
            if config.BN:
                self.mean['bn%d' % i] = nn.BatchNorm1d(config.LAYER_SIZE[i])
            self.mean['act%d' % i] = acti[config.ACT]()
        self.mean['fc_out'] = nn.Linear(config.LAYER_SIZE[-1], config.NUM_CLASSES)
        self.mean = nn.Sequential(self.mean)
        
        self.logvar = OrderedDict()
        for i in range(1, len(config.LAYER_SIZE)):
            self.logvar['fc%d' % i] = nn.Linear(config.LAYER_SIZE[i - 1], config.LAYER_SIZE[i])
            if config.BN:
                self.logvar['bn%d' % i] = nn.BatchNorm1d(config.LAYER_SIZE[i])
            self.logvar['act%d' % i] = acti[config.ACT]()
        self.logvar['fc_out']  = nn.Linear(config.LAYER_SIZE[-1], config.NUM_CLASSES)
        self.logvar['act_out'] = nn.Tanh()
        self.logvar = nn.Sequential(self.logvar)
        
        self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=loss_weight[self.key[1]])
        self.unsup_fac = config.UNSUP_FAC
        if config.CHECKPOINT:
            self.load_state_dict(torch.load(config.CHECKPOINT)['state'])
        self.predictions = {}

    def forward(self, batch):
        z       = batch[self.key[0]]
        s       = self.synth(z)
        mean    = self.mean(z)
        logvar  = self.logvar(z)
        
        if self.training:
            self.predictions[self.key[1]] = batch[self.key[1]]
            self.predictions['mean']      = mean
            self.predictions['logvar']    = logvar
        output = {
            'mean': mean,
            'logvar': logvar,
            self.key[1]: batch[self.key[1]],
        }
        
        return output
    
    def add_loss(self, ref=None):
        mean   = self.predictions['mean']
        logvar = self.predictions['logvar']
        label  = self.predictions[self.key[1]]
        if ref is not None:
            prob      = torch.sigmoid(mean * torch.exp(-logvar))
            prob_rev  = 1. - prob
            prob_ref  = torch.sigmoid(ref['mean'] * torch.exp(-ref['logvar']))
            label_ref = ref[self.key[1]]
            
            e_thresh  = torch.mean(ref['logvar'], dim=0, keepdim=True)
            p_pos1    = torch.mean(prob_ref * label_ref, dim=0, keepdim=True)
            p_pos2    = torch.max(prob_ref * (1 - label_ref), dim=0, keepdim=True)[0]
            p_pos     = torch.max(p_pos1, p_pos2)
            p_neg1    = torch.mean(prob_ref * (1 - label_ref), dim=0, keepdim=True)
            p_neg2    = torch.min(prob_ref * label_ref, dim=0, keepdim=True)[0]
            p_neg     = torch.min(p_neg1, p_neg2)
            p_mid     = (p_pos + p_neg) / 2
            
            mask_fp  = torch.logical_and(prob > p_pos, logvar > e_thresh)
            tmp_fp   = prob * (1 - p_mid) + prob_rev * p_mid
            if torch.sum(mask_fp) > 0:
                loss_fp  = torch.mean(tmp_fp[mask_fp]) - torch.mean(logvar[mask_fp])
            else:
                loss_fp  = 0.
            
            mask_tp  = torch.logical_and(prob > p_pos, logvar < e_thresh)
            if torch.sum(mask_tp) > 0:
                loss_tp  = torch.mean(prob_rev[mask_tp])
            else:
                loss_tp  = 0.
            
            mask_fn  = torch.logical_and(prob < p_neg, logvar > e_thresh)
            tmp_fn   = prob * (1 - p_mid) + prob_rev * p_mid
            if torch.sum(mask_fn) > 0:
                loss_fn  = torch.mean(tmp_fn[mask_fn]) - torch.mean(logvar[mask_fn])
            else:
                loss_fn  = 0.
            
            mask_tn  = torch.logical_and(prob < p_neg, logvar < e_thresh)
            if torch.sum(mask_tn) > 0:
                loss_tn  = torch.mean(prob[mask_tn])
            else:
                loss_tn  = 0.
            
            loss     = (loss_fp + loss_tp + loss_fn + loss_tn) * self.unsup_fac
        else:
            loss_cls = self.cls_loss(torch.exp(-logvar) * mean, label)
            loss_var = torch.mean(logvar)
            loss     = loss_cls + loss_var
        return loss

class spatial_classifier(nn.Module):
    def __init__(self, config, loss_weight):
        super(spatial_classifier, self).__init__()
        self.key  = config.KEY
        self.mean = OrderedDict()
        for i in range(1, len(config.LAYER_SIZE)):
            self.mean['fc%d' % i] = nn.Linear(config.LAYER_SIZE[i - 1], config.LAYER_SIZE[i])
            if config.BN:
                self.mean['bn%d' % i] = nn.BatchNorm1d(config.LAYER_SIZE[i])
            self.mean['act%d' % i] = acti[config.ACT]()
        self.mean['fc_out'] = nn.Linear(config.LAYER_SIZE[-1], config.NUM_CLASSES)
        self.mean = nn.Sequential(self.mean)
        
        self.logvar = OrderedDict()
        for i in range(1, len(config.LAYER_SIZE)):
            self.logvar['fc%d' % i] = nn.Linear(config.LAYER_SIZE[i - 1], config.LAYER_SIZE[i])
            if config.BN:
                self.logvar['bn%d' % i] = nn.BatchNorm1d(config.LAYER_SIZE[i])
            self.logvar['act%d' % i] = acti[config.ACT]()
        self.logvar['fc_out']  = nn.Linear(config.LAYER_SIZE[-1], config.NUM_CLASSES)
        self.logvar['act_out'] = nn.Tanh()
        self.logvar = nn.Sequential(self.logvar)
        
        self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=loss_weight[self.key[1]])
        self.unsup_fac = config.UNSUP_FAC
        if config.CHECKPOINT:
            self.load_state_dict(torch.load(config.CHECKPOINT)['state'])
        self.predictions = {}

    def forward(self, batch):
        z       = batch[self.key[0]]
        mean    = self.mean(z)
        logvar  = self.logvar(z)
        
        if self.training:
            self.predictions[self.key[1]] = batch[self.key[1]]
            self.predictions['mean']      = mean
            self.predictions['logvar']    = logvar
        output = {
            'mean': mean,
            'logvar': logvar,
            self.key[1]: batch[self.key[1]],
        }
        
        return output
    
    def add_loss(self, ref=None):
        mean   = self.predictions['mean']
        logvar = self.predictions['logvar']
        label  = self.predictions[self.key[1]]
        if ref is not None:
            prob      = torch.sigmoid(mean * torch.exp(-logvar))
            prob_rev  = 1. - prob
            prob_ref  = torch.sigmoid(ref['mean'] * torch.exp(-ref['logvar']))
            label_ref = ref[self.key[1]]
            
            e_thresh  = torch.mean(ref['logvar'], dim=0, keepdim=True)
            p_pos1    = torch.mean(prob_ref * label_ref, dim=0, keepdim=True)
            p_pos2    = torch.max(prob_ref * (1 - label_ref), dim=0, keepdim=True)[0]
            p_pos     = torch.max(p_pos1, p_pos2)
            p_neg1    = torch.mean(prob_ref * (1 - label_ref), dim=0, keepdim=True)
            p_neg2    = torch.min(prob_ref * label_ref, dim=0, keepdim=True)[0]
            p_neg     = torch.min(p_neg1, p_neg2)
            p_mid     = (p_pos + p_neg) / 2
            
            mask_fp  = torch.logical_and(prob > p_pos, logvar > e_thresh)
            tmp_fp   = prob * (1 - p_mid) + prob_rev * p_mid
            if torch.sum(mask_fp) > 0:
                loss_fp  = torch.mean(tmp_fp[mask_fp]) - torch.mean(logvar[mask_fp])
            else:
                loss_fp  = 0.
            
            mask_tp  = torch.logical_and(prob > p_pos, logvar < e_thresh)
            if torch.sum(mask_tp) > 0:
                loss_tp  = torch.mean(prob_rev[mask_tp])
            else:
                loss_tp  = 0.
            
            mask_fn  = torch.logical_and(prob < p_neg, logvar > e_thresh)
            tmp_fn   = prob * (1 - p_mid) + prob_rev * p_mid
            if torch.sum(mask_fn) > 0:
                loss_fn  = torch.mean(tmp_fn[mask_fn]) - torch.mean(logvar[mask_fn])
            else:
                loss_fn  = 0.
            
            mask_tn  = torch.logical_and(prob < p_neg, logvar < e_thresh)
            if torch.sum(mask_tn) > 0:
                loss_tn  = torch.mean(prob[mask_tn])
            else:
                loss_tn  = 0.
            
            loss     = (loss_fp + loss_tp + loss_fn + loss_tn) * self.unsup_fac
        else:
            loss_cls = self.cls_loss(torch.exp(-logvar) * mean, label)
            loss_var = torch.mean(logvar)
            loss     = loss_cls + loss_var
        return loss

class fusion_module(nn.Module):
    def __init__(self, config):
    
        super(fusion_module, self).__init__()
        
        self.wh  = nn.Parameter(torch.Tensor(1, config.NUM_CLASSES))
        self.wo  = nn.Parameter(torch.Tensor(1, config.NUM_CLASSES))
        self.wsp = nn.Parameter(torch.Tensor(1, config.NUM_CLASSES))

        self.fac = nn.Parameter(torch.Tensor(3, 117))
        self.reset_parameters()
    
    def reset_parameters(self):
        with torch.no_grad():
            self.fac[:] = 0.
            self.wh[:]  = 0.
            self.wo[:]  = 0.
            self.wsp[:] = 0.
    
    def forward(self, batch):
        sH   = batch['sH'] * torch.exp(self.wh - batch['eH'])
        sO   = batch['sO'] * torch.exp(self.wh - batch['eO'])
        ssp  = batch['ssp'] * torch.exp(self.wsp - batch['esp'])
        fac  = torch.softmax(self.fac, dim=0)
        pALL = fac[0, :] * torch.sigmoid(sH) + fac[1, :] * torch.sigmoid(sO) + fac[2, :] * torch.sigmoid(ssp)
        return {
            'sH': sH,
            'sO': sO,
            'ssp': ssp,
            'pALL': pALL
        }

class unified_classifier(nn.Module):
    def __init__(self, config, loss_weight):
    
        super(unified_classifier, self).__init__()
        
        self.human   = human_classifier(config.HUMAN, loss_weight)
        self.object  = object_classifier(config.OBJECT, loss_weight)
        self.spatial = spatial_classifier(config.SPATIAL, loss_weight)
        self.fusion  = fusion_module(config.FUSION)
        
        self.key  = config.KEY
        self.cls_loss  = nn.MSELoss()
        if config.CHECKPOINT:
            self.load_state_dict(torch.load(config.CHECKPOINT)['state'])
        self.CONSIST_fac = config.CONSIST_FAC
        self.predictions = {}

    def train(self, mode=True):
        self.training = True
        self.human.eval()
        self.object.eval()
        self.spatial.eval()
        self.fusion.train()
    
    def forward(self, batch):
        out_H  = self.human(batch)
        out_O  = self.object(batch)
        out_sp = self.spatial(batch)
        aggr   = {
            'sH': out_H['mean'], 'eH': out_H['logvar'], 
            'sO': out_O['mean'], 'eO': out_O['logvar'], 
            'ssp': out_sp['mean'], 'esp': out_sp['logvar'], 
        }
        output = self.fusion(aggr)
        if self.training:
            self.predictions = output
            self.predictions[self.key[0]] = batch[self.key[0]]
            self.predictions[self.key[1]] = batch[self.key[1]]
            self.predictions[self.key[2]] = batch[self.key[2]]
            self.predictions[self.key[2]] = batch[self.key[2]]
        return output
            
    def add_loss(self):
        batch = self.predictions
        loss_1 = self.cls_loss(batch['pALL'], batch[self.key[2]]) + \
            self.cls_loss(torch.sigmoid(batch['sH']), batch[self.key[0]]) + \
            self.cls_loss(torch.sigmoid(batch['sO']), batch[self.key[1]]) + \
            self.cls_loss(torch.sigmoid(batch['ssp']), batch[self.key[2]])
        loss_2 = self.CONSIST_fac * (
            torch.abs(torch.mean(torch.sigmoid(batch['sH'])  - torch.sigmoid(batch['sO']))) + \
            torch.abs(torch.mean(torch.sigmoid(batch['sO'])  - torch.sigmoid(batch['ssp']))) + \
            torch.abs(torch.mean(torch.sigmoid(batch['ssp']) - torch.sigmoid(batch['sH'])))
        )
        return loss_1 + loss_2