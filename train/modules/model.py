import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

def append_params(params, module, prefix):
    for child in module.children():
        for k,p in child._parameters.items():
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError('Duplicated param name: {:s}'.format(name))


def set_optimizer(model, lr_base, lr_mult, train_all=False, momentum=0.9, w_decay=0.0005):
    if train_all:
        params = model.get_all_params()
    else:
        params = model.get_learnable_params()
    # print ('----learn---- = ',params)
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    return optimizer

class Channel_attention_net(nn.Module):

    def __init__(self, channel=16, reduction=4):
        super(Channel_attention_net, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
                ('encoder1', nn.Sequential(nn.Linear(channel, channel//2,bias=True),
                                        nn.ReLU(inplace=False))),
                ('encoder2', nn.Sequential(nn.Linear(channel//2, channel//4,bias=True)))]))

        self.decoder = nn.Sequential(OrderedDict([
                ('decoder1', nn.Sequential(nn.Linear(channel//4, channel//2,bias=True),
                                        nn.ReLU(inplace=False))),
                ('decoder2', nn.Sequential(nn.Linear(channel//2, channel,bias=True)))]))
        self.soft = nn.ModuleList([nn.Sequential(nn.Softmax(dim=-1))])
    def forward(self, x):  # return 16 bands point-mul result
        b, c, w, h = x.size()# [1,16,127,127]
        c1 = x.contiguous().view(b,c,-1)
        c2 = c1.permute(0,2,1)
        for name, module in self.encoder.named_children():
            c2 = module(c2)
        for name, module in self.decoder.named_children():
            c2 = module(c2)
        res2 = c2
        res2 = res2 / res2.max() 
        res2 = self.soft[0](res2)
        res = res2.permute(0,2,1)
        att = res.contiguous().view(b,c,w,h)
        y = res.mean(dim=2)
        orderY = torch.sort(y, dim=-1, descending=True, out=None)  
        y = orderY[0]  
        y = y.contiguous().view(b, c, 1, 1) # [1,16,1,1]
        att = y.expand_as(x) # [1,16,127,127]
        res = x.mul(att) # [1,16,127,127]
        res = x
        return res,orderY


class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1, isTrain=True):
        super(MDNet, self).__init__()
        self.K = K
        self.channel = Channel_attention_net()
        self.layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.ReLU(inplace=True))),
                ('fc4',   nn.Sequential(nn.Linear(512 * 3 * 3, 512),
                                        nn.ReLU(inplace=True))),
                ('fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLU(inplace=True)))]))

        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 2)) for _ in range(K)])

        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError('Unkown model format: {:s}'.format(model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.channel.encoder.named_children():
            append_params(self.params, module, name+'_encoder')
        for name, module in self.channel.decoder.named_children():
            append_params(self.params, module, name+'_decoder')
        for name, module in self.channel.soft.named_children():
            append_params(self.params, module, name+'_soft')
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_{:d}'.format(k))
        # print ('self.params = ',self.params)

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params
    
    def get_all_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            params[k] = p
        return params

    def _split_Channel(self,feat_channel,order):
        # print ('order[0] = ',order[0])
        res = []
        b = feat_channel.size()[0]
        for i in range(5):
            gg = feat_channel[None,0,order[0,i*3:i*3+3],:,:]
            for k in range(1,b):
                gg = torch.cat((gg,feat_channel[None,k,order[k,i*3:i*3+3],:,:]),dim=0)
            res.append(gg)  # 目前这边处理batch_size=4时有问题，只选了第一个
        return res

    def singleRes(self, x, k=0, in_layer='conv1', out_layer='fc6'):
        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x = module(x)
                if name == 'conv3':
                    x = x.view(x.size(0), -1)
                if name == out_layer:
                    return x

        x = self.branches[k](x)
        if out_layer=='fc6':
            return x
        elif out_layer=='fc6_softmax':
            return F.softmax(x, dim=1)
        
    def forward(self, x, k=0, in_layer='conv1', out_layer='fc6',needOrder=False):  # 没有测试
        # forward model from in_layer to out_layer
        # print ('needOrder = ',needOrder)
        if x.size()[1] != 16:
            return self.singleRes(x,k,in_layer,out_layer)
        else:
            c_x,order = self.channel(x)
            c_res = self._split_Channel(c_x,order[1])
            res = []
            for x in c_res:
                res.append(self.singleRes(x,k,in_layer,out_layer))
            # print ('len(res) = ',len(res))
            if needOrder:
                # print ('order[0] = ',order[0])
                return res,order[0]
            else:
                return res
            

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers)
        # print ('---------self.layers-------- = ',self.layers)
        #self.channel.load_state_dict(torch.load('models/a50_22_epoch_100_channel_model.pth'))

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])
        #self.channel.load_state_dict(torch.load('models/a50_22_epoch_100_channel_model.pth'))

class BCELoss(nn.Module):
    def forward(self, pos_score_arr, neg_score_arr, average=True,pos_orderWeight=None,neg_orderWeight=None):
        tarr = pos_orderWeight
        penaltyPos = torch.stack((tarr[:,0:3].sum(dim=1),tarr[:,3:6].sum(dim=1),tarr[:,6:9].sum(dim=1),tarr[:,9:12].sum(dim=1),tarr[:,12:15].sum(dim=1)),dim=1)
        maxRes = penaltyPos.max(dim=1)[0]
        penaltyPos = penaltyPos / maxRes.repeat(5,1).permute(1,0)

        tarr = neg_orderWeight
        penaltyNeg = torch.stack((tarr[:,0:3].sum(dim=1),tarr[:,3:6].sum(dim=1),tarr[:,6:9].sum(dim=1),tarr[:,9:12].sum(dim=1),tarr[:,12:15].sum(dim=1)),dim=1)
        maxRes = penaltyNeg.max(dim=1)[0]
        penaltyNeg = penaltyNeg / maxRes.repeat(5,1).permute(1,0)
        for idx in range(len(pos_score_arr)):
            pos_score = pos_score_arr[idx]
            neg_score = neg_score_arr[idx]
            pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]
            pos_loss_p = (torch.ones(pos_loss.size()).cuda() - F.softmax(pos_score, dim=1)[:,1]) * pos_loss
            neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]
            neg_loss_p = (torch.ones(neg_loss.size()).cuda() - F.softmax(neg_score, dim=1)[:,0]) * neg_loss

            pos_loss_p = pos_loss_p * penaltyPos[:,idx]
            neg_loss_p = neg_loss_p * penaltyNeg[:,idx]
            loss = pos_loss_p.sum() + neg_loss_p.sum()
            if True:
                loss /= (pos_loss_p.size(0) + neg_loss_p.size(0))
                if idx == 0:
                    loss_sum = loss 
                else:
                    loss_sum += loss 
        return loss_sum


class Accuracy():
    def __call__(self, pos_score, neg_score):
        pos_correct = (pos_score[:, 1] > pos_score[:, 0]).sum().float()
        neg_correct = (neg_score[:, 1] < neg_score[:, 0]).sum().float()
        acc = (pos_correct + neg_correct) / (pos_score.size(0) + neg_score.size(0) + 1e-8)
        return acc.item()


class Precision():
    def __call__(self, pos_score_arr, neg_score_arr):
        for idx in range(len(pos_score_arr)):
            pos_score = pos_score_arr[idx]
            neg_score = neg_score_arr[idx]
            scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)
            topk = torch.topk(scores, pos_score.size(0))[1]
            prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)
            if idx == 0:
                res_max = prec
            else:
                res_max += prec
        return res_max
