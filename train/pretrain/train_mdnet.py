import os, sys
import pickle
import yaml
import time
import argparse
import numpy as np

import torch

sys.path.insert(0,'.')
from data_prov import RegionDataset
from modules.model import MDNet, set_optimizer, BCELoss, Precision


def getGTBbox(gt_filename):
    f = open(gt_filename, 'r')
    lines = f.readlines()
    f.close()
    gtArr = []
    for line in lines:
        kk = line.split('\t')[:-1]
        gtArr.append(list(map(int, kk)))
    return gtArr

def getSourceData(video_path):
    print ('video_path = ', video_path)
    img_list = []
    img_list_tmp = os.listdir(os.path.join(video_path, 'HSI'))
    img_list_tmp.sort()
    for img_path in img_list_tmp:
        if img_path.find('.png') != -1:
            img_list.append(os.path.join(video_path, 'HSI', img_path))
    gtArr = getGTBbox(os.path.join(video_path, 'HSI', 'groundtruth_rect.txt'))
    assert len(img_list) == len(gtArr)
    return img_list, gtArr

def generate_image_gt_list(rootDir):
    final_img_list = []
    final_gt_list = []
    for video_name in os.listdir(rootDir):
        img_list, gtArr = getSourceData(os.path.join(rootDir, video_name))
        final_img_list.append(img_list)
        final_gt_list.append(gtArr)
    return final_img_list, final_gt_list

def train_mdnet(opts,train_dataset_dir=''):
    
    # Init dataset
    final_img_list, final_gt_list = generate_image_gt_list(rootDir=train_dataset_dir)
    K = len(final_img_list)
    print ('K = ', K)
    dataset = [None] * K
    for k in range(K):
        dataset[k] = RegionDataset(final_img_list[k], final_gt_list[k], opts) 
    # Init model
    model = MDNet(opts['model_path'], K) 
    for param in model.parameters():  
        param.requires_grad = False
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])

    for name, param in model.named_parameters():
        if param.requires_grad:
            print('trained name = ',name)

    # Init criterion and optimizer
    criterion = BCELoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, opts['lr'], opts['lr_mult'])
    # Main trainig loop
    for i in range(opts['n_cycles']):
        print('==== Start Cycle {:d}/{:d} ===='.format(i + 1, opts['n_cycles']))

        if i in opts.get('lr_decay', []):
            print('decay learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opts.get('gamma', 0.1)

        # Training
        model.train()
        prec = np.zeros(K)
        k_list = np.random.permutation(K)
        for j, k in enumerate(k_list):
            tic = time.time()
            # training
            pos_regions, neg_regions = dataset[k].next() 
            if opts['use_gpu']:
                pos_regions = pos_regions.cuda()
                neg_regions = neg_regions.cuda()
            pos_score_arr,pos_weight = model(pos_regions, k, needOrder=True) # [5*[pos_b, 16, 107, 107]]
            neg_score_arr,neg_weight = model(neg_regions, k, needOrder=True) # [5*[neg_b, 16, 107, 107]]
            # print ('pos_weight = ', pos_weight)
            # print ('neg_weight = ', neg_weight)
            loss = criterion(pos_score_arr, neg_score_arr, average=False,pos_orderWeight=pos_weight,neg_orderWeight=neg_weight)
            batch_accum = opts.get('batch_accum', 1)
            if j % batch_accum == 0:
                model.zero_grad()
            loss.backward()
            if j % batch_accum == batch_accum - 1 or j == len(k_list) - 1:
                if 'grad_clip' in opts:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
                optimizer.step()

            prec[k] = evaluator(pos_score_arr, neg_score_arr)

            toc = time.time()-tic
            print('Cycle {:2d}/{:2d}, Iter {:2d}/{:2d} (Domain {:2d}), Loss {:.3f}, Precision {:.3f}, Time {:.3f}'
                    .format(i, opts['n_cycles'], j, len(k_list), k, loss.item(), prec[k], toc))

        print('--just--print--Mean Precision: {:.3f}'.format(prec.mean()))
        print('--just--print--Save model to {:s}'.format(opts['model_path']))
        if opts['use_gpu']:
            model = model.cpu()
        if (i+1)%10 == 0:
            states_channel = model.channel.state_dict()
            save_path_channel = 'models/epoch_%d_channel_model.pth' % (i+1)
            print ('save_path_channel = ',save_path_channel)
            torch.save(states_channel, save_path_channel)
        
        if opts['use_gpu']:
            model = model.cuda()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='/nas_data/lizf/HOT/dataset/train/', help='training dataset {vot, imagenet}')
    args = parser.parse_args()

    opts = yaml.safe_load(open('pretrain/options_vot.yaml', 'r'))
    train_mdnet(opts, args.dataset)
