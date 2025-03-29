import numpy as np
import os
import sys
import time
import argparse
import yaml, json
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torch.optim as optim

sys.path.insert(0, '.')
from modules.model import MDNet, BCELoss, set_optimizer
from modules.sample_generator import SampleGenerator
from modules.utils import overlap_ratio
from data_prov import RegionExtractor
from bbreg import BBRegressor
from gen_config import gen_config

sys.path.insert(0,'./gnet')
from gnet.g_init import NetG, set_optimizer_g
from gnet.g_pretrain import *

from hsi_utils import getHsiFrame

opts = yaml.safe_load(open('./tracking/options.yaml','r'))

splitNum = 5
def forward_samples(model, imageArr, samples, out_layer='conv3'):
    model.eval()
    feats_arr = []
    for image in imageArr:
        extractor = RegionExtractor(image, samples, opts)
        for i, regions in enumerate(extractor):
            if opts['use_gpu']:
                regions = regions.cuda()
            with torch.no_grad():
                feat = model(regions, out_layer=out_layer)
            if i==0:
                feats = feat.detach().clone()
            else:
                feats = torch.cat((feats, feat.detach().clone()), 0)
        feats_arr.append(feats)
    return feats_arr

def train(model, model_g, criterion, optimizer, pos_featsArr, neg_featsArr, maxiter, in_layer='fc4'):
    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)
    pos_idx = np.random.permutation(pos_featsArr[0].size(0))
    neg_idx = np.random.permutation(neg_featsArr[0].size(0))
    while(len(pos_idx) < batch_pos * maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_featsArr[0].size(0))])
    while(len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_featsArr[0].size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for i in range(maxiter):
        batch_pos_featsArr = []
        for kk in range(splitNum):
            pos_feats = pos_featsArr[kk]
            neg_feats = neg_featsArr[kk]
            # select pos idx
            pos_next = pos_pointer + batch_pos
            pos_cur_idx = pos_idx[pos_pointer:pos_next]
            pos_cur_idx = pos_feats.new(pos_cur_idx).long()
            if kk == splitNum:
                pos_pointer = pos_next

            # select neg idx
            neg_next = neg_pointer + batch_neg_cand
            neg_cur_idx = neg_idx[neg_pointer:neg_next]
            neg_cur_idx = neg_feats.new(neg_cur_idx).long()
            if kk == splitNum:
                neg_pointer = neg_next

            # create batch
            batch_pos_feats = pos_feats[pos_cur_idx]
            batch_pos_featsArr.append(batch_pos_feats)
            if model_g is not None:
                batch_asdn_feats = pos_feats.index_select(0, pos_cur_idx)
            batch_neg_feats = neg_feats[neg_cur_idx]
            # hard negative mining
            if batch_neg_cand > batch_neg:
                model.eval()
                for start in range(0, batch_neg_cand, batch_test):
                    end = min(start + batch_test, batch_neg_cand)
                    with torch.no_grad():
                        score = model(batch_neg_feats[start:end], in_layer=in_layer)
                    if start==0:
                        neg_cand_score = score.detach()[:, 1].clone()
                    else:
                        neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 1].clone()), 0)
                if neg_cand_score.size()[0] != 0:
                    _, top_idx = neg_cand_score.topk(batch_neg)
                    batch_neg_feats = batch_neg_feats[top_idx]
                model.train()

            if model_g is not None:
                model_g.eval()
                res_asdn = model_g(batch_asdn_feats)
                model_g.train()
                num = res_asdn.size(0)
                mask_asdn = torch.ones(num, 512, 3, 3)
                res_asdn = res_asdn.view(num, 3, 3)
                for i in range(num):
                    feat_ = res_asdn[i, :, :]
                    featlist = feat_.view(1, 9).squeeze()
                    feat_list = featlist.detach().cpu().numpy()
                    idlist = feat_list.argsort()
                    idxlist = idlist[:3]

                    for k in range(len(idxlist)):
                        idx = idxlist[k]
                        row = idx // 3
                        col = idx % 3
                        mask_asdn[:, :, col, row] = 0
                mask_asdn = mask_asdn.view(mask_asdn.size(0), -1)
                if opts['use_gpu']:
                    batch_asdn_feats = batch_asdn_feats.cuda()
                    mask_asdn = mask_asdn.cuda()
                batch_asdn_feats = batch_asdn_feats * mask_asdn

            # forward
            if model_g is None:
                pos_score = model(batch_pos_feats, in_layer=in_layer)
            else:
                pos_score = model(batch_asdn_feats, in_layer=in_layer)
            neg_score = model(batch_neg_feats, in_layer=in_layer)

            # optimize
            loss = criterion(pos_score, neg_score)
            if kk == 0:
                loss_sum = loss
            else:
                loss_sum += loss
        model.zero_grad()
        loss_sum.backward()
        if 'grad_clip' in opts:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
        optimizer.step()

        if model_g is not None:
            optimizer_g = set_optimizer_g(model_g)
            for kk in range(splitNum):
                batch_pos_feats = batch_pos_featsArr[kk]
                start = time.time()
                prob_k = torch.zeros(9)
                for k in range(9):
                    row = k // 3
                    col = k % 3

                    model.eval()
                    batch = batch_pos_feats.view(batch_pos, 512, 3, 3)
                    batch[:, :, col, row] = 0
                    batch = batch.view(batch.size(0), -1)

                    if opts['use_gpu']:
                        batch = batch.cuda()

                    prob = model(batch, in_layer='fc4', out_layer='fc6_softmax')[:, 1]
                    model.train()

                    prob_k[k] = prob.sum()

                _, idx = torch.min(prob_k, 0)
                idx = idx.item()
                row = idx // 3
                col = idx % 3

                
                labels = torch.ones(batch_pos, 1, 3, 3)
                labels[:, :, col, row] = 0

                batch_pos_feats = batch_pos_feats.view(batch_pos_feats.size(0), -1)
                res = model_g(batch_pos_feats)
                labels = labels.view(batch_pos, -1)
                criterion_g = torch.nn.MSELoss(reduction='mean')
                loss_g_2 = criterion_g(res.float(), labels.cuda().float())
                if kk == 0:
                    loss_g_2_sum = loss_g_2
                else:
                    loss_g_2_sum += loss_g_2

            model_g.zero_grad()
            loss_g_2_sum.backward()
            optimizer_g.step()

            end = time.time()
            print('asdn objective %.3f, %.2f s' % (loss_g_2_sum, end - start))

def run_vital(img_list, init_bbox, gt=None, channel_model_path=''):

    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list), 4))
    result_bb = np.zeros((len(img_list), 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    if gt is not None:
        overlap = np.zeros(len(img_list))
        overlap[0] = 1

    # Init model
    model = MDNet(opts['model_path'])
    model_g = NetG()
    if opts['use_gpu']:
        model = model.cuda()
        model_g = model_g.cuda()

    # Init criterion and optimizer 
    criterion = BCELoss()
    criterion_g = torch.nn.MSELoss(reduction='mean')
    model.set_learnable_params(opts['ft_layers'])
    model_g.set_learnable_params(opts['ft_layers'])
    init_optimizer = set_optimizer(model, opts['lr_init'], opts['lr_mult'])
    update_optimizer = set_optimizer(model, opts['lr_update'], opts['lr_mult'])

    tic = time.time()
    # Load first image
    image1 = Image.open(img_list[0])
    imageArr = getHsiFrame(image1,channel_model_path)
    imageArr = [Image.fromarray(image.astype('uint8')).convert('RGB') for image in imageArr]
    image = imageArr[0]
    # Draw pos/neg samples
    pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
                        target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

    neg_examples = np.concatenate([
                    SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
                        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
                    SampleGenerator('whole', image.size)(
                        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)

    # Extract pos/neg features
    pos_featsArr = forward_samples(model, imageArr, pos_examples)
    neg_featsArr = forward_samples(model, imageArr, neg_examples)
    # Initial training
    train(model, None, criterion, init_optimizer, pos_featsArr, neg_featsArr, opts['maxiter_init'])
    del init_optimizer, neg_featsArr
    torch.cuda.empty_cache()

    g_pretrain(model, model_g, criterion_g, pos_featsArr)
    torch.cuda.empty_cache()
    # Train bbox regressor
    bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'], opts['aspect_bbreg'])(
                        target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
    bbreg_featsArr = forward_samples(model, imageArr, bbreg_examples)
    bbreg = BBRegressor(image.size)
    bbreg.train(bbreg_featsArr, bbreg_examples, target_bbox)
    del bbreg_featsArr
    torch.cuda.empty_cache()

    # Init sample generators for update
    sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
    pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
    neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])

    # Init pos/neg features for update
    neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
    neg_featsArr = forward_samples(model, imageArr, neg_examples)

    pos_feats_allArr = [1]*splitNum
    neg_feats_allArr = [1]*splitNum
    for i in range(splitNum):
        pos_feats_allArr[i] = [pos_featsArr[i]]
        neg_feats_allArr[i] = [neg_featsArr[i]]
    spf_total = time.time() - tic

    # Main loop
    for i in range(1, len(img_list)):
        tic = time.time()
        # Load image
        image1= Image.open(img_list[i])
        samples = sample_generator(target_bbox, opts['n_samples'])
        imageArr = getHsiFrame(image1,channel_model_path)
        imageArr = [Image.fromarray(image.astype('uint8')).convert('RGB') for image in imageArr]

        # Estimate target bbox
        sample_scoresArr = forward_samples(model, imageArr, samples, out_layer='fc6')
        target_bboxArr = []
        for kk in range(splitNum):
            sample_scores = sample_scoresArr[kk]
            top_scores, top_idx = sample_scores[:, 1].topk(4)
            target_score = top_scores.mean()
            if 0 == kk:
                target_score_res = target_score
            else:
                target_score_res += target_score
            top_idx = top_idx.cpu()
            target_bbox = samples[top_idx]
            if top_idx.shape[0] > 1:
                target_bbox = target_bbox.mean(axis=0)
            target_bboxArr.append(target_bbox)
        target_bboxArrTT = np.array(target_bboxArr)
        target_bbox = target_bboxArrTT.mean(axis=0)
        target_score = target_score_res * 1.0 / splitNum
        success = target_score > 0
        print ('target_bbox = ',target_bbox)
        if success:
            sample_generator.set_trans(opts['trans'])
        else:
            sample_generator.expand_trans(opts['trans_limit'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None,:]
            bbreg_featsArr = forward_samples(model, imageArr, bbreg_samples)
            bbreg_samplesArr = bbreg.predict(bbreg_featsArr, bbreg_samples)
            bbreg_bbox = np.array([bbreg_samplesArr[kk].mean(axis=0) for kk in range(splitNum)]).mean(axis=0)
        else:
            bbreg_bbox = target_bbox

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox

        # Data collect
        if success:
            pos_examples = pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
            pos_featsArr = forward_samples(model, imageArr, pos_examples)
            for kk in range(splitNum):
                pos_feats_allArr[kk].append(pos_featsArr[kk])
                if len(pos_feats_allArr[kk]) > opts['n_frames_long']:
                    del pos_feats_allArr[kk][0]

            neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
            neg_featsArr = forward_samples(model, imageArr, neg_examples)
            for kk in range(splitNum):
                neg_feats_allArr[kk].append(neg_featsArr[kk])
                if len(neg_feats_allArr[kk]) > opts['n_frames_short']:
                    del neg_feats_allArr[kk][0]

        # Short term update
        if not success:
            pos_dataArr = []
            neg_dataArr = []
            for kk in range(splitNum):
                pos_feats_all = pos_feats_allArr[kk]
                neg_feats_all = neg_feats_allArr[kk]
                nframes = min(opts['n_frames_short'], len(pos_feats_all))
                pos_data = torch.cat(pos_feats_all[-nframes:], 0)
                neg_data = torch.cat(neg_feats_all, 0)
                pos_dataArr.append(pos_data)
                neg_dataArr.append(neg_data)
            train(model, None, criterion, update_optimizer, pos_dataArr, neg_dataArr, opts['maxiter_update'])

        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_dataArr = []
            neg_dataArr = []
            for kk in range(splitNum):
                pos_feats_all = pos_feats_allArr[kk]
                neg_feats_all = neg_feats_allArr[kk]
                pos_data = torch.cat(pos_feats_all, 0)
                neg_data = torch.cat(neg_feats_all, 0)
                pos_dataArr.append(pos_data)
                neg_dataArr.append(neg_data)
            train(model, model_g, criterion, update_optimizer, pos_dataArr, neg_dataArr, opts['maxiter_update'])

        torch.cuda.empty_cache()
        spf = time.time() - tic
        spf_total += spf

        if gt is None:
            print('Frame {:d}/{:d}, Score {:.3f}, Time {:.3f}'
                .format(i + 1, len(img_list), target_score, spf))
        else:
            overlap[i] = overlap_ratio(gt[i], result_bb[i])[0]
            print ('gt[',i,'] = ',gt[i],' , result_bb[',i,'] = ',result_bb[i])
            print('Frame {:d}/{:d}, Overlap {:.3f}, Score {:.3f}, Time {:.3f}'
                .format(i + 1, len(img_list), overlap[i], target_score, spf))

    if gt is not None:
        print('meanIOU: {:.3f}'.format(overlap.mean()))
    fps = len(img_list) / spf_total
    return result, result_bb, fps

def cal_iou(box1, box2):
    r"""

    :param box1: x1,y1,w,h
    :param box2: x1,y1,w,h
    :return: iou
    """
    x11 = box1[0]
    y11 = box1[1]
    x21 = box1[0] + box1[2] - 1
    y21 = box1[1] + box1[3] - 1
    area_1 = (x21 - x11 + 1) * (y21 - y11 + 1)

    x12 = box2[0]
    y12 = box2[1]
    x22 = box2[0] + box2[2] - 1
    y22 = box2[1] + box2[3] - 1
    area_2 = (x22 - x12 + 1) * (y22 - y12 + 1)

    x_left = max(x11, x12)
    x_right = min(x21, x22)
    y_top = max(y11, y12)
    y_down = min(y21, y22)

    inter_area = max(x_right - x_left + 1, 0) * max(y_down - y_top + 1, 0)
    iou = inter_area / (area_1 + area_2 - inter_area)
    return iou

def cal_success(iou):
    success_all = []
    overlap_thresholds = np.arange(0, 1.05, 0.05)
    for overlap_threshold in overlap_thresholds:
        success = sum(np.array(iou) > overlap_threshold) / len(iou)
        success_all.append(success)
    return np.array(success_all)

def calAUC(gtArr,resArr):
    # ------------ starting evaluation  -----------
    success_all_video = []
    for idx in range(len(resArr)):
        result_boxes = resArr[idx]
        result_boxes_gt = gtArr[idx]
        result_boxes_gt = [np.array(box) for box in result_boxes_gt]
        iou = list(map(cal_iou, result_boxes, result_boxes_gt))
        success = cal_success(iou)
        auc = np.mean(success)
        success_all_video.append(success)
        print ('video = ',video_dir_arr[idx],' , auc = ',auc)
    print('np.mean(success_all_video) = ', np.mean(success_all_video))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='/nas_data/lizf/HOT/dataset/test/test_HSI/', help='input seq path')
    parser.add_argument('-j', '--channel_model', default='models/final_model/11_epoch_1_channel_model_200.pth', help='channel model path')
    parser.add_argument('-f', '--savepath', default='./results/')
    args = parser.parse_args()
    video_dir_arr = os.listdir(args.seq)
    video_dir_arr.sort()

    gtArr = []
    resArr = []
    for video_dir in video_dir_arr:
        # video_dir = video_dir_arr[1]
        np.random.seed(0)
        torch.manual_seed(0)
        print ('video_dir = ',video_dir)
        img_list, init_bbox, gt = gen_config(args, video_dir)

        gtArr.append(gt)
        # print ('args.channel_model = ', args.channel_model)
        # raise Exception
        # Run tracker
        result, result_bb, fps = run_vital(img_list, init_bbox, gt=gt, channel_model_path=args.channel_model)
        result_bb = result_bb.round()
        resArr.append(result_bb)
        torch.cuda.empty_cache()

        # Save result
        result_path = os.path.join(args.savepath, video_dir+'.txt')
        f = open(result_path, 'w')
        for bbox in result_bb.tolist():
            for dd in bbox:
                f.write(str(dd))
                f.write('\t')
            f.write('\n')
        f.close()
    calAUC(gtArr,resArr) 
