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
from hsi_utils import getHsiFrameSix

opts = yaml.safe_load(open('./tracking/options.yaml','r'))

video_dir_arr = ['Griffith_ball', \
                'Griffith_basketball', \
                'Griffith_board1', \
                'Griffith_board2', \
                'Griffith_book1', \
                'Griffith_boy', \
                'Griffith_bus', \
                'Griffith_bus2', \
                'Griffith_campus', \
                'Griffith_car', \
                'Griffith_car3', \
                'Griffith_car4', \
                'Griffith_car5', \
                'Griffith_card', \
                'Griffith_coin', \
                'Griffith_coke', \
                'Griffith_drive', \
                'Griffith_excavator', \
                'Griffith_face', \
                'Griffith_face2', \
                'Griffith_forest2', \
                'Griffith_fruit', \
                'Griffith_green', \
                'Griffith_hand1', \
                'Griffith_kangroo', \
                'Griffith_kangroo2', \
                'Griffith_paper', \
                'Griffith_pedestrain', \
                'Griffith_pedestrain2', \
                'Griffith_playground', \
                'Griffith_rubik', \
                'Griffith_student', \
                'Griffith_toylight', \
                'Griffith_worker', \
                'Griffith_yellowtoy1', \
                'Griffith_yellowtoy2', \
                'Griffith_yellowtoylight', \
                'NJUST_automobile41', \
                'NJUST_automobile49', \
                'NJUST_automobile54', \
                'NJUST_bus', \
                'NJUST_car10', \
                'NJUST_car17', \
                'NJUST_car3', \
                'NJUST_car6', \
                'NJUST_car7', \
                'NJUST_pedestrian', \
                'NJUST_rider19', \
                'NJUST_rider4', \
                'NJUST_trucker']


def forward_samples(model, image, samples, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image, samples, opts)
    featsArr = [1]*5
    for i, regions in enumerate(extractor):
        if opts['use_gpu']:
            regions = regions.cuda()
        with torch.no_grad():
            featArr = model(regions, out_layer=out_layer)  # 返回的有5个数值的list
        if i==0:
            for kk in range(5):
                featsArr[kk] = featArr[kk].detach().clone()
        else:
            for kk in range(5):
                featsArr[kk] = torch.cat((featsArr[kk], featArr[kk].detach().clone()), 0)
    return featsArr


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

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_featsArr[0].new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_featsArr[0].new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_featsArr = [pos_featsArr[kk][pos_cur_idx] for kk in range(5)]
        # batch_pos_feats = pos_feats[pos_cur_idx]
        if model_g is not None:
            batch_asdn_featsArr = [pos_featsArr[kk].index_select(0, pos_cur_idx) for kk in range(5)]
            # batch_asdn_feats = pos_feats.index_select(0, pos_cur_idx)
        batch_neg_featsArr = [neg_featsArr[kk][neg_cur_idx] for kk in range(5)]

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            neg_cand_scoreArr = [1]*5
            scoreArr = [1]*5
            top_idx = [1]*5
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                with torch.no_grad():
                    for kk in range(5):
                        scoreArr[kk] = model(batch_neg_featsArr[kk][start:end], in_layer=in_layer)
                if start==0:
                    for kk in range(5):
                        neg_cand_scoreArr[kk] = scoreArr[kk].detach()[:, 1].clone()
                else:
                    neg_cand_scoreArr[kk] = torch.cat((neg_cand_scoreArr[kk], scoreArr[kk].detach()[:, 1].clone()), 0)

            for kk in range(5):
                _, top_idx[kk] = neg_cand_scoreArr[kk].topk(batch_neg)
            batch_neg_featsArr = [batch_neg_featsArr[kk][top_idx[kk]] for kk in range(5)]
            model.train()

        if model_g is not None:
            model_g.eval()
            res_asdnArr = [model_g(batch_neg_featsArr[kk]) for kk in range(5)]
            model_g.train()
            num = res_asdnArr[0].size(0)
            mask_asdn = torch.ones(num, 512, 3, 3)
            res_asdn = res_asdnArr[0].view(num, 3, 3)
            for kk in range(5):
                for i in range(num):
                    feat_ = res_asdnArr[kk][i, :, :]
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
                    batch_neg_featsArr[kk] = batch_neg_featsArr[kk].cuda()
                    mask_asdn = mask_asdn.cuda()
                batch_neg_featsArr[kk] = batch_neg_featsArr[kk] * mask_asdn

        # forward
        if model_g is None:
            pos_scoreArr = [model(batch_pos_featsArr[kk], in_layer=in_layer) for kk in range(5)]
        else:
            pos_scoreArr = [model(batch_asdn_featsArr[kk], in_layer=in_layer) for kk in range(5)]
        neg_scoreArr = [model(batch_neg_featsArr[kk], in_layer=in_layer) for kk in range(5)]

        # optimize
        loss = criterion(pos_scoreArr, neg_scoreArr)
        model.zero_grad()
        loss.backward()
        if 'grad_clip' in opts:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
        optimizer.step()

        if model_g is not None:
            start = time.time()
            prob_k = torch.zeros(9)
            for kk in range(5):
                batch_pos_feats = batch_pos_featsArr[kk]
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

                optimizer_g = set_optimizer_g(model_g)
                labels = torch.ones(batch_pos, 1, 3, 3)
                labels[:, :, col, row] = 0

                batch_pos_feats = batch_pos_feats.view(batch_pos_feats.size(0), -1)
                res = model_g(batch_pos_feats)
                labels = labels.view(batch_pos, -1)
                criterion_g = torch.nn.MSELoss(reduction='mean')
                if 0 == kk:
                    loss_g_2 = criterion_g(res.float(), labels.cuda().float())
                else:
                    loss_g_2 += criterion_g(res.float(), labels.cuda().float())

            model_g.zero_grad()
            loss_g_2.backward()
            optimizer_g.step()

            end = time.time()
            print('asdn objective %.3f, %.2f s' % (loss_g_2, end - start))


def run_vital(img_list, init_bbox, gt=None, savefig_dir='', display=False,hsi_index=0):
    # print ('hsi_index = ',hsi_index)
    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list), 4))
    result_bb = np.zeros((len(img_list), 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox
    print ('gt = ',target_bbox)

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
    image1 = Image.open(img_list[0])#.convert('RGB')
    image = getHsiFrameSix(image1)  # 16通道
    tt = image[:,:,-1,np.newaxis]
    # print ('image.shape = ',image.shape)
    # print ('image = ',image)
    # print ('type(image) = ',type(image))
    # print ('image.shape = ',image.shape)
    tArr = [Image.fromarray(image[:,:,kk*3:kk*3+3].astype('uint8')).convert('RGB') for kk in range(5)]
    image = np.concatenate((tArr[0], tArr[1],tArr[2],tArr[3],tArr[4],tt), axis=2)
    # image = Image.fromarray(image)

    # Draw pos/neg samples
    pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
                        target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

    neg_examples = np.concatenate([
                    SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
                        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
                    SampleGenerator('whole', image.size)(
                        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)
    print ('pos_examples = ',pos_examples.shape) # (500,4)  -- 和3通道一样
    print ('neg_examples.shape = ',neg_examples.shape)# (5000,4)
    
    # Extract pos/neg features
    pos_feats = forward_samples(model, image, pos_examples)  # 返回的是list 长度5
    neg_feats = forward_samples(model, image, neg_examples)
    print ('pos_feats.shape = ',pos_feats[0].shape)
    print ('neg_feats.shape = ',neg_feats[0].shape)
    
    # Initial training
    train(model, None, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])
    del init_optimizer, neg_feats
    
    torch.cuda.empty_cache()
    print ('----ok----')
    g_pretrain(model, model_g, criterion_g, pos_feats[1])   # 假设第一帧以第2个通道为准
    torch.cuda.empty_cache()

    # Train bbox regressor
    bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'], opts['aspect_bbreg'])(
                        target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])

    bbreg_feats = forward_samples(model, image, bbreg_examples) # 返回的是list 长度5
    bbreg = BBRegressor(image.size)
    bbreg.train(bbreg_feats[1], bbreg_examples, target_bbox) # 假设第一帧以第2个通道为
    del bbreg_feats
    torch.cuda.empty_cache()

    # Init sample generators for update
    sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
    pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
    neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])

    # Init pos/neg features for update
    neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
    neg_feats = forward_samples(model, image, neg_examples)
    pos_feats_all = [1]*5
    neg_feats_all = [1]*5
    for i in range(5):
        pos_feats_all[i] = [pos_feats[i]]
        neg_feats_all[i] = [neg_feats[i]]

    spf_total = time.time() - tic

    # Display
    savefig = savefig_dir != ''
    if display or savefig:
        dpi = 80.0
        figsize = (image.size[0] / dpi, image.size[1] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image, aspect='auto')

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
                                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)

        rect = plt.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3],
                             linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)

    # Main loop
    for i in range(1, len(img_list)):

        tic = time.time()
        # Load image
        image1= Image.open(img_list[i])
        image = getHsiFrameSix(image1)
        tArr = [Image.fromarray(image[:,:,kk*3:kk*3+3].astype('uint8')).convert('RGB') for kk in range(5)]
        image = np.concatenate((tArr[0], tArr[1],tArr[2],tArr[3],tArr[4],tt), axis=2)
        # print ('image = ',image)
        # image = Image.fromarray(image.astype('uint8')).convert('RGB')
        # Estimate target bbox
        samples = sample_generator(target_bbox, opts['n_samples'])
        sample_scoresArr = forward_samples(model, image, samples, out_layer='fc6')

        top_scoresArr = []
        top_idxArr = []
        maxIdx = -1
        maxScore = -1
        for kk in range(5):
            top_scores, top_idx = sample_scoresArr[kk][:, 1].topk(5)
            top_scoresArr.append(top_scores)
            top_idxArr.append(top_idx)
        # top_scores = maxScore
        for kk in range(5):
            if top_scoresArr[kk].mean() > maxScore:
                top_idx = top_idxArr[kk]
                maxScore = top_scoresArr[kk].mean()
        top_idx = top_idx.cpu()
        target_score = (top_scoresArr[0].mean() + top_scoresArr[1].mean() + top_scoresArr[2].mean() + top_scoresArr[3].mean() + top_scoresArr[4].mean()) / 5.0
        target_bbox = samples[top_idx]
        if top_idx.shape[0] > 1:
            target_bbox = target_bbox.mean(axis=0)
        success = target_score > 0
        
        # Expand search area at failure
        if success:
            sample_generator.set_trans(opts['trans'])
        else:
            sample_generator.expand_trans(opts['trans_limit'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None,:]
            bbreg_featsArr = forward_samples(model, image, bbreg_samples)
            bbreg_samplesArr = [bbreg.predict(bbreg_featsArr[kk], bbreg_samples) for kk in range(5)]
            bbreg_bbox = (bbreg_samplesArr[0].mean(axis=0) + bbreg_samplesArr[1].mean(axis=0) + bbreg_samplesArr[2].mean(axis=0) + bbreg_samplesArr[3].mean(axis=0) + bbreg_samplesArr[4].mean(axis=0)) /5.0
        else:
            bbreg_bbox = target_bbox

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox
        print ('result[',i,'] = ',result[i])
        print ('result_bb[',i,'] = ',result_bb[i])
        # Data collect
        if success:
            pos_examples = pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
            pos_feats = forward_samples(model, image, pos_examples)
            for i in range(5):
                pos_feats_all[i].append(pos_feats[i])
                if len(pos_feats_all[i]) > opts['n_frames_long']:
                    del pos_feats_all[i][0]

            neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
            neg_feats = forward_samples(model, image, neg_examples)
            for i in range(5):
                neg_feats_all[i].append(neg_feats[i])
                if len(neg_feats_all[i]) > opts['n_frames_short']:
                    del neg_feats_all[i][0]

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            pos_dataArr = [torch.cat(pos_feats_all[i][-nframes:], 0) for i in range(5)]
            neg_dataArr = [torch.cat(neg_feats_all[i], 0) for i in range(5)]
            train(model, None, criterion, update_optimizer, pos_dataArr, neg_dataArr, opts['maxiter_update'])

        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_dataArr = [torch.cat(pos_feats_all[i], 0) for i in range(5)]
            neg_dataArr = [torch.cat(neg_feats_all[i], 0) for i in range(5)]
            train(model, model_g, criterion, update_optimizer, pos_dataArr, neg_dataArr, opts['maxiter_update'])

        torch.cuda.empty_cache()
        spf = time.time() - tic
        spf_total += spf

        # Display
        if display or savefig:
            im.set_data(image)

            if gt is not None:
                gt_rect.set_xy(gt[i, :2])
                gt_rect.set_width(gt[i, 2])
                gt_rect.set_height(gt[i, 3])

            rect.set_xy(result_bb[i, :2])
            rect.set_width(result_bb[i, 2])
            rect.set_height(result_bb[i, 3])

            if display:
                plt.pause(.01)
                plt.draw()
            if savefig:
                fig.savefig(os.path.join(savefig_dir, '{:04d}.jpg'.format(i)), dpi=dpi)

        if gt is None:
            print('Frame {:d}/{:d}, Score {:.3f}, Time {:.3f}'
                .format(i + 1, len(img_list), target_score, spf))
        else:
            overlap[i] = overlap_ratio(gt[i], result_bb[i])[0]
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

def calAUC(gtArr,resArr,lent):
    # ------------ starting evaluation  -----------
    success_all_video = []
    for idx in range(lent):
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
    parser.add_argument('-s', '--seq', default='', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')
    parser.add_argument('-c', '--split', default=0)

    args = parser.parse_args()
    # assert args.seq != '' or args.json != ''

    np.random.seed(0)
    torch.manual_seed(0)
    gtArr = []
    resArr = []
    cnt = 0
    for video_dir in video_dir_arr:
        cnt += 1
        if cnt >= 3:break
        args.seq = video_dir
        print ('args.seq = ',args.seq)
        # Generate sequence config
        img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)
        gtArr.append(gt)
        # Run tracker
        result, result_bb, fps = run_vital(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display,hsi_index=args.split)
        resArr.append(result_bb)
        # Save result
        res = {}
        res['res'] = result_bb.round().tolist()
        res['type'] = 'rect'
        res['fps'] = fps
        json.dump(res, open(result_path, 'w'), indent=2)
        
    calAUC(gtArr,resArr,len(video_dir_arr))
