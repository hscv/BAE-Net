import os
import json
import numpy as np


def gen_config(args, videoname):

    if args.seq != '':
        # generate config from a sequence name

        # seq_home = 'datasets/OTB'
        seq_home = args.seq
        result_home = args.savepath 

        seq_name = videoname
        img_dir = os.path.join(seq_home, seq_name, 'HSI')
        gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')

        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list = [os.path.join(img_dir, x) for x in img_list]

        with open(gt_path) as f:
            gt = np.loadtxt((x[:-2].replace('\t',',') for x in f), delimiter=',')
        init_bbox = gt[0]

        result_dir = result_home # os.path.join(result_home, seq_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

    return img_list, init_bbox, gt
