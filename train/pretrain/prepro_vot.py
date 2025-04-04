import os
import numpy as np
import pickle
from collections import OrderedDict

seq_home = '../../hsi_dataset_new/train/trainHSI/Data'
seq_home_anno = '../../hsi_dataset_new/train/trainHSI/Annotations'
seqlist_path = 'datasets/list/train_hsi.txt'
output_path = 'pretrain/data/train_hsi.pkl'

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

with open(seqlist_path,'r') as fp:
    seq_list = fp.read().splitlines()

print ('seq_list = ',seq_list)

# Construct db
data = OrderedDict()
for i, seq in enumerate(seq_list):
    print ('seq = ',seq)
    img_list = sorted([p for p in listdir_nohidden(os.path.join(seq_home, seq, 'HSI')) if os.path.splitext(p)[1] == '.png'])
    print ('img_list = ',img_list)
    # gt = np.loadtxt((x.replace('\t',',') for x in f), delimiter=',')
    # print ('os = ',os.path.join(seq_home_anno, seq, 'groundtruth_rect.txt'))
    gt_path = os.path.join(seq_home_anno, seq, 'groundtruth_rect.txt')
    with open(gt_path) as f:
        gt = np.loadtxt((x[:-2].replace('\t',',') for x in f), delimiter=',')
    print ('gt = ',gt)
    # continue

    if seq == 'vot2014/ball':
        img_list = img_list[1:]
        gt = gt[1:]

    #assert len(img_list) == len(gt), "Lengths do not match!!"

    if gt.shape[1] == 8:
        x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

    img_list = [os.path.join(seq_home, seq, 'HSI', img) for img in img_list]
    print ('---22--- imga_list = ',img_list)
    data[seq] = {'images': img_list, 'gt': gt}

# Save db
output_dir = os.path.dirname(output_path)
os.makedirs(output_dir, exist_ok=True)
with open(output_path, 'wb') as fp:
    pickle.dump(data, fp)
