import sys
from sklearn.linear_model import Ridge
import numpy as np

from modules.utils import overlap_ratio


class BBRegressor():
    def __init__(self, img_size, alpha=1000, overlap=[0.6, 1], scale=[1, 2]):
        self.img_size = img_size
        self.alpha = alpha
        self.overlap_range = overlap
        self.scale_range = scale
        self.modelArr = [Ridge(alpha=self.alpha) for i in range(5)]

    def train(self, X_arr, bbox, gt):
        cnt = -1

        for X in X_arr:
            cnt += 1
            X = X.cpu().numpy()
            # print ('X.shape = ',X.shape)
            bbox11 = np.copy(bbox)
            gt11 = np.copy(gt)

            # print ('bbox11.shape = ',bbox11.shape) # (1000, 4)
            # print ('gt11.shape = ',gt11.shape) # (4,)
            if gt11.ndim==1:
                gt11 = gt11[None,:]

            r = overlap_ratio(bbox11, gt11)  # (1000,)
            s = np.prod(bbox11[:,2:], axis=1) / np.prod(gt11[0,2:]) # (1000,)
            # print ('cnt = ',cnt,'r.shape = ',r.shape)
            # print ('cnt = ',cnt,'s.shape = ',s.shape)
            idx = (r >= self.overlap_range[0]) * (r <= self.overlap_range[1]) * \
                  (s >= self.scale_range[0]) * (s <= self.scale_range[1])

            X11 = X[idx]
            bbox11 = bbox11[idx]

            Y = self.get_examples(bbox11, gt11)
            # print ('X11.shape = ',X11.shape)
            # print ('Y.shape = ',Y.shape)
            self.modelArr[cnt].fit(X11, Y)

    def predict(self, X_arr, bbox):
        bbox_arr = []
        cnt = -1
        for X in X_arr:
            cnt += 1
            X = X.cpu().numpy()
            bbox_ = np.copy(bbox)

            Y = self.modelArr[cnt].predict(X)
            # print ('Y = ',Y)

            bbox_[:,:2] = bbox_[:,:2] + bbox_[:,2:]/2
            bbox_[:,:2] = Y[:,:2] * bbox_[:,2:] + bbox_[:,:2]
            bbox_[:,2:] = np.exp(Y[:,2:]) * bbox_[:,2:]
            bbox_[:,:2] = bbox_[:,:2] - bbox_[:,2:]/2

            bbox_[:,:2] = np.maximum(bbox_[:,:2], 0)
            bbox_[:,2:] = np.minimum(bbox_[:,2:], self.img_size - bbox[:,:2])
            bbox_arr.append(bbox_)
            # print ('bbox_ = ',bbox_)
        return bbox_arr

    def get_examples(self, bbox, gt):
        bbox[:,:2] = bbox[:,:2] + bbox[:,2:]/2
        gt[:,:2] = gt[:,:2] + gt[:,2:]/2

        dst_xy = (gt[:,:2] - bbox[:,:2]) / bbox[:,2:]
        dst_wh = np.log(gt[:,2:] / bbox[:,2:])

        Y = np.concatenate((dst_xy, dst_wh), axis=1)
        return Y

