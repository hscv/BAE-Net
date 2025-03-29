import torch
import numpy as np
import pandas as pd
import os
import cv2
from torch.autograd import Variable
from channel_net import Channel_attention_net
from PIL import Image
import torchvision.transforms as transforms


splitNum = 5

class ToTensor(object):
    def __call__(self, sample):
        sample = sample.transpose(2, 0, 1)
        return torch.from_numpy(sample.astype(np.float32))

x_transforms = transforms.Compose([
    ToTensor()
])


def X2Cube(img):

    B = [4, 4]
    skip = [4, 4]
    # Parameters
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1

    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])

    # Generate Depth indeces
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    img = out.reshape(M//4, N//4, 16)
    img = img / img.max() * 255 #  归一化
    img.astype('uint8')
    return img


def _split_Channel(feat_channel,order):
    # print ('order[0] = ',order[0])
    res = []
    b = feat_channel.size()[0]
    for i in range(splitNum):
        gg = feat_channel[None,0,order[0,i*3:i*3+3],:,:]
        for k in range(1,b):
            gg = torch.cat((gg,feat_channel[None,k,order[k,i*3:i*3+3],:,:]),dim=0)
        res.append(gg)  
    return res


def getHsiFrame(frame, model_path=''):  # 返回,[height,width,bands]
    frame = np.array(frame)
    hsiImage = X2Cube(frame)  
    exemplar_img = x_transforms(hsiImage)[None,:,:,:]
    hsiImage_var = Variable(exemplar_img)
    model = Channel_attention_net()
    model.load_state_dict(torch.load(model_path))
    res,orderY = model(hsiImage_var)  # 1,16,xx,xx
    order = orderY[1]
    res = _split_Channel(res,order)
    orderBand = orderY[0][0]  
    hsiFrame = torch.cat((res[0],res[1],res[2],res[3],res[4]),1) 
    hsiFrame = torch.squeeze(hsiFrame)
    hsiFrame = hsiFrame.detach().numpy()
    hsiFrame = hsiFrame.transpose(1,2,0)
    hsiFrameArr = []
    for i in range(splitNum):
        hsiFrameArr.append(hsiFrame[:,:,i*3:i*3+3])
    return hsiFrameArr

if __name__ == '__main__':
    filename = '../../../hsi_dataset_new/test/testHSI50/test_HSI/Griffith_ball/img/0001.png'
    frame = np.array(Image.open(filename))
    getHsiFrame(frame)

