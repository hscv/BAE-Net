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
    # print ('out.shape = ',out.shape)
    # print ('M = ',M,' , N = ',N)
    img = out.reshape(M//4, N//4, 16)
    print ('img.shape = ',img.shape)
    img = img.transpose(1,0,2)
    print ('---222---img.shape = ',img.shape)
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
        res.append(gg)  # 目前这边处理batch_size=4时有问题，只选了第一个
    return res


def getHsiFrame(frame,hsi_index=0,gpuid=0):  # 返回,[height,width,bands]
    frame = np.array(frame)
    print (frame.shape)
    hsiImage = X2Cube(frame)  # 数据进行了归一化
    exemplar_img = x_transforms(hsiImage)[None,:,:,:]
    #print (exemplar_img.size())
    hsiImage_var = Variable(exemplar_img)

    model_path = 'models/tmp_models.pth'
    # model_pretrain.load_state_dict(torch.load(model_path))

    model = Channel_attention_net()
    model.load_state_dict(torch.load(model_path))

    #for name,parameters in model.named_parameters():
    #    print(name,':',parameters)
    #return 
    # print (type(hsiImage))
    res,orderY = model(hsiImage_var)  # 1,16,xx,xx
    order = orderY[1]
    res = _split_Channel(res,order)
    # print (len(res))
    penalty_weight = np.zeros([splitNum])
    orderBand = orderY[0][0]  # 有重大问题
    # print ('orderBand = ',orderY)
    for i in range(splitNum):
        penalty_weight[i] = orderBand[i*3] + orderBand[i*3+1] + orderBand[i*3+2]
    penalty_weight = np.exp(penalty_weight)
    penalty_weight = penalty_weight / penalty_weight.sum()
    # print ('penalty_weight = ',penalty_weight)
    hsiFrame = torch.cat((res[0],res[1],res[2],res[3],res[4]),1) 
    # for i in range(5):
    #     print ('res[',i,'] = ',res[i])
    # print ('hsiFrame = ',hsiFrame)
    
    hsiFrame = torch.squeeze(hsiFrame)
    # print (hsiFrame.size())
    hsiFrame = hsiFrame.detach().numpy()
    # print ('hsiFrame.shape = ',hsiFrame.shape)
    hsiFrame = hsiFrame.transpose(1,2,0)
    # print ('--last-- hsiFrame.shape = ',hsiFrame.shape)
    print (hsiFrame.shape)

    return hsiFrame[:,:,hsi_index*3:hsi_index*3+3],penalty_weight[hsi_index]

def getHsiFrameSix(image1):
    frame = np.array(image1)
    print (frame.shape)
    hsiImage = X2Cube(frame)  # 数据进行了归一化
    return hsiImage

if __name__ == '__main__':
    filename = '../../../hsi_dataset_new/test/testHSI50/test_HSI/Griffith_ball/img/0001.png'
    frame = np.array(Image.open(filename))
    getHsiFrame(frame)

