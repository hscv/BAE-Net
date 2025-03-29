import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
import numpy as np

from collections import OrderedDict
class Channel_attention_net(nn.Module):

    def __init__(self, channel=16, reduction=4):
        super(Channel_attention_net, self).__init__()


        self.encoder = nn.Sequential(nn.Linear(channel, channel//2,bias=True),
                                     nn.ReLU(inplace=False),
                                     nn.Linear(channel//2, channel//4,bias=True))

        self.decoder = nn.Sequential(nn.Linear(channel//4, channel//2,bias=True),
                                     nn.ReLU(inplace=False),
                                     nn.Linear(channel//2, channel,bias=True)
                                     )
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x):  # return 16 bands point-mul result
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x = module(x)
                if name == 'conv3':
                    x = x.view(x.size(0), -1)
                if name == out_layer:
                    return x
        b, c, w, h = x.size()# [1,16,127,127]
        c1 = x.view(b,c,-1)
        c2 = c1.permute(0,2,1)
        res1 = self.encoder(c2)
        res2 = self.decoder(res1)

        res2 = res2 / res2.max() # 归一化
        res2 = self.soft(res2)

        res = res2.permute(0,2,1)
        att = res.view(b,c,w,h)
        y = res.mean(dim=2)
        orderY = torch.sort(y, dim=-1, descending=True, out=None)  # 排序
        y = orderY[0]  # 排序的方向求导机制是如何的，会对y造成影响吗
        # print ('orderY = ',orderY)
        y = y.view(b, c, 1, 1) # [1,16,1,1]
        att = y.expand_as(x) # [1,16,127,127]
        res = x.mul(att) # [1,16,127,127]
        return res,orderY


class Channel_attention_net11(nn.Module):

    def __init__(self, channel=16, reduction=4):
        super(Channel_attention_net11, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
                ('encoder1', nn.Sequential(nn.Linear(channel, channel//2,bias=True),
                                        nn.ReLU(inplace=False))),
                ('encoder2', nn.Sequential(nn.Linear(channel//2, channel//4,bias=True)))]))

        self.decoder = nn.Sequential(OrderedDict([
                ('decoder1', nn.Sequential(nn.Linear(channel//4, channel//2,bias=True),
                                        nn.ReLU(inplace=False))),
                ('decoder2', nn.Sequential(nn.Linear(channel//2, channel,bias=True)))]))
        # self.soft = nn.Softmax(dim=-1)
        # self.encoder = nn.Sequential(nn.Linear(channel, channel//2,bias=True),
        #                              nn.ReLU(inplace=False),
        #                              nn.Linear(channel//2, channel//4,bias=True))

        # self.decoder = nn.Sequential(nn.Linear(channel//4, channel//2,bias=True),
        #                              nn.ReLU(inplace=False),
        #                              nn.Linear(channel//2, channel,bias=True)
        #                              )
        self.soft = nn.ModuleList([nn.Sequential(nn.Softmax(dim=-1))])
        # self.soft = nn.Softmax(dim=-1)

    def forward(self, x):  # return 16 bands point-mul result
        b, c, w, h = x.size()# [1,16,127,127]
        print ('x = ',x)
        c1 = x.view(b,c,-1)
        c2 = c1.permute(0,2,1)
        for name, module in self.encoder.named_children():
            c2 = module(c2)
        for name, module in self.decoder.named_children():
            c2 = module(c2)
        # res1 = self.encoder(c2)
        # res2 = self.decoder(res1)
        res2 = c2
        res2 = res2 / res2.max() # 归一化
        # F.softmax(x, dim=-1)
        res2 = self.soft[0](res2)
        print ('res2 = ',res2)


        res = res2.permute(0,2,1)
        att = res.view(b,c,w,h)
        y = res.mean(dim=2)
        orderY = torch.sort(y, dim=-1, descending=True, out=None)  # 排序
        y = orderY[0]  # 排序的方向求导机制是如何的，会对y造成影响吗
        # print ('orderY = ',orderY)
        y = y.view(b, c, 1, 1) # [1,16,1,1]
        att = y.expand_as(x) # [1,16,127,127]
        res = x.mul(att) # [1,16,127,127]
        print ('--last x -- = ',x)
        return res,orderY


if __name__ == "__main__":

    model22 = Channel_attention_net11()
    model22.load_state_dict(torch.load('./models/channel_ca_convert.pth'))
    print ('model22 = ',model22)
    arr = np.random.randint(0,255,[4,16,107,107])
    # print (arr)
    tarr = torch.from_numpy(arr)
    tarr = tarr.float()
    res,orderY = model22(tarr)
    print ('res = ',res)
    print ('orderY = ',orderY)
    # model = Channel_attention_net()
    # model.load_state_dict(torch.load('models/tmp_models.pth'))
    # model11 = Channel_attention_net11()

    # arr = ['encoder.encoder1.0.weight','encoder.encoder1.0.bias','encoder.encoder2.0.weight','encoder.encoder2.0.bias','decoder.decoder1.0.weight','decoder.decoder1.0.bias','decoder.decoder2.0.weight','decoder.decoder2.0.bias']


    # model_dict = model11.state_dict()
    # # # 1. filter out unnecessary keys
    # cnt = 0
    # model_pretrain = {}
    # for k, v in model.named_parameters():
    #     if arr[cnt] in model_dict:
    #         model_pretrain[arr[cnt]]=v
    #         print ('arr[cnt] = ',arr[cnt])
    #         print ('model_pretrain[arr[cnt]] = ',model_pretrain[arr[cnt]])
    #     cnt += 1
    # # model_pretrain = {k: v for k, v in model11.named_parameters() if k in model_dict}
    # # # 2. overwrite entries in the existing state dict
    # model_dict.update(model_pretrain)
    # # # 3. load the new state dict
    # model11.load_state_dict(model_dict)
    # torch.save(model11.cpu().state_dict(), "./models/channel_ca_convert.pth")


